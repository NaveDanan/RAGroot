import json
import pickle
import os
from pathlib import Path
from typing import List, Dict
import logging

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .latex_utils import preprocess_for_embedding

logger = logging.getLogger(__name__)

class VectorIndexer:
    """Handles document indexing and vector search."""
    
    def __init__(self, index_dir: str = "index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure local-only mode for embedding models
        # IMPORTANT: Must be set BEFORE importing/loading models
        from .config import config
        
        # Set up cache directory
        cache_dir = Path(config.EMBEDDING_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure environment for offline operation
        if config.EMBEDDING_LOCAL_ONLY or config.SKIP_CHECK_ST_UPDATES:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
            logger.info(f"Local-only mode enabled for embeddings (cache: {cache_dir})")
        else:
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
            logger.info(f"Online mode - will download models if not cached (cache: {cache_dir})")
        
        # Determine device based on configuration
        if config.FORCE_CPU:
            device = 'cpu'
            logger.info("Force CPU mode enabled - using CPU for embeddings")
        else:
            # Let PyTorch automatically select the best available device
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Auto-detected device: {device}")
        
        # Initialize embedding model based on type
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        
        if config.EMBEDDING_MODEL.startswith("allenai/specter2"):
            # Use SPECTER2 with adapter-transformers (specialized for scientific papers)
            from .encoders import Specter2DualEncoder
            
            logger.info("Using SPECTER2 dual encoder setup:")
            logger.info("  - Documents: allenai/specter2 (proximity adapter)")
            logger.info("  - Queries: allenai/specter2_adhoc_query (query adapter)")
            
            try:
                self.encoder = Specter2DualEncoder(
                    base_model="allenai/specter2_base",
                    device=device,
                    max_length=512,
                    cache_dir=str(cache_dir),
                    local_files_only=config.EMBEDDING_LOCAL_ONLY
                )
                self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
                self.use_specter2 = True
                logger.info(f"Loaded SPECTER2 from local cache: {cache_dir}")
            except Exception as e:
                logger.error(f"Failed to load SPECTER2 model from local cache: {e}\")")
                if config.EMBEDDING_LOCAL_ONLY:
                    logger.error(f"""{'='*70}
❌ SPECTER2 MODEL NOT FOUND IN LOCAL CACHE
{'='*70}
Cache directory: {cache_dir}

To fix this:
1. Download the model using the download script:
   python tools/download_models.py

2. Or set EMBEDDING_LOCAL_ONLY=false in .env to allow downloads
{'='*70}""")
                raise
            
        else:
            # Use standard SentenceTransformer models
            try:
                # For sentence-transformers 2.5.1, local_files_only is set via environment variable
                self.encoder = SentenceTransformer(
                    config.EMBEDDING_MODEL,
                    device=device,
                    cache_folder=str(cache_dir)
                )
                self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
                self.use_specter2 = False
                logger.info(f"Loaded model from cache: {cache_dir}/{config.EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{config.EMBEDDING_MODEL}'.")
                logger.error(f"Error: {e}")
                if config.EMBEDDING_LOCAL_ONLY:
                    logger.error(f"""\n{'='*70}
❌ EMBEDDING MODEL NOT FOUND IN LOCAL CACHE
{'='*70}
Model: {config.EMBEDDING_MODEL}
Cache directory: {cache_dir}

To fix this:
1. Download the model using the download script:
   python tools/download_models.py

2. Or set EMBEDDING_LOCAL_ONLY=false in .env to allow downloads
{'='*70}""")
                raise
        
        logger.info(f"Embedding model loaded on {device} (dim={self.embedding_dim})")
        
        self.index = None
        self.documents = []
        self.embeddings = None
        self.device = device
    
    def build_index(self, data_path: str):
        """Build vector index from JSONL dataset."""
        logger.info(f"Reading dataset from {data_path}")
        
        self.documents = []
        abstracts = []
        
        # First pass: count total lines for progress bar
        with open(data_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        # Second pass: process documents with progress bar
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Loading documents", unit="docs"), 1):
                try:
                    doc = json.loads(line.strip())
                    
                    # Validate required fields
                    if 'id' not in doc or 'title' not in doc or 'abstract' not in doc:
                        logger.warning(f"Line {line_num}: Missing required fields, skipping")
                        continue
                    
                    # Store minimal info (id, title, authors for citations)
                    self.documents.append({
                        'id': doc['id'],
                        'title': doc['title'],
                        'abstract': doc['abstract'],
                        'authors': doc.get('authors', ''),
                        'submitter': doc.get('submitter', '')
                    })
                    
                    # Preprocess abstract to handle LaTeX math
                    processed_abstract = preprocess_for_embedding(doc['abstract'])
                    # Combine title, authors, and abstract for richer embedding
                    authors_text = doc.get('authors', '')
                    combined_text = f"Title: {doc['title']}. Authors: {authors_text}. Abstract: {processed_abstract}"
                    abstracts.append(combined_text)
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error, skipping - {e}")
                except Exception as e:
                    logger.warning(f"Line {line_num}: Unexpected error, skipping - {e}")
        
        logger.info(f"Loaded {len(self.documents)} documents")
        
        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        from .config import config
        batch_size = config.EMBEDDING_BATCH_SIZE
        all_embeddings = []
        
        if self.use_specter2:
            # SPECTER2: Use title + abstract format
            logger.info("Using SPECTER2 paper format (title + abstract)")
            num_batches = (len(self.documents) + batch_size - 1) // batch_size
            
            with tqdm(total=len(self.documents), desc="Encoding embeddings (SPECTER2)", unit="docs") as pbar:
                for i in range(0, len(self.documents), batch_size):
                    batch_docs = self.documents[i:i + batch_size]
                    titles = [doc['title'] for doc in batch_docs]
                    abstracts = [doc['abstract'] for doc in batch_docs]
                    
                    batch_embeddings = self.encoder.encode_papers(
                        titles=titles,
                        abstracts=abstracts,
                        batch_size=batch_size,
                        normalize=True,  # Normalize for cosine similarity
                        show_progress=False  # Disable internal progress bar
                    )
                    all_embeddings.append(batch_embeddings)
                    pbar.update(len(batch_docs))
        else:
            # Standard SentenceTransformer: Use combined text
            logger.info("Using combined text format")
            
            with tqdm(total=len(abstracts), desc="Encoding embeddings", unit="docs") as pbar:
                for i in range(0, len(abstracts), batch_size):
                    batch = abstracts[i:i + batch_size]
                    batch_embeddings = self.encoder.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # Normalize for cosine similarity
                    )
                    all_embeddings.append(batch_embeddings)
                    pbar.update(len(batch))
        
        self.embeddings = np.vstack(all_embeddings).astype('float32')
        logger.info(f"Generated embeddings: {self.embeddings.shape}")
        
        # Build FAISS index with Inner Product (cosine similarity on normalized vectors)
        logger.info("Building FAISS index with cosine similarity (IndexFlatIP)...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine
        self.index.add(self.embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        
        # Save index and documents
        self.save_index()
    
    def save_index(self):
        """Save index and documents to disk."""
        logger.info("Saving index to disk...")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        
        # Save documents
        with open(self.index_dir / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save embeddings (optional, for debugging)
        np.save(self.index_dir / "embeddings.npy", self.embeddings)
        
        # Save embedding model name for change detection
        from .config import config
        with open(self.index_dir / "embedding_model.txt", 'w') as f:
            f.write(config.EMBEDDING_MODEL)
        
        logger.info("Index saved successfully")
    
    def load_index(self):
        """Load existing index from disk."""
        logger.info("Loading index from disk...")
        
        # Load FAISS index
        index_path = self.index_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load documents
        with open(self.index_dir / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load embeddings
        embeddings_path = self.index_dir / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        
        logger.info(f"Index loaded: {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents with reranking."""
        if self.index is None:
            raise RuntimeError("Index not built or loaded")
        
        # Retrieve more candidates for reranking
        from .config import config
        retrieval_k = min(top_k * config.RERANK_MULTIPLIER, len(self.documents))
        
        # Preprocess query to handle LaTeX math
        processed_query = preprocess_for_embedding(query)
        
        # Encode query based on model type
        if self.use_specter2:
            # Use query encoder (ad-hoc adapter)
            query_embedding = self.encoder.encode_queries(
                [processed_query],
                normalize=True  # Normalize for cosine similarity
            ).astype('float32')
        else:
            # Standard SentenceTransformer
            query_embedding = self.encoder.encode(
                [processed_query],
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            ).astype('float32')
        
        # Search (returns cosine similarity scores when using IndexFlatIP)
        scores, indices = self.index.search(query_embedding, retrieval_k)
        
        # Get candidates
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                candidates.append({
                    'document': self.documents[idx],
                    'score': float(score),  # Cosine similarity score (higher is better)
                    'index': int(idx)
                })
        
        # Rerank by computing cross-attention scores between query and titles+abstracts
        reranked = self._rerank_results(query, candidates)
        
        # Return top_k after reranking
        return reranked[:top_k]
    
    def _rerank_results(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank candidates based on semantic similarity with titles."""
        if not candidates:
            return candidates
        
        # Preprocess query for reranking
        processed_query = preprocess_for_embedding(query)
        
        # Compute similarity scores based on model type
        if self.use_specter2:
            # Use query encoder for query
            query_emb = self.encoder.encode_queries([processed_query], normalize=True)
            # Use document encoder for candidates
            titles = [c['document']['title'] for c in candidates]
            abstracts = [preprocess_for_embedding(c['document']['abstract'][:500]) for c in candidates]
            candidate_embs = self.encoder.encode_papers(
                titles=titles,
                abstracts=abstracts,
                normalize=True
            )
        else:
            # Standard SentenceTransformer
            candidate_texts = [
                f"Title: {c['document']['title']}. Authors: {c['document'].get('authors', '')}. Abstract: {preprocess_for_embedding(c['document']['abstract'][:500])}"
                for c in candidates
            ]
            query_emb = self.encoder.encode(
                [processed_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            candidate_embs = self.encoder.encode(
                candidate_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        
        # Compute cosine similarities (dot product on normalized vectors)
        similarities = []
        for cand_emb in candidate_embs:
            sim = np.dot(query_emb[0], cand_emb)  # Cosine similarity on normalized vectors
            similarities.append(float(sim))
        
        # Update scores
        for i, cand in enumerate(candidates):
            cand['rerank_score'] = similarities[i]
            cand['original_score'] = cand['score']
        
        # Sort by rerank score (higher is better)
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked
    
    def get_index_size(self) -> dict:
        """Get index size statistics."""
        if self.index is None:
            return {"vectors": 0, "dimension": 0}
        
        return {
            "vectors": self.index.ntotal,
            "dimension": self.embedding_dim
        }

if __name__ == "__main__":
    """Build index when run as script."""
    from .config import config
    
    # Print configuration
    print("\n" + "="*70)
    print("📦 Building Vector Index")
    print("="*70)
    print(f"Data source:      {config.DATA_PATH}")
    print(f"Output directory: {config.INDEX_DIR}")
    print(f"Embedding model:  {config.EMBEDDING_MODEL}")
    print(f"Batch size:       {config.EMBEDDING_BATCH_SIZE}")
    print("="*70 + "\n")
    
    indexer = VectorIndexer(index_dir=config.INDEX_DIR)
    indexer.build_index(config.DATA_PATH)
    
    print("\n" + "="*70)
    print("✅ Index Built Successfully!")
    print("="*70)
    print(f"Total documents: {len(indexer.documents)}")
    print(f"Index size: {indexer.get_index_size()}")
    print("="*70 + "\n")
