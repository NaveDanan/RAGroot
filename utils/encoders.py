"""
Custom encoders for specialized embedding models.
Supports SPECTER2 with adapter-based loading for scientific papers.
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional

logger = logging.getLogger(__name__)


class Specter2Encoder:
    """
    Document encoder for SPECTER2 using adapter-transformers.
    
    SPECTER2 has multiple task-specific adapters:
    - 'allenai/specter2' (proximity): Best for document retrieval
    - 'allenai/specter2_adhoc_query': Best for ad-hoc free-text queries
    - 'allenai/specter2_classification': For classification tasks
    
    Architecture:
    - Base: allenai/specter2_base (BERT-like encoder)
    - Adapter: Task-specific adapter loaded on top
    - Output: Embeddings from [CLS] token (dimension depends on model)
    """
    
    def __init__(
        self,
        base_model: str = "allenai/specter2_base",
        adapter_name: str = "allenai/specter2",
        device: Optional[str] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        local_files_only: bool = True
    ):
        """
        Initialize SPECTER2 encoder.
        
        Args:
            base_model: Base transformer model (default: allenai/specter2_base)
            adapter_name: Task-specific adapter (default: allenai/specter2 for retrieval)
            device: Device to run on (cuda/cpu, auto-detected if None)
            max_length: Maximum sequence length (default: 512)
            cache_dir: Local cache directory for models
            local_files_only: If True, only use local cached models (offline mode)
        """
        try:
            from transformers import AutoTokenizer
            from adapters import AutoAdapterModel
        except ImportError:
            raise ImportError(
                "SPECTER2 requires adapter-transformers. Install with:\n"
                "  pip install adapter-transformers\n"
                "or:\n"
                "  uv pip install adapter-transformers"
            )
        
        logger.info(f"Loading SPECTER2 encoder...")
        logger.info(f"  Base model: {base_model}")
        logger.info(f"  Adapter: {adapter_name}")
        logger.info(f"  Local files only: {local_files_only}")
        if cache_dir:
            logger.info(f"  Cache directory: {cache_dir}")
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                cache_dir=cache_dir,
                local_files_only=local_files_only
            )
            self.model = AutoAdapterModel.from_pretrained(
                base_model,
                cache_dir=cache_dir,
                local_files_only=local_files_only
            )
            
            # Load and activate adapter
            logger.info(f"Loading adapter from local cache...")
            self.model.load_adapter(
                adapter_name,
                source="hf",
                set_active=True,
                local_files_only=local_files_only
            )
        except Exception as e:
            logger.error(f"Failed to load SPECTER2 model from local cache: {e}")
            if local_files_only:
                logger.error(f"""\n{'='*70}
❌ SPECTER2 MODEL NOT FOUND IN LOCAL CACHE
{'='*70}
Base model: {base_model}
Adapter: {adapter_name}
Cache directory: {cache_dir or 'default'}

To fix this:
1. Download the model using the download script:
   python tools/download_models.py

2. Or set EMBEDDING_LOCAL_ONLY=false in .env to allow downloads
{'='*70}""")
            raise
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Move model to device and set to eval mode
        self.model.eval()
        self.model.to(self.device)
        
        # Store config
        self.max_length = max_length
        self.sep_token = self.tokenizer.sep_token
        
        # Get batch size from config if available
        try:
            from .config import config as cfg
            self.default_batch_size = cfg.EMBEDDING_BATCH_SIZE
            logger.info(f"Using batch size from config: {self.default_batch_size}")
        except (ImportError, AttributeError):
            self.default_batch_size = 32
            logger.info(f"Using default batch size: {self.default_batch_size}")
        
        logger.info(f"SPECTER2 encoder loaded successfully on {self.device}")
        logger.info(f"  Embedding dimension: {self.model.config.hidden_size}")
        logger.info(f"  Max length: {max_length}")
    
    @torch.inference_mode()
    def encode_papers(
        self,
        titles: List[str],
        abstracts: List[str],
        batch_size: int = None,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode academic papers (title + abstract format).
        
        Args:
            titles: List of paper titles
            abstracts: List of paper abstracts
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            show_progress: Show progress bar
        
        Returns:
            numpy array of shape (len(titles), embedding_dim)
        """
        # Use config batch size if not provided
        if batch_size is None:
            batch_size = self.default_batch_size
        
        # Combine title and abstract with SEP token
        texts = [
            f"{title}{self.sep_token}{abstract or ''}"
            for title, abstract in zip(titles, abstracts)
        ]
        
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**encoded)
            
            # Extract [CLS] token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize if requested
            if normalize:
                cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            
            # Move to CPU and convert to numpy
            embeddings.append(cls_embeddings.cpu().numpy())
            
            if show_progress and (i // batch_size + 1) % 10 == 0:
                logger.info(f"  Encoded batch {i // batch_size + 1}/{num_batches}")
        
        return np.vstack(embeddings)
    
    @torch.inference_mode()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode plain text (e.g., queries).
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        # Use config batch size if not provided
        if batch_size is None:
            batch_size = self.default_batch_size
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**encoded)
            
            # Extract [CLS] token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize if requested
            if normalize:
                cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            
            # Move to CPU and convert to numpy
            embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_sentence_embedding_dimension(self) -> int:
        """Return embedding dimension (768 for SPECTER2)."""
        return 768


class Specter2DualEncoder:
    """
    Dual encoder setup for SPECTER2:
    - Document encoder: Uses 'allenai/specter2' (proximity adapter)
    - Query encoder: Uses 'allenai/specter2_adhoc_query' (query adapter)
    
    This is optimal for retrieval where queries and documents have different characteristics.
    """
    
    def __init__(
        self,
        base_model: str = "allenai/specter2_base",
        device: Optional[str] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        local_files_only: bool = True
    ):
        """
        Initialize dual encoder setup.
        
        Args:
            base_model: Base transformer model
            device: Device to run on
            max_length: Maximum sequence length
            cache_dir: Local cache directory for models
            local_files_only: If True, only use local cached models (offline mode)
        """
        logger.info("Initializing SPECTER2 Dual Encoder setup...")
        
        # Document encoder (proximity adapter)
        self.doc_encoder = Specter2Encoder(
            base_model=base_model,
            adapter_name="allenai/specter2",
            device=device,
            max_length=max_length,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        
        # Query encoder (ad-hoc query adapter)
        logger.info("Loading query encoder with ad-hoc adapter...")
        try:
            from transformers import AutoTokenizer
            from adapters import AutoAdapterModel
            
            self.query_tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                cache_dir=cache_dir,
                local_files_only=local_files_only
            )
            self.query_model = AutoAdapterModel.from_pretrained(
                base_model,
                cache_dir=cache_dir,
                local_files_only=local_files_only
            )
            self.query_model.load_adapter(
                "allenai/specter2_adhoc_query",
                source="hf",
                set_active=True,
                local_files_only=local_files_only
            )
            
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            
            self.query_model.eval()
            self.query_model.to(self.device)
            self.max_length = max_length
            
            # Get batch size from config if available
            try:
                from .config import config as cfg
                self.default_batch_size = cfg.EMBEDDING_BATCH_SIZE
            except (ImportError, AttributeError):
                self.default_batch_size = 32
            
            logger.info(f"Query encoder loaded successfully on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load query adapter, using document encoder for queries: {e}")
            self.query_model = None
    
    def encode_papers(self, titles: List[str], abstracts: List[str], **kwargs) -> np.ndarray:
        """Encode documents using proximity adapter."""
        return self.doc_encoder.encode_papers(titles, abstracts, **kwargs)
    
    @torch.inference_mode()
    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode queries using ad-hoc query adapter.
        
        Args:
            queries: List of query strings
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            numpy array of shape (len(queries), embedding_dim)
        """
        # Use config batch size if not provided
        if batch_size is None:
            batch_size = self.default_batch_size
        
        if self.query_model is None:
            # Fallback to document encoder
            return self.doc_encoder.encode_texts(queries, batch_size, normalize)
        
        embeddings = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            # Tokenize
            encoded = self.query_tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            outputs = self.query_model(**encoded)
            
            # Extract [CLS] token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize if requested
            if normalize:
                cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            
            # Move to CPU and convert to numpy
            embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_sentence_embedding_dimension(self) -> int:
        """Return embedding dimension (768 for SPECTER2)."""
        return 768
