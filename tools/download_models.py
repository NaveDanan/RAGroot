"""
Download all required models for offline operation.
This script downloads embedding models and image generation models to local cache.

Usage:
    python tools/download_models.py [--embedding-only] [--image-only] [--all]
"""
import os
import sys
import argparse
from pathlib import Path
import logging

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_sentence_transformer(model_name: str, cache_dir: str):
    """Download a sentence-transformers model to local cache."""
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Downloading SentenceTransformer model: {model_name}")
        logger.info(f"Cache directory: {cache_dir}")
        
        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Temporarily disable offline mode to allow download
        offline_mode = os.environ.get('HF_HUB_OFFLINE')
        transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE')
        if offline_mode:
            del os.environ['HF_HUB_OFFLINE']
        if transformers_offline:
            del os.environ['TRANSFORMERS_OFFLINE']
        
        # Download model
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        
        # Restore offline mode
        if offline_mode:
            os.environ['HF_HUB_OFFLINE'] = offline_mode
        if transformers_offline:
            os.environ['TRANSFORMERS_OFFLINE'] = transformers_offline
        
        logger.info(f"✅ Successfully downloaded: {model_name}")
        logger.info(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        return False


def download_specter2(cache_dir: str):
    """Download SPECTER2 models (base model and adapters)."""
    try:
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel
        
        logger.info("Downloading SPECTER2 models and adapters...")
        logger.info(f"Cache directory: {cache_dir}")
        
        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Temporarily disable offline mode
        offline_mode = os.environ.get('HF_HUB_OFFLINE')
        transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE')
        if offline_mode:
            del os.environ['HF_HUB_OFFLINE']
        if transformers_offline:
            del os.environ['TRANSFORMERS_OFFLINE']
        
        base_model = "allenai/specter2_base"
        adapters = [
            "allenai/specter2",  # Proximity adapter for documents
            "allenai/specter2_adhoc_query"  # Query adapter
        ]
        
        # Download base model and tokenizer
        logger.info(f"  Downloading base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
        model = AutoAdapterModel.from_pretrained(base_model, cache_dir=cache_dir)
        logger.info(f"  ✅ Base model downloaded")
        
        # Download adapters
        for adapter_name in adapters:
            logger.info(f"  Downloading adapter: {adapter_name}")
            model.load_adapter(adapter_name, source="hf")
            logger.info(f"  ✅ Adapter downloaded: {adapter_name}")
        
        # Restore offline mode
        if offline_mode:
            os.environ['HF_HUB_OFFLINE'] = offline_mode
        if transformers_offline:
            os.environ['TRANSFORMERS_OFFLINE'] = transformers_offline
        
        logger.info(f"✅ Successfully downloaded SPECTER2 models")
        return True
    except ImportError:
        logger.error("❌ adapter-transformers not installed. Install with:")
        logger.error("   pip install adapter-transformers")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to download SPECTER2: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def download_image_model(model_name: str, cache_dir: str):
    """Download image generation model to local cache."""
    try:
        from diffusers import AutoPipelineForText2Image
        import torch
        
        logger.info(f"Downloading image generation model: {model_name}")
        logger.info(f"Cache directory: {cache_dir}")
        
        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Temporarily disable offline mode
        offline_mode = os.environ.get('HF_HUB_OFFLINE')
        transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE')
        if offline_mode:
            del os.environ['HF_HUB_OFFLINE']
        if transformers_offline:
            del os.environ['TRANSFORMERS_OFFLINE']
        
        # Download model
        logger.info("  This may take several minutes depending on your connection...")
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir
        )
        
        # Restore offline mode
        if offline_mode:
            os.environ['HF_HUB_OFFLINE'] = offline_mode
        if transformers_offline:
            os.environ['TRANSFORMERS_OFFLINE'] = transformers_offline
        
        logger.info(f"✅ Successfully downloaded: {model_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download models for offline RAG operation"
    )
    parser.add_argument(
        "--embedding-only",
        action="store_true",
        help="Download only embedding models"
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Download only image generation models"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models (default)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override embedding model from config"
    )
    parser.add_argument(
        "--embedding-cache",
        type=str,
        default=None,
        help="Override embedding cache directory from config"
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default=None,
        help="Override image model from config"
    )
    parser.add_argument(
        "--image-cache",
        type=str,
        default=None,
        help="Override image cache directory from config"
    )
    
    args = parser.parse_args()
    
    # Default to --all if no specific flag is set
    if not (args.embedding_only or args.image_only):
        args.all = True
    
    print("\n" + "="*70)
    print("📦 RAG Model Downloader")
    print("="*70)
    print("This script will download models for offline operation.")
    print("Models will be cached locally for faster startup and offline use.")
    print("="*70 + "\n")
    
    success = True
    
    # Download embedding models
    if args.all or args.embedding_only:
        print("\n" + "="*70)
        print("📚 DOWNLOADING EMBEDDING MODELS")
        print("="*70 + "\n")
        
        embedding_model = args.embedding_model or config.EMBEDDING_MODEL
        embedding_cache = args.embedding_cache or config.EMBEDDING_CACHE_DIR
        
        if embedding_model.startswith("allenai/specter2"):
            # Download SPECTER2
            if not download_specter2(embedding_cache):
                success = False
        else:
            # Download standard sentence-transformers model
            if not download_sentence_transformer(embedding_model, embedding_cache):
                success = False
    
    # Download image generation models
    if args.all or args.image_only:
        print("\n" + "="*70)
        print("🎨 DOWNLOADING IMAGE GENERATION MODELS")
        print("="*70 + "\n")
        
        if config.IMAGE_API_PROVIDER == "local":
            image_model = args.image_model or config.IMAGE_MODEL_NAME
            image_cache = args.image_cache or config.IMAGE_MODEL_PATH
            
            if not download_image_model(image_model, image_cache):
                success = False
        else:
            logger.info(f"Skipping image model download (provider: {config.IMAGE_API_PROVIDER})")
    
    # Summary
    print("\n" + "="*70)
    if success:
        print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY")
        print("="*70)
        print("\nYour system is now configured for offline operation!")
        print("\nNext steps:")
        print("  1. Ensure EMBEDDING_LOCAL_ONLY=true in your .env file")
        print("  2. Set HF_HUB_OFFLINE=1 in your environment if desired")
        print("  3. Run your application: python main.py")
    else:
        print("❌ SOME MODELS FAILED TO DOWNLOAD")
        print("="*70)
        print("\nPlease check the errors above and try again.")
        print("Make sure you have:")
        print("  - Internet connection")
        print("  - Sufficient disk space")
        print("  - Required Python packages installed")
    print("="*70 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
