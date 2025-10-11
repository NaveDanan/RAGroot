"""
Validation script to verify offline/on-premise setup.
This script checks if all required models are available locally.

Usage:
    python tools/validate_offline.py
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force offline mode for validation
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from utils.config import config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check environment variables."""
    print("\n" + "="*70)
    print("🔍 CHECKING ENVIRONMENT")
    print("="*70)
    
    checks = {
        "HF_HUB_OFFLINE": os.environ.get('HF_HUB_OFFLINE'),
        "TRANSFORMERS_OFFLINE": os.environ.get('TRANSFORMERS_OFFLINE'),
        "EMBEDDING_LOCAL_ONLY": config.EMBEDDING_LOCAL_ONLY,
        "SKIP_CHECK_ST_UPDATES": config.SKIP_CHECK_ST_UPDATES,
    }
    
    for key, value in checks.items():
        status = "✅" if value in ['1', True] else "⚠️"
        print(f"  {status} {key}: {value}")
    
    return all(v in ['1', True] for v in checks.values())


def check_directories():
    """Check if required directories exist."""
    print("\n" + "="*70)
    print("📁 CHECKING DIRECTORIES")
    print("="*70)
    
    dirs = {
        "Data": config.DATA_PATH,
        "Index": config.INDEX_DIR,
        "LLM Model": Path(config.MODEL_PATH).parent,
        "Embedding Cache": config.EMBEDDING_CACHE_DIR,
    }
    
    all_exist = True
    for name, path in dirs.items():
        path = Path(path)
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path} {'(exists)' if exists else '(missing)'}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_embedding_model():
    """Check if embedding model is available locally."""
    print("\n" + "="*70)
    print("🔤 CHECKING EMBEDDING MODEL")
    print("="*70)
    
    print(f"  Model: {config.EMBEDDING_MODEL}")
    print(f"  Cache: {config.EMBEDDING_CACHE_DIR}")
    print(f"  Local only: {config.EMBEDDING_LOCAL_ONLY}")
    
    try:
        from utils.indexer import VectorIndexer
        
        print("  Loading model...")
        indexer = VectorIndexer(index_dir=config.INDEX_DIR)
        
        dim = indexer.embedding_dim
        print(f"  ✅ Model loaded successfully")
        print(f"  ✅ Embedding dimension: {dim}")
        print(f"  ✅ Device: {indexer.device}")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed to load embedding model: {e}")
        return False


def check_llm_model():
    """Check if LLM model exists."""
    print("\n" + "="*70)
    print("🤖 CHECKING LLM MODEL")
    print("="*70)
    
    model_path = Path(config.MODEL_PATH)
    print(f"  Path: {model_path}")
    
    if not model_path.exists():
        print(f"  ❌ LLM model not found")
        return False
    
    size_gb = model_path.stat().st_size / (1024**3)
    print(f"  ✅ Model exists ({size_gb:.2f} GB)")
    
    return True


def check_vector_index():
    """Check if vector index exists."""
    print("\n" + "="*70)
    print("📊 CHECKING VECTOR INDEX")
    print("="*70)
    
    index_dir = Path(config.INDEX_DIR)
    index_file = index_dir / "faiss.index"
    docs_file = index_dir / "documents.pkl"
    
    print(f"  Index directory: {index_dir}")
    
    if not index_file.exists():
        print(f"  ⚠️  Index not found: {index_file}")
        print(f"     Run: python -m utils.indexer")
        return False
    
    if not docs_file.exists():
        print(f"  ❌ Documents not found: {docs_file}")
        return False
    
    print(f"  ✅ Index exists: {index_file}")
    print(f"  ✅ Documents exist: {docs_file}")
    
    # Try loading and testing
    try:
        from utils.indexer import VectorIndexer
        
        indexer = VectorIndexer(index_dir=str(index_dir))
        indexer.load_index()
        
        info = indexer.get_index_size()
        print(f"  ✅ Index loaded successfully")
        print(f"     Vectors: {info['vectors']:,}")
        print(f"     Dimension: {info['dimension']}")
        
        # Test search
        results = indexer.search("machine learning", top_k=3)
        print(f"  ✅ Search test passed ({len(results)} results)")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed to load/test index: {e}")
        return False


def check_image_model():
    """Check if image generation model is available (optional)."""
    print("\n" + "="*70)
    print("🎨 CHECKING IMAGE GENERATION (OPTIONAL)")
    print("="*70)
    
    print(f"  Provider: {config.IMAGE_API_PROVIDER}")
    
    if config.IMAGE_API_PROVIDER != "local":
        print(f"  ℹ️  Using external provider: {config.IMAGE_API_PROVIDER}")
        return True
    
    print(f"  Model: {config.IMAGE_MODEL_NAME}")
    print(f"  Cache: {config.IMAGE_MODEL_PATH}")
    
    model_cache = Path(config.IMAGE_MODEL_PATH)
    if not model_cache.exists():
        print(f"  ⚠️  Image model cache not found")
        print(f"     Run: python tools/download_models.py --image-only")
        return False
    
    # Check if model files exist
    model_files = list(model_cache.rglob("*.safetensors")) + list(model_cache.rglob("*.bin"))
    if not model_files:
        print(f"  ⚠️  No model files found in cache")
        return False
    
    print(f"  ✅ Image model cache exists ({len(model_files)} model files)")
    return True


def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("🔧 RAG OFFLINE SETUP VALIDATION")
    print("="*70)
    print("This script validates that all models are available locally")
    print("for offline/on-premise operation.")
    print("="*70)
    
    checks = [
        ("Environment", check_environment),
        ("Directories", check_directories),
        ("Embedding Model", check_embedding_model),
        ("LLM Model", check_llm_model),
        ("Vector Index", check_vector_index),
        ("Image Model", check_image_model),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n  ❌ Unexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("📋 VALIDATION SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        optional = " (optional)" if name == "Image Model" else ""
        print(f"  {status}: {name}{optional}")
    
    # Count required checks (exclude image model)
    required_checks = {k: v for k, v in results.items() if k != "Image Model"}
    passed = sum(required_checks.values())
    total = len(required_checks)
    
    print("="*70)
    
    if passed == total:
        print("✅ ALL REQUIRED CHECKS PASSED")
        print("\n🎉 Your system is ready for offline operation!")
        print("\nNext steps:")
        print("  1. Start the server: python main.py")
        print("  2. Access the UI: http://localhost:8080")
        print("="*70 + "\n")
        return 0
    else:
        print(f"❌ {total - passed} CHECKS FAILED")
        print("\n⚠️  Your system is NOT ready for offline operation.")
        print("\nTo fix:")
        print("  1. Review the failed checks above")
        print("  2. Run: python tools/download_models.py --all")
        print("  3. Build index: python -m utils.indexer")
        print("  4. Run this validation again")
        print("\nSee Documentation/OFFLINE_SETUP.md for detailed instructions.")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
