import os
import logging
from typing import Optional
import aiohttp
import asyncio
from pathlib import Path
import base64
from io import BytesIO
import re
from .config import config
logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles image generation using local or external APIs."""
    
    def __init__(self):
        # Support multiple image generation APIs
        self.api_key = config.IMAGE_API_KEY
        self.api_provider = config.IMAGE_API_PROVIDER

        # Pollinations.ai is free and doesn't require API key
        self.pollinations_base = "https://image.pollinations.ai/prompt/"
        
        # OpenAI Compatible configuration (if API key provided)
        self.openai_base = "https://api.openai.com/v1/images/generations"
        
        # Local model configuration
        self.model_path = config.IMAGE_MODEL_PATH
        self.local_model_name = config.IMAGE_MODEL_NAME

        # Lazy loading for local model
        self.local_model = None
        self.local_processor = None
        self.compel = None  # For long prompt support (>77 tokens)
        self.device = None
        
        logger.info(f"Image generator initialized with provider: {self.api_provider}")
    
    def _initialize_local_model(self) -> None:
        """Initialize local diffusion model for image generation."""
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
            
            
            # Determine device based on FORCE_CPU setting
            if config.FORCE_CPU:
                self.device = "cpu"
                logger.info("Force CPU mode enabled - image generation will use CPU")
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device for image generation: {self.device}")
            
            logger.info(f"Loading local image generation model: {self.local_model_name}")
            logger.info("This may take a few minutes on first run...")
            
            # Load SDXL Turbo pipeline using AutoPipeline for automatic pipeline detection
            # Check if we should use local-only mode
            local_only = config.EMBEDDING_LOCAL_ONLY  # Use same setting as embeddings for consistency
            
            logger.info(f"Local files only mode: {local_only}")
            
            try:
                self.local_model = AutoPipelineForText2Image.from_pretrained(
                    self.local_model_name,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    cache_dir=self.model_path,
                    local_files_only=local_only
                )
            except Exception as e:
                logger.error(f"Failed to load image model from local cache: {e}")
                if local_only:
                    logger.error(f"""
{'='*70}
❌ IMAGE MODEL NOT FOUND IN LOCAL CACHE
{'='*70}
Model: {self.local_model_name}
Cache directory: {self.model_path}

To fix this:
1. Download the model using the download script:
   python tools/download_models.py

2. Or set EMBEDDING_LOCAL_ONLY=false in .env to allow downloads
{'='*70}""")
                raise
            
            # Move model to GPU
            self.local_model = self.local_model.to(self.device)
            
            # Initialize Compel for long prompt support
            from compel import Compel, ReturnedEmbeddingsType
            self.compel = Compel(
                tokenizer=[self.local_model.tokenizer, self.local_model.tokenizer_2],
                text_encoder=[self.local_model.text_encoder, self.local_model.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            logger.info("Compel initialized - long prompts (>77 tokens)")
            
            logger.info("Local image generation model initialized successfully!")
        
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def generate_image(self, prompt: str, query: Optional[str] = None, 
                           provider_override: Optional[str] = None,
                           api_key_override: Optional[str] = None) -> Optional[str]:
        """Generate an image based on the answer/query.
        
        Args:
            prompt: The text prompt for image generation
            query: Optional original query for context
            provider_override: Override the configured image provider (local, openai, pollinations)
            api_key_override: Override the configured API key (for OpenAI)
        """
        
        # Use overrides if provided, otherwise use configured values
        provider = provider_override if provider_override is not None else self.api_provider
        api_key = api_key_override if api_key_override is not None else self.api_key
        
        logger.info(f"Generating image with provider: {provider}")
        
        if provider == "local":
            return await self._generate_local(prompt, query)
        elif provider == "pollinations":
            return await self._generate_pollinations(prompt, query)
        elif provider == "openai":
            if not api_key:
                logger.error("OpenAI provider selected but no API key provided")
                return None
            return await self._generate_openai(prompt, api_key)
        else:
            logger.warning(f"Unknown image provider: {provider}")
            return None

    async def _generate_local(self, prompt: str, query: Optional[str] = None) -> Optional[str]:
        """Generate image using local SDXL Turbo model."""
        try:
            # Initialize model if not already done
            if self.local_model is None:
                self._initialize_local_model()
            
            import torch
            from PIL import Image
            
            # Create a concise image prompt
            image_prompt = self._create_image_prompt(prompt, query)
            logger.info(f"Prompt: {image_prompt}")
            
            with torch.no_grad():
                # Encode prompt with Compel (supports >77 tokens)
                conditioning, pooled = self.compel(image_prompt)
                result = self.local_model(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    num_inference_steps=config.IMAGE_INFERENCE_STEPS,
                    guidance_scale=config.IMAGE_GUIDANCE_SCALE,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
                logger.info(f"Generated with Compel (supports long prompts)")
            
            # Extract the generated image
            generated_image = result.images[0]
            
            # Save to temporary directory
            output_dir = Path("static/generated_images")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            import time
            filename = f"gen_{int(time.time() * 1000)}.png"
            output_path = output_dir / filename
            
            # Save the generated image
            generated_image.save(output_path)
            logger.info(f"Image saved to: {output_path}")
            
            # Convert to base64 data URL for embedding in response
            buffered = BytesIO()
            generated_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            data_url = f"data:image/png;base64,{img_str}"
            
            return data_url
        
        except Exception as e:
            logger.error(f"Local image generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


    
    async def _generate_pollinations(self, prompt: str, query: Optional[str] = None) -> Optional[str]:
        """Generate image using Pollinations.ai (free, no API key needed)."""
        try:
            # Create a concise image prompt
            image_prompt = self._create_image_prompt(prompt, query)
            
            # URL encode the prompt
            import urllib.parse
            encoded_prompt = urllib.parse.quote(image_prompt)
            
            # Pollinations.ai generates images via URL
            image_url = f"{self.pollinations_base}{encoded_prompt}"
            
            logger.info(f"Generated Pollinations.ai image URL")
            return image_url
        
        except Exception as e:
            logger.error(f"Pollinations.ai image generation error: {e}")
            return None
    
    async def _generate_openai(self, prompt: str, api_key: str) -> Optional[str]:
        """Generate image using OpenAI DALL-E.
        
        Args:
            prompt: The text prompt for image generation
            api_key: OpenAI API key (passed as parameter to support runtime override)
        """
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "dall-e-3",
                "prompt": self._create_image_prompt(prompt),
                "n": 1,
                "size": "1024x1024"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.openai_base,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        image_url = result['data'][0]['url']
                        logger.info("OpenAI image generated successfully")
                        return image_url
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        return None
        
        except Exception as e:
            logger.error(f"OpenAI image generation error: {e}")
            return None
    
    def _create_image_prompt(self, answer: str, query: Optional[str] = None) -> str:
        """Build a detailed structured prompt for image generation using Compel.
        Supports prompts longer than 77 tokens.
        """
        
        def clean_text(text: str) -> str:
            text = re.sub(r"\[[^\]]*\]", "", text)  # drop citations like [1]
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        
        base_text = " ".join(part for part in (query, answer) if part)
        cleaned = clean_text(base_text)
        if not cleaned:
            return "Clean digital illustration of a scientific concept, sharp focus, white background"
        
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        concept_sentence = ""
        supporting_details = []
        
        # Extract main concept and supporting details
        for i, sentence in enumerate(sentences):
            stripped = sentence.strip().rstrip(".")
            if not stripped:
                continue
            
            if not concept_sentence:
                # Main concept from first sentence (can use longer sentences with Compel)
                concept_sentence = stripped if len(stripped.split()) <= 20 else " ".join(stripped.split()[:20])
            elif i < 3 and len(stripped.split()) > 3:
                # Add supporting details
                supporting_details.append(stripped)
        
        if not concept_sentence:
            concept_sentence = "scientific concept"
        
        # Extract keywords
        stopwords = {
            "the", "and", "with", "that", "from", "into", "their", "about", "this",
            "which", "using", "through", "such", "also", "have", "will", "over",
            "between", "abstract", "there", "these", "those", "for", "under", "each",
            "when", "where", "while", "within", "being", "very", "more", "less",
            "than", "onto", "because", "however", "among", "other", "others", "often",
            "helps", "makes", "made", "make", "like", "including", "include",
            "includes", "can", "may", "might", "different", "single", "multiple",
            "many", "most", "some", "any", "per", "via"
        }
        
        keywords = []
        seen = set()
        max_keywords = 8  # Compel supports longer prompts
        
        for token in re.findall(r"[A-Za-z0-9\-]+", cleaned.lower()):
            if len(token) < 3 or token in stopwords:
                continue
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= max_keywords:
                break
        
        # Build detailed prompt with Compel
        prompt_parts = [f"Detailed scientific illustration with no text labels showing {concept_sentence}."]
        
        if supporting_details:
            prompt_parts.append(f"The image should emphasize: {supporting_details[0]}.")
        
        if keywords:
            key_str = ", ".join(keywords[:8])
            prompt_parts.append(f"Key visual elements: {key_str}.")
        
        prompt_parts.append(
            "Style: high-quality digital art, precise technical illustration, "
            "clean composition, sharp focus, soft studio lighting, "
            "professional scientific visualization, detailed rendering, "
            "white or light gray background, no text labels, no watermarks."
        )
        
        prompt = " ".join(prompt_parts)
        logger.info(f"Generated detailed prompt with Compel ({len(prompt)} chars, ~{len(prompt.split())} words)")
        
        return prompt
    
    async def test_connection(self) -> bool:
        """Test if image generation is working."""
        try:
            if self.api_provider == "local":
                # Test local model initialization
                self._initialize_local_model()
                return self.local_model is not None
            elif self.api_provider == "pollinations":
                # Pollinations always works (URL-based)
                return True
            elif self.api_provider == "openai":
                # Test OpenAI connection
                return bool(self.api_key)
            return False
        except Exception:
            return False
