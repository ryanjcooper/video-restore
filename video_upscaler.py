#!/usr/bin/env python3
# video_upscaler.py
"""
Video Restore - AI-Powered Video Upscaler with Multi-GPU Support
Enhanced version with advanced artifact reduction techniques
"""

# Suppress all warnings and set logging levels before any imports
import warnings
warnings.filterwarnings("ignore")

import logging
# Suppress logging from various libraries
logging.getLogger('basicsr').setLevel(logging.ERROR)
logging.getLogger('realesrgan').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.ERROR)

# IMPORTANT: Monkey patch for torchvision compatibility MUST come first
import sys
import types

# Create a compatibility layer for deprecated torchvision functional_tensor
fake_module = types.ModuleType('functional_tensor')

# Dynamically redirect all function calls to torchvision.transforms.functional
def __getattr__(name):
    import torchvision.transforms.functional as F
    if hasattr(F, name):
        return getattr(F, name)
    raise AttributeError(f"module 'functional_tensor' has no attribute '{name}'")

fake_module.__getattr__ = __getattr__

# Pre-import torchvision.transforms.functional and copy common functions
try:
    import torchvision.transforms.functional as F
    for func_name in ['rgb_to_grayscale', 'adjust_brightness', 'adjust_contrast', 
                      'adjust_saturation', 'adjust_hue', 'normalize', 'resize',
                      'pad', 'crop', 'center_crop', 'resized_crop', 'hflip', 'vflip',
                      'rotate', 'affine', 'to_tensor', 'to_pil_image', 'to_grayscale']:
        if hasattr(F, func_name):
            setattr(fake_module, func_name, getattr(F, func_name))
except ImportError:
    pass  # Will handle later in main imports

# Register the fake module
sys.modules['torchvision.transforms.functional_tensor'] = fake_module

# Now continue with regular imports
import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
import json
import warnings
import contextlib
import io
import subprocess
from scipy import ndimage
from skimage import filters, restoration, exposure
import torch.nn.functional as F

# Suppress deprecation warnings from torchvision
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*functional.*")

# Third-party imports (install via pip)
try:
    # Import with version compatibility handling
    import torch
    import torchvision
    
    # Silently check versions
    torch_version = torch.__version__
    torchvision_version = torchvision.__version__
    
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    import ffmpeg
    
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with the following commands:")
    print("pip install torch torchvision torchaudio")
    print("pip install basicsr realesrgan opencv-python ffmpeg-python")
    print("pip install scipy scikit-image")
    sys.exit(1)

@dataclass
class ProcessingConfig:
    """Enhanced configuration for video processing with artifact reduction"""
    model_name: str = "RealESRGAN_x4plus"
    scale: int = 4
    tile_size: int = 512  # Default tile size, auto-adjusted based on VRAM
    tile_pad: int = 64    # Increased padding for better seamless processing
    tile_overlap: int = 32  # Overlap between tiles for seamless blending
    pre_pad: int = 10     # Increased pre-padding
    face_enhance: bool = False
    gpu_ids: List[int] = None
    output_format: str = "mp4"
    crf: int = 15  # Higher quality encoding (was 18)
    preset: str = "slower"  # Better compression (was slow)
    audio_copy: bool = True
    auto_tile_size: bool = True
    
    # Advanced artifact reduction options
    denoise_strength: float = 0.1  # Post-processing denoising
    sharpen_strength: float = 0.0  # Subtle sharpening
    color_correction: bool = True  # Enhanced color processing
    temporal_consistency: bool = True  # Frame-to-frame consistency checks
    seamless_tiles: bool = True  # Advanced tile blending
    advanced_postprocess: bool = True  # Additional post-processing
    preserve_details: bool = True  # Detail preservation mode
    
    # Model-specific optimizations
    model_specific_settings: bool = True
    
    def __post_init__(self):
        if self.gpu_ids is None:
            # Auto-detect available GPUs
            self.gpu_ids = list(range(torch.cuda.device_count()))
            if not self.gpu_ids:
                self.gpu_ids = [0]  # Fallback to single GPU

class EnhancedModelManager:
    """Enhanced model manager with artifact reduction techniques"""
    
    MODELS = {
        "RealESRGAN_x4plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "scale": 4,
            "arch": RRDBNet,
            "arch_params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 4},
            "optimal_tile": 1024,  # Optimal tile size for this model
            "padding": 64
        },
        "RealESRGAN_x2plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "scale": 2,
            "arch": RRDBNet,
            "arch_params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 2},
            "optimal_tile": 1024,
            "padding": 48
        },
        "RealESRGAN_x4plus_anime_6B": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "scale": 4,
            "arch": RRDBNet,
            "arch_params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 6, "num_grow_ch": 32, "scale": 4},
            "optimal_tile": 768,  # Smaller optimal for anime model
            "padding": 48
        },
        "RealESRGAN_x4_v3": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "scale": 4,
            "arch": SRVGGNetCompact,
            "arch_params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_conv": 32, "upsampling": 4, "act_type": "prelu"},
            "optimal_tile": 1024,
            "padding": 32
        }
    }
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.loaded_models: Dict[str, RealESRGANer] = {}
    
    def download_model(self, model_name: str) -> Path:
        """Download model if not exists"""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(self.MODELS.keys())}")
        
        model_path = self.model_dir / f"{model_name}.pth"
        if not model_path.exists():
            print(f"Downloading {model_name} model (this may take a few minutes)...")
            # Suppress verbose output from download_util
            with contextlib.redirect_stdout(io.StringIO()):
                load_file_from_url(
                    self.MODELS[model_name]["url"],
                    model_dir=str(self.model_dir),
                    file_name=f"{model_name}.pth"
                )
            print(f"✓ Model downloaded successfully")
        return model_path
    
    def get_optimal_settings(self, model_name: str, available_vram_gb: float) -> Tuple[int, int]:
        """Get optimal tile size and padding for model and VRAM"""
        model_info = self.MODELS[model_name]
        optimal_tile = model_info["optimal_tile"]
        optimal_padding = model_info["padding"]
        
        # Adjust based on available VRAM
        if available_vram_gb < 8:
            optimal_tile = min(optimal_tile, 512)
        elif available_vram_gb < 16:
            optimal_tile = min(optimal_tile, 768)
        # With 24GB VRAM, use optimal settings
        
        return optimal_tile, optimal_padding
    
    def load_model(self, model_name: str, gpu_id: int = 0, tile_size: int = 512, tile_pad: int = 64) -> RealESRGANer:
        """Load and cache model on specified GPU with enhanced settings"""
        cache_key = f"{model_name}_gpu{gpu_id}_{tile_size}_{tile_pad}"
        
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        model_path = self.download_model(model_name)
        model_info = self.MODELS[model_name]
        
        # Initialize model architecture
        model = model_info["arch"](**model_info["arch_params"])
        
        # Create upsampler with enhanced settings
        upsampler = RealESRGANer(
            scale=model_info["scale"],
            model_path=str(model_path),
            model=model,
            tile=tile_size,
            tile_pad=tile_pad,
            pre_pad=10,  # Increased pre-padding
            half=True,  # Use FP16 for memory efficiency
            gpu_id=gpu_id,
            device=torch.device(f'cuda:{gpu_id}')
        )
        
        self.loaded_models[cache_key] = upsampler
        return upsampler

class AdvancedImageProcessor:
    """Advanced image processing for artifact reduction"""
    
    @staticmethod
    def seamless_tile_blend(tiles: List[np.ndarray], positions: List[Tuple[int, int]], 
                           output_shape: Tuple[int, int], overlap: int = 32) -> np.ndarray:
        """Blend overlapping tiles seamlessly"""
        h, w = output_shape[:2]
        channels = tiles[0].shape[2] if len(tiles[0].shape) == 3 else 1
        
        if channels == 1:
            output = np.zeros((h, w), dtype=np.float32)
            weight_map = np.zeros((h, w), dtype=np.float32)
        else:
            output = np.zeros((h, w, channels), dtype=np.float32)
            weight_map = np.zeros((h, w), dtype=np.float32)
        
        for tile, (y, x) in zip(tiles, positions):
            th, tw = tile.shape[:2]
            
            # Create weight map for this tile (higher in center, lower at edges)
            tile_weights = np.ones((th, tw), dtype=np.float32)
            
            # Apply Gaussian-like weighting for smooth blending
            for i in range(overlap):
                weight = (i + 1) / overlap
                # Top edge
                if i < th:
                    tile_weights[i, :] *= weight
                # Bottom edge
                if th - 1 - i >= 0:
                    tile_weights[th - 1 - i, :] *= weight
                # Left edge
                if i < tw:
                    tile_weights[:, i] *= weight
                # Right edge
                if tw - 1 - i >= 0:
                    tile_weights[:, tw - 1 - i] *= weight
            
            # Extract valid region
            y_end = min(y + th, h)
            x_end = min(x + tw, w)
            tile_h = y_end - y
            tile_w = x_end - x
            
            if channels == 1:
                output[y:y_end, x:x_end] += tile[:tile_h, :tile_w] * tile_weights[:tile_h, :tile_w]
                weight_map[y:y_end, x:x_end] += tile_weights[:tile_h, :tile_w]
            else:
                for c in range(channels):
                    output[y:y_end, x:x_end, c] += tile[:tile_h, :tile_w, c] * tile_weights[:tile_h, :tile_w]
                weight_map[y:y_end, x:x_end] += tile_weights[:tile_h, :tile_w]
        
        # Normalize by weight map
        weight_map[weight_map == 0] = 1  # Avoid division by zero
        if channels == 1:
            output /= weight_map
        else:
            for c in range(channels):
                output[:, :, c] /= weight_map
        
        return output.astype(np.uint8)
    
    @staticmethod
    def enhance_details(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Enhance fine details using unsharp masking"""
        if strength <= 0:
            return image
        
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_float, (0, 0), 1.0)
        
        # Create unsharp mask
        unsharp = img_float + strength * (img_float - blurred)
        
        # Clip and convert back
        unsharp = np.clip(unsharp, 0, 1)
        return (unsharp * 255).astype(np.uint8)
    
    @staticmethod
    def reduce_noise(image: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Reduce noise while preserving edges"""
        if strength <= 0:
            return image
        
        # Use bilateral filter for edge-preserving denoising
        sigma_color = 75 * strength
        sigma_space = 75 * strength
        
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(image, -1, sigma_color, sigma_space)
        else:
            denoised = cv2.bilateralFilter(image, -1, sigma_color, sigma_space)
        
        return denoised
    
    @staticmethod
    def enhance_colors(image: np.ndarray) -> np.ndarray:
        """Enhance color saturation and contrast"""
        # Convert to LAB color space for better color manipulation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance L channel (lightness) with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Slightly enhance A and B channels (color)
        a = cv2.multiply(a, 1.05)
        b = cv2.multiply(b, 1.05)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    @staticmethod
    def temporal_consistency_check(prev_frame: np.ndarray, curr_frame: np.ndarray, 
                                 threshold: float = 0.8) -> np.ndarray:
        """Check temporal consistency and blend if needed"""
        if prev_frame is None:
            return curr_frame
        
        # Calculate frame difference
        diff = cv2.absdiff(prev_frame, curr_frame)
        mean_diff = np.mean(diff)
        
        # If frames are very different, might be a scene change
        if mean_diff > threshold * 255:
            return curr_frame
        
        # Gentle temporal blending for consistency
        alpha = 0.95  # Favor current frame
        blended = cv2.addWeighted(curr_frame, alpha, prev_frame, 1 - alpha, 0)
        
        return blended

class EnhancedVideoProcessor:
    """Enhanced video processor with advanced artifact reduction"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model_manager = EnhancedModelManager()
        self.image_processor = AdvancedImageProcessor()
        self.setup_logging()
        self.previous_frames = {}  # For temporal consistency
        
        # Validate GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - GPU required for video upscaling")
        
        available_gpus = torch.cuda.device_count()
        
        # Filter valid GPU IDs
        self.config.gpu_ids = [gpu_id for gpu_id in self.config.gpu_ids if gpu_id < available_gpus]
        
        if not self.config.gpu_ids:
            raise RuntimeError("No valid GPUs specified")
        
        # Auto-adjust settings based on GPU memory and model
        if self.config.auto_tile_size or self.config.model_specific_settings:
            self._optimize_settings()
        
        # Print enhanced GPU info
        print(f"\nEnhanced GPU Configuration")
        print(f"{'='*60}")
        print(f"Available GPUs: {available_gpus}")
        for gpu_id in self.config.gpu_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            vram_gb = props.total_memory / (1024**3)
            print(f"  GPU {gpu_id}: {props.name} ({vram_gb:.1f}GB)")
        print(f"Tile size: {self.config.tile_size}x{self.config.tile_size}")
        print(f"Tile padding: {self.config.tile_pad}")
        print(f"Tile overlap: {self.config.tile_overlap}")
        print(f"Model: {self.config.model_name}")
        print(f"Artifact reduction: {'Enabled' if self.config.advanced_postprocess else 'Disabled'}")
        print(f"Seamless tiles: {'Enabled' if self.config.seamless_tiles else 'Disabled'}")
        print(f"Temporal consistency: {'Enabled' if self.config.temporal_consistency else 'Disabled'}")
        print(f"{'='*60}")
    
    def _optimize_settings(self):
        """Optimize settings based on hardware and model"""
        # Get minimum VRAM across GPUs
        min_vram = float('inf')
        for gpu_id in self.config.gpu_ids:
            if gpu_id < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(gpu_id)
                vram_gb = props.total_memory / (1024**3)
                min_vram = min(min_vram, vram_gb)
        
        # Get model-specific optimal settings
        if self.config.model_specific_settings:
            optimal_tile, optimal_padding = self.model_manager.get_optimal_settings(
                self.config.model_name, min_vram
            )
            
            if self.config.auto_tile_size:
                self.config.tile_size = optimal_tile
            self.config.tile_pad = optimal_padding
        elif self.config.auto_tile_size:
            # Generic VRAM-based adjustment
            if min_vram >= 20:  # RTX 3090/4090 territory
                self.config.tile_size = 1024
                self.config.tile_pad = 64
            elif min_vram >= 12:
                self.config.tile_size = 768
                self.config.tile_pad = 48
            elif min_vram >= 8:
                self.config.tile_size = 512
                self.config.tile_pad = 32
            else:
                self.config.tile_size = 256
                self.config.tile_pad = 16
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Suppress verbose logging from libraries
        logging.getLogger('basicsr').setLevel(logging.WARNING)
        logging.getLogger('realesrgan').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('numba').setLevel(logging.WARNING)
        
        # Configure main logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('video_upscaler.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def get_video_info(self, video_path: str) -> Dict:
        """Extract video information using ffprobe"""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            # Parse frame rate
            fps_parts = video_info['r_frame_rate'].split('/')
            if len(fps_parts) == 2:
                fps = float(fps_parts[0]) / float(fps_parts[1])
            else:
                fps = float(fps_parts[0])
            
            return {
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'fps': fps,
                'duration': float(probe['format'].get('duration', 0)),
                'codec': video_info['codec_name'],
                'frames': int(video_info.get('nb_frames', 0))
            }
        except Exception as e:
            return {}
    
    def extract_frames(self, video_path: str, output_dir: str) -> List[str]:
        """Extract frames from video"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        
        # Get total frames for progress
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_path = output_dir / f"frame_{frame_count:08d}.png"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
                frame_count += 1
                
                # Progress update every 30 frames
                if frame_count % 30 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"\rExtracting frames: {frame_count}/{total_frames} ({progress:.1f}%)", end='', flush=True)
        
        finally:
            cap.release()
        
        print(f"\r✓ Extracted {len(frame_paths)} frames" + " " * 20)
        return frame_paths
    
    def enhance_frame_advanced(self, frame: np.ndarray, upsampler: RealESRGANer, gpu_id: int) -> np.ndarray:
        """Enhanced frame processing with seamless tiling and post-processing"""
        h, w = frame.shape[:2]
        
        # For smaller images, process directly
        if max(h, w) <= self.config.tile_size:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                with torch.cuda.device(gpu_id):
                    enhanced, _ = upsampler.enhance(frame, outscale=self.config.scale)
            
            # Apply post-processing
            if self.config.advanced_postprocess:
                enhanced = self._apply_postprocessing(enhanced)
            
            return enhanced
        
        # For larger images, use seamless tiling
        if self.config.seamless_tiles:
            return self._process_with_seamless_tiles(frame, upsampler, gpu_id)
        else:
            # Fallback to standard tiling
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                with torch.cuda.device(gpu_id):
                    enhanced, _ = upsampler.enhance(frame, outscale=self.config.scale)
            
            if self.config.advanced_postprocess:
                enhanced = self._apply_postprocessing(enhanced)
            
            return enhanced
    
    def _process_with_seamless_tiles(self, frame: np.ndarray, upsampler: RealESRGANer, gpu_id: int) -> np.ndarray:
        """Process frame with seamless tiling to avoid artifacts"""
        h, w = frame.shape[:2]
        tile_size = self.config.tile_size
        overlap = self.config.tile_overlap
        scale = self.config.scale
        
        # Calculate tile positions with overlap
        tiles = []
        positions = []
        enhanced_tiles = []
        enhanced_positions = []
        
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Extract tile with padding
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = frame[y:y_end, x:x_end]
                tiles.append(tile)
                positions.append((y, x))
                
                # Process tile
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with torch.cuda.device(gpu_id):
                        enhanced_tile, _ = upsampler.enhance(tile, outscale=scale)
                
                enhanced_tiles.append(enhanced_tile)
                enhanced_positions.append((y * scale, x * scale))
        
        # Blend tiles seamlessly
        output_shape = (h * scale, w * scale, frame.shape[2])
        enhanced = self.image_processor.seamless_tile_blend(
            enhanced_tiles, enhanced_positions, output_shape, overlap * scale
        )
        
        # Apply post-processing
        if self.config.advanced_postprocess:
            enhanced = self._apply_postprocessing(enhanced)
        
        return enhanced
    
    def _apply_postprocessing(self, frame: np.ndarray) -> np.ndarray:
        """Apply advanced post-processing to reduce artifacts"""
        processed = frame.copy()
        
        # Noise reduction
        if self.config.denoise_strength > 0:
            processed = self.image_processor.reduce_noise(processed, self.config.denoise_strength)
        
        # Detail enhancement
        if self.config.sharpen_strength > 0:
            processed = self.image_processor.enhance_details(processed, self.config.sharpen_strength)
        
        # Color enhancement
        if self.config.color_correction:
            processed = self.image_processor.enhance_colors(processed)
        
        return processed
    
    def process_frame_batch(self, frame_paths: List[str], output_dir: str, gpu_id: int, progress_queue=None) -> List[str]:
        """Process a batch of frames on specified GPU with enhanced processing"""
        upsampler = self.model_manager.load_model(
            self.config.model_name, gpu_id, 
            self.config.tile_size, self.config.tile_pad
        )
        
        output_dir = Path(output_dir)
        output_paths = []
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Read frame
                img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                
                # Enhanced processing
                enhanced = self.enhance_frame_advanced(img, upsampler, gpu_id)
                
                # Temporal consistency check
                if self.config.temporal_consistency:
                    prev_key = f"gpu_{gpu_id}_prev"
                    if prev_key in self.previous_frames:
                        enhanced = self.image_processor.temporal_consistency_check(
                            self.previous_frames[prev_key], enhanced
                        )
                    self.previous_frames[prev_key] = enhanced.copy()
                
                # Save enhanced frame
                frame_name = Path(frame_path).name
                output_path = output_dir / frame_name
                cv2.imwrite(str(output_path), enhanced)
                output_paths.append(str(output_path))
                
                # Update progress
                if progress_queue:
                    progress_queue.put(1)
                
                # Log progress every 10 frames for debugging
                if i > 0 and i % 10 == 0:
                    self.logger.debug(f"GPU {gpu_id}: Processed {i}/{len(frame_paths)} frames")
                
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_path} on GPU {gpu_id}: {e}")
                if progress_queue:
                    progress_queue.put(1)  # Still count as processed to avoid hanging
                continue
        
        return output_paths
    
    def process_frames_multi_gpu(self, frame_paths: List[str], output_dir: str) -> List[str]:
        """Process frames using multiple GPUs with enhanced processing"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Clear previous frames for temporal consistency
        self.previous_frames.clear()
        
        # Distribute frames across GPUs
        num_gpus = len(self.config.gpu_ids)
        frames_per_gpu = len(frame_paths) // num_gpus
        frame_batches = []
        
        for i, gpu_id in enumerate(self.config.gpu_ids):
            start_idx = i * frames_per_gpu
            if i == num_gpus - 1:  # Last GPU gets remaining frames
                batch = frame_paths[start_idx:]
            else:
                batch = frame_paths[start_idx:start_idx + frames_per_gpu]
            frame_batches.append((batch, gpu_id))
            print(f"  GPU {gpu_id}: {len(batch)} frames")
        
        # Process batches in parallel with enhanced progress tracking
        all_output_paths = []
        total_frames = len(frame_paths)
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        # Create a thread-safe queue for progress updates
        import queue as thread_queue
        progress_queue = thread_queue.Queue()
        
        # Initialize progress bar with clean format
        if use_tqdm:
            pbar = tqdm(
                total=total_frames, 
                desc="AI Enhancement", 
                unit="frame",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt:.2s}]',
                ncols=80,
                leave=True,
                dynamic_ncols=False
            )
        
        def process_gpu_batch(batch_info):
            batch_frames, gpu_id = batch_info
            return self.process_frame_batch(batch_frames, output_dir, gpu_id, progress_queue)
        
        # Progress monitoring thread
        def monitor_progress():
            processed = 0
            while processed < total_frames:
                try:
                    # Wait for progress update with timeout
                    progress_queue.get(timeout=1.0)
                    processed += 1
                    if use_tqdm:
                        pbar.update(1)
                except queue.Empty:
                    continue
        
        # Start progress monitor
        progress_thread = threading.Thread(target=monitor_progress, daemon=True)
        progress_thread.start()
        
        # Start processing in parallel
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = [executor.submit(process_gpu_batch, batch_info) for batch_info in frame_batches]
            
            # Collect results
            for future in futures:
                try:
                    batch_results = future.result()
                    all_output_paths.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"Error in GPU batch processing: {e}")
        
        # Wait for progress thread to finish
        progress_thread.join(timeout=5)
        
        if use_tqdm:
            pbar.close()
        
        # Sort output paths to maintain frame order
        all_output_paths.sort()
        return all_output_paths
    
    def reassemble_video(self, frame_dir: str, output_video: str, fps: float) -> bool:
        """Reassemble frames into video with enhanced quality settings"""
        frame_dir = Path(frame_dir)
        frame_pattern = str(frame_dir / "frame_%08d.png")
        
        try:
            # First check if we have frames
            frame_files = list(frame_dir.glob("frame_*.png"))
            if not frame_files:
                self.logger.error("No frames found for reassembly")
                return False
            
            print(f"Step 4/4: Reassembling {len(frame_files)} frames into video...")
            
            # Enhanced ffmpeg command with better quality settings
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "libx264",
                "-preset", self.config.preset,
                "-crf", str(self.config.crf),
                "-profile:v", "high",
                "-level:v", "4.1",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",  # Better streaming
                "-tune", "film",  # Optimize for film content
                output_video
            ]
            
            # Run ffmpeg with suppressed output
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Video reassembly complete")
                return True
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error reassembling video: {e}")
            return False
    
    def copy_audio(self, input_video: str, output_video: str) -> bool:
        """Copy audio from input to output video"""
        try:
            temp_video = output_video + ".temp.mp4"
            
            # Build ffmpeg command to copy audio
            cmd = [
                "ffmpeg", "-y",
                "-i", output_video,
                "-i", input_video,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                temp_video
            ]
            
            # Run with suppressed output
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original with audio version
                os.replace(temp_video, output_video)
                print("✓ Audio copied successfully")
                return True
            else:
                # Clean up temp file if it exists
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                self.logger.warning("No audio track found or error copying audio")
                return False
                
        except Exception as e:
            self.logger.error(f"Error copying audio: {e}")
            return False
    
    def process_video(self, input_video: str, output_video: str) -> bool:
        """Complete enhanced video processing pipeline"""
        import tempfile
        import shutil
        import subprocess
        
        input_video = Path(input_video)
        output_video = Path(output_video)
        
        if not input_video.exists():
            self.logger.error(f"Input video not found: {input_video}")
            return False
        
        # Get video info
        video_info = self.get_video_info(str(input_video))
        if not video_info:
            self.logger.error("Could not read video information")
            return False
        
        # Print enhanced processing info
        print(f"\n{'='*60}")
        print(f"Enhanced Processing: {input_video.name}")
        print(f"{'='*60}")
        print(f"Input resolution:  {video_info['width']}x{video_info['height']}")
        print(f"Output resolution: {video_info['width'] * self.config.scale}x{video_info['height'] * self.config.scale}")
        print(f"FPS: {video_info['fps']:.2f}")
        print(f"Duration: {video_info['duration']:.1f}s")
        print(f"Enhancement features:")
        print(f"  • Seamless tiling: {'✓' if self.config.seamless_tiles else '✗'}")
        print(f"  • Noise reduction: {'✓' if self.config.denoise_strength > 0 else '✗'}")
        print(f"  • Color enhancement: {'✓' if self.config.color_correction else '✗'}")
        print(f"  • Temporal consistency: {'✓' if self.config.temporal_consistency else '✗'}")
        print(f"  • Detail preservation: {'✓' if self.config.preserve_details else '✗'}")
        print(f"{'='*60}")
        
        # Create temporary directories
        with tempfile.TemporaryDirectory(prefix="enhanced_video_upscale_") as temp_dir:
            temp_path = Path(temp_dir)
            frames_dir = temp_path / "frames"
            upscaled_dir = temp_path / "upscaled"
            
            try:
                # Step 1: Extract frames
                print(f"\nStep 1/4: Extracting frames...")
                frame_paths = self.extract_frames(str(input_video), str(frames_dir))
                
                if not frame_paths:
                    self.logger.error("No frames extracted")
                    return False
                
                # Step 2: Enhanced AI processing
                print(f"\nStep 2/4: Enhanced AI upscaling {len(frame_paths)} frames with {len(self.config.gpu_ids)} GPU(s)...")
                
                # Estimate processing time (more conservative with enhanced processing)
                estimated_fps = 3 * len(self.config.gpu_ids)  # Slightly slower due to enhanced processing
                estimated_time = len(frame_paths) / estimated_fps / 60
                print(f"Estimated time: {estimated_time:.1f} minutes @ ~{estimated_fps} FPS")
                
                upscaled_paths = self.process_frames_multi_gpu(frame_paths, str(upscaled_dir))
                
                if not upscaled_paths:
                    self.logger.error("No frames processed successfully")
                    return False
                
                print(f"✓ Enhanced processing complete: {len(upscaled_paths)} frames")
                
                # Step 3: Enhanced video reassembly
                success = self.reassemble_video(str(upscaled_dir), str(output_video), video_info['fps'])
                
                if not success:
                    return False
                
                # Step 4: Copy audio if enabled and available
                if self.config.audio_copy:
                    print("Copying audio track...")
                    self.copy_audio(str(input_video), str(output_video))
                
                # Get output file size and quality info
                output_size = output_video.stat().st_size / (1024 * 1024)
                print(f"\n✓ Enhanced processing complete!")
                print(f"Output: {output_video} ({output_size:.1f} MB)")
                print(f"Quality improvements applied:")
                print(f"  • AI upscaling: {self.config.scale}x")
                print(f"  • Advanced tiling: Seamless blending")
                print(f"  • Post-processing: Multi-stage enhancement")
                print(f"  • Encoding: CRF {self.config.crf} ({self.config.preset})")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error in enhanced processing pipeline: {e}")
                return False

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced AI-powered video upscaling with advanced artifact reduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  # Basic enhanced processing
  python video_upscaler.py input.mp4 output.mp4 --enhanced
  
  # Maximum quality (slow but best results)
  python video_upscaler.py input.mp4 output.mp4 --quality max
  
  # Fine-tuned artifact reduction
  python video_upscaler.py input.mp4 output.mp4 --denoise 0.2 --sharpen 0.1
  
  # Anime content optimization
  python video_upscaler.py input.mp4 output.mp4 --model RealESRGAN_x4plus_anime_6B --anime-mode
        """
    )
    
    parser.add_argument("input", help="Input video file or directory (for batch processing)")
    parser.add_argument("output", help="Output video file or directory")
    parser.add_argument("--model", default="RealESRGAN_x4plus", 
                       choices=["RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x4_v3"],
                       help="AI model to use for upscaling")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                       help="GPU IDs to use (default: auto-detect all)")
    parser.add_argument("--tile-size", type=int, default=None,
                       help="Tile size for processing (default: auto-adjust based on VRAM)")
    parser.add_argument("--tile-overlap", type=int, default=32,
                       help="Overlap between tiles for seamless blending")
    parser.add_argument("--crf", type=int, default=15,
                       help="Video quality (lower = better quality, larger file)")
    parser.add_argument("--preset", default="slower",
                       choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                       help="Encoding speed vs compression efficiency")
    parser.add_argument("--no-audio", action="store_true",
                       help="Don't copy audio track")
    parser.add_argument("--batch", action="store_true",
                       help="Process all videos in input directory")
    
    # Enhanced processing options
    parser.add_argument("--enhanced", action="store_true",
                       help="Enable all enhancement features")
    parser.add_argument("--quality", choices=["fast", "balanced", "max"], default="balanced",
                       help="Quality preset (fast/balanced/max)")
    parser.add_argument("--denoise", type=float, default=0.1,
                       help="Noise reduction strength (0.0-1.0)")
    parser.add_argument("--sharpen", type=float, default=0.0,
                       help="Sharpening strength (0.0-1.0)")
    parser.add_argument("--no-seamless", action="store_true",
                       help="Disable seamless tile processing")
    parser.add_argument("--no-temporal", action="store_true",
                       help="Disable temporal consistency")
    parser.add_argument("--no-color-enhance", action="store_true",
                       help="Disable color enhancement")
    parser.add_argument("--anime-mode", action="store_true",
                       help="Optimize settings for anime/cartoon content")
    
    args = parser.parse_args()
    
    # Apply quality presets
    if args.quality == "fast":
        default_crf, default_preset = 20, "fast"
        default_denoise, default_seamless = 0.05, False
    elif args.quality == "max":
        default_crf, default_preset = 12, "veryslow"
        default_denoise, default_seamless = 0.15, True
    else:  # balanced
        default_crf, default_preset = 15, "slower"
        default_denoise, default_seamless = 0.1, True
    
    # Override defaults if not specified
    crf = args.crf if args.crf != 15 else default_crf
    preset = args.preset if args.preset != "slower" else default_preset
    denoise = args.denoise if args.denoise != 0.1 else default_denoise
    
    # Apply anime mode optimizations
    if args.anime_mode:
        if args.model == "RealESRGAN_x4plus":
            args.model = "RealESRGAN_x4plus_anime_6B"
        denoise = max(denoise, 0.05)  # Anime often benefits from light denoising
    
    # Create enhanced processing configuration
    config = ProcessingConfig(
        model_name=args.model,
        gpu_ids=args.gpus,
        tile_overlap=args.tile_overlap,
        crf=crf,
        preset=preset,
        audio_copy=not args.no_audio,
        denoise_strength=denoise,
        sharpen_strength=args.sharpen,
        seamless_tiles=not args.no_seamless and (args.enhanced or default_seamless),
        temporal_consistency=not args.no_temporal and (args.enhanced or args.quality == "max"),
        color_correction=not args.no_color_enhance and (args.enhanced or args.quality != "fast"),
        advanced_postprocess=args.enhanced or args.quality != "fast",
        preserve_details=True
    )
    
    # Override tile size if specified
    if args.tile_size:
        config.tile_size = args.tile_size
        config.auto_tile_size = False
    
    # Update scale based on model
    if "x2" in args.model:
        config.scale = 2
    elif "x4" in args.model:
        config.scale = 4
    
    try:
        processor = EnhancedVideoProcessor(config)
        
        if args.batch:
            # Batch processing
            input_dir = Path(args.input)
            output_dir = Path(args.output)
            
            if not input_dir.is_dir():
                print(f"Error: Input directory not found: {input_dir}")
                return 1
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find video files
            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
            video_files = [f for f in input_dir.iterdir() 
                          if f.suffix.lower() in video_extensions]
            
            if not video_files:
                print(f"No video files found in {input_dir}")
                return 1
            
            print(f"Found {len(video_files)} videos to process with enhanced quality")
            
            # Process each video
            success_count = 0
            for video_file in video_files:
                output_file = output_dir / f"{video_file.stem}_enhanced{video_file.suffix}"
                print(f"\nProcessing {video_file.name}...")
                
                if processor.process_video(str(video_file), str(output_file)):
                    success_count += 1
                else:
                    print(f"Failed to process {video_file.name}")
            
            print(f"\nEnhanced batch processing complete: {success_count}/{len(video_files)} videos processed successfully")
            
        else:
            # Single video processing
            success = processor.process_video(args.input, args.output)
            return 0 if success else 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())