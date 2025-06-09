#!/usr/bin/env python3
# video_upscaler.py
"""
Video Restore - Professional AI-Powered Video Upscaler
Optimized for high-end hardware with streaming processing and advanced quality
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

def __getattr__(name):
    import torchvision.transforms.functional as F
    if hasattr(F, name):
        return getattr(F, name)
    raise AttributeError(f"module 'functional_tensor' has no attribute '{name}'")

fake_module.__getattr__ = __getattr__

try:
    import torchvision.transforms.functional as F
    for func_name in ['rgb_to_grayscale', 'adjust_brightness', 'adjust_contrast', 
                      'adjust_saturation', 'adjust_hue', 'normalize', 'resize',
                      'pad', 'crop', 'center_crop', 'resized_crop', 'hflip', 'vflip',
                      'rotate', 'affine', 'to_tensor', 'to_pil_image', 'to_grayscale']:
        if hasattr(F, func_name):
            setattr(fake_module, func_name, getattr(F, func_name))
except ImportError:
    pass

sys.modules['torchvision.transforms.functional_tensor'] = fake_module

# Core imports
import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Generator
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass
import json
import warnings
import contextlib
import io
import subprocess
from scipy import ndimage
from skimage import filters, restoration, exposure, transform
import torch.nn.functional as F
from collections import deque
import multiprocessing as mp

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*functional.*")

# Third-party imports
try:
    import torch
    import torchvision
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    import ffmpeg
    
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install torch torchvision torchaudio basicsr realesrgan opencv-python ffmpeg-python scipy scikit-image")
    sys.exit(1)

@dataclass
class ProcessingConfig:
    """Professional video processing configuration optimized for high-end hardware"""
    model_name: str = "RealESRGAN_x4plus"
    scale: int = 4
    tile_size: int = 1024  # Optimized for RTX 3090
    tile_pad: int = 64
    tile_overlap: int = 64  # Increased for better blending
    pre_pad: int = 10
    face_enhance: bool = False
    gpu_ids: List[int] = None
    output_format: str = "mp4"
    crf: int = 15  # Sweet spot for quality/size
    preset: str = "slow"  # Optimal for quality
    audio_copy: bool = True
    
    # Professional quality settings
    denoise_strength: float = 0.1
    sharpen_strength: float = 0.05
    color_correction: bool = True
    temporal_consistency: bool = True
    seamless_tiles: bool = True
    advanced_postprocess: bool = True
    preserve_details: bool = True
    
    # Performance optimizations
    stream_processing: bool = True  # NEW: Stream processing instead of frame extraction
    gpu_memory_fraction: float = 0.9  # Use 90% of GPU memory
    cpu_threads: int = None  # Auto-detect optimal thread count
    buffer_size: int = 8  # Frame buffer size for streaming
    
    # Advanced quality features
    professional_mode: bool = False
    use_waifu2x_noise_reduction: bool = False
    chromatic_aberration_correction: bool = True
    motion_compensation: bool = False  # For temporal consistency
    
    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = list(range(torch.cuda.device_count()))
            if not self.gpu_ids:
                self.gpu_ids = [0]
        
        if self.cpu_threads is None:
            # Use 75% of available CPU cores for optimal performance
            self.cpu_threads = max(1, int(mp.cpu_count() * 0.75))

class StreamingFrameProcessor:
    """Streaming frame processor that eliminates file I/O bottlenecks"""
    
    def __init__(self, input_video: str, config: ProcessingConfig):
        self.input_video = input_video
        self.config = config
        self.video_info = self._get_video_info()
        self.frame_queue = queue.Queue(maxsize=config.buffer_size)
        self.result_queue = queue.Queue(maxsize=config.buffer_size)
        self._stop_flag = threading.Event()
        
    def _get_video_info(self) -> Dict:
        """Get video information using ffprobe"""
        try:
            probe = ffmpeg.probe(self.input_video)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            fps_parts = video_stream['r_frame_rate'].split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            
            return {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': fps,
                'duration': float(probe['format'].get('duration', 0)),
                'codec': video_stream['codec_name'],
                'frames': int(video_stream.get('nb_frames', fps * float(probe['format'].get('duration', 0))))
            }
        except Exception as e:
            raise RuntimeError(f"Could not read video info: {e}")
    
    def frame_generator(self) -> Generator[np.ndarray, None, None]:
        """Memory-efficient frame generator using FFmpeg"""
        # Use FFmpeg to decode frames directly to numpy arrays
        process = (
            ffmpeg
            .input(self.input_video)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True, pipe_stderr=subprocess.DEVNULL)
        )
        
        width, height = self.video_info['width'], self.video_info['height']
        frame_size = width * height * 3
        
        try:
            while True:
                raw_frame = process.stdout.read(frame_size)
                if not raw_frame:
                    break
                    
                frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
                yield frame
                
        finally:
            process.stdout.close()
            process.wait()

class ProfessionalImageProcessor:
    """Professional-grade image processing with advanced algorithms"""
    
    @staticmethod
    def seamless_tile_blend_advanced(tiles: List[np.ndarray], positions: List[Tuple[int, int]], 
                                   output_shape: Tuple[int, int], overlap: int = 64) -> np.ndarray:
        """Advanced seamless tile blending using distance-weighted interpolation"""
        h, w = output_shape[:2]
        channels = tiles[0].shape[2] if len(tiles[0].shape) == 3 else 1
        
        if channels == 1:
            output = np.zeros((h, w), dtype=np.float64)
            weight_sum = np.zeros((h, w), dtype=np.float64)
        else:
            output = np.zeros((h, w, channels), dtype=np.float64)
            weight_sum = np.zeros((h, w), dtype=np.float64)
        
        for tile, (y, x) in zip(tiles, positions):
            th, tw = tile.shape[:2]
            
            # Create smooth distance-based weights using sigmoid function
            weight_map = np.ones((th, tw), dtype=np.float64)
            
            # Apply smooth falloff at edges
            for i in range(min(overlap, th)):
                # Top edge - smooth sigmoid falloff
                alpha = 1 / (1 + np.exp(-10 * (i / overlap - 0.5)))
                weight_map[i, :] *= alpha
                
                # Bottom edge
                if th - 1 - i >= 0:
                    weight_map[th - 1 - i, :] *= alpha
            
            for j in range(min(overlap, tw)):
                # Left edge
                alpha = 1 / (1 + np.exp(-10 * (j / overlap - 0.5)))
                weight_map[:, j] *= alpha
                
                # Right edge
                if tw - 1 - j >= 0:
                    weight_map[:, tw - 1 - j] *= alpha
            
            # Apply 2D Gaussian smoothing to weights
            weight_map = ndimage.gaussian_filter(weight_map, sigma=overlap/8, mode='nearest')
            
            # Blend into output
            y_end = min(y + th, h)
            x_end = min(x + tw, w)
            tile_h = y_end - y
            tile_w = x_end - x
            
            tile_weights = weight_map[:tile_h, :tile_w]
            
            if channels == 1:
                output[y:y_end, x:x_end] += tile[:tile_h, :tile_w].astype(np.float64) * tile_weights
                weight_sum[y:y_end, x:x_end] += tile_weights
            else:
                tile_float = tile[:tile_h, :tile_w].astype(np.float64)
                for c in range(channels):
                    output[y:y_end, x:x_end, c] += tile_float[:, :, c] * tile_weights
                weight_sum[y:y_end, x:x_end] += tile_weights
        
        # Normalize and avoid division by zero
        weight_sum[weight_sum < 1e-8] = 1.0
        
        if channels == 1:
            output /= weight_sum
        else:
            for c in range(channels):
                output[:, :, c] /= weight_sum
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    @staticmethod
    def professional_denoise(image: np.ndarray, strength: float = 0.1, preserve_edges: bool = True) -> np.ndarray:
        """Professional denoising using Non-Local Means with edge preservation"""
        if strength <= 0:
            return image
        
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        if preserve_edges:
            # Use edge-preserving Non-Local Means denoising
            if len(image.shape) == 3:
                h = strength * 10  # Filter strength
                denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
            else:
                h = strength * 10
                denoised = cv2.fastNlMeansDenoising(image, None, h, 7, 21)
        else:
            # Use bilateral filter for speed
            sigma_color = 75 * strength
            sigma_space = 75 * strength
            denoised = cv2.bilateralFilter(image, -1, sigma_color, sigma_space)
        
        return denoised
    
    @staticmethod
    def professional_sharpen(image: np.ndarray, strength: float = 0.05, radius: float = 1.0) -> np.ndarray:
        """Professional unsharp masking with controlled radius"""
        if strength <= 0:
            return image
        
        img_float = image.astype(np.float32) / 255.0
        
        # Create Gaussian blur with specified radius
        blurred = cv2.GaussianBlur(img_float, (0, 0), radius)
        
        # Unsharp mask
        sharpened = img_float + strength * (img_float - blurred)
        
        # Clip and convert back
        sharpened = np.clip(sharpened, 0, 1)
        return (sharpened * 255).astype(np.uint8)
    
    @staticmethod
    def enhance_colors_professional(image: np.ndarray, method: str = "clahe") -> np.ndarray:
        """Professional color enhancement with multiple methods"""
        if method == "clahe":
            # Convert to LAB for better color handling
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE on lightness channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Enhance color channels subtly
            a = cv2.multiply(a, 1.03)
            b = cv2.multiply(b, 1.03)
            
            enhanced_lab = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        elif method == "hsv":
            # HSV-based enhancement
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Enhance saturation and value
            s = cv2.multiply(s, 1.05)
            v = cv2.multiply(v, 1.02)
            
            enhanced_hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        return image
    
    @staticmethod
    def temporal_consistency_advanced(prev_frames: deque, curr_frame: np.ndarray, 
                                   strength: float = 0.3) -> np.ndarray:
        """Advanced temporal consistency using multiple previous frames"""
        if not prev_frames:
            return curr_frame
        
        # Use weighted average of previous frames
        weights = np.array([0.7, 0.2, 0.1])[:len(prev_frames)]
        weights = weights / weights.sum()
        
        blended = curr_frame.astype(np.float32)
        
        for i, prev_frame in enumerate(prev_frames):
            if i >= len(weights):
                break
            
            # Calculate motion/change magnitude
            diff = cv2.absdiff(curr_frame, prev_frame)
            motion_mask = np.mean(diff, axis=2) if len(diff.shape) == 3 else diff
            
            # Apply temporal blending where motion is low
            blend_mask = (motion_mask < 30).astype(np.float32)  # Threshold for motion
            blend_strength = strength * weights[i] * np.expand_dims(blend_mask, -1)
            
            prev_float = prev_frame.astype(np.float32)
            blended = blended * (1 - blend_strength) + prev_float * blend_strength
        
        return np.clip(blended, 0, 255).astype(np.uint8)

class ProfessionalVideoProcessor:
    """Professional video processor optimized for high-end hardware"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model_manager = self._create_model_manager()
        self.image_processor = ProfessionalImageProcessor()
        self.setup_logging()
        self.previous_frames = {}  # Store multiple previous frames per GPU
        
        # Validate and optimize GPU setup
        self._setup_gpus()
        self._optimize_settings()
        
        # Print professional system info
        self._print_system_info()
    
    def _create_model_manager(self):
        """Create enhanced model manager"""
        class ProfessionalModelManager:
            MODELS = {
                "RealESRGAN_x4plus": {
                    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                    "scale": 4,
                    "arch": RRDBNet,
                    "arch_params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 4},
                    "optimal_tile": 1024,
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
                    "optimal_tile": 768,
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
                if model_name not in self.MODELS:
                    raise ValueError(f"Unknown model: {model_name}")
                
                model_path = self.model_dir / f"{model_name}.pth"
                if not model_path.exists():
                    print(f"Downloading {model_name} model...")
                    with contextlib.redirect_stdout(io.StringIO()):
                        load_file_from_url(
                            self.MODELS[model_name]["url"],
                            model_dir=str(self.model_dir),
                            file_name=f"{model_name}.pth"
                        )
                    print(f"✓ Model downloaded")
                return model_path
            
            def load_model(self, model_name: str, gpu_id: int, tile_size: int, tile_pad: int) -> RealESRGANer:
                cache_key = f"{model_name}_gpu{gpu_id}_{tile_size}_{tile_pad}"
                
                if cache_key in self.loaded_models:
                    return self.loaded_models[cache_key]
                
                model_path = self.download_model(model_name)
                model_info = self.MODELS[model_name]
                
                model = model_info["arch"](**model_info["arch_params"])
                
                upsampler = RealESRGANer(
                    scale=model_info["scale"],
                    model_path=str(model_path),
                    model=model,
                    tile=tile_size,
                    tile_pad=tile_pad,
                    pre_pad=10,
                    half=True,
                    gpu_id=gpu_id,
                    device=torch.device(f'cuda:{gpu_id}')
                )
                
                self.loaded_models[cache_key] = upsampler
                return upsampler
        
        return ProfessionalModelManager()
    
    def _setup_gpus(self):
        """Setup and validate GPU configuration"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - GPU required")
        
        available_gpus = torch.cuda.device_count()
        self.config.gpu_ids = [gpu_id for gpu_id in self.config.gpu_ids if gpu_id < available_gpus]
        
        if not self.config.gpu_ids:
            raise RuntimeError("No valid GPUs specified")
        
        # Set GPU memory fraction for optimal usage
        for gpu_id in self.config.gpu_ids:
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction, gpu_id)
    
    def _optimize_settings(self):
        """Optimize settings for RTX 3090 hardware"""
        # Get minimum VRAM across GPUs
        min_vram = float('inf')
        for gpu_id in self.config.gpu_ids:
            if gpu_id < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(gpu_id)
                vram_gb = props.total_memory / (1024**3)
                min_vram = min(min_vram, vram_gb)
        
        # Optimize for RTX 3090 (24GB VRAM)
        if min_vram >= 20:  # RTX 3090/4090 territory
            self.config.tile_size = 1024
            self.config.tile_pad = 64
            self.config.tile_overlap = 64
            self.config.buffer_size = 16  # Larger buffer for high-end hardware
        elif min_vram >= 12:
            self.config.tile_size = 768
            self.config.tile_pad = 48
            self.config.tile_overlap = 48
            self.config.buffer_size = 12
        else:
            self.config.tile_size = 512
            self.config.tile_pad = 32
            self.config.tile_overlap = 32
            self.config.buffer_size = 8
    
    def _print_system_info(self):
        """Print detailed system information"""
        print(f"\n{'='*80}")
        print(f"PROFESSIONAL VIDEO UPSCALER - SYSTEM CONFIGURATION")
        print(f"{'='*80}")
        
        # GPU Information
        print(f"GPU Configuration:")
        for gpu_id in self.config.gpu_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            vram_gb = props.total_memory / (1024**3)
            print(f"  GPU {gpu_id}: {props.name} ({vram_gb:.1f}GB VRAM)")
        
        # Processing Settings
        print(f"\nProcessing Configuration:")
        print(f"  Model: {self.config.model_name}")
        print(f"  Tile Size: {self.config.tile_size}x{self.config.tile_size}")
        print(f"  Tile Overlap: {self.config.tile_overlap}px")
        print(f"  Stream Processing: {'✓ Enabled' if self.config.stream_processing else '✗ Disabled'}")
        print(f"  CPU Threads: {self.config.cpu_threads}")
        print(f"  Buffer Size: {self.config.buffer_size} frames")
        
        # Quality Features
        print(f"\nProfessional Quality Features:")
        print(f"  Seamless Tiling: {'✓' if self.config.seamless_tiles else '✗'}")
        print(f"  Temporal Consistency: {'✓' if self.config.temporal_consistency else '✗'}")
        print(f"  Professional Denoising: {'✓' if self.config.denoise_strength > 0 else '✗'}")
        print(f"  Color Enhancement: {'✓' if self.config.color_correction else '✗'}")
        print(f"  Advanced Post-Processing: {'✓' if self.config.advanced_postprocess else '✗'}")
        
        print(f"{'='*80}")
    
    def setup_logging(self):
        """Setup professional logging"""
        logging.getLogger('basicsr').setLevel(logging.WARNING)
        logging.getLogger('realesrgan').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('numba').setLevel(logging.WARNING)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('professional_upscaler.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def enhance_frame_professional(self, frame: np.ndarray, upsampler: RealESRGANer, 
                                 gpu_id: int, frame_idx: int = 0) -> np.ndarray:
        """Professional frame enhancement with advanced processing"""
        h, w = frame.shape[:2]
        
        # For small frames, process directly
        if max(h, w) <= self.config.tile_size:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                with torch.cuda.device(gpu_id):
                    enhanced, _ = upsampler.enhance(frame, outscale=self.config.scale)
        else:
            # Use professional tiling for large frames
            enhanced = self._process_with_professional_tiles(frame, upsampler, gpu_id)
        
        # Apply professional post-processing
        if self.config.advanced_postprocess:
            enhanced = self._apply_professional_postprocessing(enhanced, gpu_id, frame_idx)
        
        return enhanced
    
    def _process_with_professional_tiles(self, frame: np.ndarray, upsampler: RealESRGANer, gpu_id: int) -> np.ndarray:
        """Professional tile processing with optimal overlap"""
        h, w = frame.shape[:2]
        tile_size = self.config.tile_size
        overlap = self.config.tile_overlap
        scale = self.config.scale
        
        # Calculate optimal tile positions with minimal overlap waste
        tiles = []
        positions = []
        enhanced_tiles = []
        enhanced_positions = []
        
        step_size = tile_size - overlap
        
        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = frame[y:y_end, x:x_end]
                tiles.append(tile)
                positions.append((y, x))
                
                # Process tile with error handling
                try:
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        with torch.cuda.device(gpu_id):
                            enhanced_tile, _ = upsampler.enhance(tile, outscale=scale)
                    
                    enhanced_tiles.append(enhanced_tile)
                    enhanced_positions.append((y * scale, x * scale))
                
                except Exception as e:
                    self.logger.warning(f"Tile processing failed for position ({y}, {x}): {e}")
                    # Fallback: use bilinear upscaling for failed tile
                    fallback_tile = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    enhanced_tiles.append(fallback_tile)
                    enhanced_positions.append((y * scale, x * scale))
        
        # Professional tile blending
        output_shape = (h * scale, w * scale, frame.shape[2])
        enhanced = self.image_processor.seamless_tile_blend_advanced(
            enhanced_tiles, enhanced_positions, output_shape, overlap * scale
        )
        
        return enhanced
    
    def _apply_professional_postprocessing(self, frame: np.ndarray, gpu_id: int, frame_idx: int) -> np.ndarray:
        """Apply professional-grade post-processing"""
        processed = frame.copy()
        
        # Professional denoising
        if self.config.denoise_strength > 0:
            processed = self.image_processor.professional_denoise(
                processed, self.config.denoise_strength, preserve_edges=True
            )
        
        # Professional sharpening
        if self.config.sharpen_strength > 0:
            processed = self.image_processor.professional_sharpen(
                processed, self.config.sharpen_strength, radius=1.0
            )
        
        # Professional color enhancement
        if self.config.color_correction:
            processed = self.image_processor.enhance_colors_professional(processed, method="clahe")
        
        # Temporal consistency with frame history
        if self.config.temporal_consistency:
            key = f"gpu_{gpu_id}_frames"
            if key not in self.previous_frames:
                self.previous_frames[key] = deque(maxlen=3)  # Keep last 3 frames
            
            if self.previous_frames[key]:
                processed = self.image_processor.temporal_consistency_advanced(
                    self.previous_frames[key], processed, strength=0.3
                )
            
            self.previous_frames[key].append(processed.copy())
        
        return processed
    
    def process_video_streaming(self, input_video: str, output_video: str) -> bool:
        """Professional streaming video processing pipeline"""
        try:
            # Initialize streaming processor
            stream_processor = StreamingFrameProcessor(input_video, self.config)
            video_info = stream_processor.video_info
            
            print(f"\nProcessing: {Path(input_video).name}")
            print(f"Resolution: {video_info['width']}x{video_info['height']} → {video_info['width'] * self.config.scale}x{video_info['height'] * self.config.scale}")
            print(f"Frames: {video_info['frames']} @ {video_info['fps']:.2f} FPS")
            
            # Estimate processing time more accurately
            pixel_count = video_info['width'] * video_info['height']
            complexity_factor = pixel_count / (1920 * 1080)  # Relative to 1080p
            gpu_speedup = len(self.config.gpu_ids)
            base_fps = 8 * gpu_speedup / complexity_factor  # More realistic estimate
            estimated_time = video_info['frames'] / base_fps / 60
            print(f"Estimated processing time: {estimated_time:.1f} minutes")
            
            # Setup multi-GPU processing
            frame_generator = stream_processor.frame_generator()
            processed_frames = self._process_frames_streaming(frame_generator, video_info['frames'])
            
            # Stream directly to FFmpeg for encoding
            success = self._encode_streaming_output(processed_frames, output_video, video_info)
            
            if success and self.config.audio_copy:
                self._copy_audio_track(input_video, output_video)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Streaming processing failed: {e}")
            return False
    
    def _process_frames_streaming(self, frame_generator: Generator, total_frames: int) -> Generator:
        """Process frames using streaming multi-GPU pipeline"""
        # Import tqdm for progress
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="Professional Enhancement", unit="frame",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        except ImportError:
            pbar = None
        
        # Initialize models on each GPU
        uploaders = {}
        for gpu_id in self.config.gpu_ids:
            uploaders[gpu_id] = self.model_manager.load_model(
                self.config.model_name, gpu_id, self.config.tile_size, self.config.tile_pad
            )
        
        # Multi-GPU processing with round-robin distribution
        frame_queue = queue.Queue(maxsize=self.config.buffer_size)
        result_queue = queue.Queue(maxsize=self.config.buffer_size)
        
        def gpu_worker(gpu_id: int):
            """Worker function for GPU processing"""
            upsampler = uploaders[gpu_id]
            frame_count = 0
            
            while True:
                try:
                    frame_data = frame_queue.get(timeout=1.0)
                    if frame_data is None:  # Poison pill
                        break
                    
                    frame, frame_idx = frame_data
                    enhanced = self.enhance_frame_professional(frame, upsampler, gpu_id, frame_idx)
                    result_queue.put((enhanced, frame_idx))
                    frame_count += 1
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"GPU {gpu_id} processing error: {e}")
                    result_queue.put((None, -1))  # Error marker
                finally:
                    frame_queue.task_done()
        
        # Start GPU workers
        workers = []
        for gpu_id in self.config.gpu_ids:
            worker = threading.Thread(target=gpu_worker, args=(gpu_id,), daemon=True)
            worker.start()
            workers.append(worker)
        
        # Feed frames to workers and collect results
        def frame_feeder():
            for frame_idx, frame in enumerate(frame_generator):
                frame_queue.put((frame, frame_idx))
            
            # Send poison pills to stop workers
            for _ in self.config.gpu_ids:
                frame_queue.put(None)
        
        feeder_thread = threading.Thread(target=frame_feeder, daemon=True)
        feeder_thread.start()
        
        # Collect and yield results in order
        processed_count = 0
        result_buffer = {}
        next_frame_idx = 0
        
        while processed_count < total_frames:
            try:
                enhanced, frame_idx = result_queue.get(timeout=5.0)
                
                if enhanced is None:  # Error marker
                    continue
                
                result_buffer[frame_idx] = enhanced
                processed_count += 1
                
                # Yield frames in order
                while next_frame_idx in result_buffer:
                    yield result_buffer.pop(next_frame_idx)
                    next_frame_idx += 1
                    if pbar:
                        pbar.update(1)
                
            except queue.Empty:
                self.logger.warning("Frame processing timeout")
                break
        
        if pbar:
            pbar.close()
        
        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=5.0)
    
    def _encode_streaming_output(self, processed_frames: Generator, output_video: str, video_info: Dict) -> bool:
        """Encode processed frames directly to output video"""
        try:
            output_path = Path(output_video)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Professional FFmpeg encoding settings
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24',
                      s=f"{video_info['width'] * self.config.scale}x{video_info['height'] * self.config.scale}",
                      r=video_info['fps'])
                .output(str(output_video),
                       vcodec='libx264',
                       crf=self.config.crf,
                       preset=self.config.preset,
                       pix_fmt='yuv420p',
                       movflags='+faststart',
                       tune='film')
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=subprocess.DEVNULL)
            )
            
            # Stream processed frames to FFmpeg
            for frame in processed_frames:
                process.stdin.write(frame.tobytes())
            
            process.stdin.close()
            process.wait()
            
            return process.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            return False
    
    def _copy_audio_track(self, input_video: str, output_video: str) -> bool:
        """Copy audio track to output video"""
        try:
            temp_output = output_video + ".temp.mp4"
            
            (
                ffmpeg
                .output(
                    ffmpeg.input(output_video)['v'],
                    ffmpeg.input(input_video)['a'],
                    temp_output,
                    vcodec='copy',
                    acodec='aac'
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            os.replace(temp_output, output_video)
            return True
            
        except Exception as e:
            self.logger.warning(f"Audio copying failed: {e}")
            return False
    
    def process_video(self, input_video: str, output_video: str) -> bool:
        """Main video processing entry point"""
        if self.config.stream_processing:
            return self.process_video_streaming(input_video, output_video)
        else:
            # Fallback to frame-based processing (original method)
            self.logger.warning("Using fallback frame-based processing")
            return self._process_video_fallback(input_video, output_video)
    
    def _process_video_fallback(self, input_video: str, output_video: str) -> bool:
        """Fallback to original processing method if streaming fails"""
        # This would implement the original frame extraction method as a fallback
        # For brevity, returning False to indicate streaming is required
        self.logger.error("Streaming processing is required for professional mode")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Professional AI-powered video upscaler optimized for high-end hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Professional Examples:
  # Maximum quality with streaming processing (recommended)
  python video_upscaler.py input.mp4 output.mp4 --professional
  
  # Dual GPU professional processing
  python video_upscaler.py input.mp4 output.mp4 --gpus 0 1 --professional
  
  # Fine-tuned professional settings
  python video_upscaler.py input.mp4 output.mp4 --professional --denoise 0.15 --sharpen 0.05
  
  # Batch processing with professional quality
  python video_upscaler.py input_folder/ output_folder/ --batch --professional
        """
    )
    
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("output", help="Output video file or directory")
    parser.add_argument("--model", default="RealESRGAN_x4plus",
                       choices=["RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x4_v3"],
                       help="AI model for upscaling")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                       help="GPU IDs to use (default: all available)")
    parser.add_argument("--tile-size", type=int, default=None,
                       help="Tile size (default: auto-optimized for hardware)")
    parser.add_argument("--tile-overlap", type=int, default=64,
                       help="Tile overlap for seamless blending")
    parser.add_argument("--crf", type=int, default=15,
                       help="Video quality (8-28, lower=better)")
    parser.add_argument("--preset", default="slow",
                       choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                       help="Encoding preset")
    parser.add_argument("--no-audio", action="store_true",
                       help="Skip audio track copying")
    parser.add_argument("--batch", action="store_true",
                       help="Batch process directory")
    
    # Professional quality options
    parser.add_argument("--professional", action="store_true",
                       help="Enable all professional features (recommended)")
    parser.add_argument("--quality", choices=["fast", "balanced", "professional"], default="balanced",
                       help="Quality preset")
    parser.add_argument("--denoise", type=float, default=0.1,
                       help="Denoising strength (0.0-1.0)")
    parser.add_argument("--sharpen", type=float, default=0.05,
                       help="Sharpening strength (0.0-1.0)")
    parser.add_argument("--no-streaming", action="store_true",
                       help="Disable streaming processing (not recommended)")
    parser.add_argument("--buffer-size", type=int, default=None,
                       help="Frame buffer size (auto-optimized)")
    parser.add_argument("--cpu-threads", type=int, default=None,
                       help="CPU threads (default: 75% of available)")
    
    args = parser.parse_args()
    
    # Configure professional settings
    if args.professional or args.quality == "professional":
        # Professional preset
        crf = 12 if args.crf == 15 else args.crf
        preset = "slow" if args.preset == "slow" else args.preset
        denoise = max(args.denoise, 0.15)
        sharpen = max(args.sharpen, 0.05)
        stream_processing = not args.no_streaming
    elif args.quality == "fast":
        crf = 20 if args.crf == 15 else args.crf
        preset = "fast" if args.preset == "slow" else args.preset
        denoise = min(args.denoise, 0.05)
        sharpen = 0.0
        stream_processing = not args.no_streaming
    else:  # balanced
        crf = args.crf
        preset = args.preset
        denoise = args.denoise
        sharpen = args.sharpen
        stream_processing = not args.no_streaming
    
    # Create professional configuration
    config = ProcessingConfig(
        model_name=args.model,
        gpu_ids=args.gpus,
        tile_overlap=args.tile_overlap,
        crf=crf,
        preset=preset,
        audio_copy=not args.no_audio,
        denoise_strength=denoise,
        sharpen_strength=sharpen,
        stream_processing=stream_processing,
        seamless_tiles=True,
        temporal_consistency=True,
        color_correction=True,
        advanced_postprocess=True,
        professional_mode=args.professional or args.quality == "professional",
        cpu_threads=args.cpu_threads,
        buffer_size=args.buffer_size
    )
    
    # Override tile size if specified
    if args.tile_size:
        config.tile_size = args.tile_size
    
    # Set scale based on model
    if "x2" in args.model:
        config.scale = 2
    elif "x4" in args.model:
        config.scale = 4
    
    try:
        processor = ProfessionalVideoProcessor(config)
        
        if args.batch:
            # Batch processing
            input_dir = Path(args.input)
            output_dir = Path(args.output)
            
            if not input_dir.is_dir():
                print(f"Error: Input directory not found: {input_dir}")
                return 1
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
            video_files = [f for f in input_dir.iterdir() 
                          if f.suffix.lower() in video_extensions]
            
            if not video_files:
                print(f"No video files found in {input_dir}")
                return 1
            
            print(f"\nProfessional batch processing: {len(video_files)} videos")
            
            success_count = 0
            for video_file in video_files:
                output_file = output_dir / f"{video_file.stem}_professional{video_file.suffix}"
                print(f"\nProcessing {video_file.name}...")
                
                if processor.process_video(str(video_file), str(output_file)):
                    success_count += 1
                    output_size = output_file.stat().st_size / (1024 * 1024)
                    print(f"✓ Completed: {output_file.name} ({output_size:.1f} MB)")
                else:
                    print(f"✗ Failed: {video_file.name}")
            
            print(f"\nBatch processing complete: {success_count}/{len(video_files)} videos processed")
            
        else:
            # Single video processing
            success = processor.process_video(args.input, args.output)
            if success:
                output_size = Path(args.output).stat().st_size / (1024 * 1024)
                print(f"\n✓ Professional processing complete!")
                print(f"Output: {args.output} ({output_size:.1f} MB)")
            else:
                print("✗ Processing failed")
            return 0 if success else 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())