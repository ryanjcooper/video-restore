#!/usr/bin/env python3
# video_upscaler_fixed.py
"""
Video Restore - AI-Powered Video Upscaler with Multi-GPU Support
Fixed version with proper method signatures
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
    sys.exit(1)

@dataclass
class ProcessingConfig:
    """Configuration for video processing parameters"""
    model_name: str = "RealESRGAN_x4plus"
    scale: int = 4
    tile_size: int = 512  # Default tile size, auto-adjusted based on VRAM
    tile_pad: int = 32
    pre_pad: int = 0
    face_enhance: bool = False
    gpu_ids: List[int] = None
    output_format: str = "mp4"
    crf: int = 18  # High quality encoding
    preset: str = "slow"  # Better compression
    audio_copy: bool = True
    auto_tile_size: bool = True  # Automatically adjust tile size based on VRAM
    
    def __post_init__(self):
        if self.gpu_ids is None:
            # Auto-detect available GPUs
            self.gpu_ids = list(range(torch.cuda.device_count()))
            if not self.gpu_ids:
                self.gpu_ids = [0]  # Fallback to single GPU

class ModelManager:
    """Manages AI models and their configurations"""
    
    MODELS = {
        "RealESRGAN_x4plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "scale": 4,
            "arch": RRDBNet,
            "arch_params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 4}
        },
        "RealESRGAN_x2plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "scale": 2,
            "arch": RRDBNet,
            "arch_params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 2}
        },
        "RealESRGAN_x4plus_anime_6B": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "scale": 4,
            "arch": RRDBNet,
            "arch_params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 6, "num_grow_ch": 32, "scale": 4}
        },
        "RealESRGAN_x4_v3": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "scale": 4,
            "arch": SRVGGNetCompact,
            "arch_params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_conv": 32, "upsampling": 4, "act_type": "prelu"}
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
    
    def load_model(self, model_name: str, gpu_id: int = 0) -> RealESRGANer:
        """Load and cache model on specified GPU"""
        cache_key = f"{model_name}_gpu{gpu_id}"
        
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        model_path = self.download_model(model_name)
        model_info = self.MODELS[model_name]
        
        # Initialize model architecture
        model = model_info["arch"](**model_info["arch_params"])
        
        # Create upsampler
        upsampler = RealESRGANer(
            scale=model_info["scale"],
            model_path=str(model_path),
            model=model,
            tile=512,  # Will be overridden in processing
            tile_pad=32,
            pre_pad=0,
            half=True,  # Use FP16 for memory efficiency
            gpu_id=gpu_id,
            device=torch.device(f'cuda:{gpu_id}')
        )
        
        self.loaded_models[cache_key] = upsampler
        return upsampler

class VideoProcessor:
    """Handles video processing with multi-GPU support"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model_manager = ModelManager()
        self.setup_logging()
        
        # Validate GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - GPU required for video upscaling")
        
        available_gpus = torch.cuda.device_count()
        
        # Filter valid GPU IDs
        self.config.gpu_ids = [gpu_id for gpu_id in self.config.gpu_ids if gpu_id < available_gpus]
        
        if not self.config.gpu_ids:
            raise RuntimeError("No valid GPUs specified")
        
        # Auto-adjust tile size based on GPU memory if enabled
        if self.config.auto_tile_size:
            self._adjust_tile_size()
        
        # Print GPU info
        print(f"\nGPU Configuration")
        print(f"{'='*60}")
        print(f"Available GPUs: {available_gpus}")
        for gpu_id in self.config.gpu_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            print(f"  GPU {gpu_id}: {props.name} ({props.total_memory / (1024**3):.1f}GB)")
        print(f"Tile size: {self.config.tile_size}x{self.config.tile_size}")
        print(f"Model: {self.config.model_name}")
        print(f"{'='*60}")
    
    def _adjust_tile_size(self):
        """Automatically adjust tile size based on available GPU memory"""
        min_vram = float('inf')
        
        # Find minimum VRAM across all selected GPUs
        for gpu_id in self.config.gpu_ids:
            if gpu_id < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(gpu_id)
                vram_gb = props.total_memory / (1024**3)
                min_vram = min(min_vram, vram_gb)
        
        # Adjust tile size based on minimum VRAM
        if min_vram < 6:
            self.config.tile_size = 256
        elif min_vram < 10:
            self.config.tile_size = 512
        elif min_vram < 16:
            self.config.tile_size = 768
        else:
            self.config.tile_size = 1024
    
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
            # Don't print error here, will be handled by caller
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
                
                # Progress update
                if frame_count % 30 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"\rExtracting frames: {frame_count}/{total_frames} ({progress:.1f}%)", end='', flush=True)
        
        finally:
            cap.release()
        
        print(f"\r✓ Extracted {len(frame_paths)} frames" + " " * 20)  # Extra spaces to clear line
        return frame_paths
    
    def process_frame_batch(self, frame_paths: List[str], output_dir: str, gpu_id: int, progress_queue=None) -> List[str]:
        """Process a batch of frames on specified GPU - WITH CORRECT SIGNATURE"""
        upsampler = self.model_manager.load_model(self.config.model_name, gpu_id)
        
        # Update tile settings
        upsampler.tile = self.config.tile_size
        upsampler.tile_pad = self.config.tile_pad
        upsampler.pre_pad = self.config.pre_pad
        
        output_dir = Path(output_dir)
        output_paths = []
        
        # Suppress tile output by temporarily redirecting stderr and stdout
        import os
        import sys
        
        # Save original streams
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Open null device
        devnull = open(os.devnull, 'w')
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Read frame
                img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                
                # Suppress all output
                sys.stdout = devnull
                sys.stderr = devnull
                
                try:
                    # Upscale
                    with torch.cuda.device(gpu_id):
                        output, _ = upsampler.enhance(img, outscale=self.config.scale)
                finally:
                    # Always restore streams
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                
                # Save upscaled frame
                frame_name = Path(frame_path).name
                output_path = output_dir / frame_name
                cv2.imwrite(str(output_path), output)
                output_paths.append(str(output_path))
                
                # Update progress
                if progress_queue:
                    progress_queue.put(1)
                
                # Log progress every 10 frames for debugging
                if i > 0 and i % 10 == 0:
                    self.logger.info(f"GPU {gpu_id}: Processed {i}/{len(frame_paths)} frames")
                
            except Exception as e:
                # Restore streams in case of error
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                self.logger.error(f"Error processing frame {frame_path} on GPU {gpu_id}: {e}")
                if progress_queue:
                    progress_queue.put(1)  # Still count as processed to avoid hanging
                continue
        
        # Close devnull
        devnull.close()
        
        return output_paths
    
    def process_frames_multi_gpu(self, frame_paths: List[str], output_dir: str) -> List[str]:
        """Process frames using multiple GPUs"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
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
        
        # Process batches in parallel with progress tracking
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
        
        if use_tqdm:
            pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        def process_gpu_batch(batch_info):
            batch_frames, gpu_id = batch_info
            # Call with ALL parameters including progress_queue
            return self.process_frame_batch(batch_frames, output_dir, gpu_id, progress_queue)
        
        # Start processing in threads
        import threading
        all_done = threading.Event()
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor: