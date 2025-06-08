#!/usr/bin/env python3
# video_upscaler.py
"""
Video Restore - AI-Powered Video Upscaler with Multi-GPU Support
Uses state-of-the-art models like Real-ESRGAN for video restoration and enhancement
Automatically detects and utilizes available GPUs for optimal performance
"""

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

# Suppress deprecation warnings from torchvision
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*functional.*")

# Third-party imports (install via pip)
try:
    # Import with version compatibility handling
    import torch
    import torchvision
    
    # Check PyTorch version compatibility
    torch_version = torch.__version__
    torchvision_version = torchvision.__version__
    print(f"Using PyTorch {torch_version}, torchvision {torchvision_version}")
    
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
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_dir / f"{model_name}.pth"
        if not model_path.exists():
            print(f"Downloading {model_name}...")
            load_file_from_url(
                self.MODELS[model_name]["url"],
                model_dir=str(self.model_dir),
                file_name=f"{model_name}.pth"
            )
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
            gpu_id=gpu_id
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
            raise RuntimeError("CUDA not available")
        
        available_gpus = torch.cuda.device_count()
        print(f"Detected {available_gpus} GPU{'s' if available_gpus != 1 else ''}")
        
        # Filter valid GPU IDs
        self.config.gpu_ids = [gpu_id for gpu_id in self.config.gpu_ids if gpu_id < available_gpus]
        
        if not self.config.gpu_ids:
            raise RuntimeError("No valid GPUs specified")
        
        # Auto-adjust tile size based on GPU memory if enabled
        if self.config.auto_tile_size:
            self._adjust_tile_size()
        
        print(f"Using GPU{'s' if len(self.config.gpu_ids) > 1 else ''}: {self.config.gpu_ids}")
        print(f"Tile size: {self.config.tile_size}x{self.config.tile_size}")
    
    def _adjust_tile_size(self):
        """Automatically adjust tile size based on available GPU memory"""
        min_vram = float('inf')
        
        # Find minimum VRAM across all selected GPUs
        for gpu_id in self.config.gpu_ids:
            if gpu_id < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(gpu_id)
                vram_gb = props.total_memory / (1024**3)
                min_vram = min(min_vram, vram_gb)
                print(f"GPU {gpu_id} ({props.name}): {vram_gb:.1f}GB VRAM")
        
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
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('video_upscaler.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_video_info(self, video_path: str) -> Dict:
        """Extract video information using ffprobe"""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            return {
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'fps': eval(video_info['r_frame_rate']),
                'duration': float(video_info.get('duration', 0)),
                'codec': video_info['codec_name'],
                'frames': int(video_info.get('nb_frames', 0))
            }
        except Exception as e:
            self.logger.error(f"Error getting video info: {e}")
            return {}
    
    def extract_frames(self, video_path: str, output_dir: str) -> List[str]:
        """Extract frames from video"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_path = output_dir / f"frame_{frame_count:08d}.png"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Extracted {frame_count} frames", end='\r')
        
        finally:
            cap.release()
        
        print(f"\nExtracted {len(frame_paths)} frames")
        return frame_paths
    
    def process_frame_batch(self, frame_paths: List[str], output_dir: str, gpu_id: int) -> List[str]:
        """Process a batch of frames on specified GPU"""
        upsampler = self.model_manager.load_model(self.config.model_name, gpu_id)
        
        # Update tile settings
        upsampler.tile = self.config.tile_size
        upsampler.tile_pad = self.config.tile_pad
        upsampler.pre_pad = self.config.pre_pad
        
        output_dir = Path(output_dir)
        output_paths = []
        
        for frame_path in frame_paths:
            try:
                # Read frame
                img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                
                # Upscale
                with torch.cuda.device(gpu_id):
                    output, _ = upsampler.enhance(img, outscale=self.config.scale)
                
                # Save upscaled frame
                frame_name = Path(frame_path).name
                output_path = output_dir / frame_name
                cv2.imwrite(str(output_path), output)
                output_paths.append(str(output_path))
                
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_path}: {e}")
                continue
        
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
        
        # Process batches in parallel
        all_output_paths = []
        
        def process_gpu_batch(batch_info):
            batch_frames, gpu_id = batch_info
            return self.process_frame_batch(batch_frames, output_dir, gpu_id)
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = [executor.submit(process_gpu_batch, batch_info) for batch_info in frame_batches]
            
            for i, future in enumerate(futures):
                try:
                    gpu_results = future.result()
                    all_output_paths.extend(gpu_results)
                    print(f"GPU {self.config.gpu_ids[i]} completed processing")
                except Exception as e:
                    self.logger.error(f"GPU {self.config.gpu_ids[i]} error: {e}")
        
        # Sort paths to maintain frame order
        all_output_paths.sort()
        return all_output_paths
    
    def reassemble_video(self, frame_dir: str, output_path: str, video_info: Dict):
        """Reassemble frames into video with optimal encoding"""
        frame_pattern = str(Path(frame_dir) / "frame_%08d.png")
        
        # Build ffmpeg command for high-quality encoding
        input_stream = ffmpeg.input(frame_pattern, framerate=video_info['fps'])
        
        # Video encoding options optimized for quality
        video_args = {
            'vcodec': 'libx264',
            'crf': self.config.crf,
            'preset': self.config.preset,
            'pix_fmt': 'yuv420p',
            'movflags': '+faststart'
        }
        
        if self.config.output_format == 'mp4':
            output_stream = ffmpeg.output(input_stream, output_path, **video_args)
        else:
            video_args['vcodec'] = 'libx265'  # HEVC for better compression
            output_stream = ffmpeg.output(input_stream, output_path, **video_args)
        
        # Run ffmpeg
        try:
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            self.logger.info(f"Video reassembled: {output_path}")
        except ffmpeg.Error as e:
            self.logger.error(f"FFmpeg error: {e}")
            raise
    
    def add_audio_track(self, video_path: str, original_video: str, output_path: str):
        """Add original audio track to upscaled video"""
        if not self.config.audio_copy:
            return
        
        try:
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(original_video)
            
            output = ffmpeg.output(
                video_input['v'],
                audio_input['a'],
                output_path,
                vcodec='copy',
                acodec='copy'
            )
            
            ffmpeg.run(output, overwrite_output=True, quiet=True)
            self.logger.info(f"Audio track added: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error adding audio: {e}")
            # Copy video without audio if audio processing fails
            import shutil
            shutil.copy2(video_path, output_path)
    
    def process_video(self, input_path: str, output_path: str, temp_dir: str = "temp") -> bool:
        """Process a single video file"""
        input_path = Path(input_path)
        output_path = Path(output_path)
        temp_dir = Path(temp_dir)
        
        if not input_path.exists():
            self.logger.error(f"Input video not found: {input_path}")
            return False
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup temporary directories
        frames_dir = temp_dir / "frames"
        upscaled_dir = temp_dir / "upscaled"
        
        try:
            # Clean temp directories
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True)
            
            self.logger.info(f"Processing video: {input_path}")
            
            # Get video information
            video_info = self.get_video_info(str(input_path))
            if not video_info:
                return False
            
            print(f"Input resolution: {video_info['width']}x{video_info['height']}")
            print(f"Output resolution: {video_info['width']*self.config.scale}x{video_info['height']*self.config.scale}")
            print(f"FPS: {video_info['fps']:.2f}")
            
            # Extract frames
            print("Extracting frames...")
            frame_paths = self.extract_frames(str(input_path), str(frames_dir))
            
            if not frame_paths:
                self.logger.error("No frames extracted")
                return False
            
            # Process frames with multi-GPU
            print("Processing frames with AI upscaling...")
            start_time = time.time()
            
            upscaled_paths = self.process_frames_multi_gpu(frame_paths, str(upscaled_dir))
            
            processing_time = time.time() - start_time
            fps_processed = len(upscaled_paths) / processing_time
            print(f"Processing completed in {processing_time:.1f}s ({fps_processed:.2f} FPS)")
            
            if not upscaled_paths:
                self.logger.error("No frames were successfully processed")
                return False
            
            # Reassemble video
            print("Reassembling video...")
            temp_video = temp_dir / "temp_video.mp4"
            self.reassemble_video(str(upscaled_dir), str(temp_video), video_info)
            
            # Add audio track
            print("Adding audio track...")
            self.add_audio_track(str(temp_video), str(input_path), str(output_path))
            
            self.logger.info(f"Video processing completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing video {input_path}: {e}")
            return False
        
        finally:
            # Cleanup temp files
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def process_batch(self, input_dir: str, output_dir: str, extensions: List[str] = None):
        """Process multiple videos in a directory"""
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv']
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            self.logger.error(f"Input directory not found: {input_dir}")
            return
        
        # Find all video files
        video_files = []
        for ext in extensions:
            video_files.extend(input_dir.glob(f"*{ext}"))
            video_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            self.logger.error("No video files found")
            return
        
        print(f"Found {len(video_files)} video files")
        
        # Process each video
        successful = 0
        failed = 0
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
            
            output_file = output_dir / f"{video_file.stem}_upscaled{video_file.suffix}"
            temp_dir = Path("temp") / video_file.stem
            
            if self.process_video(str(video_file), str(output_file), str(temp_dir)):
                successful += 1
            else:
                failed += 1
        
        print(f"\nBatch processing completed:")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(description="Video Restore - AI-Powered Video Upscaler with Multi-GPU Support")
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("output", help="Output video file or directory")
    parser.add_argument("--model", default="RealESRGAN_x4plus", 
                       choices=["RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x4_v3"],
                       help="AI model to use")
    parser.add_argument("--scale", type=int, help="Upscaling factor (overrides model default)")
    parser.add_argument("--tile-size", type=int, help="Tile size for processing (auto-detected by default)")
    parser.add_argument("--gpus", nargs="+", type=int, help="GPU IDs to use (auto-detect all by default)")
    parser.add_argument("--crf", type=int, default=18, help="Video quality (lower = better)")
    parser.add_argument("--preset", default="slow", help="Encoding preset")
    parser.add_argument("--batch", action="store_true", help="Process directory of videos")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio copying")
    parser.add_argument("--no-auto-tile", action="store_true", help="Disable automatic tile size adjustment")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ProcessingConfig(
        model_name=args.model,
        crf=args.crf,
        preset=args.preset,
        audio_copy=not args.no_audio,
        auto_tile_size=not args.no_auto_tile
    )
    
    # Set GPU IDs if specified
    if args.gpus:
        config.gpu_ids = args.gpus
    
    # Set tile size if specified (overrides auto-detection)
    if args.tile_size:
        config.tile_size = args.tile_size
        config.auto_tile_size = False
    
    # Override scale if specified
    if args.scale:
        config.scale = args.scale
    
    # Create processor
    processor = VideoProcessor(config)
    
    # Process videos
    if args.batch:
        processor.process_batch(args.input, args.output)
    else:
        success = processor.process_video(args.input, args.output)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()