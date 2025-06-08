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
            futures = [executor.submit(process_gpu_batch, batch_info) for batch_info in frame_batches]
            
            # Monitor progress in separate thread to avoid blocking
            def monitor_progress():
                processed_count = 0
                last_update_time = time.time()
                
                while not all_done.is_set() or not progress_queue.empty():
                    try:
                        progress_queue.get(timeout=0.1)
                        processed_count += 1
                        
                        if use_tqdm:
                            pbar.update(1)
                        else:
                            current_time = time.time()
                            if current_time - last_update_time > 1.0:  # Update every second
                                elapsed = current_time - start_time
                                fps = processed_count / elapsed if elapsed > 0 else 0
                                eta = (total_frames - processed_count) / fps if fps > 0 else 0
                                print(f"\rProcessing: {processed_count}/{total_frames} ({processed_count/total_frames*100:.1f}%) | "
                                      f"{fps:.1f} FPS | ETA: {eta/60:.1f} min", end='', flush=True)
                                last_update_time = current_time
                    except thread_queue.Empty:
                        continue
            
            # Start progress monitor
            start_time = time.time()
            if not use_tqdm:
                monitor_thread = threading.Thread(target=monitor_progress)
                monitor_thread.start()
            else:
                monitor_thread = threading.Thread(target=monitor_progress)
                monitor_thread.start()
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    gpu_results = future.result()
                    all_output_paths.extend(gpu_results)
                except Exception as e:
                    print(f"\n⚠ GPU {self.config.gpu_ids[i]} error: {e}")
                    self.logger.error(f"GPU {self.config.gpu_ids[i]} processing error: {e}", exc_info=True)
            
            # Signal completion
            all_done.set()
            monitor_thread.join(timeout=2.0)
        
        if use_tqdm:
            pbar.close()
        else:
            print()  # New line after progress
        
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
        
        # Run ffmpeg with simple progress indicator
        try:
            # Run with progress tracking
            process = ffmpeg.run_async(output_stream, pipe_stderr=True, overwrite_output=True)
            
            # Simple progress indicator
            print("Encoding video", end='', flush=True)
            dot_count = 0
            while True:
                output = process.stderr.readline()
                if output == b'' and process.poll() is not None:
                    break
                # Show dots for progress (limit to avoid too many)
                if output and dot_count < 50:
                    print(".", end='', flush=True)
                    dot_count += 1
            print(" Done!")
            
            if process.returncode != 0:
                stderr_output = process.stderr.read()
                if stderr_output:
                    raise ffmpeg.Error('ffmpeg', '', stderr_output)
            
        except ffmpeg.Error as e:
            print(f"\n✗ Error during video encoding: Check ffmpeg installation")
            self.logger.debug(f"FFmpeg error details: {e}")
            raise RuntimeError("Video encoding failed")
    
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
            
            ffmpeg.run(output, overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)
            print("✓ Audio track added successfully")
            
        except Exception as e:
            # Check if the original video has audio
            try:
                probe = ffmpeg.probe(original_video)
                has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
                if not has_audio:
                    print("✓ No audio track in source video")
                else:
                    print("⚠ Warning: Could not copy audio track")
                    self.logger.debug(f"Audio error: {e}")
            except:
                print("⚠ Warning: Could not copy audio track")
                self.logger.debug(f"Audio error: {e}")
            
            # Copy video without audio if audio processing fails
            import shutil
            shutil.copy2(video_path, output_path)
    
    def process_video(self, input_path: str, output_path: str, temp_dir: str = "temp") -> bool:
        """Process a single video file"""
        input_path = Path(input_path)
        output_path = Path(output_path)
        temp_dir = Path(temp_dir)
        
        if not input_path.exists():
            print(f"✗ Error: Input video not found: {input_path}")
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
            
            print(f"\n{'='*60}")
            print(f"Processing: {input_path.name}")
            print(f"{'='*60}")
            
            # Get video information
            video_info = self.get_video_info(str(input_path))
            if not video_info:
                print("✗ Error: Could not read video information. File may be corrupted.")
                return False
            
            print(f"Input resolution:  {video_info['width']}x{video_info['height']}")
            print(f"Output resolution: {video_info['width']*self.config.scale}x{video_info['height']*self.config.scale}")
            print(f"FPS: {video_info['fps']:.2f}")
            print(f"Duration: {video_info['duration']:.1f}s")
            print(f"{'='*60}\n")
            
            # Extract frames
            print("Step 1/4: Extracting frames...")
            frame_paths = self.extract_frames(str(input_path), str(frames_dir))
            
            if not frame_paths:
                print("✗ Error: No frames could be extracted from the video")
                return False
            
            # Process frames with multi-GPU
            total_frames = len(frame_paths)
            print(f"\nStep 2/4: AI upscaling {total_frames} frames with {len(self.config.gpu_ids)} GPU(s)...")
            print(f"Estimated time: {total_frames / 5 / 60:.1f} minutes @ ~5 FPS")
            start_time = time.time()
            
            upscaled_paths = self.process_frames_multi_gpu(frame_paths, str(upscaled_dir))
            
            processing_time = time.time() - start_time
            fps_processed = len(upscaled_paths) / processing_time if processing_time > 0 else 0
            print(f"\n✓ AI upscaling completed: {processing_time:.1f}s ({fps_processed:.2f} FPS)")
            
            if not upscaled_paths:
                print("✗ Error: No frames were successfully upscaled")
                return False
            
            # Reassemble video
            print(f"\nStep 3/4: Reassembling video...")
            temp_video = temp_dir / "temp_video.mp4"
            self.reassemble_video(str(upscaled_dir), str(temp_video), video_info)
            
            # Add audio track
            print(f"\nStep 4/4: Adding audio track...")
            self.add_audio_track(str(temp_video), str(input_path), str(output_path))
            
            print(f"\n{'='*60}")
            print(f"✓ Success! Output saved to: {output_path}")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error processing video: {e}")
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
            print(f"✗ Error: Input directory not found: {input_dir}")
            return
        
        # Find all video files
        video_files = []
        for ext in extensions:
            video_files.extend(input_dir.glob(f"*{ext}"))
            video_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            print("✗ Error: No video files found in the directory")
            return
        
        print(f"\nBatch Processing")
        print(f"{'='*60}")
        print(f"Found {len(video_files)} video files to process")
        print(f"{'='*60}\n")
        
        # Process each video
        successful = 0
        failed = 0
        
        for i, video_file in enumerate(video_files, 1):
            print(f"[{i}/{len(video_files)}] {video_file.name}")
            
            output_file = output_dir / f"{video_file.stem}_upscaled{video_file.suffix}"
            temp_dir = Path("temp") / video_file.stem
            
            if self.process_video(str(video_file), str(output_file), str(temp_dir)):
                successful += 1
            else:
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete")
        print(f"{'='*60}")
        print(f"✓ Successful: {successful}")
        if failed > 0:
            print(f"✗ Failed: {failed}")
        print(f"{'='*60}\n")

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