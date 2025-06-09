#!/usr/bin/env python3
# video_upscaler.py
"""
Video Restore - Enhanced AI Video Upscaler with High Performance
Combines artifact reduction with multi-GPU parallel processing
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Set minimal logging before imports
import logging
for logger_name in ['basicsr', 'realesrgan', 'torch', 'torchvision', 'numba', 'PIL']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# CRITICAL: Fix torchvision compatibility BEFORE basicsr imports
import types

# First, make sure torchvision is imported
import torch
import torchvision
import torchvision.transforms.functional as F

# Create compatibility module with all needed functions
fake_module = types.ModuleType('torchvision.transforms.functional_tensor')

# List of functions basicsr might need
needed_functions = [
    'rgb_to_grayscale', 'adjust_brightness', 'adjust_contrast',
    'adjust_saturation', 'adjust_hue', 'normalize', 'resize',
    'pad', 'crop', 'center_crop', 'resized_crop', 'hflip', 'vflip',
    'rotate', 'affine', 'to_tensor', 'to_pil_image', 'to_grayscale'
]

# Copy functions to fake module
for func_name in needed_functions:
    if hasattr(F, func_name):
        setattr(fake_module, func_name, getattr(F, func_name))

# Also copy all other attributes just in case
for attr in dir(F):
    if not attr.startswith('_') and not hasattr(fake_module, attr):
        try:
            setattr(fake_module, attr, getattr(F, attr))
        except:
            pass

# Register the module
sys.modules['torchvision.transforms.functional_tensor'] = fake_module

# Now we can import the rest
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Generator
import threading
import queue
import time
from dataclasses import dataclass, field
import subprocess
import multiprocessing as mp
from collections import deque
import gc
import contextlib
import io

# Optimize OpenCV for your hardware
cv2.setNumThreads(0)  # Let TBB handle threading
cv2.setUseOptimized(True)

# Third-party imports with better error handling
try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    print(f"PyTorch not found: {e}")
    print("Install with: pip install torch torchvision")
    sys.exit(1)

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
except ImportError as e:
    print(f"BasicSR not found: {e}")
    print("Install with: pip install basicsr")
    sys.exit(1)

try:
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
except ImportError as e:
    print(f"Real-ESRGAN not found: {e}")
    print("Install with: pip install realesrgan")
    sys.exit(1)

try:
    import ffmpeg
except ImportError as e:
    print(f"ffmpeg-python not found: {e}")
    print("Install with: pip install ffmpeg-python")
    sys.exit(1)

# Enable CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

@dataclass
class OptimizedConfig:
    """Optimized configuration for high-end hardware with enhancement options"""
    model_name: str = "RealESRGAN_x4plus"
    scale: int = 4
    
    # GPU settings optimized for RTX 3090
    gpu_ids: List[int] = field(default_factory=list)
    tile_size: int = 512  # Smaller for better quality with overlap
    tile_overlap: int = 32  # Default overlap
    use_fp16: bool = True
    
    # Enhanced mode settings
    enhanced_mode: bool = False
    light_denoise: bool = False  # Light denoising
    
    # Output settings
    output_format: str = "mp4"
    crf: int = 15
    preset: str = "slow"
    audio_copy: bool = True
    
    # Optimized processing
    prefetch_frames: int = 32  # Larger prefetch buffer
    
    def __post_init__(self):
        if not self.gpu_ids:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        if not self.gpu_ids:
            raise RuntimeError("No CUDA GPUs available")

class OptimizedFrameProcessor:
    """Optimized frame processing with minimal CPU usage"""
    
    def __init__(self, input_path: str, output_path: str, config: OptimizedConfig):
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.video_info = self._get_video_info()
        
        # Queues with larger capacity
        self.input_queue = queue.Queue(maxsize=config.prefetch_frames)
        self.output_queue = queue.PriorityQueue(maxsize=config.prefetch_frames)
        
        # Setup CUDA streams for each GPU
        self.cuda_streams = {}
        for gpu_id in config.gpu_ids:
            self.cuda_streams[gpu_id] = torch.cuda.Stream(device=f'cuda:{gpu_id}')
        
        # Stats tracking
        self.processed_count = 0
        self.processed_lock = threading.Lock()
    
    def _get_video_info(self) -> Dict:
        """Get video information efficiently"""
        try:
            probe = ffmpeg.probe(self.input_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            # Parse framerate
            fps_str = video_stream['r_frame_rate']
            if '/' in fps_str:
                num, den = map(float, fps_str.split('/'))
                fps = num / den
            else:
                fps = float(fps_str)
            
            # Get frame count - try multiple methods
            nb_frames = int(video_stream.get('nb_frames', 0))
            
            if nb_frames == 0:
                # Try getting from format
                if 'nb_frames' in probe.get('format', {}):
                    nb_frames = int(probe['format']['nb_frames'])
            
            if nb_frames == 0:
                # Estimate from duration
                duration = float(video_stream.get('duration', 0))
                if duration == 0 and 'duration' in probe.get('format', {}):
                    duration = float(probe['format']['duration'])
                if duration > 0:
                    nb_frames = int(duration * fps)
            
            # If still 0, count frames (slow but accurate)
            if nb_frames == 0:
                print("Warning: Could not get frame count from metadata, counting frames...")
                cmd = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                       '-show_entries', 'stream=nb_read_frames', '-of', 'default=nokey=1:noprint_wrappers=1',
                       self.input_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    nb_frames = int(result.stdout.strip())
            
            return {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': fps,
                'frames': nb_frames,
                'codec': video_stream['codec_name']
            }
        except Exception as e:
            raise RuntimeError(f"Failed to read video info: {e}")
    
    def decode_frames_gpu(self) -> Generator[np.ndarray, None, None]:
        """Decode frames using hardware acceleration if available"""
        # Try hardware decoding first
        hw_accel = self._detect_hw_accel()
        
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
        ]
        
        # Hardware acceleration must come BEFORE input file
        if hw_accel:
            cmd.extend(['-hwaccel', hw_accel])
        
        cmd.extend([
            '-i', self.input_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ])
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        frame_size = self.video_info['width'] * self.video_info['height'] * 3
        
        try:
            while True:
                data = process.stdout.read(frame_size)
                if not data or len(data) != frame_size:
                    break
                
                frame = np.frombuffer(data, dtype=np.uint8).reshape(
                    self.video_info['height'], self.video_info['width'], 3
                )
                yield frame
                
        finally:
            process.stdout.close()
            stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
            process.stderr.close()
            process.terminate()
            process.wait()
            
            if process.returncode != 0 and stderr_output:
                print(f"FFmpeg error: {stderr_output}")
    
    def _detect_hw_accel(self) -> Optional[str]:
        """Detect available hardware acceleration"""
        # Check for NVIDIA GPU decoding
        try:
            result = subprocess.run(
                ['ffmpeg', '-hwaccels'], 
                capture_output=True, text=True
            )
            if 'cuda' in result.stdout:
                return 'cuda'
            elif 'nvdec' in result.stdout:
                return 'nvdec'
        except:
            pass
        return None
    
    def increment_processed(self):
        """Thread-safe increment of processed count"""
        with self.processed_lock:
            self.processed_count += 1
            return self.processed_count

class OptimizedVideoUpscaler:
    """Main upscaler class with optimized pipeline"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.models = {}
        
        # Initialize models on each GPU
        self._setup_models()
        
        print(f"\n{'='*60}")
        print("OPTIMIZED VIDEO UPSCALER")
        print(f"{'='*60}")
        print(f"GPUs: {len(config.gpu_ids)} x RTX 3090")
        print(f"Model: {config.model_name}")
        print(f"Tile Size: {config.tile_size}x{config.tile_size}")
        if config.enhanced_mode:
            print(f"Enhanced Mode: ON")
            print(f"  - Tile Padding: {config.tile_overlap} pixels")
            print(f"  - Light Denoise: ON")
        print(f"{'='*60}\n")
    
    def _setup_models(self):
        """Setup Real-ESRGAN models on each GPU"""
        model_path = self._download_model()
        
        for gpu_id in self.config.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            
            # Create model architecture
            if self.config.model_name == "RealESRGAN_x4plus":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                              num_block=23, num_grow_ch=32, scale=4)
            elif self.config.model_name == "RealESRGAN_x4_v3":
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                      num_conv=32, upscale=4, act_type='prelu')
            elif self.config.model_name == "RealESRGAN_x4plus_anime_6B":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                              num_block=6, num_grow_ch=32, scale=4)
            else:
                raise ValueError(f"Unsupported model: {self.config.model_name}")
            
            # Create upsampler with optimized settings
            tile_pad = self.config.tile_overlap if self.config.enhanced_mode else 10
            
            upsampler = RealESRGANer(
                scale=self.config.scale,
                model_path=str(model_path),
                model=model,
                tile=self.config.tile_size,
                tile_pad=tile_pad,  # Use overlap as padding for seamless results
                pre_pad=0,
                half=self.config.use_fp16,
                gpu_id=gpu_id,
                device=device
            )
            
            self.models[gpu_id] = upsampler
    
    def _download_model(self) -> Path:
        """Download model if needed"""
        model_urls = {
            "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRGAN_x4_v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        }
        
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{self.config.model_name}.pth"
        
        if not model_path.exists():
            print(f"Downloading {self.config.model_name}...")
            url = model_urls.get(self.config.model_name)
            if not url:
                raise ValueError(f"Unknown model: {self.config.model_name}")
            
            # Use basicsr downloader (silent)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                load_file_from_url(url, model_dir=str(model_dir), 
                                 file_name=f"{self.config.model_name}.pth")
            print("✓ Download complete")
        
        return model_path
    
    def process_video(self, input_path: str, output_path: str) -> bool:
        """Main processing pipeline"""
        try:
            processor = OptimizedFrameProcessor(input_path, output_path, self.config)
            video_info = processor.video_info
            
            print(f"Input: {Path(input_path).name}")
            print(f"Resolution: {video_info['width']}x{video_info['height']} → "
                  f"{video_info['width'] * self.config.scale}x{video_info['height'] * self.config.scale}")
            print(f"Frames: {video_info['frames']} @ {video_info['fps']:.2f} FPS")
            
            # Start processing threads
            decode_thread = threading.Thread(
                target=self._decode_worker,
                args=(processor,),
                name="DecodeThread"
            )
            
            process_threads = []
            for i, gpu_id in enumerate(self.config.gpu_ids):
                thread = threading.Thread(
                    target=self._process_worker,
                    args=(processor, gpu_id, i),
                    name=f"ProcessThread-{i}"
                )
                process_threads.append(thread)
            
            encode_thread = threading.Thread(
                target=self._encode_worker,
                args=(processor,),
                name="EncodeThread"
            )
            
            # Start all threads
            decode_thread.start()
            for thread in process_threads:
                thread.start()
            encode_thread.start()
            
            # Show progress
            self._show_progress(processor)
            
            # Wait for completion
            decode_thread.join()
            for thread in process_threads:
                thread.join()
            encode_thread.join()
            
            # Copy audio if needed
            if self.config.audio_copy:
                self._copy_audio(input_path, output_path)
            
            print(f"\n✓ Processing complete: {output_path}")
            return True
            
        except Exception as e:
            print(f"\n✗ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _decode_worker(self, processor: OptimizedFrameProcessor):
        """Decode frames and add to queue"""
        frame_idx = 0
        
        try:
            for frame in processor.decode_frames_gpu():
                # Distribute frames to workers in round-robin
                worker_id = frame_idx % len(self.config.gpu_ids)
                processor.input_queue.put((frame_idx, frame, worker_id))
                frame_idx += 1
        except Exception as e:
            print(f"Decode error: {e}")
            import traceback
            traceback.print_exc()
        
        # Signal end of stream to all workers
        for worker_id in range(len(self.config.gpu_ids)):
            processor.input_queue.put((None, None, worker_id))
        
        # Update frame count if it was 0
        if processor.video_info['frames'] == 0:
            processor.video_info['frames'] = frame_idx
    
    def _process_worker(self, processor: OptimizedFrameProcessor, gpu_id: int, worker_id: int):
        """Process frames on specific GPU"""
        torch.cuda.set_device(gpu_id)
        
        upscaler = self.models[gpu_id]
        
        while True:
            try:
                frame_data = processor.input_queue.get(timeout=2.0)
                
                if not hasattr(frame_data, '__len__') or len(frame_data) != 3:
                    continue
                    
                frame_idx, frame, target_worker = frame_data
                
                if frame_idx is None:  # End signal
                    break
                
                # Only process frames assigned to this worker
                if target_worker != worker_id:
                    continue
                
                # Process frame immediately
                try:
                    self._process_frame(frame_idx, frame, upscaler, processor, gpu_id)
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except queue.Empty:
                continue
        
        # If this is the last worker, signal encode worker
        if worker_id == len(self.config.gpu_ids) - 1:
            processor.output_queue.put((None, None))
    
    def _process_frame(self, frame_idx: int, frame: np.ndarray, upscaler, 
                      processor: OptimizedFrameProcessor, gpu_id: int):
        """Process single frame"""
        with torch.cuda.stream(processor.cuda_streams[gpu_id]):
            # Light preprocessing if enabled
            if self.config.enhanced_mode and self.config.light_denoise:
                frame = cv2.bilateralFilter(frame, 5, 25, 25)
            
            # Use Real-ESRGAN
            with torch.no_grad():
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    enhanced, _ = upscaler.enhance(frame, outscale=self.config.scale)
            
            # Add to output queue
            processor.output_queue.put((frame_idx, enhanced))
            processor.increment_processed()
    
    def _encode_worker(self, processor: OptimizedFrameProcessor):
        """Encode frames to output video"""
        video_info = processor.video_info
        width = video_info['width'] * self.config.scale
        height = video_info['height'] * self.config.scale
        
        # Setup FFmpeg encoder with optimized settings
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(video_info['fps']),
            '-i', '-',
            '-an',
            '-vcodec', 'libx264',
            '-crf', str(self.config.crf),
            '-preset', self.config.preset,
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            processor.output_path
        ]
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        # Write frames in order
        expected_idx = 0
        frame_buffer = {}
        timeout_count = 0
        
        # If frame count is 0, process until we get a None signal
        expected_frames = video_info['frames'] if video_info['frames'] > 0 else float('inf')
        
        while expected_idx < expected_frames:
            try:
                data = processor.output_queue.get(timeout=5.0)
                
                # Check for end signal
                if data is None or (isinstance(data, tuple) and data[0] is None):
                    break
                
                idx, frame = data
                frame_buffer[idx] = frame
                timeout_count = 0  # Reset timeout counter
                
                # Write all consecutive frames
                while expected_idx in frame_buffer:
                    process.stdin.write(frame_buffer[expected_idx].tobytes())
                    del frame_buffer[expected_idx]
                    expected_idx += 1
                    
            except queue.Empty:
                timeout_count += 1
                if timeout_count >= 3:  # 15 seconds total
                    if expected_idx == 0:
                        print(f"\nError: No frames received. Check processing threads.")
                    else:
                        print(f"\nWarning: Timeout at frame {expected_idx}")
                    break
        
        process.stdin.close()
        process.wait()
    
    def _show_progress(self, processor: OptimizedFrameProcessor):
        """Show processing progress"""
        total_frames = processor.video_info['frames']
        start_time = time.time()
        
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="Processing", unit="frames")
            
            last_count = 0
            while last_count < total_frames:
                current_count = processor.processed_count
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                    
                    # Calculate and display FPS
                    elapsed = time.time() - start_time
                    if elapsed > 0 and current_count > 0:
                        fps = current_count / elapsed
                        pbar.set_postfix({'FPS': f'{fps:.1f}'})
                
                time.sleep(0.1)
            
            pbar.close()
            
        except ImportError:
            # Fallback progress
            print("Processing frames...")
            while processor.processed_count < total_frames:
                time.sleep(1)
    
    def _copy_audio(self, input_path: str, output_path: str):
        """Copy audio track efficiently"""
        temp_path = output_path + '.temp.mp4'
        
        try:
            (
                ffmpeg
                .output(
                    ffmpeg.input(output_path)['v'],
                    ffmpeg.input(input_path)['a'],
                    temp_path,
                    vcodec='copy',
                    acodec='copy'
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            os.replace(temp_path, output_path)
            
        except ffmpeg.Error:
            # No audio track or error
            if os.path.exists(temp_path):
                os.remove(temp_path)

def main():
    parser = argparse.ArgumentParser(
        description="Optimized AI Video Upscaler for High-End Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detects GPUs)
  python video_upscaler.py input.mp4 output.mp4
  
  # Enhanced mode for artifact reduction
  python video_upscaler.py input.mp4 output.mp4 --enhanced
  
  # Use specific GPUs
  python video_upscaler.py input.mp4 output.mp4 --gpus 0 1
  
  # Maximum quality
  python video_upscaler.py input.mp4 output.mp4 --quality max
        """
    )
    
    parser.add_argument("input", help="Input video or directory")
    parser.add_argument("output", help="Output video or directory")
    
    parser.add_argument("--model", default="RealESRGAN_x4plus",
                       choices=["RealESRGAN_x4plus", "RealESRGAN_x4_v3", "RealESRGAN_x4plus_anime_6B"],
                       help="AI model (x4plus=quality, v3=speed, anime=animation)")
    
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                       help="GPU IDs to use (default: all)")
    
    parser.add_argument("--quality", choices=["fast", "balanced", "max"], 
                       default="balanced", help="Quality preset")
    
    parser.add_argument("--enhanced", action="store_true",
                       help="Enable artifact reduction features")
    
    parser.add_argument("--tile-size", type=int, default=None,
                       help="Tile size (default: auto)")
    
    parser.add_argument("--tile-overlap", type=int, default=None,
                       help="Tile overlap for seamless mode")
    
    parser.add_argument("--crf", type=int, default=None,
                       help="Video quality (0-51, lower=better)")
    
    parser.add_argument("--preset", default=None,
                       choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
                       help="Encoding speed/quality")
    
    parser.add_argument("--no-audio", action="store_true",
                       help="Don't copy audio")
    
    parser.add_argument("--batch", action="store_true",
                       help="Process directory")
    
    args = parser.parse_args()
    
    # Configure quality presets
    if args.quality == "max":
        crf = args.crf or 12
        preset = args.preset or "veryslow"
        tile_size = args.tile_size or (512 if args.enhanced else 1536)
        tile_overlap = args.tile_overlap or (64 if args.enhanced else 32)
    elif args.quality == "fast":
        crf = args.crf or 18
        preset = args.preset or "fast"
        tile_size = args.tile_size or 1024
        tile_overlap = args.tile_overlap or 16
    else:  # balanced
        crf = args.crf or 15
        preset = args.preset or "slow"
        tile_size = args.tile_size or (512 if args.enhanced else 1024)
        tile_overlap = args.tile_overlap or (32 if args.enhanced else 16)
    
    # Create configuration
    config = OptimizedConfig(
        model_name=args.model,
        gpu_ids=args.gpus,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        crf=crf,
        preset=preset,
        audio_copy=not args.no_audio,
        enhanced_mode=args.enhanced,
        light_denoise=args.enhanced,
        use_fp16=True
    )
    
    # Set scale based on model
    config.scale = 4  # All current models are 4x
    
    try:
        upscaler = OptimizedVideoUpscaler(config)
        
        if args.batch:
            # Batch processing
            input_dir = Path(args.input)
            output_dir = Path(args.output)
            
            if not input_dir.is_dir():
                print(f"Error: {input_dir} is not a directory")
                return 1
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            videos = [f for f in input_dir.iterdir() 
                     if f.suffix.lower() in video_extensions]
            
            if not videos:
                print(f"No videos found in {input_dir}")
                return 1
            
            print(f"\nBatch processing {len(videos)} videos\n")
            
            for video in videos:
                output_path = output_dir / f"{video.stem}_upscaled{video.suffix}"
                upscaler.process_video(str(video), str(output_path))
                
        else:
            # Single video
            upscaler.process_video(args.input, args.output)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())