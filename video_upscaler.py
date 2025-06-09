#!/usr/bin/env python3
# video_upscaler.py
"""
Video Restore - Optimized AI Video Upscaler
Rewritten for maximum performance and quality on high-end hardware
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

# Optimize OpenCV for your hardware
cv2.setNumThreads(0)  # Let TBB handle threading
cv2.setUseOptimized(True)

# Third-party imports with better error handling
try:
    import torch
    import torchvision
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
    """Optimized configuration for high-end hardware"""
    model_name: str = "RealESRGAN_x4plus"
    scale: int = 4
    
    # GPU settings optimized for RTX 3090
    gpu_ids: List[int] = field(default_factory=list)
    tile_size: int = 1536  # Larger tiles for 24GB VRAM
    tile_overlap: int = 32  # Reduced overlap
    batch_size: int = 2  # Process multiple frames per GPU
    use_fp16: bool = True
    
    # Output settings
    output_format: str = "mp4"
    crf: int = 15
    preset: str = "slow"
    audio_copy: bool = True
    
    # Optimized processing
    use_gpu_postprocess: bool = True  # GPU-accelerated post-processing
    temporal_buffer_size: int = 5  # Frames for temporal processing
    prefetch_frames: int = 32  # Larger prefetch buffer
    
    # Quality settings (simplified)
    denoise: float = 0.0  # Disabled by default - ESRGAN handles this
    sharpen: float = 0.0  # Disabled by default - causes artifacts
    color_enhance: bool = False  # Often makes things worse
    
    def __post_init__(self):
        if not self.gpu_ids:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        if not self.gpu_ids:
            raise RuntimeError("No CUDA GPUs available")

class GPUPostProcessor:
    """GPU-accelerated post-processing using PyTorch"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
    @torch.no_grad()
    def process_batch(self, frames: torch.Tensor, denoise: float = 0.0, 
                     sharpen: float = 0.0) -> torch.Tensor:
        """Process batch of frames on GPU"""
        # frames shape: [B, H, W, C] in uint8
        
        # Convert to float32 tensor on GPU
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames).to(self.device)
        
        frames = frames.float() / 255.0
        
        # Minimal processing - less is more!
        if denoise > 0:
            # Simple bilateral filter approximation on GPU
            frames = self._gpu_bilateral_filter(frames, denoise)
        
        if sharpen > 0:
            # Unsharp mask on GPU
            frames = self._gpu_unsharp_mask(frames, sharpen)
        
        # Convert back to uint8
        frames = (frames * 255).clamp(0, 255).byte()
        return frames
    
    def _gpu_bilateral_filter(self, frames: torch.Tensor, strength: float) -> torch.Tensor:
        """Approximate bilateral filter on GPU"""
        # Simple Gaussian blur as approximation (much faster)
        kernel_size = int(5 + strength * 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply Gaussian blur
        frames = frames.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        # Create Gaussian kernel
        sigma = kernel_size / 3.0
        kernel = self._gaussian_kernel(kernel_size, sigma).to(self.device)
        
        # Apply convolution
        frames = torch.nn.functional.conv2d(
            frames, kernel.expand(3, 1, kernel_size, kernel_size),
            padding=kernel_size//2, groups=3
        )
        
        frames = frames.permute(0, 2, 3, 1)  # BCHW -> BHWC
        return frames
    
    def _gpu_unsharp_mask(self, frames: torch.Tensor, strength: float) -> torch.Tensor:
        """Unsharp masking on GPU"""
        blurred = self._gpu_bilateral_filter(frames, 0.3)
        sharpened = frames + strength * (frames - blurred)
        return sharpened.clamp(0, 1)
    
    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        
        return g.unsqueeze(0) * g.unsqueeze(1)

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
            
            return {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': fps,
                'frames': int(video_stream.get('nb_frames', 0)),
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
            '-i', self.input_path,
        ]
        
        if hw_accel:
            cmd.extend(['-hwaccel', hw_accel])
        
        cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ])
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        frame_size = self.video_info['width'] * self.video_info['height'] * 3
        
        try:
            while True:
                data = process.stdout.read(frame_size)
                if not data:
                    break
                
                frame = np.frombuffer(data, dtype=np.uint8).reshape(
                    self.video_info['height'], self.video_info['width'], 3
                )
                yield frame
                
        finally:
            process.stdout.close()
            process.terminate()
            process.wait()
    
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

class OptimizedVideoUpscaler:
    """Main upscaler class with optimized pipeline"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.models = {}
        self.post_processors = {}
        
        # Initialize models on each GPU
        self._setup_models()
        
        # Setup post-processors
        for gpu_id in config.gpu_ids:
            device = torch.device(f'cuda:{gpu_id}')
            self.post_processors[gpu_id] = GPUPostProcessor(device)
        
        print(f"\n{'='*60}")
        print("OPTIMIZED VIDEO UPSCALER")
        print(f"{'='*60}")
        print(f"GPUs: {len(config.gpu_ids)} x RTX 3090")
        print(f"Model: {config.model_name}")
        print(f"Tile Size: {config.tile_size}x{config.tile_size}")
        print(f"Batch Size: {config.batch_size}")
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
            else:
                raise ValueError(f"Unsupported model: {self.config.model_name}")
            
            # Create upsampler with optimized settings
            upsampler = RealESRGANer(
                scale=self.config.scale,
                model_path=str(model_path),
                model=model,
                tile=self.config.tile_size,
                tile_pad=10,  # Minimal padding
                pre_pad=0,  # No pre-padding needed
                half=self.config.use_fp16,
                gpu_id=gpu_id,
                device=device
            )
            
            self.models[gpu_id] = upsampler
    
    def _download_model(self) -> Path:
        """Download model if needed"""
        model_urls = {
            "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRGAN_x4_v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
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
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
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
                args=(processor,)
            )
            
            process_threads = []
            for i, gpu_id in enumerate(self.config.gpu_ids):
                thread = threading.Thread(
                    target=self._process_worker,
                    args=(processor, gpu_id, i)
                )
                process_threads.append(thread)
            
            encode_thread = threading.Thread(
                target=self._encode_worker,
                args=(processor,)
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
        for frame in processor.decode_frames_gpu():
            processor.input_queue.put((frame_idx, frame))
            frame_idx += 1
        
        # Signal end of stream
        for _ in self.config.gpu_ids:
            processor.input_queue.put((None, None))
    
    def _process_worker(self, processor: OptimizedFrameProcessor, gpu_id: int, worker_id: int):
        """Process frames on specific GPU"""
        torch.cuda.set_device(gpu_id)
        model = self.models[gpu_id]
        post_processor = self.post_processors[gpu_id]
        
        # Frame buffer for batch processing
        batch_buffer = []
        batch_indices = []
        
        while True:
            try:
                frame_idx, frame = processor.input_queue.get(timeout=1.0)
                
                if frame_idx is None:  # End signal
                    break
                
                # Only process frames assigned to this worker
                if frame_idx % len(self.config.gpu_ids) != worker_id:
                    processor.input_queue.put((frame_idx, frame))
                    continue
                
                batch_buffer.append(frame)
                batch_indices.append(frame_idx)
                
                # Process batch when full
                if len(batch_buffer) >= self.config.batch_size:
                    self._process_batch(
                        batch_buffer, batch_indices, model, 
                        post_processor, processor, gpu_id
                    )
                    batch_buffer = []
                    batch_indices = []
                    
            except queue.Empty:
                continue
        
        # Process remaining frames
        if batch_buffer:
            self._process_batch(
                batch_buffer, batch_indices, model,
                post_processor, processor, gpu_id
            )
    
    def _process_batch(self, frames: List[np.ndarray], indices: List[int],
                      model: RealESRGANer, post_processor: GPUPostProcessor,
                      processor: OptimizedFrameProcessor, gpu_id: int):
        """Process batch of frames efficiently"""
        with torch.cuda.stream(self.cuda_streams[gpu_id]):
            # Process each frame
            enhanced_frames = []
            
            for frame in frames:
                # Use Real-ESRGAN (already optimized internally)
                with torch.no_grad():
                    enhanced, _ = model.enhance(frame, outscale=self.config.scale)
                enhanced_frames.append(enhanced)
            
            # Optional GPU post-processing
            if self.config.use_gpu_postprocess and (self.config.denoise > 0 or self.config.sharpen > 0):
                # Stack frames for batch processing
                batch = np.stack(enhanced_frames)
                batch_tensor = torch.from_numpy(batch).to(f'cuda:{gpu_id}')
                
                # Process on GPU
                processed = post_processor.process_batch(
                    batch_tensor, 
                    self.config.denoise,
                    self.config.sharpen
                )
                
                # Convert back to numpy
                enhanced_frames = [
                    processed[i].cpu().numpy() 
                    for i in range(len(enhanced_frames))
                ]
            
            # Add to output queue
            for idx, enhanced in zip(indices, enhanced_frames):
                processor.output_queue.put((idx, enhanced))
    
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
        
        while expected_idx < video_info['frames']:
            try:
                idx, frame = processor.output_queue.get(timeout=5.0)
                frame_buffer[idx] = frame
                
                # Write all consecutive frames
                while expected_idx in frame_buffer:
                    process.stdin.write(frame_buffer[expected_idx].tobytes())
                    del frame_buffer[expected_idx]
                    expected_idx += 1
                    
            except queue.Empty:
                if expected_idx < video_info['frames'] - 1:
                    print(f"\nWarning: Timeout waiting for frame {expected_idx}")
                break
        
        process.stdin.close()
        process.wait()
    
    def _show_progress(self, processor: OptimizedFrameProcessor):
        """Show processing progress"""
        total_frames = processor.video_info['frames']
        
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="Processing", unit="frames")
            
            last_count = 0
            while last_count < total_frames:
                current_count = processor.output_queue.qsize()
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                time.sleep(0.1)
            
            pbar.close()
            
        except ImportError:
            # Fallback progress
            print("Processing frames...")
            while processor.output_queue.qsize() < total_frames:
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
  
  # Maximum quality
  python video_upscaler.py input.mp4 output.mp4 --quality max
  
  # Use specific GPUs
  python video_upscaler.py input.mp4 output.mp4 --gpus 0 1
  
  # Batch processing
  python video_upscaler.py input_dir/ output_dir/ --batch
        """
    )
    
    parser.add_argument("input", help="Input video or directory")
    parser.add_argument("output", help="Output video or directory")
    
    parser.add_argument("--model", default="RealESRGAN_x4plus",
                       choices=["RealESRGAN_x4plus", "RealESRGAN_x4_v3"],
                       help="AI model (x4plus=quality, v3=speed)")
    
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                       help="GPU IDs to use (default: all)")
    
    parser.add_argument("--quality", choices=["fast", "balanced", "max"], 
                       default="balanced", help="Quality preset")
    
    parser.add_argument("--tile-size", type=int, default=None,
                       help="Tile size (default: auto)")
    
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Frames per batch (default: auto)")
    
    parser.add_argument("--crf", type=int, default=None,
                       help="Video quality (0-51, lower=better)")
    
    parser.add_argument("--preset", default=None,
                       choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
                       help="Encoding speed/quality")
    
    parser.add_argument("--no-audio", action="store_true",
                       help="Don't copy audio")
    
    parser.add_argument("--batch", action="store_true",
                       help="Process directory")
    
    # Post-processing (usually makes things worse)
    parser.add_argument("--denoise", type=float, default=0.0,
                       help="Denoising (0-1, default: 0)")
    parser.add_argument("--sharpen", type=float, default=0.0,
                       help="Sharpening (0-1, default: 0)")
    
    args = parser.parse_args()
    
    # Configure quality presets
    if args.quality == "max":
        crf = args.crf or 12
        preset = args.preset or "veryslow"
        tile_size = args.tile_size or 2048
        batch_size = args.batch_size or 1
    elif args.quality == "fast":
        crf = args.crf or 18
        preset = args.preset or "fast"
        tile_size = args.tile_size or 1024
        batch_size = args.batch_size or 4
    else:  # balanced
        crf = args.crf or 15
        preset = args.preset or "slow"
        tile_size = args.tile_size or 1536
        batch_size = args.batch_size or 2
    
    # Create configuration
    config = OptimizedConfig(
        model_name=args.model,
        gpu_ids=args.gpus,
        tile_size=tile_size,
        batch_size=batch_size,
        crf=crf,
        preset=preset,
        audio_copy=not args.no_audio,
        denoise=args.denoise,
        sharpen=args.sharpen,
        use_gpu_postprocess=args.denoise > 0 or args.sharpen > 0
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