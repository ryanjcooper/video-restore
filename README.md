# Video Restore - AI-Powered Video Upscaler

A professional-grade video restoration and upscaling tool leveraging state-of-the-art AI models (Real-ESRGAN, BSRGAN) with multi-GPU support for home video restoration and enhancement.

## Features

- **Multiple AI Models**: Support for Real-ESRGAN x2/x4, anime-optimized models, and more
- **Multi-GPU Processing**: Automatically distributes frame processing across available GPUs
- **Batch Processing**: Process entire directories of videos
- **High-Quality Output**: Configurable encoding settings with H.264/H.265 support
- **Audio Preservation**: Maintains original audio tracks
- **Memory Efficient**: FP16 processing and optimized tile-based rendering
- **Comprehensive Logging**: Detailed progress tracking and error reporting

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: 11.8 or higher
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB free space for models and temporary files

### Recommended Specifications
- **GPU**: NVIDIA RTX 3060 or better (12GB+ VRAM)
- **CPU**: 8+ cores for efficient frame extraction
- **RAM**: 32GB for processing 4K videos
- **Storage**: SSD with 50GB+ free space

### Software
- Python 3.8+
- CUDA 11.8+ 
- FFmpeg (for video processing)

## Installation

### Quick Setup (Recommended)
```bash
git clone https://github.com/ryanjcooper/video-restore.git
cd video-restore
make setup  # Installs dependencies and downloads test videos
make check  # Verify system requirements
```

### Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-upscaler.git
cd video-upscaler
```

2. Check system requirements:
```bash
python quick_start.py
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install FFmpeg if not already installed:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Quick Start

### Single Video
```bash
python video_upscaler.py input_video.mp4 output_video.mp4
```

### Batch Processing
```bash
python video_upscaler.py input_directory/ output_directory/ --batch
```

### Using Multiple GPUs
```bash
# Use GPUs 0 and 1
python video_upscaler.py input.mp4 output.mp4 --gpus 0 1

# Use all available GPUs (auto-detect)
python video_upscaler.py input.mp4 output.mp4
```

## Advanced Usage

### Model Selection
```bash
# Default: RealESRGAN_x4plus
python video_upscaler.py input.mp4 output.mp4

# 2x upscaling
python video_upscaler.py input.mp4 output.mp4 --model RealESRGAN_x2plus

# Anime-optimized model
python video_upscaler.py input.mp4 output.mp4 --model RealESRGAN_x4plus_anime_6B

# Fast model (RealESRGAN v3)
python video_upscaler.py input.mp4 output.mp4 --model RealESRGAN_x4_v3
```

### Quality Settings
```bash
# Higher quality (lower CRF = better quality, larger file)
python video_upscaler.py input.mp4 output.mp4 --crf 15

# Faster encoding (but larger file)
python video_upscaler.py input.mp4 output.mp4 --preset fast

# Best compression (slower)
python video_upscaler.py input.mp4 output.mp4 --preset veryslow
```

### Memory Optimization
```bash
# Smaller tile size for GPUs with less VRAM
python video_upscaler.py input.mp4 output.mp4 --tile-size 256

# Larger tile size for high-end GPUs (24GB+ VRAM)
python video_upscaler.py input.mp4 output.mp4 --tile-size 1024
```

## Performance Guidelines

### Tile Size Recommendations
- 4GB VRAM: `--tile-size 256`
- 8GB VRAM: `--tile-size 512` (default)
- 12GB VRAM: `--tile-size 768`
- 24GB VRAM: `--tile-size 1024`

### Processing Speed Estimates
Processing speed varies based on hardware, model, and video resolution:
- RTX 3090 (24GB): ~5-10 FPS for 1080p→4K
- RTX 4090 (24GB): ~8-15 FPS for 1080p→4K
- Dual GPUs: Nearly 2x performance

## Testing

Download test videos:
```bash
chmod +x test_video_downloads.sh  # Make executable (first time only)
./test_video_downloads.sh
```

Run tests on sample videos:
```bash
# Test on compressed 360p video
python video_upscaler.py test_videos/BigBuckBunny_360p_1MB.mp4 output_360p_upscaled.mp4

# Test on very low quality 144p
python video_upscaler.py test_videos/Jellyfish_144p_1MB.mp4 output_144p_upscaled.mp4
```

## Supported Formats

### Input
- MP4, AVI, MOV, MKV, M4V, WMV
- Any format supported by FFmpeg

### Output
- MP4 (H.264) - Default, best compatibility
- MP4 (H.265/HEVC) - Better compression
- Original container format with `--format` option

## Troubleshooting

### CUDA Out of Memory
- Reduce tile size: `--tile-size 256`
- Process on single GPU: `--gpus 0`
- Close other GPU applications

### FFmpeg Errors
- Ensure FFmpeg is installed and in PATH
- Check input video isn't corrupted
- Try different output format

### Slow Processing
- Use faster model: `--model RealESRGAN_x4_v3`
- Enable multiple GPUs
- Use faster encoding preset: `--preset fast`

## Models

| Model | Scale | Best For | Speed |
|-------|-------|----------|-------|
| RealESRGAN_x4plus | 4x | General videos | Medium |
| RealESRGAN_x2plus | 2x | Mild upscaling | Fast |
| RealESRGAN_x4plus_anime_6B | 4x | Anime/cartoons | Fast |
| RealESRGAN_x4_v3 | 4x | General (faster) | Fast |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and follow the existing code style.

## License

This project uses models and code from:
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - BSD-3-Clause License
- [BasicSR](https://github.com/xinntao/BasicSR) - Apache-2.0 License

## Acknowledgments

- Real-ESRGAN team for the amazing upscaling models
- FFmpeg team for video processing capabilities
- PyTorch team for the deep learning framework

## Citation

If you use this in research, please cite the original Real-ESRGAN paper:
```bibtex
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```