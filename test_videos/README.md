# Test Videos for Video Upscaling

This directory contains scripts and videos for testing the AI video upscaler with various types of degradation.

## Quick Start

```bash
# Make the script executable
chmod +x generate_test_videos.sh

# Generate all test videos
./generate_test_videos.sh

# Or use the Python version for more options
python create_test_videos.py
```

## Generated Test Videos

The scripts create multiple degraded versions of BigBuckBunny.mp4 and ElephantsDream.mp4:

### Clean Low-Resolution Versions
- **clean_144p** (256x144): Extremely low resolution, but clean encoding
- **clean_240p** (426x240): Old YouTube quality
- **clean_360p** (640x360): Standard definition web video
- **clean_480p** (854x480): DVD quality

### Compression Artifacts
- **heavy_compression**: 100kbps bitrate causing blocking and color banding
- **extreme_compression**: 50kbps bitrate with severe artifacts

### Old Format Styles
- **interlaced**: Simulates old TV broadcasts with interlacing artifacts
- **vhs_style**: VHS tape degradation with color shifts, noise, and audio degradation
- **blocky_mpeg**: MPEG-2 compression artifacts common in old DVDs

### Extreme Degradation
- **blurry_noisy**: Heavy blur and noise, simulating poor camera/transmission
- **old_webcam**: 160x120 @ 15fps, simulating early 2000s webcam quality
- **damaged_film**: Film grain, scratches, and faded colors
- **extreme_low_quality**: 144x108 @ 10fps - worst case scenario

## Testing Recommendations

### Quick Test (Fastest)
```bash
# Test on the most degraded video
python video_upscaler.py test_videos/degraded/BigBuckBunny_extreme_low_quality.mp4 output_test.mp4
```

### Comprehensive Test Suite
```bash
# Test different degradation types
python video_upscaler.py test_videos/degraded/BigBuckBunny_clean_144p.mp4 output_144p.mp4
python video_upscaler.py test_videos/degraded/BigBuckBunny_vhs_style.mp4 output_vhs.mp4
python video_upscaler.py test_videos/degraded/BigBuckBunny_old_webcam.mp4 output_webcam.mp4
```

### Multi-GPU Performance Test
```bash
# Process all test videos with both GPUs
python video_upscaler.py test_videos/degraded/ output_results/ --batch --gpus 0 1
```

## Expected Results

- **Clean low-res**: Should upscale well with sharp details
- **Compressed**: AI should reduce blocking artifacts
- **VHS/Interlaced**: Should clean up analog artifacts
- **Extreme degradation**: Test of model limits - expect improvement but not miracles

## Customization

Edit the scripts to:
- Change clip duration (default 10s)
- Adjust start time (default 30s into video)
- Add custom degradation filters
- Modify compression settings

## File Sizes

Degraded videos range from ~100KB (extreme low quality) to ~5MB (interlaced) for 10-second clips, making them perfect for quick testing without consuming much disk space.