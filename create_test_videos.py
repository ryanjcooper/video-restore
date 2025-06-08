#!/usr/bin/env python3
# create_test_videos.py

import os
import sys
import subprocess
from pathlib import Path
import random
import numpy as np

def run_ffmpeg(cmd):
    """Run ffmpeg command and handle errors"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

def extract_clip(input_file, output_file, start_time=30, duration=10):
    """Extract a clip from the video"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(input_file),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        str(output_file)
    ]
    return run_ffmpeg(cmd)

def create_low_res_clean(input_file, output_file, resolution):
    """Create clean low-resolution version"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vf", f"scale={resolution}:-1",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_file)
    ]
    return run_ffmpeg(cmd)

def create_heavy_compression(input_file, output_file, resolution="480:360", bitrate="100k"):
    """Create heavily compressed version with artifacts"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vf", f"scale={resolution}",
        "-c:v", "libx264",
        "-b:v", bitrate,
        "-maxrate", bitrate,
        "-bufsize", "50k",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-b:a", "32k",
        str(output_file)
    ]
    return run_ffmpeg(cmd)

def create_interlaced(input_file, output_file):
    """Create interlaced video (old TV format)"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vf", "scale=720:480,interlace",
        "-c:v", "libx264",
        "-flags", "+ilme+ildct",
        "-top", "1",
        "-c:a", "copy",
        str(output_file)
    ]
    return run_ffmpeg(cmd)

def create_vhs_style(input_file, output_file):
    """Create VHS-style degradation"""
    filters = [
        "scale=640:480",
        "noise=alls=20:allf=t+u",  # Add noise
        "curves=vintage",  # Vintage color curves
        "chromashift=rx=-2:ry=1:bx=2:by=-1",  # Chromatic aberration
        "unsharp=5:5:1.5:5:5:0.0",  # Oversharpening
        "eq=saturation=0.7:contrast=1.1",  # Reduce saturation, increase contrast
        "format=yuv420p"
    ]
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vf", ",".join(filters),
        "-c:v", "libx264",
        "-crf", "28",
        "-c:a", "aac",
        "-af", "highpass=f=300,lowpass=f=3000,volume=0.8",  # Degrade audio too
        str(output_file)
    ]
    return run_ffmpeg(cmd)

def create_blocky_compressed(input_file, output_file):
    """Create blocky MPEG-style compression artifacts"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vf", "scale=352:288",
        "-c:v", "mpeg2video",
        "-b:v", "300k",
        "-maxrate", "400k",
        "-bufsize", "100k",
        "-g", "15",  # Short GOP for more keyframes
        str(output_file)
    ]
    return run_ffmpeg(cmd)

def create_blurry_noisy(input_file, output_file):
    """Create blurry and noisy version"""
    filters = [
        "scale=320:240",
        "boxblur=2:1",  # Blur
        "noise=alls=25:allf=t+u",  # Heavy noise
        "unsharp=5:5:-1.5:5:5:0.0",  # Make it look oversharpened but still blurry
        "format=yuv420p"
    ]
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vf", ",".join(filters),
        "-c:v", "libx264",
        "-crf", "32",
        "-c:a", "aac",
        "-b:a", "64k",
        str(output_file)
    ]
    return run_ffmpeg(cmd)

def create_old_webcam(input_file, output_file):
    """Create old webcam quality (very low res, bad colors, compression)"""
    filters = [
        "scale=160:120",
        "fps=15",  # Low framerate
        "curves=preset=lighter",  # Overexposed look
        "eq=saturation=0.5:contrast=1.3",  # Washed out colors
        "noise=alls=15:allf=t",
        "format=yuv420p"
    ]
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vf", ",".join(filters),
        "-c:v", "libx264",
        "-b:v", "50k",
        "-maxrate", "75k",
        "-bufsize", "25k",
        "-c:a", "aac",
        "-b:a", "16k",
        "-ar", "11025",  # Low sample rate audio
        str(output_file)
    ]
    return run_ffmpeg(cmd)

def create_damaged_film(input_file, output_file):
    """Create damaged film effect with scratches and dust"""
    # This creates a grainy, scratched look
    filters = [
        "scale=480:360",
        "curves=preset=vintage",
        "noise=alls=30:allf=t+u",  # Heavy grain
        "eq=brightness=-0.1:contrast=1.2:saturation=0.6",  # Faded colors
        # Add vertical scratches effect using geq
        "geq='lum=lum(X,Y)*(1-0.5*random(1)*lt(mod(X,40),2)):cb=cb(X,Y):cr=cr(X,Y)'",
        "format=yuv420p"
    ]
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vf", ",".join(filters),
        "-c:v", "libx264",
        "-crf", "25",
        "-c:a", "aac",
        "-af", "highpass=f=200,lowpass=f=4000,volume=0.7",
        str(output_file)
    ]
    return run_ffmpeg(cmd)

def create_extreme_low_quality(input_file, output_file):
    """Create extremely low quality version - worst case scenario"""
    filters = [
        "scale=144:108",  # Very low resolution
        "fps=10",  # Very low framerate
        "noise=alls=40:allf=t+u",  # Extreme noise
        "eq=brightness=-0.2:contrast=1.5:saturation=0.3",  # Bad exposure
        "format=yuv420p"
    ]
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vf", ",".join(filters),
        "-c:v", "libx264",
        "-b:v", "30k",
        "-maxrate", "40k",
        "-bufsize", "10k",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-b:a", "8k",
        "-ar", "8000",
        str(output_file)
    ]
    return run_ffmpeg(cmd)

def main():
    # Input videos
    input_videos = [
        "test_videos/BigBuckBunny.mp4",
        "test_videos/ElephantsDream.mp4"
    ]
    
    # Create output directory
    output_dir = Path("test_videos/degraded")
    output_dir.mkdir(exist_ok=True)
    
    # Test configurations
    test_configs = [
        # Clean versions at different resolutions
        ("clean_144p", lambda i, o: create_low_res_clean(i, o, "256:144")),
        ("clean_240p", lambda i, o: create_low_res_clean(i, o, "426:240")),
        ("clean_360p", lambda i, o: create_low_res_clean(i, o, "640:360")),
        ("clean_480p", lambda i, o: create_low_res_clean(i, o, "854:480")),
        
        # Compressed/artifact versions
        ("heavy_compression", lambda i, o: create_heavy_compression(i, o, "480:360", "150k")),
        ("extreme_compression", lambda i, o: create_heavy_compression(i, o, "320:240", "50k")),
        
        # Old format styles
        ("interlaced", create_interlaced),
        ("vhs_style", create_vhs_style),
        ("blocky_mpeg", create_blocky_compressed),
        
        # Degraded versions
        ("blurry_noisy", create_blurry_noisy),
        ("old_webcam", create_old_webcam),
        ("damaged_film", create_damaged_film),
        ("extreme_low_quality", create_extreme_low_quality),
    ]
    
    # Process each video
    for input_video in input_videos:
        if not Path(input_video).exists():
            print(f"Warning: {input_video} not found, skipping...")
            continue
        
        video_name = Path(input_video).stem
        print(f"\nProcessing {video_name}...")
        
        # First extract 10-second clip
        clip_path = output_dir / f"{video_name}_10s_clip.mp4"
        if not extract_clip(input_video, clip_path, start_time=30, duration=10):
            print(f"Failed to extract clip from {input_video}")
            continue
        
        # Create all test versions
        for test_name, process_func in test_configs:
            output_path = output_dir / f"{video_name}_{test_name}.mp4"
            print(f"\nCreating {test_name} version...")
            
            if not process_func(clip_path, output_path):
                print(f"Failed to create {test_name} version")
            else:
                # Get file size
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ Created {output_path.name} ({size_mb:.1f} MB)")
    
    # Create a combined showcase reel
    print("\nCreating showcase reel with all degradation types...")
    create_showcase_reel(output_dir)
    
    print("\n✓ Test video generation complete!")
    print(f"Generated videos are in: {output_dir}")
    
    # List all generated files
    print("\nGenerated files:")
    for video_file in sorted(output_dir.glob("*.mp4")):
        size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"  {video_file.name} ({size_mb:.1f} MB)")

def create_showcase_reel(output_dir):
    """Create a side-by-side comparison video of all degradation types"""
    # This would require more complex ffmpeg commands with multiple inputs
    # For now, we'll skip this feature
    pass

if __name__ == "__main__":
    # Check if ffmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH")
        print("Install with: sudo apt install ffmpeg")
        sys.exit(1)
    
    main()