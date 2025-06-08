#!/bin/bash

# Quick test video downloads for AI upscaling evaluation
# Run this to get some immediate test files

echo "🎬 Quick Test Video Downloads for AI Upscaling"
echo "=============================================="

# Create test directory
mkdir -p test_videos
cd test_videos

echo ""
echo "📥 Downloading test videos..."

# Big Buck Bunny samples (Google's test videos)
echo "Downloading Big Buck Bunny (720p)..."
wget -O "BigBuckBunny_720p.mp4" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" || echo "Failed to download Big Buck Bunny"

echo "Downloading Elephants Dream..."
wget -O "ElephantsDream.mp4" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4" || echo "Failed to download Elephants Dream"

# Pre-compressed low quality samples
echo "Downloading compressed test samples..."
wget -O "BigBuckBunny_360p_1MB.mp4" "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4" || echo "Failed to download 360p sample"

wget -O "Jellyfish_144p_1MB.mp4" "https://test-videos.co.uk/vids/jellyfish/mp4/h264/144/Jellyfish_144_10s_1MB.mp4" || echo "Failed to download 144p sample"

# VHS glitch sample (smaller portion)
echo "Downloading VHS artifact sample..."
wget -O "vhs_artifacts.mp4" "https://archive.org/download/vhs_glitch_vol_1/vhs_glitch_vol_1.mp4" || echo "Failed to download VHS sample"

echo ""
echo "🔧 Creating additional test samples with FFmpeg..."

# Check if FFmpeg is available
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg found - creating degraded samples..."
    
    # Create 240p heavily compressed version
    if [ -f "BigBuckBunny_720p.mp4" ]; then
        echo "Creating 240p compressed version..."
        ffmpeg -i BigBuckBunny_720p.mp4 -vf "scale=320:240" -c:v libx264 -crf 45 -preset fast -c:a aac -b:a 64k -y BigBuckBunny_240p_compressed.mp4 2>/dev/null
        
        echo "Creating VHS-style degraded version..."
        ffmpeg -i BigBuckBunny_720p.mp4 -vf "scale=352:240,noise=alls=15:allf=t,unsharp=5:5:-0.5:5:5:-0.5" -c:v libx264 -crf 32 -preset fast -c:a aac -b:a 64k -y BigBuckBunny_VHS_style.mp4 2>/dev/null
        
        echo "Creating old mobile phone quality..."
        ffmpeg -i BigBuckBunny_720p.mp4 -vf "scale=176:144" -c:v libx264 -b:v 64k -preset fast -c:a aac -b:a 32k -y BigBuckBunny_mobile_64k.mp4 2>/dev/null
    fi
else
    echo "FFmpeg not found - skipping custom degraded samples"
    echo "Install FFmpeg to create additional test samples"
fi

echo ""
echo "📊 Download Summary"
echo "=================="

# List downloaded files with sizes
for file in *.mp4; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "✓ $file ($size)"
    fi
done

echo ""
echo "🚀 Ready to test! Try these commands:"
echo ""

# Suggest test commands
if [ -f "BigBuckBunny_240p_compressed.mp4" ]; then
    echo "# Test 240p heavily compressed:"
    echo "python ../video_upscaler.py BigBuckBunny_240p_compressed.mp4 output_240p_upscaled.mp4"
    echo ""
fi

if [ -f "BigBuckBunny_360p_1MB.mp4" ]; then
    echo "# Test 360p compressed sample:"
    echo "python ../video_upscaler.py BigBuckBunny_360p_1MB.mp4 output_360p_upscaled.mp4"
    echo ""
fi

if [ -f "Jellyfish_144p_1MB.mp4" ]; then
    echo "# Test extreme low quality (144p):"
    echo "python ../video_upscaler.py Jellyfish_144p_1MB.mp4 output_144p_upscaled.mp4"
    echo ""
fi

if [ -f "BigBuckBunny_VHS_style.mp4" ]; then
    echo "# Test VHS-style artifacts:"
    echo "python ../video_upscaler.py BigBuckBunny_VHS_style.mp4 output_vhs_upscaled.mp4"
    echo ""
fi

echo "# Batch process multiple files:"
echo "python ../video_upscaler.py . ../upscaled_output --batch"
echo ""

echo "💡 Tips:"
echo "- Start with the smallest files first (144p, 360p samples)"
echo "- These are only 10-second clips, perfect for testing"
echo "- Use --tile-size 1024 for your RTX 3090s"
echo "- Monitor GPU usage with nvidia-smi"

echo ""
echo "📁 All files are in: $(pwd)"