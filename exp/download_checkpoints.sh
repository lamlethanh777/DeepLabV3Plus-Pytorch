#!/bin/bash

# Script to download pretrained DeepLabV3/V3+ models from Dropbox
# Usage: ./download_checkpoints.sh [checkpoint_dir]
# Default checkpoint_dir: checkpoints/

# Set checkpoint directory (default: checkpoints/)
CHECKPOINT_DIR="${1:-checkpoints/}"

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

echo "Downloading pretrained models to: $CHECKPOINT_DIR"
echo "================================================"

# Detect best download tool (aria2c preferred, fallback to wget)
if command -v aria2c &> /dev/null; then
    DOWNLOADER="aria2c"
    echo "Using aria2c for fast parallel downloads"
elif command -v wget &> /dev/null; then
    DOWNLOADER="wget"
    echo "Using wget for downloads"
else
    echo "Error: Neither aria2c nor wget is installed!"
    echo "Please install one of them: sudo apt-get install aria2 or sudo apt-get install wget"
    exit 1
fi
echo "================================================"

# Function to download a file from Dropbox
download_model() {
    local url=$1
    local filename=$2
    local output_path="${CHECKPOINT_DIR}${filename}"
    
    # Convert Dropbox URL from dl=0 to dl=1 for direct download
    local download_url="${url/dl=0/dl=1}"
    
    echo "Downloading: $filename"
    
    if [ "$DOWNLOADER" = "aria2c" ]; then
        aria2c -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=0 \
               -d "$(dirname "$output_path")" -o "$(basename "$output_path")" "$download_url" &
    else
        wget -q --show-progress -O "$output_path" "$download_url" &
    fi
}

# Pascal VOC2012 Aug models
echo "Downloading Pascal VOC2012 Aug models..."
download_model "https://www.dropbox.com/s/uhksxwfcim3nkpo/best_deeplabv3_mobilenet_voc_os16.pth?dl=1" "best_deeplabv3_mobilenet_voc_os16.pth"
# download_model "https://www.dropbox.com/s/3eag5ojccwiexkq/best_deeplabv3_resnet50_voc_os16.pth?dl=0" "best_deeplabv3_resnet50_voc_os16.pth"
# download_model "https://www.dropbox.com/s/vtenndnsrnh4068/best_deeplabv3_resnet101_voc_os16.pth?dl=0" "best_deeplabv3_resnet101_voc_os16.pth"
download_model "https://www.dropbox.com/s/0idrhwz6opaj7q4/best_deeplabv3plus_mobilenet_voc_os16.pth?dl=1" "best_deeplabv3plus_mobilenet_voc_os16.pth"
# download_model "https://www.dropbox.com/s/dgxyd3jkyz24voa/best_deeplabv3plus_resnet50_voc_os16.pth?dl=0" "best_deeplabv3plus_resnet50_voc_os16.pth"
# download_model "https://www.dropbox.com/s/bm3hxe7wmakaqc5/best_deeplabv3plus_resnet101_voc_os16.pth?dl=0" "best_deeplabv3plus_resnet101_voc_os16.pth"

# Cityscapes models
echo "Downloading Cityscapes models..."
download_model "https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=1" "best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
# download_model "https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing" "best_deeplabv3plus_resnet101_cityscapes_os16.pth"

# Wait for all background downloads to complete
echo ""
echo "Waiting for all downloads to complete..."
wait

echo ""
echo "================================================"
echo "All downloads completed!"
echo "Models saved to: $CHECKPOINT_DIR"