#!/bin/bash
# UTKFace Dataset Download Script

echo "Downloading UTKFace Dataset..."

# Create directory
mkdir -p data\images\utkface

# Download parts (you may need to adjust URLs)
wget -O data\images\utkface/part1.tar.gz "https://susanqq.github.io/UTKFace/part1.tar.gz"
wget -O data\images\utkface/part2.tar.gz "https://susanqq.github.io/UTKFace/part2.tar.gz"
wget -O data\images\utkface/part3.tar.gz "https://susanqq.github.io/UTKFace/part3.tar.gz"

# Extract files
cd data\images\utkface
tar -xzf part1.tar.gz
tar -xzf part2.tar.gz
tar -xzf part3.tar.gz

echo "UTKFace dataset downloaded and extracted!"
