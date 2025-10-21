#!/bin/bash

# Setup script for Python Inference Service

echo "🚀 Setting up Python Inference Service..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
echo "This may take 5-10 minutes for ML libraries..."
pip install -r requirements.txt

# Download models
echo -e "\n${YELLOW}Downloading ML models...${NC}"
echo "CLIP ViT-L/14 (~1.7GB)..."
python3 << EOF
import clip
import torch
print("Downloading CLIP ViT-L/14...")
model, preprocess = clip.load("ViT-L/14", device="cpu")
print("✓ CLIP downloaded successfully")
EOF

# Create models directory
mkdir -p models

# Test installation
echo -e "\n${YELLOW}Testing installation...${NC}"
python3 << EOF
import torch
import clip
from paddleocr import PaddleOCR
print("✓ PyTorch:", torch.__version__)
print("✓ CLIP: installed")
print("✓ PaddleOCR: installed")
print("✓ CUDA available:", torch.cuda.is_available())
EOF

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo -e "\nTo start the service:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo -e "  ${YELLOW}python -m app.main${NC}"
echo -e "\nOr with uvicorn:"
echo -e "  ${YELLOW}uvicorn app.main:app --reload --port 8000${NC}"
echo -e "\nOr with Docker:"
echo -e "  ${YELLOW}docker-compose up --build${NC}"

