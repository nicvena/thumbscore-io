# Training Pipeline Guide

## Overview

Complete guide for training the YouTube thumbnail ranking model from scratch using real data.

## Pipeline Architecture

```
YouTube Data API
       ↓
1) Data Collection (100k+ videos)
       ↓
2) Supabase Storage (Postgres)
       ↓
3) Pair Building (±30 days, same channel)
       ↓
4) Embedding Generation (CLIP + MiniLM)
       ↓
5) Multi-Task Training
   ├─ Pairwise Ranking Head (Margin Loss)
   ├─ Absolute CTR Head (MSE Loss)
   └─ Auxiliary Sub-Score Heads
       ↓
6) Evaluation (AUC, Spearman ρ)
       ↓
7) Model Deployment
```

## Prerequisites

### 1. API Keys

```bash
# YouTube Data API v3
export YOUTUBE_API_KEY="YOUR_YOUTUBE_API_KEY"

# Get key from: https://console.cloud.google.com/apis/credentials
```

### 2. Supabase Setup

```bash
# Supabase project
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-service-role-key"

# Sign up at: https://supabase.com
```

### 3. Install Dependencies

```bash
cd python-service/training
pip install -r requirements-training.txt
```

## Step-by-Step Training

### Step 1: Setup Database Schema

```bash
# Run this SQL in your Supabase SQL editor
```

```sql
-- Copy schema from data_preparation.py
-- Creates tables: thumbnails, channel_baselines, training_pairs, model_evaluations
```

### Step 2: Collect YouTube Data

```bash
# Collect data from YouTube
python data_preparation.py
```

This will:
- Search for channels in each niche (tech, gaming, education, etc.)
- Fetch 50 recent videos per channel
- Store in Supabase with metadata
- Compute engagement proxy (views_per_hour)
- Generate CLIP and MiniLM embeddings

**Target:** 120k-200k thumbnails across 500-1,500 channels

**Time:** ~24-48 hours (with API rate limits)

### Step 3: Build Training Pairs

```python
# Already done in data_preparation.py

# Pairwise samples:
# - Same channel
# - Within ±30 days
# - Labeled by views_per_hour (winner = higher)
# - Margin = |proxy_A - proxy_B| / max(proxy_A, proxy_B)
```

### Step 4: Train Model

```bash
# Train with default config
python pipeline.py

# Or with custom config
python pipeline.py --epochs 200 --batch-size 16 --lr 2e-5
```

**Training Configuration:**

```python
# In pipeline.py
model = ThumbnailRankingModel(
    clip_dim=768,
    text_dim=384,
    freeze_clip=True,      # Phase 1: frozen
    fine_tune_blocks=0,    # Phase 2: fine-tune last 2-4 blocks
    hidden_dim=512
)

pipeline = TrainingPipeline(
    model=model,
    learning_rate=2e-5,
    label_smoothing=0.15,   # Smoothing factor
    mixup_alpha=0.3,        # Mixup augmentation
    use_swa=True,           # Stochastic Weight Averaging
    swa_start=150           # Start SWA at epoch 150
)
```

**Expected Training Time:**
- CPU only: ~48-72 hours (not recommended)
- Single GPU (T4): ~12-18 hours
- Single GPU (A100): ~4-6 hours
- 4x GPU: ~1-2 hours

### Step 5: Monitor Training

**TensorBoard:**
```bash
tensorboard --logdir runs/
# Visit http://localhost:6006
```

**Weights & Biases:**
```bash
wandb login
# Training will auto-log to wandb.ai
```

**Console Output:**
```
Epoch   0: Pair Loss=0.2453, CTR Loss=0.0234, Val AUC=0.5234, Val Corr=0.1234
Epoch  10: Pair Loss=0.1823, CTR Loss=0.0189, Val AUC=0.6012, Val Corr=0.2567
Epoch  20: Pair Loss=0.1456, CTR Loss=0.0145, Val AUC=0.6523, Val Corr=0.3234
...
Epoch 150: Pair Loss=0.0823, CTR Loss=0.0089, Val AUC=0.7234, Val Corr=0.4789
Epoch 200: Pair Loss=0.0756, CTR Loss=0.0078, Val AUC=0.7312, Val Corr=0.5012
```

### Step 6: Evaluate Model

```bash
# Run comprehensive evaluation
python ../evaluation/evaluate.py --model models/thumbnail_ranking_model.pt
```

**Target Metrics:**
- ✅ Pairwise AUC: ≥0.65 (baseline), ≥0.72 (target)
- ✅ Spearman ρ: >0.3 (minimum), >0.5 (good)
- ✅ CTR Correlation: >0.4

### Step 7: Deploy Model

```bash
# Copy trained model to inference service
cp models/thumbnail_ranking_model.pt ../app/models/

# Update main.py to load your model
# See INTEGRATION.md for details
```

## Training Phases

### Phase 1: Baseline (Frozen CLIP)

```python
model = ThumbnailRankingModel(
    freeze_clip=True,
    fine_tune_blocks=0
)
```

**Target:** AUC ≥0.65, Correlation >0.3  
**Training time:** 12 hours  
**Data:** 120k pairs minimum

### Phase 2: Fine-Tuned (Last 2-4 Blocks)

```python
model = ThumbnailRankingModel(
    freeze_clip=False,
    fine_tune_blocks=4  # Fine-tune last 4 transformer blocks
)
```

**Target:** AUC ≥0.72, Correlation >0.5  
**Training time:** 18 hours  
**Data:** 200k pairs + YouTube Analytics CTR

### Phase 3: Production (Full Fine-Tune + Analytics)

```python
model = ThumbnailRankingModel(
    freeze_clip=False,
    fine_tune_blocks=12  # Fine-tune entire CLIP
)
```

**Target:** AUC ≥0.75, Correlation >0.6  
**Training time:** 24+ hours  
**Data:** 300k+ pairs + gold-label CTR from Analytics API

## Model Architecture Details

### Input Features

```python
# Per thumbnail:
img_embedding:   torch.Tensor  # (768,) - CLIP ViT-L/14
title_embedding: torch.Tensor  # (384,) - MiniLM
# Combined: (1152,) → projected to (512,)
```

### Three Heads

**1. Pairwise Ranking Head:**
```python
Input: features_a (512), features_b (512)
Architecture:
  Linear(512 → 256) → ReLU → Dropout(0.15)
  → Linear(256 → 128) → ReLU → Dropout(0.1)
  → Linear(128 → 1)
Output: score_a, score_b
Loss: MarginRankingLoss(margin=0.1)
```

**2. Absolute CTR Head:**
```python
Input: features (512)
Architecture:
  Linear(512 → 512) → ReLU → BatchNorm → Dropout(0.2)
  → Linear(512 → 256) → ReLU → BatchNorm → Dropout(0.15)
  → Linear(256 → 1) → Sigmoid
Output: ctr (0-1)
Loss: MSE with label smoothing (0.15)
```

**3. Auxiliary Heads (6 sub-scores):**
```python
Each head:
  Linear(512 → 256) → ReLU → Dropout(0.1)
  → Linear(256 → 1) → Sigmoid
Output: clarity, prominence, contrast, emotion, hierarchy, title_match
Loss: MSE (weak supervision from CTR)
```

## Data Requirements

### Minimum (Baseline)

- **Thumbnails**: 120,000
- **Channels**: 500
- **Pairwise samples**: 50,000
- **Niches**: 6 (tech, gaming, education, entertainment, beauty, news)
- **Target AUC**: ≥0.65

### Recommended (Production)

- **Thumbnails**: 200,000+
- **Channels**: 1,000-1,500
- **Pairwise samples**: 150,000+
- **YouTube Analytics**: True CTR for 10,000+ videos
- **Target AUC**: ≥0.72

### Data Quality

**Inclusion Criteria:**
- Videos published in last 90 days
- Minimum 1,000 views
- Active channels (≥10 videos in last 90 days)
- English language (or target language)

**Exclusion Criteria:**
- Copyrighted/removed videos
- Channels with <100 subscribers
- Videos <1 minute or >60 minutes
- Outliers (>3 std from channel mean)

## Optimization Techniques

### 1. Label Smoothing (0.15)

```python
# Instead of hard targets (0 or 1)
# Use smoothed targets: 0 → 0.075, 1 → 0.925

target_smooth = target * (1 - 0.15) + 0.5 * 0.15
```

**Benefit:** Reduces overfitting, better generalization

### 2. Mixup Augmentation (α=0.3)

```python
lam = np.random.beta(0.3, 0.3)
img_mixed = lam * img_a + (1 - lam) * img_b
```

**Benefit:** Creates synthetic training samples, improves robustness

### 3. Stochastic Weight Averaging (SWA)

```python
# After epoch 150, average model weights
swa_model.update_parameters(model)

# At end, update batch norm statistics
torch.optim.swa_utils.update_bn(dataloader, swa_model)
```

**Benefit:** More stable inference, better generalization

### 4. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Benefit:** Prevents exploding gradients, stable training

## Monitoring & Debugging

### Check Training Progress

```python
# In pipeline.py, metrics are logged every 10 epochs

if epoch % 10 == 0:
    print(f"Epoch {epoch}: AUC={val_auc:.4f}, Corr={val_corr:.4f}")
```

### Common Issues

**Issue 1: AUC stuck at 0.5**
- Model not learning
- Check learning rate (try 1e-5 to 1e-4)
- Verify labels are correct
- Increase batch size

**Issue 2: Correlation very low (<0.1)**
- Engagement proxy may be noisy
- Need more data
- Try different normalization

**Issue 3: Overfitting (train AUC >> val AUC)**
- Increase dropout (0.2 → 0.3)
- Add weight decay (0.01 → 0.02)
- Use more data augmentation
- Enable label smoothing

## Hardware Requirements

### Minimum
- **CPU**: 8+ cores
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **Training time**: 48-72 hours

### Recommended
- **GPU**: NVIDIA T4 or better (16GB VRAM)
- **RAM**: 64GB
- **Storage**: 500GB SSD
- **Training time**: 12-18 hours

### Optimal
- **GPU**: NVIDIA A100 (40GB VRAM)
- **RAM**: 128GB
- **Storage**: 1TB NVMe
- **Training time**: 4-6 hours

## Checkpointing

```python
# Save checkpoint every 50 epochs
if epoch % 50 == 0:
    pipeline.save_checkpoint(f'checkpoints/model_epoch_{epoch}.pt', epoch)
```

**Resume from checkpoint:**
```python
pipeline.load_checkpoint('checkpoints/model_epoch_100.pt')
pipeline.train(start_epoch=100, num_epochs=200)
```

## Distributed Training

### Multi-GPU (Single Machine)

```python
# Use PyTorch DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])
```

### Multi-Machine

```bash
# Use accelerate for easy distributed training
accelerate config
accelerate launch pipeline.py
```

## Next Steps After Training

1. ✅ Evaluate on held-out test set
2. ✅ Run A/B tests on live YouTube videos (Phase 2)
3. ✅ Deploy model to inference service
4. ✅ Monitor production metrics
5. 🔄 Collect feedback and retrain monthly

## Support

For training issues:
- Check logs in `runs/` directory
- Review training history in `models/training_history.json`
- Monitor GPU usage with `nvidia-smi`
- Post questions with training curves

**Ready to train your production model!** 🚀

