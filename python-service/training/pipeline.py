"""
Training Pipeline for YouTube Thumbnail Ranking Model

Pipeline Steps:
1) Ingest via YouTube Data API → store in Postgres (Supabase)
2) Build pairs within channel, ±30 days, label by views_per_hour
3) Train pairwise + absolute heads with margin ranking loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
import json
import os

# ============================================================================
# DATASET CLASSES
# ============================================================================

class PairDataset(Dataset):
    """
    Pairwise comparison dataset
    Items: (imgA, titleA, imgB, titleB, label)
    Label: 1 if A > B, 0 if B > A (based on views_per_hour)
    """
    def __init__(self, pairs: List[Tuple]):
        """
        Args:
            pairs: List of (imgA, titleA, imgB, titleB, label, margin)
                   imgA/B: CLIP embeddings (768-dim)
                   titleA/B: MiniLM embeddings (384-dim)
                   label: 1 if A better, 0 if B better
                   margin: |views_per_hour_A - views_per_hour_B|
        """
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, i):
        return self.pairs[i]

class AbsoluteCTRDataset(Dataset):
    """
    Single-thumbnail CTR prediction dataset
    Items: (img, title, ctr_normalized, channel_baseline)
    """
    def __init__(self, samples: List[Tuple]):
        """
        Args:
            samples: List of (img_embedding, title_embedding, ctr_normalized, channel_baseline)
                    img_embedding: CLIP (768-dim)
                    title_embedding: MiniLM (384-dim)
                    ctr_normalized: 0-1 normalized CTR
                    channel_baseline: z-score within channel
        """
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        return self.samples[i]

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class RankHead(nn.Module):
    """
    Pairwise ranking head with margin ranking loss
    Predicts which thumbnail will perform better
    """
    def __init__(self, d: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xa: Thumbnail A features (batch, d)
            xb: Thumbnail B features (batch, d)
        Returns:
            sa: Score for A (batch, 1)
            sb: Score for B (batch, 1)
        """
        sa = self.mlp(xa)
        sb = self.mlp(xb)
        return sa, sb

class CTRHead(nn.Module):
    """
    Absolute CTR prediction head
    Predicts normalized CTR score for single thumbnail
    """
    def __init__(self, d: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output 0-1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Thumbnail features (batch, d)
        Returns:
            ctr: Predicted CTR (batch, 1)
        """
        return self.mlp(x)

class AuxiliaryHeads(nn.Module):
    """
    6 auxiliary sub-score heads for explainability
    """
    def __init__(self, d: int, hidden_dim: int = 256):
        super().__init__()
        
        # Individual heads for each sub-score
        self.clarity_head = self._make_head(d, hidden_dim)
        self.prominence_head = self._make_head(d, hidden_dim)
        self.contrast_head = self._make_head(d, hidden_dim)
        self.emotion_head = self._make_head(d, hidden_dim)
        self.hierarchy_head = self._make_head(d, hidden_dim)
        self.title_match_head = self._make_head(d, hidden_dim)
    
    def _make_head(self, d: int, hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Thumbnail features (batch, d)
        Returns:
            Dict of sub-scores (each batch, 1)
        """
        return {
            'clarity': self.clarity_head(x),
            'prominence': self.prominence_head(x),
            'contrast': self.contrast_head(x),
            'emotion': self.emotion_head(x),
            'hierarchy': self.hierarchy_head(x),
            'title_match': self.title_match_head(x)
        }

class ThumbnailRankingModel(nn.Module):
    """
    Complete two-stage multi-task model
    """
    def __init__(
        self,
        clip_dim: int = 768,
        text_dim: int = 384,
        freeze_clip: bool = True,
        fine_tune_blocks: int = 0,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        # Combined feature dimension
        self.feature_dim = clip_dim + text_dim  # 768 + 384 = 1152
        
        # CLIP encoder (pre-trained, optionally fine-tuned)
        self.clip_encoder = None  # Load from clip.load() externally
        self.freeze_clip = freeze_clip
        self.fine_tune_blocks = fine_tune_blocks
        
        # Text encoder (MiniLM)
        self.text_encoder = None  # Load from sentence_transformers externally
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Three heads
        self.rank_head = RankHead(hidden_dim, hidden_dim // 2)
        self.ctr_head = CTRHead(hidden_dim, hidden_dim)
        self.auxiliary_heads = AuxiliaryHeads(hidden_dim, hidden_dim // 2)
    
    def encode_features(
        self,
        img_embedding: torch.Tensor,
        title_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine image and title embeddings
        
        Args:
            img_embedding: CLIP embedding (batch, 768)
            title_embedding: MiniLM embedding (batch, 384)
        Returns:
            Combined features (batch, hidden_dim)
        """
        # Concatenate CLIP + text embeddings
        combined = torch.cat([img_embedding, title_embedding], dim=1)
        
        # Project to hidden dimension
        features = self.feature_proj(combined)
        
        return features
    
    def forward_pairwise(
        self,
        img_a: torch.Tensor,
        title_a: torch.Tensor,
        img_b: torch.Tensor,
        title_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pairwise ranking forward pass
        """
        features_a = self.encode_features(img_a, title_a)
        features_b = self.encode_features(img_b, title_b)
        
        score_a, score_b = self.rank_head(features_a, features_b)
        
        return score_a, score_b
    
    def forward_absolute(
        self,
        img: torch.Tensor,
        title: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Absolute CTR prediction forward pass
        """
        features = self.encode_features(img, title)
        
        ctr = self.ctr_head(features)
        subscores = self.auxiliary_heads(features)
        
        return ctr, subscores

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MarginRankingLoss(nn.Module):
    """
    Margin ranking loss for pairwise comparison
    Loss = max(0, margin - (score_winner - score_loser))
    """
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        score_a: torch.Tensor,
        score_b: torch.Tensor,
        label: torch.Tensor,
        margin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            score_a: Scores for thumbnail A (batch, 1)
            score_b: Scores for thumbnail B (batch, 1)
            label: 1 if A > B, 0 if B > A (batch,)
            margin: Dynamic margins based on performance gap (batch,) [optional]
        Returns:
            loss: Scalar loss
        """
        # Convert label to +1/-1 for margin ranking
        y = 2 * label - 1  # 1 → 1, 0 → -1
        
        # Use dynamic margin if provided, otherwise use fixed margin
        margin_val = margin if margin is not None else self.margin
        
        # Margin ranking loss
        loss = torch.mean(torch.clamp(margin_val - y * (score_a - score_b), min=0))
        
        return loss

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for better generalization
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (batch, 1)
            target: Targets (batch, 1)
        Returns:
            loss: MSE with label smoothing
        """
        # Apply label smoothing
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # MSE loss
        loss = nn.functional.mse_loss(pred, target_smooth)
        
        return loss

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """
    Complete training pipeline with SWA, mixup, and multi-task learning
    """
    def __init__(
        self,
        model: ThumbnailRankingModel,
        device: str = 'cuda',
        learning_rate: float = 2e-5,
        label_smoothing: float = 0.15,
        mixup_alpha: float = 0.3,
        use_swa: bool = True,
        swa_start: int = 150,
        swa_lr: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizers
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2
        )
        
        # Loss functions
        self.margin_loss = MarginRankingLoss(margin=0.1)
        self.ctr_loss = LabelSmoothingLoss(smoothing=label_smoothing)
        self.aux_loss = nn.MSELoss()
        
        # Training config
        self.mixup_alpha = mixup_alpha
        self.use_swa = use_swa
        self.swa_start = swa_start
        
        # Stochastic Weight Averaging
        if use_swa:
            from torch.optim.swa_utils import AveragedModel, SWALR
            self.swa_model = AveragedModel(model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=swa_lr)
        
        # Metrics tracking
        self.train_history = []
    
    def train_epoch_pairwise(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train one epoch on pairwise data
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            img_a, title_a, img_b, title_b, label, margin = batch
            
            # Move to device
            img_a = img_a.to(self.device)
            title_a = title_a.to(self.device)
            img_b = img_b.to(self.device)
            title_b = title_b.to(self.device)
            label = label.to(self.device).float()
            margin = margin.to(self.device).float()
            
            # Apply mixup augmentation
            if self.mixup_alpha > 0 and np.random.rand() < 0.5:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                img_a = lam * img_a + (1 - lam) * img_b
                title_a = lam * title_a + (1 - lam) * title_b
                # For mixup, we interpolate features but keep original pairs for B
            
            # Forward pass
            score_a, score_b = self.model.forward_pairwise(img_a, title_a, img_b, title_b)
            
            # Compute loss
            loss = self.margin_loss(score_a, score_b, label, margin)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def train_epoch_absolute(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train one epoch on absolute CTR data
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            img, title, ctr_target, channel_baseline = batch
            
            # Move to device
            img = img.to(self.device)
            title = title.to(self.device)
            ctr_target = ctr_target.to(self.device).float().unsqueeze(1)
            
            # Forward pass
            ctr_pred, subscores = self.model.forward_absolute(img, title)
            
            # Multi-task loss
            # 1. Main CTR loss
            loss_ctr = self.ctr_loss(ctr_pred, ctr_target)
            
            # 2. Auxiliary sub-score losses (if ground truth available)
            # For now, use self-supervised learning from CTR
            loss_aux = 0
            for key, score in subscores.items():
                # Weak supervision: sub-scores should correlate with CTR
                loss_aux += self.aux_loss(score, ctr_target)
            loss_aux = loss_aux / len(subscores)
            
            # Combined loss
            loss = loss_ctr + 0.3 * loss_aux  # Weight auxiliary loss at 0.3
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def train(
        self,
        pair_dataloader: DataLoader,
        ctr_dataloader: DataLoader,
        num_epochs: int = 200,
        val_pair_dataloader: Optional[DataLoader] = None,
        val_ctr_dataloader: Optional[DataLoader] = None
    ):
        """
        Main training loop with multi-task learning
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Pairwise batches: {len(pair_dataloader)}")
        print(f"CTR batches: {len(ctr_dataloader)}")
        
        for epoch in range(num_epochs):
            # Train both heads
            pair_loss = self.train_epoch_pairwise(pair_dataloader, epoch)
            ctr_loss = self.train_epoch_absolute(ctr_dataloader, epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Stochastic Weight Averaging
            if self.use_swa and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            
            # Validation
            if val_pair_dataloader and val_ctr_dataloader and epoch % 10 == 0:
                val_pair_auc = self.evaluate_pairwise_auc(val_pair_dataloader)
                val_ctr_corr = self.evaluate_ctr_correlation(val_ctr_dataloader)
                
                print(f"Epoch {epoch:3d}: "
                      f"Pair Loss={pair_loss:.4f}, "
                      f"CTR Loss={ctr_loss:.4f}, "
                      f"Val AUC={val_pair_auc:.4f}, "
                      f"Val Corr={val_ctr_corr:.4f}")
                
                # Save metrics
                self.train_history.append({
                    'epoch': epoch,
                    'pair_loss': pair_loss,
                    'ctr_loss': ctr_loss,
                    'val_auc': val_pair_auc,
                    'val_corr': val_ctr_corr,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            else:
                print(f"Epoch {epoch:3d}: Pair Loss={pair_loss:.4f}, CTR Loss={ctr_loss:.4f}")
        
        # Final SWA
        if self.use_swa:
            print("Updating final SWA model...")
            torch.optim.swa_utils.update_bn(pair_dataloader, self.swa_model, device=self.device)
        
        print("Training completed!")
        return self.train_history
    
    @torch.no_grad()
    def evaluate_pairwise_auc(self, dataloader: DataLoader) -> float:
        """
        Evaluate pairwise AUC on validation set
        """
        self.model.eval()
        predictions = []
        targets = []
        
        for batch in dataloader:
            img_a, title_a, img_b, title_b, label, margin = batch
            
            img_a = img_a.to(self.device)
            title_a = title_a.to(self.device)
            img_b = img_b.to(self.device)
            title_b = title_b.to(self.device)
            
            score_a, score_b = self.model.forward_pairwise(img_a, title_a, img_b, title_b)
            
            pred_diff = (score_a - score_b).cpu().numpy().flatten()
            predictions.extend(pred_diff)
            targets.extend(label.numpy().flatten())
        
        # Compute AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(targets, predictions)
        except:
            auc = 0.5
        
        return auc
    
    @torch.no_grad()
    def evaluate_ctr_correlation(self, dataloader: DataLoader) -> float:
        """
        Evaluate Spearman correlation for CTR predictions
        """
        self.model.eval()
        predictions = []
        targets = []
        
        for batch in dataloader:
            img, title, ctr_target, _ = batch
            
            img = img.to(self.device)
            title = title.to(self.device)
            
            ctr_pred, _ = self.model.forward_absolute(img, title)
            
            predictions.extend(ctr_pred.cpu().numpy().flatten())
            targets.extend(ctr_target.numpy().flatten())
        
        # Compute Spearman correlation
        from scipy.stats import spearmanr
        try:
            corr, _ = spearmanr(predictions, targets)
        except:
            corr = 0.0
        
        return corr
    
    def save_checkpoint(self, path: str, epoch: int):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history
        }
        
        if self.use_swa:
            checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        
        if self.use_swa and 'swa_model_state_dict' in checkpoint:
            self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
        
        print(f"Checkpoint loaded: {path}")

# ============================================================================
# DATA INGESTION (YouTube Data API → Supabase)
# ============================================================================

class YouTubeDataIngestion:
    """
    Ingest data from YouTube Data API and store in Supabase
    """
    def __init__(self, youtube_api_key: str, supabase_url: str, supabase_key: str):
        self.youtube_api_key = youtube_api_key
        
        # Initialize Supabase client
        try:
            from supabase import create_client
            self.supabase = create_client(supabase_url, supabase_key)
            print("✓ Supabase client initialized")
        except Exception as e:
            print(f"⚠ Supabase initialization failed: {e}")
            self.supabase = None
    
    def fetch_channel_videos(
        self,
        channel_id: str,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Fetch videos from YouTube channel
        
        Returns list of video metadata including:
        - videoId, title, thumbnailUrl
        - viewCount, likeCount, commentCount
        - publishedAt, categoryId, duration
        """
        # In production: Use YouTube Data API v3
        # from googleapiclient.discovery import build
        # youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
        
        # For now, return mock data
        videos = []
        for i in range(max_results):
            videos.append({
                'videoId': f'video_{channel_id}_{i}',
                'channelId': channel_id,
                'title': f'Sample Video {i}',
                'thumbnailUrl': f'https://example.com/thumb_{i}.jpg',
                'publishedAt': (datetime.now() - timedelta(days=i)).isoformat(),
                'viewCount': int(np.random.lognormal(10, 2)),
                'likeCount': int(np.random.lognormal(7, 2)),
                'commentCount': int(np.random.lognormal(5, 2)),
                'categoryId': '22',  # People & Blogs
                'duration': int(np.random.uniform(180, 1200))
            })
        
        return videos
    
    def compute_engagement_proxy(self, video: Dict) -> float:
        """
        Compute views_per_hour engagement proxy
        """
        published = datetime.fromisoformat(video['publishedAt'].replace('Z', '+00:00'))
        hours_since = (datetime.now() - published.replace(tzinfo=None)).total_seconds() / 3600
        
        if hours_since < 1:
            hours_since = 1
        
        views_per_hour = video['viewCount'] / hours_since
        
        # Adjust for engagement (likes, comments)
        like_ratio = video['likeCount'] / max(1, video['viewCount'])
        comment_ratio = video['commentCount'] / max(1, video['viewCount'])
        
        proxy = views_per_hour * (1 + like_ratio * 0.2 + comment_ratio * 0.1)
        
        return proxy
    
    def store_in_supabase(self, videos: List[Dict]):
        """
        Store video metadata in Supabase
        """
        if not self.supabase:
            print("⚠ Supabase not initialized, skipping storage")
            return
        
        # Compute engagement proxy
        for video in videos:
            video['engagement_proxy'] = self.compute_engagement_proxy(video)
        
        # Insert into thumbnails table
        try:
            result = self.supabase.table('thumbnails').insert(videos).execute()
            print(f"✓ Stored {len(videos)} videos in Supabase")
        except Exception as e:
            print(f"⚠ Supabase insert failed: {e}")
    
    def build_pairwise_dataset(
        self,
        channel_id: str,
        time_window_days: int = 30
    ) -> List[Tuple]:
        """
        Build pairwise comparison dataset
        
        Steps:
        1. Get all videos from channel
        2. For each pair within ±30 days
        3. Label by views_per_hour (winner = higher proxy)
        4. Compute margin = |proxy_A - proxy_B|
        """
        # Fetch videos (from Supabase or API)
        if self.supabase:
            result = self.supabase.table('thumbnails')\
                .select('*')\
                .eq('channelId', channel_id)\
                .order('publishedAt')\
                .execute()
            videos = result.data
        else:
            videos = self.fetch_channel_videos(channel_id)
        
        if len(videos) < 2:
            return []
        
        pairs = []
        
        # Build pairs within time window
        for i in range(len(videos)):
            for j in range(i + 1, len(videos)):
                video_a = videos[i]
                video_b = videos[j]
                
                # Check time window
                date_a = datetime.fromisoformat(video_a['publishedAt'].replace('Z', '+00:00'))
                date_b = datetime.fromisoformat(video_b['publishedAt'].replace('Z', '+00:00'))
                days_diff = abs((date_b - date_a).days)
                
                if days_diff > time_window_days:
                    break  # Videos are sorted, no need to check further
                
                # Compute engagement proxies
                proxy_a = self.compute_engagement_proxy(video_a)
                proxy_b = self.compute_engagement_proxy(video_b)
                
                # Determine winner
                label = 1 if proxy_a > proxy_b else 0
                margin = abs(proxy_a - proxy_b) / max(proxy_a, proxy_b)  # Normalized margin
                
                # Generate embeddings (in production, use real CLIP + MiniLM)
                img_a = torch.randn(768)  # Mock CLIP embedding
                title_a = torch.randn(384)  # Mock MiniLM embedding
                img_b = torch.randn(768)
                title_b = torch.randn(384)
                
                pairs.append((img_a, title_a, img_b, title_b, label, margin))
        
        print(f"Built {len(pairs)} pairs from {len(videos)} videos (channel: {channel_id})")
        return pairs

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """
    Complete training pipeline
    """
    print("=" * 60)
    print("YouTube Thumbnail Ranking - Training Pipeline")
    print("=" * 60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # 1. Initialize model
    print("\n1. Initializing model...")
    model = ThumbnailRankingModel(
        clip_dim=768,
        text_dim=384,
        freeze_clip=True,  # Start with frozen CLIP
        fine_tune_blocks=0,  # Fine-tune later
        hidden_dim=512
    )
    print(f"✓ Model initialized ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    # 2. Data ingestion
    print("\n2. Ingesting data from YouTube...")
    ingestion = YouTubeDataIngestion(
        youtube_api_key=os.getenv('YOUTUBE_API_KEY', ''),
        supabase_url=os.getenv('SUPABASE_URL', ''),
        supabase_key=os.getenv('SUPABASE_KEY', '')
    )
    
    # Fetch and store data for multiple channels
    channels = ['channel_1', 'channel_2', 'channel_3']  # Example channels
    all_pairs = []
    
    for channel_id in channels:
        videos = ingestion.fetch_channel_videos(channel_id, max_results=100)
        ingestion.store_in_supabase(videos)
        pairs = ingestion.build_pairwise_dataset(channel_id, time_window_days=30)
        all_pairs.extend(pairs)
    
    print(f"✓ Total pairwise samples: {len(all_pairs)}")
    
    # 3. Create datasets and dataloaders
    print("\n3. Creating dataloaders...")
    
    # Split data
    train_split = int(0.8 * len(all_pairs))
    train_pairs = all_pairs[:train_split]
    val_pairs = all_pairs[train_split:]
    
    train_dataset = PairDataset(train_pairs)
    val_dataset = PairDataset(val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Mock CTR dataset (would be built similarly from absolute CTR data)
    mock_ctr_samples = [(torch.randn(768), torch.randn(384), torch.rand(1), torch.randn(1)) 
                         for _ in range(1000)]
    ctr_dataset = AbsoluteCTRDataset(mock_ctr_samples)
    ctr_loader = DataLoader(ctr_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    print(f"✓ Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    
    # 4. Initialize training pipeline
    print("\n4. Initializing training pipeline...")
    pipeline = TrainingPipeline(
        model=model,
        device=device,
        learning_rate=2e-5,
        label_smoothing=0.15,
        mixup_alpha=0.3,
        use_swa=True,
        swa_start=150
    )
    print("✓ Pipeline ready with SWA, label smoothing, and mixup")
    
    # 5. Train
    print("\n5. Starting training...")
    history = pipeline.train(
        pair_dataloader=train_loader,
        ctr_dataloader=ctr_loader,
        num_epochs=200,
        val_pair_dataloader=val_loader,
        val_ctr_dataloader=ctr_loader
    )
    
    # 6. Save model
    print("\n6. Saving model...")
    os.makedirs('models', exist_ok=True)
    pipeline.save_checkpoint('models/thumbnail_ranking_model.pt', epoch=200)
    
    # Save training history
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nModel saved to: models/thumbnail_ranking_model.pt")
    print(f"History saved to: models/training_history.json")
    
    if history:
        final_metrics = history[-1]
        print(f"\nFinal Metrics:")
        print(f"  Pairwise AUC: {final_metrics.get('val_auc', 0):.4f}")
        print(f"  CTR Correlation: {final_metrics.get('val_corr', 0):.4f}")

if __name__ == '__main__':
    main()

