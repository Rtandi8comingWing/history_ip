"""
Track Encoder: 将对象历史点轨迹编码为固定维度的 token

从 HistRISE 的 TrackEncoder 精简迁移，适配 HA-IGD：
1. Point Patch Embedding
2. Temporal Positional Encoding
3. Cross-Attention + MLP 聚合
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class TimeEmbedding(nn.Module):
    """时间位置编码（正弦版本）"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步 [B, T] 或 [B]
        Returns:
            time_emb: [B, T, dim] 或 [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class PointPatchEmbedding(nn.Module):
    """点轨迹 Patch 嵌入"""
    def __init__(
        self,
        input_dim: int = 3,
        patch_size: int = 5,
        embed_dim: int = 256
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 每个 patch 线性投影
        self.proj = nn.Linear(patch_size * input_dim, embed_dim)

    def forward(self, point_tracks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_tracks: 点轨迹 [B, N, H, P, 3] 或 [B*N, H, P, 3]
        Returns:
            patches: [B*N, H, embed_dim]
        """
        if point_tracks.ndim == 5:
            B, N, H, P, D = point_tracks.shape
            point_tracks = point_tracks.view(B * N, H, P, D)
        else:
            B, H, P, D = point_tracks.shape

        # 将 P 个点聚合成一个 patch
        # [B, H, P, D] -> [B, H, patch_size, D] -> [B, H, patch_size*D]
        if P < self.patch_size:
            # Padding
            padding = torch.zeros(B, H, self.patch_size - P, D, device=point_tracks.device)
            point_tracks = torch.cat([point_tracks, padding], dim=2)

        # 取前 patch_size 个点
        patches = point_tracks[:, :, :self.patch_size, :]  # [B, H, patch_size, D]
        patches = patches.reshape(B, H, self.patch_size * D)

        # 线性投影
        embeddings = self.proj(patches)  # [B, H, embed_dim]

        return embeddings


class CrossAttentionBlock(nn.Module):
    """Cross-Attention 聚合历史信息"""
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, kv):
        """
        Args:
            q: query [B, 1, D]
            kv: key/value [B, H, D]
        Returns:
            output: [B, 1, D]
        """
        # Cross attention
        attn_out, _ = self.attn(q, kv, kv)
        x = self.norm1(q + attn_out)

        # MLP
        mlp_out = self.mlp(x)
        out = self.norm2(x + mlp_out)

        return out


class TrackEncoder(nn.Module):
    """
    轨迹编码器：将对象历史点序列编码为固定维度 token

    输入: [B, N, H, P, 3]  (batch, num_objects, history_len, points_per_obj, dim)
    输出: [B, N, output_dim]
    """
    def __init__(
        self,
        input_dim: int = 3,
        patch_size: int = 5,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        track_age_embed_dim: int = 32
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim
        self.track_age_embed_dim = track_age_embed_dim

        # Point patch embedding
        self.patch_embed = PointPatchEmbedding(
            input_dim=input_dim,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        # Temporal positional encoding
        self.time_embed = TimeEmbedding(embed_dim)

        # Learnable query for summarization
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Cross-attention + MLP
        self.cross_attn = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # Track age embedding
        self.age_embed = nn.Linear(track_age_embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, self.output_dim)

    def forward(
        self,
        point_tracks: torch.Tensor,
        track_ages: Optional[torch.Tensor] = None,
        track_valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            point_tracks: [B, N, H, P, 3]
            track_ages: [B, N, 1] 或 [B, N] (可选，秒归一化到 [0,1])
            track_valid: [B, N] bool (可选)

        Returns:
            track_tokens: [B, N, output_dim]
        """
        B, N, H, P, D = point_tracks.shape

        # Reshape for patch embedding: [B*N, H, P, 3]
        point_tracks = point_tracks.view(B * N, H, P, D)

        # Patch embedding
        patches = self.patch_embed(point_tracks)  # [B*N, H, embed_dim]

        # Temporal positional encoding
        time_steps = torch.arange(H, device=point_tracks.device).float()
        time_emb = self.time_embed(time_steps)  # [H, embed_dim]
        patches = patches + time_emb.unsqueeze(0)  # [B*N, H, embed_dim]

        # Learnable query (expand to batch)
        query = self.query.expand(B * N, -1, -1)  # [B*N, 1, embed_dim]

        # Cross-attention summarization
        # patches: [B*N, H, D], query: [B*N, 1, D]
        summary = self.cross_attn(query, patches)  # [B*N, 1, embed_dim]
        summary = summary.squeeze(1)  # [B*N, embed_dim]

        # Add track age information
        if track_ages is not None:
            track_ages = track_ages.view(B * N, -1)  # [B*N, 1]
            # Create age embedding
            age_emb = self._get_age_embedding(track_ages)  # [B*N, embed_dim]
            summary = summary + age_emb

        # Mask invalid tracks
        if track_valid is not None:
            track_valid = track_valid.view(B * N)
            summary = summary * track_valid.float().unsqueeze(-1)

        # Output projection
        track_tokens = self.out_proj(summary)  # [B*N, output_dim]

        # Reshape back
        track_tokens = track_tokens.view(B, N, self.output_dim)

        return track_tokens

    def _get_age_embedding(self, ages: torch.Tensor) -> torch.Tensor:
        """
        将归一化的 age [0,1] 转换为 embedding

        Args:
            ages: [B, 1] 或 [B*N, 1], values in [0, 1]
        Returns:
            age_emb: [B, embed_dim] 或 [B*N, embed_dim]
        """
        # 使用正弦编码
        half_dim = self.track_age_embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=ages.device) * -embeddings)
        embeddings = ages * embeddings  # [B, half_dim]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # 投影到 embed_dim
        age_emb = self.age_embed(embeddings)
        return age_emb


# ============================================================================
# 简化版本：适用于图节点的轻量 Track Encoder
# ============================================================================
class LightTrackEncoder(nn.Module):
    """
    轻量级轨迹编码器：直接对点序列做 MLP + MaxPool
    适用于图中节点数较多的情况
    """
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 512,
        track_age_embed_dim: int = 32
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 点序列 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Track age embedding
        self.age_embed = nn.Sequential(
            nn.Linear(track_age_embed_dim, hidden_dim),
            nn.GELU()
        )

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        point_tracks: torch.Tensor,  # [B, N, H*P, 3] 或 [B*N, H*P, 3]
        track_ages: Optional[torch.Tensor] = None,  # [B, N, 1]
        track_valid: Optional[torch.Tensor] = None  # [B, N]
    ) -> torch.Tensor:
        """
        Args:
            point_tracks: 展平后的点轨迹 [B*N, H*P, 3]
            track_ages: [B, N, 1]
            track_valid: [B, N]
        """
        if point_tracks.ndim == 4:
            B, N, HP, D = point_tracks.shape
            point_tracks = point_tracks.reshape(B * N, HP, D)
        else:
            B = 1
            N = point_tracks.shape[0] if point_tracks.ndim == 3 else 1
            HP = point_tracks.shape[1] if point_tracks.ndim == 3 else point_tracks.shape[0]
            D = point_tracks.shape[-1]

        # MLP per point
        features = self.mlp(point_tracks)  # [B*N, H*P, hidden]

        # Max pool over time
        features = features.max(dim=1)[0]  # [B*N, hidden]

        # Add age
        if track_ages is not None:
            track_ages = track_ages.reshape(-1, 1)  # [B*N, 1]
            age_emb = self.age_embed(self._age_to_embedding(track_ages))
            features = features + age_emb

        # Mask invalid
        if track_valid is not None:
            track_valid = track_valid.reshape(-1)
            features = features * track_valid.float().unsqueeze(-1)

        # Output
        out = self.out_proj(features)

        if track_valid is not None:
            out = out.view(-1, N, self.output_dim)
        else:
            out = out.view(B, N, self.output_dim)

        return out

    def _age_to_embedding(self, ages: torch.Tensor) -> torch.Tensor:
        """将 [0,1] 年龄转换为 embedding"""
        half_dim = 16
        embeddings = torch.exp(torch.arange(half_dim, device=ages.device).float() * -math.log(10000) / half_dim)
        embeddings = ages * embeddings
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


# ============================================================================
# 测试
# ============================================================================
if __name__ == '__main__':
    torch.manual_seed(42)

    # Test full TrackEncoder
    B, N, H, P = 2, 3, 16, 5
    point_tracks = torch.randn(B, N, H, P, 3)
    track_ages = torch.rand(B, N, 1)  # [0,1]
    track_valid = torch.ones(B, N, dtype=torch.bool)
    track_valid[0, 2] = False

    encoder = TrackEncoder(
        input_dim=3,
        patch_size=5,
        embed_dim=128,
        output_dim=256,
        track_age_embed_dim=32
    )

    tokens = encoder(point_tracks, track_ages, track_valid)
    print("TrackEncoder output:", tokens.shape)  # [B, N, output_dim]

    # Test light encoder
    light_encoder = LightTrackEncoder(
        input_dim=3,
        hidden_dim=128,
        output_dim=256
    )

    # Flatten for light encoder
    pt_flat = point_tracks.reshape(B, N, H * P, 3)
    tokens_light = light_encoder(pt_flat, track_ages, track_valid)
    print("LightTrackEncoder output:", tokens_light.shape)

    print("\n✓ track_encoder.py test passed")
