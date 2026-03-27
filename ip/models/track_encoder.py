"""
Track Encoder: 将对象历史点轨迹编码为固定维度的 token

升级后的版本更接近 HistRISE：
1. Temporal patch embedding over each tracked point
2. Learnable query cross-attention over temporal patches
3. Optional self-attention refinement over queries
4. Object-level aggregation across point tracks
"""
import math
from typing import Optional

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """时间位置编码（正弦版本）"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        device = positions.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = positions[..., None] * embeddings
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        if embeddings.shape[-1] < self.dim:
            pad = self.dim - embeddings.shape[-1]
            embeddings = torch.cat([embeddings, torch.zeros(*embeddings.shape[:-1], pad, device=device)], dim=-1)
        return embeddings


class TemporalPointPatchEmbedding(nn.Module):
    """沿时间轴分块，每个 tracked point 单独做 patch embedding。"""
    def __init__(self, input_dim: int = 3, patch_size: int = 4, embed_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size * input_dim, embed_dim)

    def forward(self, point_tracks: torch.Tensor, track_lengths: Optional[torch.Tensor] = None):
        """
        Args:
            point_tracks: [B, N, H, P, D] or [B*N, H, P, D]
            track_lengths: [B, N] or [B*N]
        Returns:
            patches: [B*N, P, num_patches, embed_dim]
            patch_lengths: [B*N]
        """
        if point_tracks.ndim == 5:
            B, N, H, P, D = point_tracks.shape
            point_tracks = point_tracks.reshape(B * N, H, P, D)
        elif point_tracks.ndim == 4:
            BN, H, P, D = point_tracks.shape
            B, N = 1, BN
        else:
            raise ValueError(f'Unsupported point_tracks shape: {tuple(point_tracks.shape)}')

        BN, H, P, D = point_tracks.shape
        patch_size = self.patch_size

        if track_lengths is not None:
            track_lengths = track_lengths.reshape(BN).to(point_tracks.device)
            padded_points = []
            padded_lengths = []
            max_padded_len = 0
            for idx in range(BN):
                actual_len = int(track_lengths[idx].item())
                actual_len = max(1, min(actual_len, H))
                seq = point_tracks[idx, :actual_len]
                if actual_len % patch_size != 0:
                    pad_len = patch_size - (actual_len % patch_size)
                    padding = seq[-1:].repeat(pad_len, 1, 1)
                    seq = torch.cat([seq, padding], dim=0)
                padded_points.append(seq)
                padded_lengths.append(seq.shape[0])
                max_padded_len = max(max_padded_len, seq.shape[0])

            final_tracks = torch.zeros(BN, max_padded_len, P, D, dtype=point_tracks.dtype, device=point_tracks.device)
            for idx, seq in enumerate(padded_points):
                final_tracks[idx, :seq.shape[0]] = seq
            point_tracks = final_tracks
            patch_lengths = torch.tensor([length // patch_size for length in padded_lengths], dtype=torch.long, device=point_tracks.device)
        else:
            if H % patch_size != 0:
                pad_len = patch_size - (H % patch_size)
                padding = point_tracks[:, -1:].repeat(1, pad_len, 1, 1)
                point_tracks = torch.cat([point_tracks, padding], dim=1)
            patch_lengths = torch.full((BN,), point_tracks.shape[1] // patch_size, dtype=torch.long, device=point_tracks.device)

        BN, H_pad, P, D = point_tracks.shape
        num_patches = H_pad // patch_size
        tracks = point_tracks.reshape(BN, num_patches, patch_size, P, D)
        tracks = tracks.permute(0, 3, 1, 2, 4).reshape(BN, P, num_patches, patch_size * D)
        patches = self.proj(tracks)
        return patches, patch_lengths


class CrossAttentionBlock(nn.Module):
    """Cross-Attention 聚合时间 patch。"""
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm_q(query), kv, kv, key_padding_mask=key_padding_mask)
        x = query + attn_out
        x = x + self.mlp(self.norm_out(x))
        return x


class SelfAttentionBlock(nn.Module):
    """Refine query tokens after cross-attention."""
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm_q(x), self.norm_q(x), self.norm_q(x))
        x = x + attn_out
        x = x + self.mlp(self.norm_out(x))
        return x


class TrackEncoder(nn.Module):
    """
    HistRISE-style track encoder.

    输入: [B, N, H, P, 3]
    输出: [B, N, output_dim]
    """
    def __init__(
        self,
        input_dim: int = 3,
        patch_size: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        track_age_embed_dim: int = 32,
        num_queries: int = 1,
        num_self_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim
        self.track_age_embed_dim = track_age_embed_dim
        self.num_queries = num_queries

        self.patch_embed = TemporalPointPatchEmbedding(
            input_dim=input_dim,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.time_embed = TimeEmbedding(embed_dim)
        self.query = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        nn.init.xavier_uniform_(self.query)

        self.cross_attn = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.self_attn_blocks = nn.ModuleList([
            SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(max(0, num_self_layers))
        ])
        self.age_embed = nn.Linear(track_age_embed_dim, embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, self.output_dim)

    def forward(
        self,
        point_tracks: torch.Tensor,
        track_lengths: Optional[torch.Tensor] = None,
        track_ages: Optional[torch.Tensor] = None,
        track_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, H, P, D = point_tracks.shape
        patches, patch_lengths = self.patch_embed(point_tracks, track_lengths=track_lengths)
        BN, num_points, num_patches, _ = patches.shape

        patch_positions = torch.arange(num_patches, device=patches.device).float()
        pos_emb = self.time_embed(patch_positions).view(1, 1, num_patches, self.embed_dim)
        patches = patches + pos_emb

        query = self.query.expand(BN, -1, -1)
        outputs = []
        patch_index = torch.arange(num_patches, device=patches.device).unsqueeze(0)
        key_padding_mask = patch_index >= patch_lengths.unsqueeze(1)

        for point_idx in range(num_points):
            point_patches = patches[:, point_idx, :, :]
            point_query = query
            point_query = self.cross_attn(point_query, point_patches, key_padding_mask=key_padding_mask)
            for self_attn in self.self_attn_blocks:
                point_query = self_attn(point_query)
            outputs.append(point_query)

        point_tokens = torch.stack(outputs, dim=1)  # [BN, P, Q, D]
        point_tokens = point_tokens.mean(dim=2)      # [BN, P, D]
        object_tokens = point_tokens.mean(dim=1)     # [BN, D]
        object_tokens = self.final_norm(object_tokens)

        if track_ages is not None:
            track_ages = track_ages.view(BN, -1)
            object_tokens = object_tokens + self._get_age_embedding(track_ages)

        valid_mask = None
        if track_valid is not None:
            track_valid = track_valid.view(BN)
            valid_mask = track_valid.float().unsqueeze(-1)
            object_tokens = object_tokens * valid_mask

        track_tokens = self.out_proj(object_tokens)
        if valid_mask is not None:
            track_tokens = track_tokens * valid_mask
        return track_tokens.view(B, N, self.output_dim)

    def _get_age_embedding(self, ages: torch.Tensor) -> torch.Tensor:
        half_dim = max(1, self.track_age_embed_dim // 2)
        embeddings = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=ages.device) * -embeddings)
        embeddings = ages * embeddings
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        if embeddings.shape[-1] < self.track_age_embed_dim:
            pad = self.track_age_embed_dim - embeddings.shape[-1]
            embeddings = torch.cat([embeddings, torch.zeros(embeddings.shape[0], pad, device=ages.device)], dim=-1)
        return self.age_embed(embeddings)


class LightTrackEncoder(nn.Module):
    """Fallback lightweight encoder used for quick ablations."""
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 512,
        track_age_embed_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.age_embed = nn.Sequential(
            nn.Linear(track_age_embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        point_tracks: torch.Tensor,
        track_lengths: Optional[torch.Tensor] = None,
        track_ages: Optional[torch.Tensor] = None,
        track_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if point_tracks.ndim == 5:
            B, N, H, P, D = point_tracks.shape
            point_tracks = point_tracks.reshape(B * N, H, P, D)
        elif point_tracks.ndim == 4:
            BN, H, P, D = point_tracks.shape
            B, N = 1, BN
        else:
            raise ValueError(f'Unsupported point_tracks shape: {tuple(point_tracks.shape)}')

        features = self.mlp(point_tracks).max(dim=2)[0]  # [B*N, H, hidden]
        if track_lengths is not None:
            track_lengths = track_lengths.reshape(-1).to(point_tracks.device)
            time_idx = torch.arange(features.shape[1], device=point_tracks.device).unsqueeze(0)
            valid_time = time_idx < track_lengths.unsqueeze(1)
            masked = features.masked_fill(~valid_time.unsqueeze(-1), float('-inf'))
            features = masked.max(dim=1)[0]
            all_invalid = track_lengths <= 0
            if all_invalid.any():
                features[all_invalid] = 0.0
        else:
            features = features.max(dim=1)[0]

        if track_ages is not None:
            track_ages = track_ages.reshape(-1, 1)
            age_emb = self.age_embed(self._age_to_embedding(track_ages))
            features = features + age_emb

        valid_mask = None
        if track_valid is not None:
            track_valid = track_valid.reshape(-1)
            valid_mask = track_valid.float().unsqueeze(-1)
            features = features * valid_mask

        out = self.out_proj(features)
        if valid_mask is not None:
            out = out * valid_mask
        return out.view(B, N, self.output_dim)

    def _age_to_embedding(self, ages: torch.Tensor) -> torch.Tensor:
        half_dim = 16
        embeddings = torch.exp(torch.arange(half_dim, device=ages.device).float() * -math.log(10000) / half_dim)
        embeddings = ages * embeddings
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings
