"""
HA-IGD Graph Representation: Track 节点扩展

为 GraphRep 添加 track 节点支持：
1. Track 节点编码
2. Track 相关边构建（track-geo, demo-current track, track-action）
3. Soft membership 计算
4. Curriculum dropout
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def build_track_encoder(config):
    """延迟加载 TrackEncoder"""
    from ip.models.track_encoder import TrackEncoder, LightTrackEncoder

    if config.get('use_light_track_encoder', False):
        return LightTrackEncoder(
            input_dim=3,
            hidden_dim=config.get('track_hidden_dim', 512),
            output_dim=config.get('local_nn_dim', 512),
            track_age_embed_dim=config.get('track_age_embed_dim', 32)
        )

    return TrackEncoder(
        input_dim=3,
        patch_size=config.get('track_patch_size', 4),
        embed_dim=config.get('track_hidden_dim', 512),
        num_heads=config.get('track_num_heads', 8),
        mlp_dim=config.get('track_mlp_dim', 1024),
        output_dim=config.get('local_nn_dim', 512),
        dropout=config.get('track_dropout', 0.1),
        track_age_embed_dim=config.get('track_age_embed_dim', 32),
        num_queries=config.get('track_num_queries', 1),
        num_self_layers=config.get('track_num_self_layers', 2),
    )


def compute_soft_membership_rbf(
    geo_points: torch.Tensor,      # [M, 3]
    track_points: torch.Tensor,    # [N, P, 3]
    sigma: float = 0.05
) -> torch.Tensor:
    """
    计算 Geometry 到 Track 的 RBF 软重叠

    Args:
        geo_points: [M, 3]
        track_points: [N, P, 3]
    Returns:
        overlap: [N, M]
    """
    # 取末帧
    track_last = track_points[:, -1, :, :]  # [N, P, 3]

    # 计算距离矩阵
    # geo: [M, 1, 3], track: [1, N*P, 3] -> [M, N*P]
    geo_exp = geo_points.unsqueeze(1)  # [M, 1, 3]
    track_flat = track_points.reshape(track_last.shape[0], -1, 3)  # 已经是 flat 了

    # 更高效的实现
    M = geo_points.shape[0]
    N = track_last.shape[0]
    P = track_last.shape[1]

    # [M, N, P]
    dist = torch.norm(
        geo_points[:, None, None, :] - track_last[None, :, :, :],
        dim=-1
    )  # [M, N, P]

    # 取最近
    min_dist = dist.min(dim=-1)[0]  # [M, N]

    # RBF
    overlap = torch.exp(- (min_dist ** 2) / (2 * sigma ** 2))  # [M, N]

    return overlap.T  # [N, M]


class HAIGDGraphMixin:
    """
    HA-IGD Graph 扩展 Mixin
    添加 track 节点和相关边的处理逻辑
    """

    def init_track_encoder(self, config):
        """初始化 Track 编码器"""
        if self.enable_track_nodes and self._track_encoder is None:
            self._track_encoder = build_track_encoder(config)

    def encode_tracks(self, data) -> Optional[torch.Tensor]:
        """
        编码 track 序列为节点特征

        Args:
            data: 包含 current_track_seq, current_track_valid, current_track_age_sec
        Returns:
            track_emb: [B, N, hidden_dim]
        """
        if not self.enable_track_nodes:
            return None

        if not hasattr(data, 'current_track_seq'):
            return None

        # 输入: [B, Nmax, H, P, 3]
        track_seq = data.current_track_seq
        track_valid = data.current_track_valid
        track_age = data.current_track_age_sec

        if track_seq is None:
            return None

        track_lengths = getattr(data, 'current_track_lengths', None)

        # 编码
        track_emb = self._track_encoder(
            point_tracks=track_seq,
            track_lengths=track_lengths,
            track_ages=track_age,
            track_valid=track_valid
        )

        return track_emb

    def apply_curriculum_dropout(self, track_emb: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        应用 curriculum dropout 到 track 节点

        Args:
            track_emb: [B, N, D]
            training: 是否在训练模式
        Returns:
            track_emb after dropout
        """
        if not training:
            dropout_rate = self.track_modality_dropout_eval
        else:
            dropout_rate = self.curriculum_dropout_rate

        if dropout_rate > 0 and track_emb is not None:
            # 随机 mask 掉 track 节点
            mask = torch.rand_like(track_emb[:, :, 0]) > dropout_rate  # [B, N]
            mask = mask.unsqueeze(-1)  # [B, N, 1]
            track_emb = track_emb * mask.float()

        return track_emb

    def update_curriculum_dropout(self, global_step: int):
        """
        根据训练步数更新 dropout 率

        Args:
            global_step: 当前训练步数
        """
        warmup_steps = 50000
        hold_steps = 200000
        start_rate = 0.05
        end_rate = 0.25

        if global_step < warmup_steps:
            self.curriculum_dropout_rate = start_rate
        elif global_step < warmup_steps + hold_steps:
            # Linear increase
            progress = (global_step - warmup_steps) / hold_steps
            self.curriculum_dropout_rate = start_rate + (end_rate - start_rate) * progress
        else:
            self.curriculum_dropout_rate = end_rate

    def get_track_geo_edge_attr(
        self,
        geo_pos: torch.Tensor,      # [M, 3]
        track_pos: torch.Tensor,   # [N, 3] (取末帧中心)
        sigma: float = 0.05
    ) -> torch.Tensor:
        """
        构建 Track -> Geometry 边的属性（含 soft overlap）

        Returns:
            edge_attr: [N*M, edge_dim + 1]
        """
        # 计算 soft overlap
        overlap = compute_soft_membership_rbf(
            geo_points=geo_pos,
            track_points=track_pos.unsqueeze(1),  # [N, 1, P, 3] 简化
            sigma=sigma
        )  # [N, M]

        # 相对位置编码
        rel_pos = geo_pos.unsqueeze(0) - track_pos.unsqueeze(1)  # [N, M, 3]
        rel_pos_emb = self.pos_embd(rel_pos.reshape(-1, 3))  # [N*M, pos_dim]

        # 拼接 overlap score
        overlap_flat = overlap.reshape(-1, 1)  # [N*M, 1]

        edge_attr = torch.cat([rel_pos_emb, overlap_flat], dim=-1)

        return edge_attr


# ============================================================================
# 独立的 HA-IGD Graph 构建器
# 用于更精细控制 graph 构建
# ============================================================================
class HAIGDGraphBuilder:
    """
    HA-IGD 图构建器
    独立于原有 GraphRep，提供更灵活的 track 节点管理
    """

    def __init__(self, config):
        self.config = config
        self.enable_track_nodes = config.get('enable_track_nodes', False)

        # Track encoder
        if self.enable_track_nodes:
            self.track_encoder = build_track_encoder(config)

        # Curriculum dropout state
        self.global_step = 0
        self._init_dropout_schedule(config)

    def _init_dropout_schedule(self, config):
        self.dropout_start = config.get('curriculum_dropout_start', 0.05)
        self.dropout_end = config.get('curriculum_dropout_end', 0.25)
        self.warmup_steps = config.get('curriculum_dropout_warmup_steps', 50000)
        self.hold_steps = config.get('curriculum_dropout_hold_steps', 200000)
        self.dropout_eval = config.get('track_modality_dropout_eval', 0.0)
        self.current_dropout = self.dropout_start

    def step(self):
        """更新训练步数"""
        self.global_step += 1
        self._update_dropout_rate()

    def _update_dropout_rate(self):
        if self.global_step < self.warmup_steps:
            self.current_dropout = self.dropout_start
        elif self.global_step < self.warmup_steps + self.hold_steps:
            progress = (self.global_step - self.warmup_steps) / self.hold_steps
            self.current_dropout = self.dropout_start + (self.dropout_end - self.dropout_start) * progress
        else:
            self.current_dropout = self.dropout_end

    def encode_tracks(
        self,
        track_seq: torch.Tensor,     # [B, N, H, P, 3]
        track_valid: torch.Tensor,  # [B, N]
        track_age: torch.Tensor     # [B, N, 1]
    ) -> torch.Tensor:
        """编码 track 序列"""
        if not self.enable_track_nodes or track_seq is None:
            return None

        track_lengths = getattr(data, 'current_track_lengths', None)

        return self.track_encoder(
            point_tracks=track_seq,
            track_lengths=track_lengths,
            track_ages=track_age,
            track_valid=track_valid
        )

    def apply_dropout(self, track_emb: torch.Tensor, training: bool = True) -> torch.Tensor:
        """应用 dropout"""
        if track_emb is None:
            return None

        rate = self.dropout_eval if not training else self.current_dropout

        if rate > 0:
            mask = torch.rand_like(track_emb[:, :, 0]) > rate
            track_emb = track_emb * mask.unsqueeze(-1).float()

        return track_emb


# ============================================================================
# 测试
# ============================================================================
if __name__ == '__main__':
    torch.manual_seed(42)

    # Test soft membership
    M, N, P = 16, 3, 5
    geo_points = torch.randn(M, 3)
    track_points = torch.randn(N, P, 3)

    overlap = compute_soft_membership_rbf(geo_points, track_points, sigma=0.05)
    print("Overlap shape:", overlap.shape)  # [N, M]
    print("Overlap sample:", overlap[0, :5])

    # Test graph builder
    config = {
        'enable_track_nodes': True,
        'track_hidden_dim': 256,
        'track_age_embed_dim': 32,
        'local_nn_dim': 512,
        'curriculum_dropout_start': 0.05,
        'curriculum_dropout_end': 0.25,
    }

    builder = HAIGDGraphBuilder(config)
    print("\n✓ HA-IGD graph builder initialized")

    # Test track encoding
    B = 2
    track_seq = torch.randn(B, 3, 16, 5, 3)
    track_valid = torch.ones(B, 3, dtype=torch.bool)
    track_age = torch.rand(B, 3, 1)

    track_emb = builder.encode_tracks(track_seq, track_valid, track_age)
    print("Track emb shape:", track_emb.shape)

    # Test dropout
    track_emb = builder.apply_dropout(track_emb, training=True)
    print("After dropout:", track_emb.shape)

    print("\n✓ HA-IGD graph mixin test passed")
