# models_modern.py
import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """
    Pre-norm residual MLP block:
      x -> LN -> Linear(dim -> expansion*dim) -> GELU -> Dropout
         -> Linear(expansion*dim -> dim) -> Dropout -> +x
    """

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden_dim = expansion * dim
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class ModernPairEncoder(nn.Module):
    """
    Encodes pairwise interactions (f, c) into a message vector m_{f,c}.

    Input per pair: [s_f^{t-1}, s_f^t, s_c^{t-1}, s_c^t] \in R^{4D}
    Output: message \in R^{msg_dim}
    """

    def __init__(
        self,
        state_dim: int,
        msg_dim: int = 96,
        num_blocks: int = 3,
        dropout: float = 0.0,
        expansion: int = 2,
    ):
        super().__init__()
        in_dim = 4 * state_dim

        self.fc_in = nn.Linear(in_dim, msg_dim, bias=True)
        self.act_in = nn.GELU()
        self.dropout_in = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    dim=msg_dim,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, pair_feats: torch.Tensor) -> torch.Tensor:
        """
        pair_feats: [B*N*N, 4*D]
        returns:    [B*N*N, msg_dim]
        """
        x = self.fc_in(pair_feats)
        x = self.act_in(x)
        x = self.dropout_in(x)
        for block in self.blocks:
            x = block(x)
        return x


class ModernDecoder(nn.Module):
    """
    Decodes aggregated message and self state into next-step velocity.

    Input per object:
      [agg_msg_f, s_f^{t-1}, s_f^t] \in R^{msg_dim + 2D}
    Output:
      v^{t+1}_f \in R^2  (normalized velocity)
    """

    def __init__(
        self,
        state_dim: int,
        msg_dim: int = 96,
        num_blocks: int = 3,
        dropout: float = 0.0,
        expansion: int = 2,
    ):
        super().__init__()
        in_dim = msg_dim + 2 * state_dim

        self.fc_in = nn.Linear(in_dim, msg_dim, bias=True)
        self.act_in = nn.GELU()
        self.dropout_in = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    dim=msg_dim,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.norm_out = nn.LayerNorm(msg_dim)
        self.fc_out = nn.Linear(msg_dim, 2, bias=True)  # predict vx, vy

    def forward(self, dec_in: torch.Tensor) -> torch.Tensor:
        """
        dec_in: [B*N, msg_dim + 2D]
        returns: [B*N, 2]
        """
        x = self.fc_in(dec_in)
        x = self.act_in(x)
        x = self.dropout_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        x = self.fc_out(x)
        return x


class ModernNeuralPhysicsEngine(nn.Module):
    """
    Modern NPE:
      - Pairwise encoder with residual MLP + biases
      - Neighborhood mask + sum aggregation
      - Residual decoder to velocities at t+1

    Input:  window [B, 3, N, D]  (frames t-1, t, t+1)
    Output: (pred_v, target_v) each [B, N, 2] in normalized units.
    """

    def __init__(
        self,
        state_dim: int,
        neighborhood_radius: float,
        msg_dim: int = 96,
        enc_blocks: int = 3,
        dec_blocks: int = 3,
        dropout: float = 0.0,
        expansion: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.msg_dim = msg_dim
        self.neighborhood_radius = neighborhood_radius

        self.encoder = ModernPairEncoder(
            state_dim=state_dim,
            msg_dim=msg_dim,
            num_blocks=enc_blocks,
            dropout=dropout,
            expansion=expansion,
        )
        self.decoder = ModernDecoder(
            state_dim=state_dim,
            msg_dim=msg_dim,
            num_blocks=dec_blocks,
            dropout=dropout,
            expansion=expansion,
        )

    # ---------- internals ----------

    def _build_pair_features(self, s_tm1, s_t):
        """
        s_tm1, s_t: [B, N, D]
        returns pair_feats_flat: [B*N*N, 4D]
        """
        B, N, D = s_tm1.shape

        # focus
        s_tm1_f = s_tm1.unsqueeze(2).expand(B, N, N, D)
        s_t_f = s_t.unsqueeze(2).expand(B, N, N, D)
        # context
        s_tm1_c = s_tm1.unsqueeze(1).expand(B, N, N, D)
        s_t_c = s_t.unsqueeze(1).expand(B, N, N, D)

        pair_feats = torch.cat([s_tm1_f, s_t_f, s_tm1_c, s_t_c], dim=-1)  # [B,N,N,4D]
        pair_feats_flat = pair_feats.reshape(B * N * N, 4 * D)
        return pair_feats_flat

    def _build_neighbor_mask(self, s_t):
        """
        s_t: [B, N, D]
        returns neighbor_mask: [B, N, N] bool, excluding self.
        """
        B, N, D = s_t.shape
        pos_t = s_t[..., 0:2]  # [B,N,2]

        rel = pos_t.unsqueeze(2) - pos_t.unsqueeze(1)  # [B,N,N,2]
        dist = torch.linalg.norm(rel, dim=-1)          # [B,N,N]
        mask = dist <= self.neighborhood_radius

        eye = torch.eye(N, dtype=torch.bool, device=s_t.device).unsqueeze(0)
        mask = mask & (~eye)  # no self
        return mask

    # ---------- forward ----------

    def forward(self, window: torch.Tensor):
        """
        window: [B, 3, N, D]
        Returns:
            pred_v:   [B, N, 2] normalized velocities at t+1
            target_v: [B, N, 2] from ground truth at t+1
        """
        assert window.dim() == 4, "window should be [B, 3, N, D]"
        B, T, N, D = window.shape
        assert T >= 3
        assert D == self.state_dim

        s_tm1 = window[:, 0]  # [B,N,D]
        s_t   = window[:, 1]
        s_tp1 = window[:, 2]

        # ----- encoder: pairwise messages -----
        pair_feats_flat = self._build_pair_features(s_tm1, s_t)  # [B*N*N, 4D]
        msg_flat = self.encoder(pair_feats_flat)                 # [B*N*N, M]
        msg = msg_flat.view(B, N, N, self.msg_dim)               # [B,N,N,M]

        # neighborhood mask
        neighbor_mask = self._build_neighbor_mask(s_t)           # [B,N,N]
        msg = msg * neighbor_mask.unsqueeze(-1)                  # zero out non-neighbors

        # aggregate over context
        agg = msg.sum(dim=2)  # [B,N,M]

        # ----- decoder: per-object prediction -----
        self_feat = torch.cat([s_tm1, s_t], dim=-1)  # [B,N,2D]
        dec_in = torch.cat([agg, self_feat], dim=-1) # [B,N,M+2D]
        dec_in_flat = dec_in.view(B * N, self.msg_dim + 2 * self.state_dim)

        pred_v_flat = self.decoder(dec_in_flat)      # [B*N,2]
        pred_v = pred_v_flat.view(B, N, 2)

        target_v = s_tp1[..., 2:4]                   # [B,N,2]

        return pred_v, target_v


def count_parameters(model: nn.Module) -> int:
    """Utility: total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
