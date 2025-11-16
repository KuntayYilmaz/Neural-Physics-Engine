# models.py
from typing import Sequence

import torch
import torch.nn as nn


class NPEEncoder(nn.Module):
    """
    Pairwise encoder f_enc:

    - Input: concat(o_f^{t-1}, o_f^t, o_c^{t-1}, o_c^t) => 4 * state_dim
    - Pairwise layer: 25 units, no bias
    - Then 5-layer MLP, 50 units each, ReLU, no bias in any encoder layer.

    This matches the paper's implementation choices.
    """

    def __init__(
        self,
        state_dim: int,
        pair_hidden: int = 25,
        hidden_dim: int = 50,
        num_layers: int = 5,
    ):
        super().__init__()
        input_dim = 4 * state_dim
        self.pair_layer = nn.Linear(input_dim, pair_hidden, bias=False)

        layers = []
        in_dim = pair_hidden
        for _ in range(num_layers):
            # No bias in encoder so that masked (zero) inputs stay zero.
            layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, pair_feats: torch.Tensor) -> torch.Tensor:
        """
        pair_feats: [B, N, N, 4 * state_dim]
        returns: [B, N, N, enc_dim]
        """
        x = self.pair_layer(pair_feats)
        x = self.mlp(x)
        return x


class NPEDecoder(nn.Module):
    """
    Decoder f_dec:

    - Input: concat(sum_c e_{f,c}, o_f^{t-1}, o_f^t)
    - 5-layer MLP, 50 units each, ReLU, with bias
    - Output: Δv_f (2D: Δvx, Δvy)
    """

    def __init__(
        self,
        state_dim: int,
        enc_dim: int,
        hidden_dim: int = 50,
        num_layers: int = 5,
        vel_dim: int = 2,
    ):
        super().__init__()
        input_dim = enc_dim + 2 * state_dim
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, vel_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        enc_sum: torch.Tensor,
        focus_prev: torch.Tensor,
        focus_cur: torch.Tensor,
    ) -> torch.Tensor:
        """
        enc_sum:   [B, N, enc_dim]
        focus_prev, focus_cur: [B, N, state_dim]
        returns: Δv [B, N, vel_dim]
        """
        x = torch.cat([enc_sum, focus_prev, focus_cur], dim=-1)
        delta_v = self.mlp(x)
        return delta_v


class NeuralPhysicsEngine(nn.Module):
    """
    Full NPE model:

    - Takes [B, 3, N, D] windows: (t-1, t, t+1)
    - Uses positions at t to build a neighborhood mask
    - Encodes pairwise interactions for all (f, c)
    - Masks non-neighbors and sums over c
    - Decodes Δv_f, predicts v_f^{t+1} = v_f^t + Δv_f
    - Trains with MSE on normalized velocities v^{t+1}
    """

    def __init__(
        self,
        state_dim: int,
        neighborhood_radius: float,
        pos_indices: Sequence[int] = (0, 1),  # x,y
        vel_indices: Sequence[int] = (2, 3),  # vx,vy
        pair_hidden: int = 25,
        enc_hidden: int = 50,
        enc_layers: int = 5,
        dec_hidden: int = 50,
        dec_layers: int = 5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.pos_indices = tuple(pos_indices)
        self.vel_indices = tuple(vel_indices)
        self.neighborhood_radius = float(neighborhood_radius)

        self.encoder = NPEEncoder(
            state_dim=state_dim,
            pair_hidden=pair_hidden,
            hidden_dim=enc_hidden,
            num_layers=enc_layers,
        )
        self.decoder = NPEDecoder(
            state_dim=state_dim,
            enc_dim=self.encoder.output_dim,
            hidden_dim=dec_hidden,
            num_layers=dec_layers,
            vel_dim=len(self.vel_indices),
        )

    def forward(self, states_3: torch.Tensor):
        """
        states_3: [B, 3, N, D]  (t-1, t, t+1)

        Returns
        -------
        pred_v_next : [B, N, 2]   (normalized predicted velocities at t+1)
        target_v_next : [B, N, 2] (normalized ground-truth velocities at t+1)
        """
        B, T, N, D = states_3.shape
        assert T == 3, "Input must be 3 timesteps: (t-1, t, t+1)"

        prev = states_3[:, 0]  # [B, N, D]
        cur = states_3[:, 1]
        nxt = states_3[:, 2]

        pos_start, pos_end = self.pos_indices[0], self.pos_indices[-1] + 1
        vel_start, vel_end = self.vel_indices[0], self.vel_indices[-1] + 1

        pos_cur = cur[..., pos_start:pos_end]  # [B, N, 2]

        # Build pairwise features: (o_f^{t-1}, o_f^t, o_c^{t-1}, o_c^t)
        prev_i = prev[:, :, None, :]  # [B, N, 1, D]
        prev_j = prev[:, None, :, :]  # [B, 1, N, D]
        cur_i = cur[:, :, None, :]
        cur_j = cur[:, None, :, :]

        prev_i = prev_i.expand(-1, -1, N, -1)
        prev_j = prev_j.expand(-1, N, -1, -1)
        cur_i = cur_i.expand(-1, -1, N, -1)
        cur_j = cur_j.expand(-1, N, -1, -1)

        pair_feats = torch.cat([prev_i, cur_i, prev_j, cur_j], dim=-1)  # [B,N,N,4D]

        # Neighborhood mask based on distance at time t
        pi = pos_cur[:, :, None, :]  # [B, N, 1, 2]
        pj = pos_cur[:, None, :, :]  # [B, 1, N, 2]
        dist = torch.linalg.norm(pi - pj, dim=-1)  # [B, N, N]

        # neighbors: dist < R, exclude self: dist > 0
        mask = (dist > 0.0) & (dist < self.neighborhood_radius)
        mask = mask.float()  # [B, N, N]

        # Encode all pairs, then mask and sum
        enc = self.encoder(pair_feats)        # [B, N, N, enc_dim]
        enc = enc * mask[..., None]          # zero out non-neighbors
        enc_sum = enc.sum(dim=2)             # [B, N, enc_dim]

        # Decode Δv and add to current velocity
        v_cur = cur[..., vel_start:vel_end]  # [B, N, 2]
        delta_v = self.decoder(enc_sum, prev, cur)  # [B, N, 2]
        pred_v_next = v_cur + delta_v

        target_v_next = nxt[..., vel_start:vel_end]
        return pred_v_next, target_v_next
