#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        """
        Initialize the TimeEmbedding module.

        Args:
            n_channels (int): Number of channels for the embedding.
        """
        super().__init__()

        self.n_channels = n_channels
        self.model = nn.Sequential(
            nn.Linear(n_channels // 4, n_channels // 2),
            nn.SiLU(),
            nn.Linear(n_channels // 2, n_channels),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TimeEmbedding module.

        Args:
            t (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after embedding.
        """
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.model(emb)

        return emb


class ConditionEmbedding(nn.Module):
    def __init__(self, n_channels: int, condition_dim: int, emb_type: str = "linear"):
        super().__init__()
        self.emb_type = emb_type
        
        if emb_type == "sinusoidal":
            self.embedding_dim = n_channels // 2
            self.model = nn.Sequential(
                nn.Linear(condition_dim * self.embedding_dim * 2, n_channels * 4),
                nn.SiLU(),
                nn.Linear(n_channels * 4, n_channels * 4),
                nn.SiLU(),
                nn.Linear(n_channels * 4, n_channels) 
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(condition_dim, n_channels * 4),
                nn.SiLU(),
                nn.Linear(n_channels * 4, n_channels * 4),
                nn.SiLU(),
                nn.Linear(n_channels * 4, n_channels * 2),
                nn.SiLU(),
                nn.Linear(n_channels * 2, n_channels),
            )
        
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        if self.emb_type == "sinusoidal":
            half_dim = self.embedding_dim
            emb = math.log(10_000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=c.device) * -emb)
            emb = c.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            emb = emb.view(c.shape[0], -1)
            return self.model(emb)
        else:
            return self.model(c)


class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."

    def __init__(self, n_channels):
        super().__init__()
        self.query, self.key, self.value = [
            self._conv(n_channels, c)
            for c in (n_channels // 4, n_channels // 4, n_channels)
        ]
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def _conv(self, n_in, n_out):
        return nn.Conv1d(n_in, n_out, 1, bias=False)

    def forward(self, x):
        # Notation from the paper.
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x

        return o.view(*size).contiguous()


class CrossAttention(nn.Module):
    def __init__(self, n_channels: int, condition_dim: int):
        super().__init__()
        self.query = nn.Conv3d(n_channels, n_channels // 4, kernel_size=1, bias=False)
        self.key = nn.Linear(condition_dim, n_channels // 4, bias=False)
        self.value = nn.Linear(condition_dim, n_channels, bias=False)
        self.out_conv = nn.Conv3d(n_channels, n_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        size = x.size()  # [batch_size, channels, height, width, depth]
        x_flat = x.view(
            size[0], size[1], -1
        )  # Flatten spatial dimensions [batch_size, channels, height*width*depth]

        # Apply query projection
        query = self.query(x)  # [batch_size, n_channels // 4, height, width, depth]
        query_flat = query.view(
            size[0], query.size(1), -1
        )  # Flatten spatial dimensions

        # Apply key and value projections
        key = self.key(c).unsqueeze(-1)  # [batch_size, n_channels // 4, 1]
        value = self.value(c).unsqueeze(-1)  # [batch_size, n_channels, 1]

        # Attention mechanism
        attn = torch.bmm(
            query_flat.permute(0, 2, 1), key
        )  # [batch_size, height*width*depth, 1]
        attn = F.softmax(attn, dim=1)

        # Compute weighted sum
        weighted_value = torch.bmm(
            value, attn.permute(0, 2, 1)
        )  # [batch_size, n_channels, height*width*depth]
        weighted_value = weighted_value.view(
            size
        )  # Reshape to original dimensions [batch_size, n_channels, height, width, depth]

        # Apply output convolution
        out = self.out_conv(weighted_value)

        return self.gamma * out + x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        n_groups: int = 8,
        dropout: float = 0.1,
        condition_injection: str = "add",
    ):
        super().__init__()
        self.condition_injection = condition_injection

        self.conv1 = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity(),
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))
        self.dropout = nn.Dropout3d(dropout)

        # Initialize the FiLM adapter if toggled
        if self.condition_injection == "film":
            self.cond_adapter = nn.Sequential(
                nn.SiLU(), 
                nn.Linear(time_channels, out_channels * 2) 
            )
            nn.init.zeros_(self.cond_adapter[1].weight)
            nn.init.zeros_(self.cond_adapter[1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        h = self.conv1(x)
        time = self.time_emb(t)
        h += time[:, :, None, None, None]

        # Apply FiLM if toggled and condition exists
        if self.condition_injection == "film" and c is not None:
            cond_vec = self.cond_adapter(c)
            scale, shift = cond_vec.chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]

        h = self.conv2(h)
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool, dropout: float, condition_injection: str = "add"):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, dropout=dropout, condition_injection=condition_injection)
        self.attn = SelfAttention(out_channels) if has_attn else nn.Identity()
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        x = self.res(x, t, c)
        x = self.attn(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool, dropout: float, condition_injection: str = "add"):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels, dropout=dropout, condition_injection=condition_injection)
        self.attn = SelfAttention(out_channels) if has_attn else nn.Identity()
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        x = self.res(x, t, c)
        x = self.attn(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int, middle_attn=False, condition_injection: str = "add"):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, dropout=0, condition_injection=condition_injection)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, dropout=0, condition_injection=condition_injection)
        self.attn = SelfAttention(n_channels) if middle_attn else nn.Identity()
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        x = self.res1(x, t, c)
        x = self.attn(x)
        x = self.res2(x, t, c)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        # self.conv = nn.ConvTranspose3d(n_channels, n_channels, 4, 2, 1)
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(n_channels, n_channels, kernel_size=3, stride=1, padding="same"),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,c: torch.Tensor = None) -> torch.Tensor:
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self, image_channels: int = 1, n_channels: int = 64, ch_mults=(1, 2, 2, 4),
        is_attn=(False, False, False, True), n_blocks: int = 2, middle_attn=True,
        dropout=0.05, condition_dim=None, cross_attn=False,
        condition_emb_type="linear", condition_injection="add"
    ):
        super(UNet, self).__init__()
        self.condition_injection = condition_injection

        if (type(dropout) == float) or (type(dropout) == int):
            dropout = [dropout] * len(ch_mults)
        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv3d(image_channels, n_channels, kernel_size=3, padding=1)
        self.time_emb = TimeEmbedding(n_channels * 4)

        if condition_dim is not None:
            self.condition_emb = ConditionEmbedding(n_channels * 4, condition_dim, emb_type=condition_emb_type)
            self.cross_attn = CrossAttention(n_channels, n_channels * 4) if cross_attn else None
            
            # Baseline uses addition mapping
            if self.condition_injection == "add":
                self.time_concat = nn.Linear(n_channels * 4, n_channels * 4)
                with torch.no_grad():
                    self.time_concat.weight.zero_()
                    self.time_concat.bias.zero_()
            else:
                self.time_concat = None
        else:
            self.condition_emb = None
            self.cross_attn = None
            self.time_concat = None

        down = []
        out_channels = in_channels = n_channels
        for i, drop_val in zip(range(n_resolutions), dropout):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i], drop_val, condition_injection))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_channels * 4, middle_attn, condition_injection)

        up = []
        in_channels = out_channels
        for i, drop_val in zip(reversed(range(n_resolutions)), reversed(dropout)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i], drop_val, condition_injection))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i], 0, condition_injection))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv3d(in_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        t = self.time_emb(t)
        x = self.image_proj(x)

        if (self.condition_emb is not None) and (c is not None):
            c_emb = self.condition_emb(c)
            
            if self.condition_injection == "add":
                t = t + self.time_concat(c_emb)
                c_pass = None # It's folded into 't' now
            else:
                c_pass = c_emb # Pass separately for FiLM

            if (self.cross_attn is not None):
                x = self.cross_attn(x, c_emb)
        else:
            c_pass = None

        h = [x]
        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x, t, c_pass)
            else:
                x = m(x, t, c_pass)
            h.append(x)

        x = self.middle(x, t, c_pass)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t, c_pass)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t, c_pass)

        return self.final(self.act(self.norm(x)))
