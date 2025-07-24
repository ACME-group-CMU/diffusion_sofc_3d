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
    def __init__(self, n_channels: int, condition_dim: int):
        """
        Initialize the ConditionEmbedding module.

        Args:
            n_channels (int): Number of channels for the embedding.
            condition_dim (int): Dimension of the condition.
        """
        super().__init__()
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
        """
        Forward pass for the ConditionEmbedding module.

        Args:
            c (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after embedding.
        """
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
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        n_groups: int = 8,
        dropout: float = 0.1,
    ):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()

        # Group normalization and the first convolution layer
        self.conv1 = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity(),
        )

        # Group normalization and the second convolution layer
        self.conv2 = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(),
            # nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))

        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        * `x` has shape `[batch_size, in_channels, height, width,depth]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(x)
        time = self.time_emb(t)
        # Add time embeddings
        h += time[:, :, None, None, None]
        # Second convolution layer
        h = self.conv2(h)

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        dropout: float,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels, out_channels, time_channels, dropout=dropout
        )
        if has_attn:
            self.attn = SelfAttention(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res(x, t)
        x = self.attn(x)

        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        dropout: float,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels, time_channels, dropout=dropout
        )
        if has_attn:
            self.attn = SelfAttention(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(
        self, n_channels: int, time_channels: int, middle_attn=False,
    ):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, dropout=0)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, dropout=0)
        if middle_attn:
            self.attn = SelfAttention(n_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(
        self,
        image_channels: int = 1,
        n_channels: int = 64,
        ch_mults=(1, 2, 2, 4),
        is_attn=(False, False, False, True),
        n_blocks: int = 2,
        middle_attn=True,
        dropout=0.05,
        condition_dim=None,
        cross_attn=False,
    ):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        * `middle_attn` for whether you want attention block
        """
        super(UNet, self).__init__()

        if (type(dropout) == float) or (type(dropout) == int):
            dropout = [dropout] * len(ch_mults)
        else:
            pass

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv3d(
            image_channels, n_channels, kernel_size=3, padding=1
        )

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # Conditional Embedding. Conditional Embedding currently dense layers to `n_channels * 4`

        if condition_dim is not None:
            self.condition_emb = ConditionEmbedding(n_channels * 4, condition_dim)
            self.cross_attn = (
                CrossAttention(n_channels, n_channels * 4)
                if cross_attn is True
                else None
            )
            self.time_concat = nn.Linear(n_channels * 8, n_channels * 4)

        else:
            self.condition_emb = None
            self.cross_attn = None
            self.time_concat = None

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i, drop_val in zip(range(n_resolutions), dropout):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels, out_channels, n_channels * 4, is_attn[i], drop_val
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, middle_attn)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i, drop_val in zip(reversed(range(n_resolutions)), reversed(dropout)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels, out_channels, n_channels * 4, is_attn[i], drop_val
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i], 0))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv3d(in_channels, image_channels, kernel_size=3, padding=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None
    ) -> torch.Tensor:
        """
        * `x` has shape `[batch_size, in_channels, height, width,depth]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        if (self.condition_emb is not None) and (c is not None):
            c = self.condition_emb(c)
            t = self.time_concat(torch.concat((t, c), dim=1))
            if (self.cross_attn is not None) and (c is not None):
                x = self.cross_attn(x, c)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))
