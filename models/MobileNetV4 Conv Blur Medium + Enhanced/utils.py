# models/pytorch/utils.py

import torch
import torch.nn as nn

# ─── Res2Net Block ──────────────────────────────────────────────────────────
class Res2NetBlock(nn.Module):
    def __init__(self, channels, scale=4):
        super().__init__()
        assert channels % scale == 0, "channels must be divisible by scale"
        self.scale = scale
        self.width = channels // scale
        # one depthwise conv per branch except the first
        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1,
                      groups=self.width, bias=False)
            for _ in range(scale-1)
        ])
        self.bn  = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        parts = torch.split(x, self.width, dim=1)
        y, outs = parts[0], [parts[0]]
        for i in range(1, self.scale):
            sp = parts[i] + y
            sp = self.convs[i-1](sp)
            y = sp
            outs.append(sp)
        out = torch.cat(outs, dim=1)
        return self.act(self.bn(out))


# ─── Temporal Shift Module ─────────────────────────────────────────────────
class TemporalShift(nn.Module):
    def __init__(self, channels, shift_div=8):
        super().__init__()
        self.fold = channels // shift_div

    def forward(self, x):
        # x: (B, C, T, F) or (B, C, H, W)
        B, C, T, F = x.size()
        t = x.permute(0,2,1,3).contiguous()       # B×T×C×F
        out = torch.zeros_like(t)
        out[:, :-1, :self.fold, :]               = t[:, 1:, :self.fold, :]
        out[:, 1:,  self.fold:2*self.fold, :]    = t[:, :-1, self.fold:2*self.fold, :]
        out[:, :, 2*self.fold:, :]               = t[:, :, 2*self.fold:, :]
        return out.permute(0,2,1,3)               # B×C×T×F


# ─── Res2TSM Block ─────────────────────────────────────────────────────────
class Res2TSMBlock(nn.Module):
    def __init__(self, channels, scale=4, shift_div=8):
        super().__init__()
        assert channels % scale == 0, "channels must be divisible by scale"
        self.scale = scale
        self.width = channels // scale
        self.temporal_shift = TemporalShift(channels, shift_div)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, (3,1), padding=(1,0),
                      groups=self.width, bias=False)
            for _ in range(scale-1)
        ])
        self.bn  = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1) apply TSM
        x = self.temporal_shift(x)           
        # 2) Res2Net-style split + hierarchical fusion
        splits = torch.split(x, self.width, dim=1)
        y = splits[0]; outs = [y]
        for i in range(1, self.scale):
            sp = splits[i] + y
            sp = self.convs[i-1](sp)
            y = sp
            outs.append(sp)
        out = torch.cat(outs, dim=1)
        return self.act(self.bn(out))
