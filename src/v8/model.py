"""V8 gated multi-evidence CNN for AI image detection.

This version keeps the V7 RGB + residual evidence idea, but uses a learned gate
so the residual branch can help without dominating the RGB evidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(8, channels // max(1, reduction))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock(out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.main(x)
        out = self.se(out)
        out = self.activation(out + residual)
        out = self.dropout(out)
        return out


class EvidenceBranch(nn.Module):
    """CNN branch for one evidence view: RGB content or residual artifacts."""

    def __init__(self, in_channels: int, widths: tuple[int, ...], dropouts: tuple[float, ...]) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("EvidenceBranch needs at least two channel widths.")
        if len(dropouts) != len(widths) - 1:
            raise ValueError("dropouts must have one value for each residual block.")

        first_width = widths[0]
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, first_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_width),
            nn.SiLU(inplace=True),
        ]

        for in_width, out_width, dropout in zip(widths, widths[1:], dropouts):
            layers.append(ResidualSEBlock(in_width, out_width, stride=2, dropout=dropout))

        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ])

        self.features = nn.Sequential(*layers)
        self.out_features = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class HighPassResidual(nn.Module):
    """Fixed blur-subtract residual map used as forensic evidence."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        blurred = F.avg_pool2d(padded, kernel_size=self.kernel_size, stride=1)
        residual = x - blurred
        return torch.clamp(residual * 2.0, min=-4.0, max=4.0)


class CNN(nn.Module):
    """Two-branch detector with gated residual evidence and auxiliary heads."""

    def __init__(self) -> None:
        super().__init__()

        self.residual_view = HighPassResidual(kernel_size=5)
        self.rgb_branch = EvidenceBranch(
            in_channels=3,
            widths=(32, 64, 128, 192, 256, 384),
            dropouts=(0.05, 0.10, 0.12, 0.15, 0.18),
        )
        self.residual_branch = EvidenceBranch(
            in_channels=3,
            widths=(16, 32, 64, 96),
            dropouts=(0.03, 0.05, 0.08),
        )
        self.residual_projection = nn.Sequential(
            nn.Linear(self.residual_branch.out_features, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
        )

        fused_features = self.rgb_branch.out_features + 96
        self.residual_gate = nn.Sequential(
            nn.Linear(fused_features, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 96),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(fused_features, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(96, 1),
        )
        self.rgb_head = nn.Sequential(
            nn.Linear(self.rgb_branch.out_features, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

        gate_linear = self.residual_gate[-2]
        if isinstance(gate_linear, nn.Linear):
            nn.init.zeros_(gate_linear.weight)
            nn.init.constant_(gate_linear.bias, -1.0)

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        rgb_features = self.rgb_branch(x)
        residual_features = self.residual_projection(self.residual_branch(self.residual_view(x)))
        gate_input = torch.cat([rgb_features, residual_features], dim=1)
        residual_gate = self.residual_gate(gate_input)
        gated_residual_features = residual_features * residual_gate
        fused = torch.cat([rgb_features, gated_residual_features], dim=1)
        fused_logits = self.classifier(fused)

        if not return_aux:
            return fused_logits

        return {
            "logits": fused_logits,
            "rgb_logits": self.rgb_head(rgb_features),
            "residual_logits": self.residual_head(residual_features),
            "residual_gate_mean": residual_gate.mean(),
        }
