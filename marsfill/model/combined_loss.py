import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self
from dataclasses import dataclass

from torchmetrics.image import StructuralSimilarityIndexMeasure

@dataclass
class LossWights:
    l1: float
    gradenty: float
    ssim: float

class _GradentLoss:
    def __init__(self) -> None:
        self._l1 = nn.L1Loss()
        self._sabel_x = nn.Parameter(
            data=torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3), 
            requires_grad=False
        )
        self._sabel_y = nn.Parameter(
            data=torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3), 
            requires_grad=False
        )

    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.gradient_loss(predicted, target)
    
    def to(self, device: torch.device) -> Self:
        self._sabel_x = nn.Parameter(
            data=torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(device), 
            requires_grad=False
        )
        self._sabel_y = nn.Parameter(
            data=torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3).to(device), 
            requires_grad=False
        )

        return self

    def gradient_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_gradent_x = F.conv2d(predicted, self._sabel_x, padding="same")
        predicted_gradent_y = F.conv2d(predicted, self._sabel_y, padding="same")

        target_gradent_x = F.conv2d(target, self._sabel_x, padding="same")
        target_gradent_y = F.conv2d(target, self._sabel_y, padding="same")

        loss_x = self._l1(predicted_gradent_x, target_gradent_x)
        loss_y = self._l1(predicted_gradent_y, target_gradent_y)

        return loss_x + loss_y

class CombinedLoss(nn.Module):
    def __init__(self, lossWeights: LossWights, device: torch.device) -> None:
        super(CombinedLoss, self).__init__()

        self._lossWeights = lossWeights

        self._l1 = nn.L1Loss()
        self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)
        self._gradent = _GradentLoss().to(device=device)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, ...]:
        loss_l1 = self._l1(predicted, target)
        loss_ssim = 1.0 - self._ssim(predicted, target)
        loss_gradient = self._gradent(predicted, target)

        total_loss = (self._lossWeights.l1 * loss_l1) + (self._lossWeights.gradenty * loss_gradient) + (self._lossWeights.ssim * loss_ssim)

        return total_loss, loss_l1, loss_gradient, loss_ssim
