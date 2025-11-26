import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
from torchmetrics.image import StructuralSimilarityIndexMeasure

@dataclass
class LossWeights:
    """
    Estrutura de dados para armazenar os pesos de cada componente da função de perda combinada.
    """
    l1_weight: float
    gradient_weight: float
    ssim_weight: float

class GradientLoss(nn.Module):
    """
    Calcula a perda baseada no gradiente da imagem (detecção de bordas) usando operadores de Sobel.
    Isso ajuda o modelo a preservar a nitidez das bordas na estimativa de profundidade.
    """

    def __init__(self) -> None:
        """
        Inicializa os filtros de Sobel e a função de perda L1 interna.
        Os filtros são registrados como buffers para gestão automática de dispositivo (CPU/GPU).
        """
        super(GradientLoss, self).__init__()
        self.l1_loss_function = nn.L1Loss()
        
        sobel_x_data = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        
        sobel_y_data = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=torch.float32
        ).reshape(1, 1, 3, 3)

        self.register_buffer('sobel_kernel_x', sobel_x_data)
        self.register_buffer('sobel_kernel_y', sobel_y_data)

    def forward(self, predicted_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calcula a perda de gradiente entre a previsão e o alvo.

        Parâmetros:
            predicted_tensor (torch.Tensor): O tensor de profundidade predito pelo modelo (Batch, Canais, Altura, Largura).
            target_tensor (torch.Tensor): O tensor de profundidade real (Ground Truth).

        Retorno:
            torch.Tensor: O valor escalar da perda de gradiente somada (eixos X e Y).
        """
        predicted_gradient_x = F.conv2d(predicted_tensor, self.sobel_kernel_x, padding="same")
        predicted_gradient_y = F.conv2d(predicted_tensor, self.sobel_kernel_y, padding="same")

        target_gradient_x = F.conv2d(target_tensor, self.sobel_kernel_x, padding="same")
        target_gradient_y = F.conv2d(target_tensor, self.sobel_kernel_y, padding="same")

        loss_x = self.l1_loss_function(predicted_gradient_x, target_gradient_x)
        loss_y = self.l1_loss_function(predicted_gradient_y, target_gradient_y)

        return loss_x + loss_y

class CombinedLoss(nn.Module):
    """
    Função de perda composta que agrega L1 (erro absoluto), Gradiente (bordas) e SSIM (similaridade estrutural).
    """

    def __init__(
        self, 
        loss_weights: LossWeights, 
        ssim_module: Optional[nn.Module] = None,
        gradient_module: Optional[nn.Module] = None
    ) -> None:
        """
        Inicializa a perda combinada.

        Parâmetros:
            loss_weights (LossWeights): Pesos para ponderar cada componente da perda final.
            ssim_module (Optional[nn.Module]): Módulo injetável para cálculo de SSIM. Útil para testes unitários (mocking). 
                                             Se None, usa o StructuralSimilarityIndexMeasure padrão.
            gradient_module (Optional[nn.Module]): Módulo injetável para cálculo de gradiente. Útil para testes unitários.
                                                  Se None, usa o GradientLoss padrão.
        """
        super(CombinedLoss, self).__init__()

        self.weights = loss_weights
        self.l1_loss_function = nn.L1Loss()
        
        if ssim_module:
            self.structural_similarity_module = ssim_module
        else:
            self.structural_similarity_module = StructuralSimilarityIndexMeasure(data_range=1.0)
            
        if gradient_module:
            self.gradient_loss_module = gradient_module
        else:
            self.gradient_loss_module = GradientLoss()

    def forward(self, predicted_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcula a perda total ponderada e retorna os componentes individuais para monitoramento.

        Parâmetros:
            predicted_tensor (torch.Tensor): Imagem/Profundidade predita.
            target_tensor (torch.Tensor): Imagem/Profundidade alvo.

        Retorno:
            Tuple[torch.Tensor, ...]: Uma tupla contendo (perda_total, perda_l1, perda_gradiente, perda_ssim).
        """
        l1_loss_value = self.l1_loss_function(predicted_tensor, target_tensor)
        
        gradient_loss_value = self.gradient_loss_module(predicted_tensor, target_tensor)
        
        ssim_value = self.structural_similarity_module(predicted_tensor, target_tensor)
        ssim_loss_value = 1.0 - ssim_value

        total_loss_value = (
            (self.weights.l1_weight * l1_loss_value) + 
            (self.weights.gradient_weight * gradient_loss_value) + 
            (self.weights.ssim_weight * ssim_loss_value)
        )

        return total_loss_value, l1_loss_value, gradient_loss_value, ssim_loss_value
