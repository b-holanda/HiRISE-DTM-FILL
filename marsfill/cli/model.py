import os

from pathlib import Path
from marsfill.model.combined_loss import LossWights
from marsfill.model.train import AvaliabeDevices, AvaliableModels, Train
from marsfill.utils import Logger
from marsfill.utils.profiler import get_profile

logger = Logger()

class Model:
    """Lidar com o treinamento e teste do modelo DPT-ViT Marsfill."""

    def __init__(self):
        pass

    def train(self, p: str = "prod"):
        """Constrói o dataset de treinamento e teste e salva na pasta especificada.

        Args:
            [Parâmetros de treinamento]

            selected_device (AvaliabeDevices): Use gpu para melhor performance, opções: [gpu, cpu]. Default gpu
            selected_model (AvaliableModels): Por enquanto o único modelo disponível é o INTEL_DPT_LARGE. Default INTEL_DPT_LARGE
            batch_size (int): "Tamanho do Lote". Quantas imagens a IA vai "olhar" de uma vez antes de atualizar o que aprendeu. Default 4
            learning_rate (float): "Taxa de Aprendizado". É o "tamanho do passo" que a IA dá ao aprender. Default 1e-5
            epochs (int): Uma época é uma passagem completa por todos os dados de treinamento. Default 50
            weight_decay (float): Adiciona uma pequena penalidade ao otimizador. A IA é forçada a ser mais "eficiente". Ela só vai usar um peso grande se for absolutamente essencial para diminuir o valor de perda total, valores muito altos causam underfitting e valores muito baixos causam overfitting. Default 1e-2
            data_dir (str): Pasta raiz dos datasets. Default datasets
            out_dir (str): Pasta raiz para salvar arquivos gerados. Default outputs

            [Pesos da função de perda combinada]

            w_l1 (float): "Mean Absolute Error" peso para o valor calculado da diferença absoluta média entre a elevação prevista e a elevação real, pixel por pixel. Default 0.6
            w_grad (float): "Perda de Gradiente" peso para o valor da comparação a inclinação (slope) do terreno previsto com a inclinação do terreno real. Default 0.3
            w_ssim (float): "Perda de Similaridade Estrutural" Ela olha para "janelas" de pixels e compara luminância, contraste e estrutura. Default 0.1

            [Ajuste de pesos na função de perda (Fine-Tuning)] L_total = (w_l1 * L_l1) + (w_grad * L_Gradiente_Sobel) + (w_ssim * L_SSIM)

            Sintoma: O terreno gerado é muito borrado (blurry) e as crateras não têm bordas definidas.

            Diagnóstico: O peso W_GRAD está muito baixo. A IA está ignorando a "forma".
            Cura: Aumente W_GRAD (ex: para 0.4) e diminua W_L1 (ex: para 0.5).

            Sintoma: As bordas são nítidas, mas a elevação geral está errada (ex: o fundo da cratera está na altura errada).

            Diagnóstico: W_L1 está muito baixo. A IA está focando só na forma, não no valor.
            Cura: Aumente W_L1.
        """
        profile = get_profile(p)

        if not profile:
            logger.error("Nenhum perfil compatível encontrado para o hardware do sistema.")
            return

        selected_model: AvaliableModels = AvaliableModels.INTEL_DPT_LARGE
        batch_size: int = profile["train"].get("batch_size", 8)
        learning_rate: float = profile["train"].get("learning_rate", 0.00001)
        epochs: int = profile["train"].get("epochs", 50)
        weight_decay: float = profile["train"].get("weight_decay", 0.01)
        w_l1: float = 0.6
        w_grad: float = 0.3
        w_ssim: float = 0.1

        logger.info("Iniciando rotina de treinamento...")
        logger.info(f"selected_model={selected_model}")
        logger.info(f"batch_size={batch_size}")
        logger.info(f"learning_rate={learning_rate}")
        logger.info(f"epochs={epochs}")
        logger.info(f"weight_decay={weight_decay}")
        logger.info(f"w_l1={w_l1}")
        logger.info(f"w_grad={w_grad}")
        logger.info(f"w_ssim={w_ssim}")

        Train(
            selected_model=selected_model,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            weight_decay=weight_decay,
            loss_weights=LossWights(l1=w_l1, gradenty=w_grad, ssim=w_ssim)
        ).run()
