import os
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

# ATENÇÃO: Verifique se este import abaixo está correto no seu projeto. 
# Se 'AvailableModels' estiver em train.py, mantenha. Se não, ajuste o caminho.
from marsfill.model.train import AvailableModels 
from marsfill.utils import Logger

logger = Logger()

class Evaluator:
    """
    Classe responsável pela inferência (avaliação) do modelo de estimativa de profundidade.
    """

    def __init__(
        self,
        pretrained_model_name: AvailableModels,
        model_path_uri: str,
    ) -> None:
        """
        Inicializa o modelo base e carrega os pesos treinados.
        """
        logger.info("Iniciando arquitetura base do avaliador")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Carrega arquitetura
        self.model = DPTForDepthEstimation.from_pretrained(pretrained_model_name.value)

        # Carrega pesos treinados
        self._load_model_weights(model_path_uri)

        self.model.to(self.device)
        self.model.eval()

        self.image_processor = DPTImageProcessor.from_pretrained(
            pretrained_model_name.value, do_rescale=False
        )
        logger.info("Modelo carregado e pronto para inferência")

    def _load_model_weights(self, model_uri: str) -> None:
        logger.info(f"Carregando pesos do modelo de: {model_uri}")

        if not os.path.exists(model_uri):
            logger.error(f"Caminho do modelo {model_uri} não encontrado.")
            raise FileNotFoundError(f"Caminho do modelo {model_uri} não encontrado.")

        self._apply_state_dict_from_file(model_uri)

    def _apply_state_dict_from_file(self, file_path: str) -> None:
        """
        Carrega o dicionário de estados do disco e remove prefixos específicos de treinamento distribuído (DDP).
        """
        # weights_only=True é mais seguro, mas se der erro em versões antigas do torch, remova esse argumento
        try:
            raw_state_dict = torch.load(file_path, map_location=self.device, weights_only=True)
        except TypeError:
            # Fallback para versões mais antigas do PyTorch que não suportam weights_only
            raw_state_dict = torch.load(file_path, map_location=self.device)

        sanitized_state_dict = {}
        for key, value in raw_state_dict.items():
            # Remove prefixo 'module.' se o treino foi feito com DDP (Distributed Data Parallel)
            clean_key = key[7:] if key.startswith("module.") else key
            sanitized_state_dict[clean_key] = value

        self.model.load_state_dict(sanitized_state_dict)

    def predict_depth(
        self, orthophoto_image: np.ndarray, target_height: int, target_width: int
    ) -> np.ndarray:
        """
        Executa a inferência na imagem fornecida.
        """
        # Prepara tensores
        model_inputs = self.image_processor(images=orthophoto_image, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            prediction_output = self.model(**model_inputs)
            predicted_depth_tensor = prediction_output.predicted_depth

        # Interpolação bicúbica para ajustar a resolução da predição ao tamanho do recorte original
        interpolated_depth = torch.nn.functional.interpolate(
            predicted_depth_tensor.unsqueeze(1),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        )

        return interpolated_depth.squeeze().cpu().numpy()
