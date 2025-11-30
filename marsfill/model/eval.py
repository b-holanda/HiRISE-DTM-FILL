import os
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

from marsfill.model.train import AvailableModels
from marsfill.utils import Logger

logger = Logger()


class Evaluator:
    """
    Classe responsável pela inferência (avaliação) do modelo de estimativa de profundidade.
    Gerencia o carregamento do modelo local e realiza a predição,
    retornando os dados brutos em memória.
    """

    def __init__(
        self,
        pretrained_model_name: AvailableModels,
        model_path_uri: str,
    ) -> None:
        """
        Inicializa o modelo base e carrega os pesos treinados.

        Args:
            pretrained_model_name: Nome do backbone HuggingFace.
            model_path_uri: Caminho local do checkpoint .pth.
        """
        logger.info("Iniciando arquitetura base do avaliador")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DPTForDepthEstimation.from_pretrained(pretrained_model_name.value)

        self._load_model_weights(model_path_uri)

        self.model.to(self.device)
        self.model.eval()

        self.image_processor = DPTImageProcessor.from_pretrained(
            pretrained_model_name.value, do_rescale=False
        )
        logger.info("Modelo carregado e pronto para inferência")

    def _load_model_weights(self, model_uri: str) -> None:
        """
        Gerencia o carregamento do state_dict de um caminho local.

        Args:
            model_uri: Caminho de origem do arquivo .pth.

        Raises:
            FileNotFoundError: Se o arquivo local não existir.
        """
        logger.info(f"Carregando pesos do modelo de: {model_uri}")

        if not os.path.exists(model_uri):
            logger.error(f"Caminho do modelo {model_uri} não encontrado.")
            raise FileNotFoundError(f"Caminho do modelo {model_uri} não encontrado.")

        self._apply_state_dict_from_file(model_uri)

    def _apply_state_dict_from_file(self, file_path: str) -> None:
        """
        Carrega o dicionário de estados do disco e remove prefixos específicos de treinamento distribuído (DDP).

        Args:
            file_path: Caminho físico local para o arquivo .pth.
        """
        raw_state_dict = torch.load(file_path, map_location=self.device, weights_only=True)

        sanitized_state_dict = {}
        for key, value in raw_state_dict.items():
            clean_key = key[7:] if key.startswith("module.") else key
            sanitized_state_dict[clean_key] = value

        self.model.load_state_dict(sanitized_state_dict)

    def predict_depth(
        self, orthophoto_image: np.ndarray, target_height: int, target_width: int
    ) -> np.ndarray:
        """
        Executa a inferência na imagem fornecida e redimensiona a saída para as dimensões alvo.

        Args:
            orthophoto_image: Imagem de entrada (RGB) como array Numpy.
            target_height: Altura desejada na saída.
            target_width: Largura desejada na saída.

        Returns:
            Mapa de profundidade predito (float32) no tamanho solicitado.
        """
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
