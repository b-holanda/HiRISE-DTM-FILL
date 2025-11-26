import os
import torch
import numpy as np
import boto3
import tempfile
from typing import Optional, Any, Tuple
from transformers import DPTForDepthEstimation, DPTImageProcessor

from marsfill.model.train import AvailableModels
from marsfill.utils import Logger

logger = Logger()


class Evaluator:
    """
    Classe responsável pela inferência (avaliação) do modelo de estimativa de profundidade.
    Gerencia o carregamento do modelo (seja de caminho Local ou URI S3) e realiza a predição,
    retornando os dados brutos em memória.
    """

    def __init__(
        self,
        pretrained_model_name: AvailableModels,
        model_path_uri: str,
        s3_client: Optional[Any] = None,
    ) -> None:
        """
        Inicializa o modelo base e carrega os pesos treinados.

        Args:
            pretrained_model_name: Nome do backbone HuggingFace.
            model_path_uri: Caminho local ou URI S3 do checkpoint .pth.
            s3_client: Cliente Boto3 opcional (para testes/injeção).
        """
        logger.info("Iniciando arquitetura base do avaliador")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s3_client = s3_client

        self.model = DPTForDepthEstimation.from_pretrained(pretrained_model_name.value)

        self._load_model_weights(model_path_uri)

        self.model.to(self.device)
        self.model.eval()

        self.image_processor = DPTImageProcessor.from_pretrained(
            pretrained_model_name.value, do_rescale=False
        )
        logger.info("Modelo carregado e pronto para inferência")

    def _get_s3_client(self) -> Any:
        """
        Retorna o cliente S3 existente ou cria um novo se necessário.

        Returns:
            Cliente Boto3.
        """
        if self.s3_client is None:
            self.s3_client = boto3.client("s3")
        return self.s3_client

    def _extract_s3_bucket_and_key(self, uri: str) -> Tuple[str, str]:
        """
        Analisa uma URI S3 e extrai o nome do bucket e a chave do objeto.

        Args:
            uri: URI no formato s3://bucket-name/path/to/file.

        Returns:
            Nome do bucket e chave.
        """
        clean_uri = uri.replace("s3://", "")
        parts = clean_uri.split("/", 1)
        return parts[0], parts[1]

    def _load_model_weights(self, model_uri: str) -> None:
        """
        Gerencia o carregamento do state_dict. Se a URI for S3, baixa para um arquivo temporário primeiro.

        Args:
            model_uri: URI de origem do arquivo .pth.

        Raises:
            FileNotFoundError: Se o arquivo local não existir.
        """
        logger.info(f"Carregando pesos do modelo de: {model_uri}")

        if model_uri.startswith("s3://"):
            s3_interface = self._get_s3_client()
            bucket_name, object_key = self._extract_s3_bucket_and_key(model_uri)

            with tempfile.NamedTemporaryFile(suffix=".pth") as temporary_file:
                logger.info(f"Baixando modelo do S3 ({bucket_name}/{object_key})...")
                s3_interface.download_file(bucket_name, object_key, temporary_file.name)

                self._apply_state_dict_from_file(temporary_file.name)
        else:
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
