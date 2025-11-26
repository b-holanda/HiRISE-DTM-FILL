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
        s3_client: Optional[Any] = None
    ) -> None:
        """
        Inicializa a arquitetura do modelo e carrega os pesos treinados.

        Parâmetros:
            pretrained_model_name (AvailableModels): O nome do modelo base (backbone) da HuggingFace.
            model_path_uri (str): URI para os pesos do modelo (.pth). 
                                  Pode ser local (ex: 'data/model.pth') ou S3 (ex: 's3://bucket/model.pth').
            s3_client (Optional[Any]): Cliente Boto3 injetável para facilitar testes. 
                                       Se None e o URI for S3, cria uma instância padrão.
        """
        logger.info("Iniciando arquitetura base do avaliador")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s3_client = s3_client
        
        # Inicializa a arquitetura do modelo (pesos aleatórios/padrão do backbone)
        self.model = DPTForDepthEstimation.from_pretrained(pretrained_model_name.value)
        
        # Carrega os pesos específicos do treinamento (checkpoint)
        self._load_model_weights(model_path_uri)

        self.model.to(self.device)
        self.model.eval()
        
        self.image_processor = DPTImageProcessor.from_pretrained(pretrained_model_name.value, do_rescale=False)
        logger.info("Modelo carregado e pronto para inferência")

    def _get_s3_client(self) -> Any:
        """
        Retorna o cliente S3 existente ou cria um novo se necessário.
        
        Retorno:
            Any: Cliente do Boto3.
        """
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        return self.s3_client

    def _extract_s3_bucket_and_key(self, uri: str) -> Tuple[str, str]:
        """
        Analisa uma URI S3 e extrai o nome do bucket e a chave do objeto.

        Parâmetros:
            uri (str): URI no formato s3://bucket-name/path/to/file

        Retorno:
            Tuple[str, str]: Uma tupla (nome_do_bucket, chave_do_arquivo).
        """
        clean_uri = uri.replace("s3://", "")
        parts = clean_uri.split("/", 1)
        return parts[0], parts[1]

    def _load_model_weights(self, model_uri: str) -> None:
        """
        Gerencia o carregamento do state_dict. Se a URI for S3, baixa para um arquivo temporário primeiro.

        Parâmetros:
            model_uri (str): URI de origem do arquivo .pth.
        
        Levanta:
            FileNotFoundError: Se o arquivo local não existir.
        """
        logger.info(f"Carregando pesos do modelo de: {model_uri}")
        
        if model_uri.startswith("s3://"):
            s3_interface = self._get_s3_client()
            bucket_name, object_key = self._extract_s3_bucket_and_key(model_uri)
            
            # NamedTemporaryFile garante que o arquivo seja deletado após o uso (context exit)
            with tempfile.NamedTemporaryFile(suffix=".pth") as temporary_file:
                logger.info(f"Baixando modelo do S3 ({bucket_name}/{object_key})...")
                s3_interface.download_file(bucket_name, object_key, temporary_file.name)
                
                # Aplica os pesos do arquivo temporário baixado
                self._apply_state_dict_from_file(temporary_file.name)
        else:
            if not os.path.exists(model_uri):
                logger.error(f"Caminho do modelo {model_uri} não encontrado.")
                raise FileNotFoundError(f"Caminho do modelo {model_uri} não encontrado.")
            
            self._apply_state_dict_from_file(model_uri)

    def _apply_state_dict_from_file(self, file_path: str) -> None:
        """
        Carrega o dicionário de estados do disco e remove prefixos específicos de treinamento distribuído (DDP).

        Parâmetros:
            file_path (str): Caminho físico local para o arquivo .pth.
        """
        raw_state_dict = torch.load(file_path, map_location=self.device)
        
        sanitized_state_dict = {}
        for key, value in raw_state_dict.items():
            # Remove o prefixo 'module.' que o DistributedDataParallel adiciona
            clean_key = key[7:] if key.startswith('module.') else key
            sanitized_state_dict[clean_key] = value

        self.model.load_state_dict(sanitized_state_dict)

    def predict_depth(self, orthophoto_image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """
        Executa a inferência na imagem fornecida e redimensiona a saída para as dimensões alvo.

        Parâmetros:
            orthophoto_image (np.ndarray): Imagem de entrada (RGB) como array Numpy.
            target_height (int): Altura desejada para o array de saída.
            target_width (int): Largura desejada para o array de saída.

        Retorno:
            np.ndarray: Mapa de profundidade predito (float32) com as dimensões especificadas.
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
            align_corners=False
        )

        return interpolated_depth.squeeze().cpu().numpy()
