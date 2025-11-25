
from ast import mod
from pathlib import Path
from marsfill.model.train import AvaliableModels
from marsfill.utils import Logger
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
import os
import numpy as np

logger = Logger()

class Evaluator:
    def __init__(
            self,
            pretrained_model_name: AvaliableModels,
            model_path: Path
    ):
        logger.info("Iniciando arquitetura base")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = DPTForDepthEstimation.from_pretrained(pretrained_model_name.value)

        if not os.path.exists(model_path):
            logger.error(f"Model path {model_path} does not exist.")

            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        
        logger.info(f"Carregando pesos do modelo de {model_path}")

        state_dict = self._load_state_dict(model_path)

        self._model.load_state_dict(state_dict)

        self._model.to(self._device)
        self._model.eval()
        self._processor = DPTImageProcessor.from_pretrained(pretrained_model_name.value, do_rescale=False)

        logger.info("Modelo carregado com sucesso")

    def _load_state_dict(self, model_path: Path)-> dict:
        state_dict = torch.load(model_path, map_location=self._device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k 
                new_state_dict[name] = v

        return new_state_dict

    def predict(self, orthoimage: np.ndarray, height: int, width: int) -> np.ndarray:
        inputs = self._processor(images=orthoimage, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth,
            size=(height, width),
            mode="bicubic",
            align_corners=False
        )

        return prediction.squeeze().cpu().numpy()
