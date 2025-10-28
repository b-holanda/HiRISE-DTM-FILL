from operator import contains
import os
import logging

from attr import dataclass
from pathlib import Path
from functools import reduce

@dataclass(frozen=True)
class CandidateFile:
    filename: str
    href: str

class Logger:
    _instance = None

    def __new__(cls, *args, **keyargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, '_initialized'):
            return
 
        self._logger = logging.getLogger("marsfill")

        self._logger.setLevel(logging.INFO)
        
        console_hadler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(Path(os.path.join(__name__)).parent.parent.parent / "app.log")

        console_hadler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self._logger.addHandler(console_hadler)
        self._logger.addHandler(file_handler)

        self._initialized = True

    def info(self, message: str) -> None:
        self._logger.info(message)

    def debug(self, message:str) -> None:
        self._logger.debug(message)

    def error(self, exception: Exception) -> None:
        self._logger.error(exception, stack_info=True, exc_info=True)

def make_hirise_expected_filenames(hirise_id: str) -> tuple[str, str]:
    hirise_id_parts = hirise_id.split("_")

    return f"{hirise_id}.tif", f"ESP_{hirise_id_parts[1]}_{hirise_id_parts[2]}_RED_B_01_ORTHO.tif"

def get_dtm_candidate(pair: list[CandidateFile]) -> CandidateFile:
    return pair[0] if pair[0].filename == "DTM.tif" else pair[1]

def get_ortho_candidate(pair: list[CandidateFile]) -> CandidateFile:
    return pair[0] if pair[0].filename == "ORTHO.tif" else pair[1]