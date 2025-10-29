import os
import logging
from re import L

from attr import dataclass
from pathlib import Path
from enum import Enum

class CandidateType(Enum):
    DTM=1
    ORTHO=2

@dataclass(frozen=True)
class CandidateFile:
    filename: str
    href: str
    type: CandidateType

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

    def error(self, message: str, exc_info=True) -> None:
        self._logger.error(message, stack_info=True, exc_info=exc_info)

def make_hirise_expected_filenames(hirise_id: str) -> tuple[str, str]:
    hirise_id_parts = hirise_id.split("_")

    return f"{hirise_id}.tif", f"ESP_{hirise_id_parts[1]}_{hirise_id_parts[2]}_RED_B_01_ORTHO.tif"

def get_dtm_candidate(pair: list[CandidateFile]) -> CandidateFile | None:
    return get_candidate(pair, CandidateType.DTM)

def get_ortho_candidate(pair: list[CandidateFile]) -> CandidateFile | None:
    return get_candidate(pair, CandidateType.ORTHO)

def get_candidate(pair: list[CandidateFile], type: CandidateType) -> CandidateFile | None:
    return next((item for item in pair if item.type == type), None)
