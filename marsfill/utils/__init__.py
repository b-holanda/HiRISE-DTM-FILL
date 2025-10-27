import os
import logging

from pathlib import Path

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
