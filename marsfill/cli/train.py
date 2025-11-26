import argparse
import sys
import os
import torch.distributed as dist
from typing import Tuple, Type, Callable, Optional

from marsfill.model.combined_loss import LossWeights
from marsfill.model.train import AvailableModels, MarsDepthTrainer
from marsfill.utils import Logger
from marsfill.utils.profiler import get_profile


class TrainingCLI:
    """
    Interface de Linha de Comando (CLI) para orquestrar o treinamento do modelo.

    Responsabilidades principais:
    - Ler perfis de configuração.
    - Definir ambiente (local, S3, distribuído).
    - Instanciar e executar o laço de treinamento.
    """

    def __init__(
        self,
        trainer_class: Type[MarsDepthTrainer] = MarsDepthTrainer,
        profile_loader: Callable = get_profile,
        logger_instance: Optional[Logger] = None,
    ) -> None:
        """
        Inicializa o orquestrador do CLI.

        Args:
            trainer_class: Classe do treinador a ser instanciada (permite injeção em testes).
            profile_loader: Função que retorna um dicionário de perfil.
            logger_instance: Logger opcional para reutilização em testes.
        """
        self.trainer_class = trainer_class
        self.profile_loader = profile_loader
        self.logger = logger_instance if logger_instance else Logger()

    def _setup_distributed_environment(self) -> Tuple[bool, int, int, int]:
        """
        Inicializa (se necessário) o ambiente distribuído (DDP) a partir de variáveis de ambiente.

        Returns:
            Uma tupla com: (is_distributed, local_rank, global_rank, world_size).
        """
        if "RANK" in os.environ:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            return (
                True,
                int(os.environ.get("LOCAL_RANK", 0)),
                int(os.environ["RANK"]),
                int(os.environ["WORLD_SIZE"]),
            )
        return False, 0, 0, 1

    def execute_training(self, profile_name: str, storage_mode: str) -> None:
        """
        Carrega as configurações e executa o pipeline de treinamento.

        Args:
            profile_name: Nome do perfil de configuração (ex.: `prod`, `dev`).
            storage_mode: Modo de armazenamento (`local` ou `s3`).

        Raises:
            RuntimeError: Se o perfil não existir.
            ValueError: Se o modo for inválido ou faltarem campos obrigatórios.
        """
        is_distributed, local_rank, global_rank, world_size = self._setup_distributed_environment()

        training_configuration = self.profile_loader(profile_name)

        if not training_configuration:
            error_message = f"Perfil '{profile_name}' não encontrado."
            self.logger.error(error_message)
            raise RuntimeError(error_message)

        train_config = training_configuration.get("train", {})

        dataset_root_path = ""

        if storage_mode == "s3":
            dataset_root_path = train_config.get("s3_bucket")
            if not dataset_root_path:
                error_message = "Modo S3 exige a chave 's3_bucket' definida no profile."
                self.logger.error(error_message)
                raise ValueError(error_message)

        elif storage_mode == "local":
            dataset_root_path = train_config.get("local_data_dir", "data")

        else:
            error_message = "Modo inválido. Use 'local' ou 's3'."
            self.logger.error(error_message)
            raise ValueError(error_message)

        if global_rank == 0:
            self.logger.info(f"🚀 Configuração: Profile={profile_name}, Mode={storage_mode}")
            self.logger.info(f"📂 Root: {dataset_root_path}")

        trainer = self.trainer_class(
            selected_model_name=AvailableModels.INTEL_DPT_LARGE,
            batch_size=train_config.get("batch_size", 8),
            learning_rate=train_config.get("learning_rate", 1e-5),
            total_epochs=train_config.get("epochs", 50),
            weight_decay=train_config.get("weight_decay", 0.01),
            loss_weights=LossWeights(
                l1_weight=train_config.get("w_l1", 1.0),
                gradient_weight=train_config.get("w_grad", 1.0),
                ssim_weight=train_config.get("w_ssim", 1.0),
            ),
            storage_mode=storage_mode,
            dataset_root=dataset_root_path,
            dataset_prefix="dataset/v1",
            output_prefix="models",
            is_distributed=is_distributed,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
        )

        trainer.run_training_loop()


def main():
    """Ponto de entrada da CLI de treinamento."""
    parser = argparse.ArgumentParser(description="CLI para Treinamento MarsFill")
    parser.add_argument(
        "--profile", type=str, default="prod", help="Nome do perfil de configuração"
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["local", "s3"], help="Onde ler/salvar dados"
    )

    arguments = parser.parse_args()

    cli_orchestrator = TrainingCLI()

    try:
        cli_orchestrator.execute_training(arguments.profile, arguments.mode)
    except Exception as error:
        print(f"Erro Crítico na Execução: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
