import argparse
import sys
import os
import torch.distributed as dist
from typing import Tuple, Type, Callable, Optional

from marsfill.model.combined_loss import LossWights
from marsfill.model.train import AvailableModels, MarsDepthTrainer
from marsfill.utils import Logger
from marsfill.utils.profiler import get_profile

class TrainingCLI:
    """
    Interface de Linha de Comando (CLI) para orquestrar o treinamento do modelo.
    Responsável por carregar perfis de configuração, definir o ambiente (Local/S3/Distribuído)
    e instanciar o treinador.
    """

    def __init__(
        self, 
        trainer_class: Type[MarsDepthTrainer] = MarsDepthTrainer,
        profile_loader: Callable = get_profile,
        logger_instance: Optional[Logger] = None
    ) -> None:
        """
        Inicializa o orquestrador do CLI.

        Parâmetros:
            trainer_class (Type[MarsDepthTrainer]): A classe do treinador a ser instanciada. Permite injeção de mocks para testes.
            profile_loader (Callable): Função para carregar o dicionário de perfil. Permite injeção para testes.
            logger_instance (Optional[Logger]): Instância de logger customizada.
        """
        self.trainer_class = trainer_class
        self.profile_loader = profile_loader
        self.logger = logger_instance if logger_instance else Logger()

    def _setup_distributed_environment(self) -> Tuple[bool, int, int, int]:
        """
        Verifica e inicializa o ambiente de treinamento distribuído (DDP) baseado em variáveis de ambiente.

        Retorno:
            Tuple[bool, int, int, int]: Uma tupla contendo:
                - is_distributed (bool): Se o ambiente é distribuído.
                - local_rank (int): Rank local do processo.
                - global_rank (int): Rank global do processo.
                - world_size (int): Tamanho total do mundo (número de processos).
        """
        if 'RANK' in os.environ:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            
            return (
                True, 
                int(os.environ.get('LOCAL_RANK', 0)), 
                int(os.environ['RANK']), 
                int(os.environ['WORLD_SIZE'])
            )
        return False, 0, 0, 1

    def execute_training(self, profile_name: str, storage_mode: str) -> None:
        """
        Carrega as configurações e executa o pipeline de treinamento.

        Parâmetros:
            profile_name (str): Nome do perfil de configuração (ex: 'prod', 'dev').
            storage_mode (str): Modo de armazenamento dos dados ('local' ou 's3').

        Levanta:
            RuntimeError: Se o perfil não for encontrado.
            ValueError: Se as configurações do modo de armazenamento estiverem incompletas.
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
            loss_weights=LossWights(
                l1=train_config.get("w_l1", 1.0), 
                gradenty=train_config.get("w_grad", 1.0), 
                ssim=train_config.get("w_ssim", 1.0)
            ),
            storage_mode=storage_mode,
            dataset_root=dataset_root_path,
            dataset_prefix="dataset/v1",
            output_prefix="models",
            is_distributed=is_distributed,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size
        )
        
        trainer.run_training_loop()

def main():
    parser = argparse.ArgumentParser(description="CLI para Treinamento MarsFill")
    parser.add_argument("--profile", type=str, default="prod", help="Nome do perfil de configuração")
    parser.add_argument("--mode", type=str, required=True, choices=["local", "s3"], help="Onde ler/salvar dados")
    
    arguments = parser.parse_args()
    
    cli_orchestrator = TrainingCLI()
    
    try:
        cli_orchestrator.execute_training(arguments.profile, arguments.mode)
    except Exception as error:
        print(f"Erro Crítico na Execução: {error}")
        sys.exit(1)

if __name__ == "__main__":
    main()
