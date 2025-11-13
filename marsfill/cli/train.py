import os
import torch
import torch.distributed as dist
from marsfill.model.combined_loss import LossWights
from marsfill.model.train import AvaliableModels, Train
from marsfill.utils import Logger
from marsfill.utils.profiler import get_profile

logger = Logger()

def setup_ddp():
    """Configura DDP se estiver rodando com torchrun"""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        return True, local_rank, rank, world_size
    return False, 0, 0, 1

def main():
    is_distributed, local_rank, rank, world_size = setup_ddp()

    profile_name = "prod"
    profile = get_profile(profile_name)

    if not profile:
        if rank == 0:
            logger.error("Nenhum perfil compatível encontrado.")
        return

    # Parâmetros de treino
    selected_model = AvaliableModels.INTEL_DPT_LARGE
    batch_size = profile["train"].get("batch_size", 8)
    learning_rate = profile["train"].get("learning_rate", 1e-5)
    epochs = profile["train"].get("epochs", 50)
    weight_decay = profile["train"].get("weight_decay", 0.01)
    w_l1 = profile["train"].get("w_l1", 1.0)
    w_grad = profile["train"].get("w_grad", 1.0)
    w_ssim = profile["train"].get("w_ssim", 1.0)

    if rank == 0:
        logger.info("🚀 Iniciando treinamento...")
        if is_distributed:
            logger.info(f"   Modo: DDP com {world_size} GPUs")
            logger.info(f"   Batch size efetivo: {batch_size * world_size}")
        else:
            logger.info("   Modo: Single GPU")
        logger.info(f"   Modelo: {selected_model}")
        logger.info(f"   Batch size (per GPU): {batch_size}")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Epochs: {epochs}")
        logger.info(f"   Weight decay: {weight_decay}")
        logger.info(f"   Loss weights: L1={w_l1}, Grad={w_grad}, SSIM={w_ssim}")

    Train(
        selected_model=selected_model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        weight_decay=weight_decay,
        loss_weights=LossWights(l1=w_l1, gradenty=w_grad, ssim=w_ssim),
        is_distributed=is_distributed,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size
    ).run()

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()

    if rank == 0:
        logger.info("✅ Treinamento finalizado!")

if __name__ == '__main__':
    main()
