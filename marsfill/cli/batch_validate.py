import os
import sys
import json
import argparse
import gc
import torch
import numpy as np
import csv  # Movido para o topo para garantir disponibilidade
from pathlib import Path
from tqdm import tqdm

from marsfill.fill.dtm_filler import DTMFiller
from marsfill.model.eval import Evaluator
from marsfill.model.train import AvailableModels
from marsfill.utils import Logger
from marsfill.fill.filler_stats import FillerStats
from tabulate import tabulate

logger = Logger()

def find_dataset_pairs(root_dir):
    """
    Encontra os pares de arquivos (Input, Ortho, GT).
    Ajustado para o padrão onde o nome do arquivo base é consistente na pasta.
    Ex: Pasta 'dunes' -> Input: 'dune_with_nodata.tif', GT: 'dune.IMG', Ortho: 'dune.JP2'
    """
    cases = []
    root = Path(root_dir)
    
    # Busca recursiva por todos os inputs
    inputs = list(root.rglob("*_with_nodata.tif"))
    
    if not inputs:
        logger.error(f"Nenhum arquivo '*_with_nodata.tif' encontrado em {root_dir}")
        return []

    logger.info(f"Encontrados {len(inputs)} arquivos de input. Buscando pares...")

    for input_path in inputs:
        folder = input_path.parent
        
        # --- LÓGICA ATUALIZADA ---
        # Extrai o nome base removendo o sufixo do input
        # Ex: 'dune_with_nodata.tif' -> base_name = 'dune'
        base_name = input_path.stem.replace("_with_nodata", "")
        
        # 1. Busca pelo Ground Truth (GT)
        gt_candidates = [
            folder / f"{base_name}.IMG",
            folder / f"{base_name}.tif",
            folder / f"{base_name}.GTiff"
        ]
        gt_path = next((p for p in gt_candidates if p.exists()), None)
        
        # 2. Busca pelo Ortomosaico (Ortho)
        ortho_candidates = [
            folder / f"{base_name}.JP2",
            folder / f"{base_name}.tif"
        ]
        
        # Filtra para garantir que o ortho não seja o próprio input ou GT (caso sejam todos .tif)
        ortho_path = next((p for p in ortho_candidates 
                           if p.exists() 
                           and p != input_path 
                           and p != gt_path
                           and "with_nodata" not in p.name), None)

        # Se encontrou o trio completo
        if gt_path and ortho_path:
            cases.append({
                "id": base_name,          # Ex: dune
                "input": input_path,
                "gt": gt_path,
                "ortho": ortho_path,
                "folder": folder.name     # Ex: dunes
            })

    return cases

def run_batch_process(test_dir, output_root_dir, model_path_str, profile="prod"):
    cases = find_dataset_pairs(test_dir)
    if not cases:
        return

    logger.info(f"Carregando modelo de IA: {model_path_str}")
    
    try:
        model_enum = AvailableModels.INTEL_DPT_LARGE 
        evaluator = Evaluator(pretrained_model_name=model_enum, model_path_uri=model_path_str)
        filler = DTMFiller(evaluator=evaluator, padding_size=128, tile_size=512)
        
    except Exception as e:
        logger.error(f"Erro fatal ao carregar modelo: {e}")
        return

    results = []
    logger.info(f"Iniciando processamento em lote de {len(cases)} cenas...")
    
    pbar = tqdm(cases, desc="Progresso Global", unit="cena")

    for case in pbar:
        # Mostra o ID atual na barra de progresso
        pbar.set_description(f"Processando {case['folder']}/{case['id']}")
        
        # Estrutura de saída: output_root / nome_da_pasta / nome_do_arquivo
        case_out_dir = Path(output_root_dir) / case['folder'] / case['id']
        case_out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Preenchimento (Inpainting)
            final_dtm, final_mask, _, _, _ = filler.fill(
                dtm_path=case['input'],
                ortho_path=case['ortho'],
                output_root=str(case_out_dir)
            )
            
            # 2. Cálculo de Métricas e Gráficos
            stats = FillerStats(output_dir=case_out_dir)
            
            metrics = stats.calculate_metrics(
                gt_path=case['gt'],
                filled_path=final_dtm,
                mask_path=final_mask
            )
            
            if metrics and metrics.get('evaluated_pixels', 0) > 0:
                stats.generate_all_outputs(
                    gt_path=case['gt'],
                    input_path=case['input'],
                    ortho_path=case['ortho'],
                    filled_path=final_dtm,
                    mask_path=final_mask,
                    metrics=metrics
                )
                
                res_data = metrics.copy()
                # Cria um ID composto para o relatório (ex: dunes/dune)
                res_data['id'] = f"{case['folder']}/{case['id']}"
                res_data['type'] = case['folder']
                results.append(res_data)
            else:
                pass 

        except Exception as e:
            tqdm.write(f"Erro em {case['id']}: {e}")

        finally:
            # --- OTIMIZAÇÃO DE MEMÓRIA ---
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Relatório Final ---
    if results:
        print("\n" + "="*65)
        print("📊 RELATÓRIO FINAL DE VALIDAÇÃO")
        print("="*65)
        
        table_data = []
        rmses = []
        ssims = []
        times = []
        
        for r in results:
            rmses.append(r.get('rmse_m', 0))
            ssims.append(r.get('ssim', 0))
            times.append(r.get('execution_time_s', 0))
            
            table_data.append([
                r['type'], 
                r['id'], 
                f"{r.get('rmse_m', 0):.2f}", 
                f"{r.get('ssim', 0):.4f}", 
                f"{r.get('execution_time_s', 0):.2f}s"
            ])

        headers = ["Tipo", "ID", "RMSE (m)", "SSIM", "Tempo"]
        
        print(tabulate(table_data, headers=headers, tablefmt="github"))

        print("-" * 65)
        if len(results) > 0:
            print(f"MÉDIA GLOBAL ({len(results)} amostras):")
            print(f"RMSE Médio: {np.mean(rmses):.4f} m")
            print(f"SSIM Médio: {np.mean(ssims):.4f}")
            print(f"Tempo Total: {np.sum(times):.2f} s")
        print("="*65)
        
        csv_path = Path(output_root_dir) / "batch_summary_optimized.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(table_data)
        print(f"📂 CSV salvo em: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch Validation Tool (Optimized)")
    parser.add_argument("--test_dir", default="data/dataset/v1/test", help="Diretório de testes")
    parser.add_argument("--out_dir", default="data/filled_batch_results", help="Saída dos resultados")
    parser.add_argument("--model", default="data/models/marsfill_model.pth", help="Caminho do modelo .pth")
    parser.add_argument("--profile", default="prod", help="Perfil de execução")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        logger.error(f"Modelo não encontrado em: {args.model}")
        sys.exit(1)
        
    run_batch_process(args.test_dir, args.out_dir, args.model, args.profile)

if __name__ == "__main__":
    main()
