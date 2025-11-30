import os
import sys
import json
import argparse
import gc
import torch
import numpy as np
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
    """Encontra os pares de arquivos (Input, Ortho, GT)."""
    cases = []
    root = Path(root_dir)
    inputs = list(root.rglob("*_with_nodata.tif"))
    
    if not inputs:
        logger.error(f"Nenhum arquivo '*_with_nodata.tif' encontrado em {root_dir}")
        return []

    logger.info(f"Encontrados {len(inputs)} arquivos de input. Buscando pares...")

    for input_path in inputs:
        folder = input_path.parent
        base_name = input_path.name.replace("_with_nodata.tif", "")
        
        gt_candidates = [
            folder / (base_name + ".IMG"),
            folder / (base_name + ".tif"),
            folder / (base_name + ".GTiff")
        ]
        gt_path = next((p for p in gt_candidates if p.exists()), None)
        
        orbit_id_parts = base_name.split('_')[1:3]
        if len(orbit_id_parts) >= 2:
            orbit_key = f"{orbit_id_parts[0]}_{orbit_id_parts[1]}"
            ortho_candidates = list(folder.glob(f"*{orbit_key}*.JP2")) + \
                               list(folder.glob(f"*{orbit_key}*.tif"))
            
            ortho_path = next((p for p in ortho_candidates 
                               if "DTEPC" not in p.name and "with_nodata" not in p.name), None)
        else:
            ortho_path = None

        if gt_path and ortho_path:
            cases.append({
                "id": base_name,
                "input": input_path,
                "gt": gt_path,
                "ortho": ortho_path,
                "folder": folder.name
            })

    return cases

def run_batch_process(test_dir, output_root_dir, model_path_str, profile="prod"):
    cases = find_dataset_pairs(test_dir)
    if not cases:
        return

    logger.info(f"Carregando modelo de IA: {model_path_str}")
    
    try:
        model_enum = AvailableModels.DPT_LARGE 
        evaluator = Evaluator(pretrained_model_name=model_enum, model_path_uri=model_path_str)
        filler = DTMFiller(evaluator=evaluator, padding_size=128, tile_size=512)
        
    except Exception as e:
        logger.error(f"Erro fatal ao carregar modelo: {e}")
        return

    results = []
    logger.info(f"Iniciando processamento em lote de {len(cases)} cenas...")
    
    pbar = tqdm(cases, desc="Progresso Global", unit="cena")

    for case in pbar:
        pbar.set_description(f"Processando {case['id'][:15]}")
        
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
            
            # --- CORREÇÃO AQUI: Recebe apenas 1 valor (metrics) ---
            metrics = stats.calculate_metrics(
                gt_path=case['gt'],
                filled_path=final_dtm,
                mask_path=final_mask
            )
            
            if metrics and metrics.get('evaluated_pixels', 0) > 0:
                # Gera os 6 gráficos/imagens separados
                # O método generate_all_outputs carrega os arquivos por conta própria
                stats.generate_all_outputs(
                    gt_path=case['gt'],
                    input_path=case['input'],
                    ortho_path=case['ortho'],
                    filled_path=final_dtm,
                    mask_path=final_mask,
                    metrics=metrics
                )
                
                res_data = metrics.copy()
                res_data['id'] = case['id']
                res_data['type'] = case['folder']
                results.append(res_data)
            else:
                pass 

        except Exception as e:
            tqdm.write(f"Erro em {case['id']}: {e}")
            # import traceback
            # traceback.print_exc()

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
                r['id'][:25], 
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
        
        import csv
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
