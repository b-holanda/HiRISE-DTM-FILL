import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# Tenta importar tabulate para tabelas bonitas (opcional)
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

def find_dataset_pairs(root_dir):
    """
    Encontra trios: (Input com NoData, Ortho, Ground Truth)
    baseado na convenção de nomes do HiRISE.
    """
    cases = []
    root = Path(root_dir)
    
    # Encontra todos os arquivos de input (com buracos)
    inputs = list(root.rglob("*_with_nodata.tif"))
    
    if not inputs:
        print(f"❌ Nenhum arquivo '*_with_nodata.tif' encontrado em {root_dir}")
        return []

    print(f"🔍 Encontrados {len(inputs)} arquivos de input. Buscando pares correspondentes...")

    for input_path in inputs:
        folder = input_path.parent
        base_name = input_path.name.replace("_with_nodata.tif", "")
        
        # 1. Tenta achar o Ground Truth (Original sem _with_nodata)
        gt_candidates = [
            folder / (base_name + ".IMG"),
            folder / (base_name + ".tif"),
            folder / (base_name + ".GTiff")
        ]
        # Pega o primeiro que existir
        gt_path = next((p for p in gt_candidates if p.exists()), None)
        
        # 2. Tenta achar a Ortoimagem (Baseado no Orbit ID)
        # Ex: DTEPC_088676_2540 -> ID: 088676_2540 -> Busca ESP_088676_2540...
        orbit_id_parts = base_name.split('_')[1:3]
        if len(orbit_id_parts) >= 2:
            orbit_key = f"{orbit_id_parts[0]}_{orbit_id_parts[1]}"
            # Procura JP2 ou TIF que contenha o ID mas não seja DTEPC
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

def run_batch(test_dir, output_root_dir, profile="prod"):
    # Verifica se o script shell existe
    fill_script = Path("fill.sh").resolve()
    if not fill_script.exists():
        print("❌ Erro: 'fill.sh' não encontrado na raiz do diretório atual.")
        sys.exit(1)

    # Busca os casos
    cases = find_dataset_pairs(test_dir)
    if not cases:
        print("Nenhum caso completo encontrado.")
        return

    results = []
    
    print(f"\n🚀 Iniciando processamento em lote de {len(cases)} casos usando ./fill.sh\n")

    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Processando: {case['id']} ({case['folder']})")
        
        # Cria pasta de saída específica para este caso
        case_out_dir = Path(output_root_dir) / case['folder'] / case['id']
        case_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Constrói o comando chamando o script bash
        cmd = [
            str(fill_script),
            "--profile", profile,
            "--dtm", str(case['input']),
            "--ortho", str(case['ortho']),
            "--gt", str(case['gt']),
            "--out_dir", str(case_out_dir)
        ]
        
        try:
            # Executa e aguarda terminar
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Verifica se o JSON de métricas foi gerado
            metrics_file = case_out_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    data['id'] = case['id']
                    data['type'] = case['folder'] 
                    results.append(data)
                    print(f"   ✅ Sucesso! RMSE: {data.get('rmse_m', 0):.2f}m | SSIM: {data.get('ssim', 0):.4f}")
            else:
                print("   ⚠️  Executou, mas metrics.json não encontrado.")
                # Opcional: imprimir stderr se falhar silenciosamente
                # print(result.stderr)

        except subprocess.CalledProcessError as e:
            print(f"   ❌ Falha na execução do fill.sh.")
            # Mostra o erro real do script shell se falhar
            print(f"   Erro: {e.stderr.strip()[-300:]}") # Mostra os últimos 300 chars do erro

    # --- Geração do Relatório ---
    if results:
        print("\n" + "="*65)
        print("📊 RELATÓRIO FINAL DE VALIDAÇÃO (BATCH)")
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
                r['id'][:25], # Trunca nome longo
                f"{r.get('rmse_m', 0):.2f}", 
                f"{r.get('ssim', 0):.4f}",
                f"{r.get('execution_time_s', 0):.2f}s"
            ])

        headers = ["Tipo", "ID", "RMSE (m)", "SSIM", "Tempo"]
        
        if HAS_TABULATE:
            print(tabulate(table_data, headers=headers, tablefmt="github"))
        else:
            # Fallback simples
            print(f"{'Tipo':<10} {'ID':<30} {'RMSE':<10} {'SSIM':<10}")
            for row in table_data:
                print(f"{row[0]:<10} {row[1]:<30} {row[2]:<10} {row[3]:<10}")

        print("-" * 65)
        print(f"MÉDIA GLOBAL ({len(results)} amostras):")
        print(f"RMSE Médio: {np.mean(rmses):.4f} m")
        print(f"SSIM Médio: {np.mean(ssims):.4f}")
        print(f"Tempo Total: {np.sum(times):.2f} s")
        print("="*65)
        
        # Salva CSV
        import csv
        csv_path = Path(output_root_dir) / "batch_summary.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(table_data)
        print(f"📂 CSV salvo em: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Caminho onde estão as pastas 'dunes', 'craters', etc.
    parser.add_argument("--test_dir", default="data/dataset/v1/test", help="Diretório de testes")
    parser.add_argument("--out_dir", default="data/filled_batch_results", help="Saída dos resultados")
    args = parser.parse_args()
    
    run_batch(args.test_dir, args.out_dir)
