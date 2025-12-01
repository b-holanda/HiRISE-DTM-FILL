import pyarrow.dataset as ds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# --- Configuração ---
DATASET_PATH = 'data/dataset/v1/train' # Aponte para sua pasta de treino
BATCH_SIZE = 1000 # Processa 1000 tiles por vez para não estourar a RAM

def analisar_dataset(path):
    print(f"🚀 Iniciando análise do dataset em: {path}")
    
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(batch_size=BATCH_SIZE)
    total_rows = dataset.count_rows()
    
    # Acumuladores para estatísticas
    stats_ortho = {'means': [], 'stds': []}
    stats_dtm = {'means': [], 'stds': []}
    
    print(f"📊 Total de amostras encontradas: {total_rows}")
    print("Processando lotes (isso pode demorar um pouco)...")
    
    # Barra de progresso para acompanhar
    for batch in tqdm(scanner.to_batches(), total=total_rows // BATCH_SIZE):
        df = batch.to_pandas()
        
        # Itera sobre as linhas do batch
        for _, row in df.iterrows():
            # Decodifica (conforme seu build.py)
            ortho = np.frombuffer(row['ortho_bytes'], dtype=np.float32)
            dtm = np.frombuffer(row['dtm_bytes'], dtype=np.float32)
            
            # Coleta métricas rápidas de cada tile (muito mais leve que guardar a imagem toda)
            stats_ortho['means'].append(ortho.mean())
            stats_ortho['stds'].append(ortho.std())
            
            stats_dtm['means'].append(dtm.mean())
            stats_dtm['stds'].append(dtm.std())

    return stats_ortho, stats_dtm

def plotar_resultados(stats_ortho, stats_dtm):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Distribuição de Brilho (Ortho)
    sns.histplot(stats_ortho['means'], bins=50, kde=True, ax=axes[0, 0], color='gray')
    axes[0, 0].set_title("Distribuição do Albedo Médio (Ortoimagens)")
    axes[0, 0].set_xlabel("Valor Médio de Pixel [0-1]")
    axes[0, 0].set_ylabel("Contagem de Tiles")
    axes[0, 0].text(0.05, 0.9, "Interpretação: Se houver picos nos extremos,\nmuitas imagens são totalmente pretas ou brancas.", 
                    transform=axes[0, 0].transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    # 2. Distribuição de Contraste (Ortho Std)
    sns.histplot(stats_ortho['stds'], bins=50, kde=True, ax=axes[0, 1], color='blue')
    axes[0, 1].set_title("Complexidade Visual (Desvio Padrão - Ortho)")
    axes[0, 1].set_xlabel("Desvio Padrão")
    
    # 3. Distribuição de Elevação Média (DTM)
    # Nota: Como é normalizado por tile, isso deve tender a 0.5 se for bem distribuído
    sns.histplot(stats_dtm['means'], bins=50, kde=True, ax=axes[1, 0], color='green')
    axes[1, 0].set_title("Distribuição de Elevação Relativa (DTM)")
    axes[1, 0].set_xlabel("Valor Médio de Elevação [0-1]")

    # 4. Rugosidade do Terreno (DTM Std)
    sns.histplot(stats_dtm['stds'], bins=50, kde=True, ax=axes[1, 1], color='red')
    axes[1, 1].set_title("Rugosidade do Terreno (Desvio Padrão - DTM)")
    axes[1, 1].set_xlabel("Desvio Padrão (Variação de altura no tile)")
    axes[1, 1].text(0.05, 0.9, "Interpretação: Valores altos indicam\nterrenos acidentados (escarpas, crateras).", 
                    transform=axes[1, 1].transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("dataset_quality_report.png")
    print("\n✅ Gráficos salvos em 'dataset_quality_report.png'")
    plt.show()

# --- Execução ---
if __name__ == "__main__":
    s_ortho, s_dtm = analisar_dataset(DATASET_PATH)
    plotar_resultados(s_ortho, s_dtm)
