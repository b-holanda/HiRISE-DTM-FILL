from pathlib import Path
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from osgeo import gdal
from marsfill.utils import Logger

logger = Logger()

class FillerStats:
    def __init__(self, output_dir: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_geotiff(self, path):
        ds = gdal.Open(str(path))
        if not ds:
            return None, None
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        
        if nodata is None:
            if np.nanmin(array) < -1e30:
                nodata = -3.4028235e38
            else:
                nodata = None
        return array, nodata

    def calculate_metrics(self, gt_path, filled_path, mask_path=None):
        logger.info("⚡ Calculando métricas...")
        start_time = time.time()

        gt_arr, gt_nodata = self.load_geotiff(gt_path)
        filled_arr, _ = self.load_geotiff(filled_path)

        if gt_arr is None or filled_arr is None:
            return self._return_empty_metrics()

        # Definição da Máscara
        if mask_path:
            mask_arr, _ = self.load_geotiff(mask_path)
            h = min(mask_arr.shape[0], gt_arr.shape[0])
            w = min(mask_arr.shape[1], gt_arr.shape[1])
            mask_arr = mask_arr[:h, :w]
            gt_arr = gt_arr[:h, :w]
            filled_arr = filled_arr[:h, :w]
            hole_mask = (mask_arr > 0)
        else:
            hole_mask = np.ones_like(gt_arr, dtype=bool)

        if np.sum(hole_mask) == 0:
            return self._return_empty_metrics()

        valid_gt_mask = np.isfinite(gt_arr) & (gt_arr > -1e30)
        if gt_nodata is not None:
             valid_gt_mask &= ~np.isclose(gt_arr, gt_nodata)

        eval_mask = hole_mask & valid_gt_mask
        
        num_pixels = int(np.sum(eval_mask))
        if num_pixels == 0:
            return self._return_empty_metrics()

        # Métricas
        y_true = gt_arr[eval_mask]
        y_pred = filled_arr[eval_mask]

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        ssim_score = self._calculate_masked_ssim(gt_arr, filled_arr, eval_mask)

        metrics = {
            "rmse_m": float(rmse),
            "mae_m": float(mae),
            "ssim": float(ssim_score),
            "execution_time_s": float(time.time() - start_time),
            "evaluated_pixels": num_pixels,
        }

        self._save_metrics(metrics)
        return metrics, gt_arr, filled_arr, eval_mask

    def generate_all_outputs(self, gt_path, input_path, ortho_path, filled_path, mask_path, metrics):
        """
        Gera todas as imagens e gráficos solicitados separadamente.
        """
        gt_arr, _ = self.load_geotiff(gt_path)
        input_arr, _ = self.load_geotiff(input_path)
        ortho_arr, _ = self.load_geotiff(ortho_path)
        filled_arr, _ = self.load_geotiff(filled_path)
        mask_arr, _ = self.load_geotiff(mask_path)

        if gt_arr is None: return

        # Recorte Automático (Crop)
        rows = np.any(mask_arr > 0, axis=1)
        cols = np.any(mask_arr > 0, axis=0)
        
        if np.any(rows):
            pad = 50
            y_min, y_max = max(0, np.where(rows)[0][0] - pad), min(gt_arr.shape[0], np.where(rows)[0][-1] + pad)
            x_min, x_max = max(0, np.where(cols)[0][0] - pad), min(gt_arr.shape[1], np.where(cols)[0][-1] + pad)
        else:
            y_min, y_max, x_min, x_max = 0, gt_arr.shape[0], 0, gt_arr.shape[1]

        def crop(arr): 
            if arr is None: return None
            if arr.ndim == 3: return arr[:, y_min:y_max, x_min:x_max] if arr.shape[0] < arr.shape[1] else arr[y_min:y_max, x_min:x_max, :]
            return arr[y_min:y_max, x_min:x_max]

        gt_c = crop(gt_arr)
        inp_c = crop(input_arr)
        ortho_c = crop(ortho_arr)
        filled_c = crop(filled_arr)
        mask_c = crop(mask_arr)

        # Escala de Cores para DTM (Ignora NoData)
        valid_pixels = gt_c[np.isfinite(gt_c) & (gt_c > -1e30)]
        if valid_pixels.size > 0:
            vmin, vmax = np.percentile(valid_pixels, 2), np.percentile(valid_pixels, 98)
        else:
            vmin, vmax = None, None

        logger.info("🎨 Gerando imagens individuais...")

        # 1. Orthophoto (Correção de contraste aplicada aqui)
        self._save_single_map(ortho_c, "orthophoto.jpg", "Ortoimagem (Referência Visual)", cmap="gray")

        # 2. Ground Truth
        self._save_single_map(gt_c, "dtm_ground_truth.jpg", "Ground Truth (Real)", cmap="terrain", vmin=vmin, vmax=vmax)

        # 3. Input com NoData
        self._save_single_map(inp_c, "dtm_nodata.jpg", "Input (Com Lacunas)", cmap="terrain", vmin=vmin, vmax=vmax)

        # 4. Preenchido
        self._save_single_map(filled_c, "dtm_preenchido.jpg", "Resultado (Preenchido)", cmap="terrain", vmin=vmin, vmax=vmax)

        # 5. Mapa de Erro
        diff = np.abs(gt_c - filled_c)
        diff[mask_c == 0] = 0 
        diff[gt_c < -1e30] = 0
        self._save_single_map(diff, "error_map.jpg", f"Erro Absoluto (RMSE: {metrics.get('rmse_m', 0):.2f}m)", cmap="inferno")

        # 6. Gráfico de Perfil
        self._save_profile_graph(gt_c, filled_c, mask_c, "profile_graph.jpg")

    def _save_single_map(self, data, filename, title, cmap="terrain", vmin=None, vmax=None):
        if data is None: return
        
        plt.figure(figsize=(10, 8))
        
        # Lógica especial para contraste da ortoimagem
        if cmap == "gray":
            # Ignora o valor 0 (preto) para calcular o histograma de cores
            valid_viz = data[data > 0]
            if valid_viz.size > 0:
                # Recalcula vmin/vmax localmente para a orto
                local_vmin, local_vmax = np.percentile(valid_viz, 2), np.percentile(valid_viz, 98)
                plot_data = data
                # Usa os limites calculados
                plt.imshow(plot_data, cmap=cmap, vmin=local_vmin, vmax=local_vmax)
            else:
                plt.imshow(data, cmap=cmap)
        else:
            # Para DTMs, mascara NoData para ficar branco/transparente
            plot_data = np.where(data < -1e30, np.nan, data)
            plt.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.title(title, fontsize=14)
        
        # Barra de cores
        if cmap != "gray":
            plt.colorbar(fraction=0.046, pad=0.04, label="Metros")
        
        plt.axis('off')
        
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()

    def _save_profile_graph(self, gt, filled, mask, filename):
        plt.figure(figsize=(12, 6))
        
        mid_y = gt.shape[0] // 2
        line_gt = gt[mid_y, :]
        line_pred = filled[mid_y, :]
        x_axis = np.arange(len(line_gt))
        
        # Filtro Rigoroso: Só plota pontos onde AMBOS (GT e Pred) são dados válidos
        # Assume que a elevação em Marte não é menor que -10000m (Gale Crater é ~ -4500m)
        valid_mask = (line_gt > -10000) & (line_pred > -10000)
        
        # Se a linha inteira for inválida no meio, tenta achar outra linha
        if np.sum(valid_mask) == 0:
            # Tenta achar uma linha que atravesse um buraco
            rows_with_holes = np.where(np.any(mask > 0, axis=1))[0]
            if len(rows_with_holes) > 0:
                mid_y = rows_with_holes[len(rows_with_holes)//2]
                line_gt = gt[mid_y, :]
                line_pred = filled[mid_y, :]
                valid_mask = (line_gt > -10000) & (line_pred > -10000)

        x_plot = x_axis[valid_mask]
        gt_plot = line_gt[valid_mask]
        pred_plot = line_pred[valid_mask]
        
        if len(x_plot) > 0:
            plt.plot(x_plot, gt_plot, 'k-', lw=2.5, label='Ground Truth', alpha=0.8)
            plt.plot(x_plot, pred_plot, 'r--', lw=2, label='Predição IA')

            # Destaque da área preenchida (precisa filtrar pelos x válidos também)
            hole_indices = np.where(mask[mid_y, :] > 0)[0]
            # Intersecção entre onde é buraco e onde é válido plotar
            valid_hole_indices = [h for h in hole_indices if h in x_plot]
            
            if len(valid_hole_indices) > 0:
                from itertools import groupby, count
                for _, g in groupby(valid_hole_indices, key=lambda n, c=count(): n-next(c)):
                    group = list(g)
                    plt.axvspan(group[0], group[-1], color='yellow', alpha=0.3)
                plt.axvspan(valid_hole_indices[0], valid_hole_indices[0], color='yellow', alpha=0.3, label='Lacuna')

        plt.title(f"Perfil Topográfico (Transecto Y={mid_y})", fontsize=14)
        plt.xlabel("Pixels")
        plt.ylabel("Elevação (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()

    # -- Métodos Auxiliares Mantidos --
    def _calculate_masked_ssim(self, gt, pred, mask):
        valid_pixels = gt[np.isfinite(gt) & (gt > -1e30)]
        if valid_pixels.size == 0: return 0.0
        mn, mx = valid_pixels.min(), valid_pixels.max()
        scale = mx - mn + 1e-6
        gt_t = torch.tensor(gt, device=self.device).unsqueeze(0).unsqueeze(0)
        pred_t = torch.tensor(pred, device=self.device).unsqueeze(0).unsqueeze(0)
        mask_t = torch.tensor(mask, device=self.device).unsqueeze(0).unsqueeze(0)
        pred_masked = torch.where(mask_t, pred_t, gt_t)
        gt_norm = torch.clamp((gt_t - mn) / scale, 0, 1)
        pred_norm = torch.clamp((pred_masked - mn) / scale, 0, 1)
        return self.ssim_calc(pred_norm, gt_norm).item()

    def _return_empty_metrics(self):
        empty = {"rmse_m": 0.0, "mae_m": 0.0, "ssim": 0.0, "execution_time_s": 0.0, "evaluated_pixels": 0}
        self._save_metrics(empty)
        return empty, None, None, None

    def _save_metrics(self, metrics):
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
