import sys
import numpy as np
from osgeo import gdal

def inspect_tif(path):
    print(f"🔍 Inspecionando: {path}")
    ds = gdal.Open(path)
    if not ds:
        print("❌ Erro: Não foi possível abrir o arquivo.")
        return

    band = ds.GetRasterBand(1)
    nodata_val = band.GetNoDataValue()
    array = band.ReadAsArray()
    
    print(f"📋 Metadado NoData (GDAL): {nodata_val}")
    print(f"📊 Estatísticas dos Pixels:")
    print(f"   Min: {np.nanmin(array)}")
    print(f"   Max: {np.nanmax(array)}")
    print(f"   Média: {np.nanmean(array)}")
    print("-" * 30)
    
    # Verificações de Lacunas
    count_nan = np.sum(np.isnan(array))
    print(f"   Pixels NaN: {count_nan}")
    
    if nodata_val is not None:
        count_nodata = np.sum(array == nodata_val)
        print(f"   Pixels iguais ao NoData ({nodata_val}): {count_nodata}")
        
        # Verificação com tolerância (para floats)
        if isinstance(nodata_val, float):
            count_close = np.sum(np.isclose(array, nodata_val, atol=1e-5))
            print(f"   Pixels 'próximos' ao NoData (tolerância): {count_close}")
    
    # Verifica valores muito baixos (comuns para NoData não declarado)
    count_low = np.sum(array < -1e30)
    print(f"   Pixels muito baixos (<-1e30): {count_low}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python check_nodata.py <caminho_do_tif>")
    else:
        inspect_tif(sys.argv[1])
