import numpy as np
from osgeo import gdal
import pytest

from marsfill.fill.hole_gen import apply_holes_to_raster, generate_holes_in_array


def test_generate_holes_in_array_seeded():
    data = np.ones((10, 10), dtype=np.float32)
    out = generate_holes_in_array(
        data=data,
        nodata_value=-9999.0,
        num_holes=1,
        min_radius=2,
        max_radius=2,
        seed=123,
    )
    assert out.shape == data.shape
    assert np.any(out != -9999.0)  # sempre deve haver pixels válidos
    # buracos podem não aparecer dependendo do ruído; apenas garantimos que valores novos são nodata
    assert np.all((out == data) | (out == -9999.0))


def test_generate_holes_invalid_radius():
    with pytest.raises(ValueError):
        generate_holes_in_array(
            data=np.ones((4, 4), dtype=np.float32),
            nodata_value=-1.0,
            num_holes=1,
            min_radius=3,
            max_radius=2,
        )


def test_apply_holes_to_raster(tmp_path):
    src_path = tmp_path / "src.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(src_path), 8, 8, 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(np.ones((8, 8), dtype=np.float32))
    ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    ds.FlushCache()
    ds = None

    out_path = tmp_path / "out.tif"
    apply_holes_to_raster(
        input_path=src_path,
        output_path=out_path,
        num_holes=1,
        min_radius=1,
        max_radius=2,
        seed=7,
    )

    out_ds = gdal.Open(str(out_path))
    band = out_ds.GetRasterBand(1)
    out_arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()

    assert np.any(out_arr == nodata)
    assert np.any(out_arr != nodata)
