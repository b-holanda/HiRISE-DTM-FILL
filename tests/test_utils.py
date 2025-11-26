import builtins
import os
from pathlib import Path

import pytest

from marsfill.utils import (
    generate_expected_filenames,
    CandidateFile,
    CandidateType,
    find_candidate_by_type,
    find_digital_terrain_model_candidate,
    find_orthophoto_candidate,
    validate_dataset_pairs,
    convert_strings_to_paths,
    list_parquet_files,
)


def test_generate_expected_filenames():
    dtm, ortho = generate_expected_filenames("ESP_012345_1234")
    assert dtm == "ESP_012345_1234.tif"
    assert ortho == "ESP_012345_1234_RED_B_01_ORTHO.tif"


def test_candidate_helpers():
    candidates = [
        CandidateFile("a", "u1", CandidateType.ORTHOPHOTO),
        CandidateFile("b", "u2", CandidateType.DIGITAL_TERRAIN_MODEL),
    ]
    assert find_candidate_by_type(candidates, CandidateType.ORTHOPHOTO).filename == "a"
    assert find_digital_terrain_model_candidate(candidates).filename == "b"
    assert find_orthophoto_candidate(candidates).filename == "a"


def test_validate_dataset_pairs(tmp_path):
    ortho = tmp_path / "o.tif"
    dtm = tmp_path / "d.tif"
    ortho.write_text("x")
    dtm.write_text("y")
    o_list, d_list = validate_dataset_pairs([str(ortho)], [str(dtm)])
    assert o_list[0].endswith("o.tif")
    with pytest.raises(FileNotFoundError):
        validate_dataset_pairs([str(ortho)], [str(tmp_path / "missing.tif")])
    with pytest.raises(ValueError):
        validate_dataset_pairs([], [])


def test_convert_strings_to_paths():
    paths = convert_strings_to_paths(["a", "b"])
    assert all(isinstance(p, Path) for p in paths)
    assert [p.name for p in paths] == ["a", "b"]


def test_list_parquet_files_local(tmp_path):
    (tmp_path / "nested").mkdir()
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "nested" / "b.parquet"
    f1.write_text("x")
    f2.write_text("y")
    found = list_parquet_files(str(tmp_path))
    assert str(f1) in found and str(f2) in found


def test_list_parquet_files_s3(monkeypatch):
    class FakeS3FS:
        def glob(self, pattern):
            return ["bucket/path/a.parquet", "bucket/path/b.parquet"]

    monkeypatch.setattr("marsfill.utils.s3fs.S3FileSystem", lambda anon=False: FakeS3FS())
    found = list_parquet_files("s3://bucket/path")
    assert found == ["s3://bucket/path/a.parquet", "s3://bucket/path/b.parquet"]
