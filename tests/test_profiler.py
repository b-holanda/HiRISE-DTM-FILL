import os
from pathlib import Path

from marsfill.utils.profiler import load_all_profiles, get_profile


def test_load_all_and_get_profile(tmp_path):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    file_path = profiles_dir / "demo.profile.yml"
    file_path.write_text("foo: 1\nbar: test\n")

    loaded = load_all_profiles(str(profiles_dir))
    assert "demo.profile" in loaded
    assert loaded["demo.profile"]["foo"] == 1

    profile = get_profile("demo.profile", str(profiles_dir))
    assert profile["bar"] == "test"
