from polydata.paths import DATA_DIR, METADATA_CACHE, parquet_files


def test_data_dir_exists():
    assert DATA_DIR.exists(), "Raw data directory missing"

def test_parquet_files_nonempty():
    files = parquet_files()
    assert len(files) > 1000
    assert all(f.suffix == ".parquet" for f in files)
    assert all(f.name.startswith("polymarket_orderbook_") for f in files)

def test_metadata_cache_path_is_writable_parent():
    assert METADATA_CACHE.parent.exists()
