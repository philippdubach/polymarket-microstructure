from polydata.inventory import InventoryReport, scan_inventory


def test_inventory_structure():
    r = scan_inventory(limit=3)
    assert isinstance(r, InventoryReport)
    assert r.n_files == 3
    assert r.total_rows > 0
    assert r.first_hour < r.last_hour
    assert isinstance(r.gaps, list)
