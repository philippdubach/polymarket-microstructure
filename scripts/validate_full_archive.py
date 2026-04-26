from polydata.inventory import scan_inventory

if __name__ == "__main__":
    r = scan_inventory()
    print(f"files={r.n_files} rows={r.total_rows:,} gb={r.total_bytes/1e9:.1f}")
    print(f"first={r.first_hour} last={r.last_hour}")
    print(f"gaps={len(r.gaps)} schema_errors={len(r.schema_errors)}")
    for g in r.gaps[:10]:
        print(f"  GAP {g}")
    for p, e in r.schema_errors[:10]:
        print(f"  SCHEMA_ERR {p.name}: {e}")
