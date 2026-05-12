# read_log.py
for fname in ["DtldKHJY.log", "Past.Log", "PoorKHJY.log", "BptnKHJY.log"]:
    path = f"C:\\Users\\25B1029\\Desktop\\{fname}"
    try:
        with open(path, encoding="cp932", errors="ignore") as f:
            content = f.read()
        print(f"\n=== {fname} ===")
        print(content[:300])
    except Exception as e:
        print(f"{fname}: {e}")