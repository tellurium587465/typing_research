TWL_PATH = r"C:\Users\25B1029\Desktop\ComJR.twl"

with open(TWL_PATH, "rb") as f:
    raw = f.read()

# ASCII部分だけ表示
content = raw.decode("ascii", errors="replace")
print(content[:2000])