t = open("detection.py", encoding="utf-8").read()
old = '    "SALARY": r"(luong|salary|income|thu_nhap)",\n'
new = old + '    "CCCD": r"(^|_)(cccd|so_cccd|can_cuoc)($|_)",\n    "CMND": r"(^|_)(cmnd|so_cmnd)($|_)",\n'
assert old in t
open("detection.py", "w", encoding="utf-8", newline="\n").write(t.replace(old, new, 1))
print("ok")