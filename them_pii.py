import hashlib
import random
import sys
import unicodedata

import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "healthcare_dataset.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "healthcare_vn_pii.csv"

TINH = {
    "Hà Nội": ("001", ["Cầu Giấy", "Đống Đa", "Hà Đông", "Thanh Xuân"], 16),
    "Hồ Chí Minh": ("079", ["Quận 1", "Quận 7", "Bình Thạnh", "Thủ Đức"], 22),
    "Đà Nẵng": ("048", ["Hải Châu", "Sơn Trà", "Liên Chiểu"], 8),
    "Lâm Đồng": ("068", ["Đà Lạt", "Bảo Lộc", "Đức Trọng"], 7),
    "Gia Lai": ("064", ["Pleiku", "An Khê", "Chư Sê"], 5),
    "Khánh Hòa": ("056", ["Nha Trang", "Cam Ranh"], 6),
    "Cần Thơ": ("092", ["Ninh Kiều", "Cái Răng"], 6),
    "Hải Phòng": ("031", ["Hồng Bàng", "Lê Chân"], 6),
    "Đắk Lắk": ("066", ["Buôn Ma Thuột", "Buôn Hồ"], 6),
    "Nghệ An": ("040", ["Vinh", "Cửa Lò"], 6),
    "Thanh Hóa": ("038", ["Thanh Hóa", "Sầm Sơn"], 5),
    "Đồng Nai": ("075", ["Biên Hòa", "Long Khánh"], 7),
}
NHA_MANG = ["090", "093", "089", "070", "079", "077", "076", "078", "091", "094", "088", "083", "084", "085",
            "081", "082", "096", "097", "098", "086", "032", "033", "034", "035", "036", "037", "038", "039", "056", "058"]
DOMAIN = ["gmail.com"] * 7 + ["yahoo.com", "outlook.com", "hotmail.com"]


def bo_dau(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def sinh(row):
    key = "|".join(str(v) for v in row.values)
    rng = random.Random(int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16))
    nam_gioi = str(row["Gender"]).strip().lower().startswith("m")
    nhap_vien = pd.to_datetime(row["Date of Admission"])
    nam_sinh = (nhap_vien - pd.DateOffset(years=int(row["Age"])) - pd.Timedelta(days=rng.randint(0, 364))).year

    tinh = rng.choices(list(TINH), weights=[v[2] for v in TINH.values()])[0]
    ma_tinh = TINH[tinh][0]

    so_the = (0 if nam_gioi else 1) if nam_sinh < 2000 else (2 if nam_gioi else 3)
    cccd = f"{ma_tinh}{so_the}{nam_sinh % 100:02d}{''.join(rng.choices('0123456789', k=6))}"

    tu = [t for t in bo_dau(str(row["Name"])).replace(".", " ").split() if t.isalpha()]
    local = ".".join(tu[:2]) if tu else "user"
    email = f"{local}{rng.randint(1, 999)}@{rng.choice(DOMAIN)}"
    if rng.random() < 0.1:
        email = email.upper()

    s = rng.choice(NHA_MANG) + "".join(rng.choices("0123456789", k=7))
    sdt = rng.choices([s, "+84" + s[1:], f"{s[:4]} {s[4:7]} {s[7:]}", f"{s[:4]}.{s[4:7]}.{s[7:]}"],
                      weights=[60, 15, 15, 10])[0]
    return {"Phone": sdt, "Email": email, "CCCD": cccd}


df = pd.read_csv(SRC)
extra = pd.DataFrame([sinh(r) for _, r in df.iterrows()])
pos = df.columns.get_loc("Gender") + 1
out = pd.concat([df.iloc[:, :pos], extra, df.iloc[:, pos:]], axis=1)
out.to_csv(OUT, index=False, encoding="utf-8-sig")
print(out.shape, "->", OUT)