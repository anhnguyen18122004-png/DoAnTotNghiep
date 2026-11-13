import random

import pandas as pd

HO = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Võ", "Đặng", "Bùi"]
DEM = ["Văn", "Thị", "Minh", "Quốc", "Hữu", "Ngọc", "Thanh", "Đức"]
TEN = ["An", "Bình", "Cường", "Dũng", "Hà", "Hương", "Khoa", "Lan", "Nam", "Phúc", "Trang", "Tuấn"]
TINH = ["Lâm Đồng", "Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Khánh Hòa", "Gia Lai"]
HUYEN = ["Quận 1", "Quận 7", "Đà Lạt", "Cầu Giấy", "Hải Châu", "Nha Trang", "Pleiku"]


def make_sample(n: int = 500, path: str = "sample.csv", seed: int = 42) -> str:
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        phone = "09" + "".join(rnd.choices("0123456789", k=8))
        phone = rnd.choice([phone, "+84" + phone[1:], phone[:4] + " " + phone[4:7] + " " + phone[7:]])
        name = f"{rnd.choice(HO)} {rnd.choice(DEM)} {rnd.choice(TEN)}"
        d, m, y = rnd.randint(1, 28), rnd.randint(1, 12), rnd.randint(1985, 2004)
        rows.append({
            "Họ tên": name,
            "Ngày sinh": rnd.choice([f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}"]),
            "Tuổi": 2026 - y,
            "SĐT": phone,
            "Email": f"user{i}{rnd.choice(['@Gmail.com', '@yahoo.com', '@dlu.edu.vn'])}",
            "CCCD": "0" + str(rnd.randint(10, 96)).zfill(2) + "".join(rnd.choices("0123456789", k=9)),
            "Địa chỉ": f"{rnd.randint(1, 300)} Nguyễn Trãi, {rnd.choice(HUYEN)}, {rnd.choice(TINH)}",
            "Lương": rnd.randint(8, 60) * 1_000_000,
            "Ghi chú": rnd.choice(["", "N/A", f"Liên hệ lại qua {phone}", "Khách VIP", "-"]),
        })
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path