import pandas as pd
import pytest

import ingestion
from anonymization import (dp_histogram, generalize_address, generalize_numeric,
                           laplace_perturb, mask, mask_email, tokenize)
from cleaning import clean_and_standardize, snake
from detection import detect

KEY = b"k" * 32


@pytest.fixture(autouse=True)
def tmp_bronze(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion, "BRONZE", tmp_path)


def test_snake():
    assert snake("Họ và Tên") == "ho_va_ten"
    assert snake("Địa chỉ") == "dia_chi"


def test_ingest_csv_keeps_raw(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("sdt,ten\n0912345678,An\n", encoding="utf-8")
    df, batch = ingestion.ingest_file(str(f))
    assert df.loc[0, "sdt"] == "0912345678"
    assert (batch / "manifest.json").exists() and (batch / "a.csv").exists()


def test_ingest_sql_select_only(tmp_path):
    url = f"sqlite:///{tmp_path / 't.db'}"
    from sqlalchemy import create_engine, text
    with create_engine(url).begin() as c:
        c.execute(text("CREATE TABLE u(id INTEGER, email TEXT)"))
        c.execute(text("INSERT INTO u VALUES (1,'a@b.com')"))
    df, _ = ingestion.ingest_sql(url, table="u")
    assert len(df) == 1
    with pytest.raises(ValueError):
        ingestion.ingest_sql(url, query="DROP TABLE u")
    with pytest.raises(ValueError):
        ingestion.ingest_sql(url, table="u; DROP TABLE u")


def test_cleaning_standardizes():
    df = pd.DataFrame({
        "SĐT": ["+84912345678", "0912 345 679", "091.234.5680", "N/A"],
        "Email": ["A@Gmail.com", "b@x.vn", "c@x.vn", None],
        "Ghi chú": ["gọi 0912345678", "ok", "ok", "ok"],
        "Tuổi": ["20", "21", "22", "23"],
    })
    out, rep = clean_and_standardize(df.astype("string"))
    assert out["sdt"].dropna().tolist() == ["0912345678", "0912345679", "0912345680"]
    assert out["email"].iloc[0] == "a@gmail.com"
    assert out["ghi_chu"].iloc[0] == "gọi 0912345678"
    assert pd.api.types.is_numeric_dtype(out["tuoi"])


def test_cleaning_dedup():
    df = pd.DataFrame({"a": ["x", "x", " x "]}).astype("string")
    out, rep = clean_and_standardize(df)
    assert len(out) == 1 and rep["duplicates_removed"] == 2


def test_detect_regex_and_inline():
    df = pd.DataFrame({
        "c1": ["0912345678"] * 5,
        "c2": ["a@b.com"] * 5,
        "c3": ["012345678901"] * 5,
        "c4": ["gọi 0912345678", "ok", "ok", "ok", "ok"],
        "c5": ["hello"] * 5,
    }).astype("string")
    f = {x["column"]: x for x in detect(df)}
    assert f["c1"]["type"] == "PHONE" and f["c1"]["scope"] == "column"
    assert f["c2"]["type"] == "EMAIL"
    assert f["c3"]["type"] == "CCCD"
    assert f["c4"]["scope"] == "inline"
    assert "c5" not in f


def test_mask_tokenize():
    assert mask("0912345678", keep_last=3) == "*******678"
    assert mask_email("abc@x.com") == "a***@x.com"
    assert tokenize("x", KEY) == tokenize("x", KEY)
    assert tokenize("x", KEY) != tokenize("y", KEY)


def test_generalize():
    assert generalize_numeric(pd.Series([23, 31]), 5).tolist() == ["20-24", "30-34"]
    assert generalize_address("1 A, Đà Lạt, Lâm Đồng", 2) == "Đà Lạt, Lâm Đồng"


def test_dp_bounds_and_seed():
    s = pd.Series(range(100, 200))
    a = laplace_perturb(s, 1.0, 100, 200, seed=1)
    b = laplace_perturb(s, 1.0, 100, 200, seed=1)
    assert a.equals(b) and a.between(100, 200).all()
    assert (dp_histogram(pd.Series(["a"] * 50 + ["b"] * 10), 1.0, seed=0) >= 0).all()