# Data Platform đánh giá rủi ro quyền riêng tư (Medallion)

Pipeline Bronze – Silver – Gold: nạp dữ liệu, làm sạch và chuẩn hóa, phát hiện PII, ẩn danh hóa, đánh giá rủi ro.

## Cài đặt

    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python -m spacy download xx_ent_wiki_sm

## Các module (gọi từ API hoặc dòng lệnh)

- ingestion.py: ingest_file(path), ingest_sql(url, table hoặc query)
- cleaning.py: clean_and_standardize(df)
- detection.py: detect(df, ner)
- anonymization.py: anonymize(df, findings, key, policy, seed)
- gold.py: assess(df, qi, sa, findings, cleaning, masked) trả về hồ sơ dữ liệu, k-anonymity, l-diversity
- risk.py: prepare(path, age_band, no_ner, seed), score_records(df, qi, sa), dataset_risk(scored, n_pii_unmasked), validate(...)

Ví dụ gọi từ API:

    from risk import prepare, score_records, dataset_risk
    silver, anon, findings, masked = prepare(duong_dan_tep, age_band=5)
    scored = score_records(anon, qi, sa)
    ket_qua = dataset_risk(scored, len({f["column"] for f in findings} - masked))

Dữ liệu trung gian ghi vào thư mục data (bronze, silver, gold), tự tạo khi chạy.

## Dòng lệnh

    python gold.py --file <tệp>.csv --qi age gender blood_type --sa medical_condition --age-band 5
    python risk.py --file <tệp>.csv --qi age gender blood_type --sa medical_condition --age-band 5

Biến môi trường PII_TOKEN_KEY đặt khóa cho tokenization.

## Kiểm thử

    pytest -q