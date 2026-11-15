"""FastAPI upload interface for the PII processing pipeline."""
import json
import os
import re
import secrets
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool

from anonymization import anonymize
from cleaning import clean_and_standardize, save_silver
from config import GOLD, OUTPUT
from detection import detect, load_ner
from evaluation import evaluate, save_gold
from ingestion import ingest_file
from ml_experiment import run_privacy_utility_experiment

app = FastAPI(title="PII Anonymization Pipeline", version="1.1.0")
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PII Pipeline</title><style>
body{font:16px system-ui,sans-serif;max-width:950px;margin:40px auto;padding:0 20px;color:#172033;background:#f5f7fb}
main{background:white;padding:28px;border-radius:14px;box-shadow:0 4px 24px #18263b12}h1{margin-top:0;color:#173b69}
form{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:18px;background:#f2f6fc;border-radius:10px}
input[type=text]{display:block;width:95%;margin-top:6px;padding:10px;border:1px solid #ccd5e2;border-radius:6px}button{background:#1769aa;color:white;border:0;padding:11px 18px;border-radius:7px;cursor:pointer;font-weight:600}
button:disabled{opacity:.55;cursor:wait}.muted{color:#596579}.error{color:#b42318}.success{color:#137333}.full{grid-column:1/-1}
li{margin:7px 0}a{color:#1769aa}table{border-collapse:collapse;width:100%;margin:14px 0}td,th{border:1px solid #dce2eb;padding:8px;text-align:left}
</style></head><body><main><h1>Phát hiện và ẩn danh hóa PII</h1>
<p class="muted">Tải CSV, TSV hoặc Excel để chạy Bronze → Silver → Detection → Anonymization → Gold/Evaluation.</p>
<form id="upload-form"><input class="full" type="file" name="file" accept=".csv,.tsv,.xlsx,.xls" required>
<label><input type="checkbox" name="use_ner" value="true"> Thử dùng NER (nếu model đã cài)</label><span></span>
<label>Quasi-identifiers (tùy chọn, cách nhau bằng dấu phẩy)<input type="text" name="quasi_identifiers" placeholder="age, gender, address"></label>
<label>Sensitive attributes (tùy chọn)<input type="text" name="sensitive_attributes" placeholder="diagnosis, salary"></label>
<label class="full">Nhãn phân loại để thử nghiệm privacy–utility (tùy chọn)<input type="text" name="utility_target" placeholder="medical_condition"></label>
<button id="submit" class="full">Xử lý tệp</button></form>
<p class="muted">Tối đa 50 MB mỗi tệp. Kết quả đánh giá là chỉ báo heuristic, không thay thế đánh giá bảo mật hoặc kiểm định khoa học.</p><section id="result" aria-live="polite"></section>
</main><script>
const form=document.querySelector('#upload-form'), result=document.querySelector('#result'), button=document.querySelector('#submit');
function el(tag,text,cls){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x;}
form.addEventListener('submit',async e=>{e.preventDefault();result.replaceChildren(el('p','Đang xử lý...'));button.disabled=true;
try{const fd=new FormData(form);const r=await fetch('/api/process',{method:'POST',body:fd});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Không xử lý được tệp');
const box=el('div');box.append(el('h2','Kết quả xử lý','success'));box.append(el('p',`Batch ${d.batch_id}: ${d.rows_in} dòng đầu vào, ${d.rows_out} dòng sau làm sạch; đã loại ${d.duplicates_removed} dòng trùng.`));
const before=d.evaluation.privacy_before, after=d.evaluation.privacy_after;
box.append(el('p',`Privacy risk heuristic: ${before.risk_score ?? 'N/A'} (${before.risk_label}) → ${after.risk_score ?? 'N/A'} (${after.risk_label}); k-anonymity: ${before.k_anonymity ?? 'N/A'} → ${after.k_anonymity ?? 'N/A'}.`));
box.append(el('p',`Quasi-identifiers dùng để tính: ${before.quasi_identifier_columns.join(', ') || 'chưa xác định'}.`,'muted'));
if(d.ml_privacy_utility){box.append(el('h3',`Privacy–utility: ${d.ml_privacy_utility.target_column}`));const mt=el('table'),mh=el('tr');['Mô hình','F1 trước','F1 sau','ROC-AUC trước','ROC-AUC sau'].forEach(v=>mh.append(el('th',v)));mt.append(mh);Object.entries(d.ml_privacy_utility.models).forEach(([name,v])=>{const tr=el('tr');const a=v.before||{},b=v.after||{};[name,a.f1_weighted??v.reason??'—',b.f1_weighted??'—',a.roc_auc??'—',b.roc_auc??'—'].forEach(x=>tr.append(el('td',String(x))));mt.append(tr)});box.append(mt)}
box.append(el('h3','Các trường được phát hiện'));const table=el('table'),thead=el('tr');['Cột','Loại PII','Phương pháp','Điểm'].forEach(v=>thead.append(el('th',v)));table.append(thead);
d.findings.forEach(f=>{const tr=el('tr');[f.column,f.type,f.method,String(f.score)].forEach(v=>tr.append(el('td',v)));table.append(tr)});if(!d.findings.length)box.append(el('p','Không phát hiện trường PII theo cấu hình hiện tại.'));else box.append(table);
box.append(el('h3','Tải kết quả'));const ul=el('ul');Object.entries(d.downloads).forEach(([label,url])=>{const li=el('li'),a=el('a',label);a.href=url;li.append(a);ul.append(li)});box.append(ul);
box.append(el('p','Risk score là heuristic minh bạch, chưa phải mô hình ML đã huấn luyện. Kết quả tự động có thể bỏ sót dữ liệu nhạy cảm; hãy rà soát trước khi chia sẻ.','muted'));result.replaceChildren(box)
}catch(err){result.replaceChildren(el('p',err.message,'error'))}finally{button.disabled=false}});
</script></body></html>"""


def _split_columns(value: str) -> list[str] | None:
    return [c.strip() for c in value.split(",") if c.strip()] or None


def _run_pipeline(path: Path, use_ner: bool, quasi_identifiers: str = "",
                  sensitive_attributes: str = "", utility_target: str = "") -> dict:
    raw, batch_dir = ingest_file(str(path))
    silver, cleaning_report = clean_and_standardize(raw)
    save_silver(silver, cleaning_report, batch_dir)
    findings = detect(silver, load_ner() if use_ner else None)
    key = os.environ.get("PII_TOKEN_KEY", "").encode() or secrets.token_bytes(32)
    anonymized, applied = anonymize(silver, findings, key)
    transformed = {x["column"] for x in applied}
    remaining_findings = [f for f in detect(anonymized, None) if f["column"] not in transformed]
    evaluation = evaluate(silver, anonymized, findings, remaining_findings,
                          _split_columns(quasi_identifiers), _split_columns(sensitive_attributes))
    ml_report = run_privacy_utility_experiment(silver, anonymized, utility_target) if utility_target.strip() else None
    if ml_report is not None:
        evaluation["ml_privacy_utility"] = ml_report
    save_gold(anonymized, batch_dir.name, evaluation)

    output_dir = OUTPUT / batch_dir.name
    output_dir.mkdir(parents=True, exist_ok=False)
    anonymized.to_csv(output_dir / "anonymized.csv", index=False, encoding="utf-8-sig")
    artifacts = {"batch_id": batch_dir.name, "detection": findings,
                 "anonymization": applied, "cleaning": cleaning_report, "evaluation": evaluation}
    (output_dir / "report.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "detection.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "anonymization.json").write_text(json.dumps(applied, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if ml_report is not None:
        (output_dir / "ml_experiment.json").write_text(json.dumps(ml_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "batch_id": batch_dir.name, "rows_in": len(raw), "rows_out": len(anonymized),
        "duplicates_removed": cleaning_report["duplicates_removed"], "findings": findings,
        "evaluation": evaluation, "ml_privacy_utility": ml_report,
        "downloads": {
            "CSV đã ẩn danh": f"/download/{batch_dir.name}/anonymized.csv",
            "Báo cáo tổng hợp JSON": f"/download/{batch_dir.name}/report.json",
            "Kết quả phát hiện JSON": f"/download/{batch_dir.name}/detection.json",
            "Kết quả ẩn danh hóa JSON": f"/download/{batch_dir.name}/anonymization.json",
            "Báo cáo Evaluation JSON": f"/download/{batch_dir.name}/evaluation.json",
            **({"Báo cáo ML privacy–utility JSON": f"/download/{batch_dir.name}/ml_experiment.json"} if ml_report is not None else {}),
            "Gold Layer Parquet": f"/download/{batch_dir.name}/gold.parquet",
        },
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.post("/api/process")
async def process_upload(file: UploadFile = File(...), use_ner: bool = Form(False),
                        quasi_identifiers: str = Form(""), sensitive_attributes: str = Form(""),
                        utility_target: str = Form("")):
    filename = (file.filename or "upload").replace("\\", "/").split("/")[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(415, "Chỉ nhận CSV, TSV, XLSX hoặc XLS.")
    try:
        with tempfile.TemporaryDirectory(prefix="pii_upload_") as temp_dir:
            source = Path(temp_dir) / ("upload" + suffix)
            size = 0
            with source.open("wb") as dst:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Tệp vượt quá giới hạn 50 MB.")
                    dst.write(chunk)
            if size == 0:
                raise HTTPException(400, "Tệp tải lên đang trống.")
            return await run_in_threadpool(_run_pipeline, source, use_ner, quasi_identifiers, sensitive_attributes, utility_target)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Pipeline gặp lỗi: {type(exc).__name__}: {exc}") from exc
    finally:
        await file.close()


@app.get("/download/{batch_id}/{name}")
def download(batch_id: str, name: str):
    if not re.fullmatch(r"\d{8}_\d{6}(?:_\d{6})?", batch_id):
        raise HTTPException(404, "Không tìm thấy kết quả.")
    allowed = {"anonymized.csv", "report.json", "detection.json", "anonymization.json", "evaluation.json", "ml_experiment.json", "gold.parquet"}
    if name not in allowed:
        raise HTTPException(404, "Không tìm thấy kết quả.")
    root = GOLD if name == "gold.parquet" else OUTPUT
    path = root / batch_id / ("data.parquet" if name == "gold.parquet" else name)
    if not path.is_file():
        raise HTTPException(404, "Không tìm thấy kết quả.")
    return FileResponse(path, filename=name)
