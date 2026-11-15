"""Local web interface for uploading data to the PII pipeline."""
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
from config import OUTPUT
from detection import detect, load_ner
from ingestion import ingest_file

app = FastAPI(title="PII Anonymization Pipeline", version="1.0.0")
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PII Pipeline</title><style>
body{font:16px system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#172033;background:#f5f7fb}
main{background:white;padding:28px;border-radius:14px;box-shadow:0 4px 24px #18263b12}h1{margin-top:0;color:#173b69}
form{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:18px;background:#f2f6fc;border-radius:10px}
button{background:#1769aa;color:white;border:0;padding:11px 18px;border-radius:7px;cursor:pointer;font-weight:600}
button:disabled{opacity:.55;cursor:wait}.muted{color:#596579}.error{color:#b42318}.success{color:#137333}
li{margin:7px 0}a{color:#1769aa}table{border-collapse:collapse;width:100%;margin:14px 0}td,th{border:1px solid #dce2eb;padding:8px;text-align:left}
</style></head><body><main><h1>PII Detection &amp; Anonymization</h1>
<p class="muted">Tải tệp CSV, TSV hoặc Excel lên để chạy pipeline Bronze → Silver → Detection → Anonymization.</p>
<form id="upload-form"><input type="file" name="file" accept=".csv,.tsv,.xlsx,.xls" required>
<label><input type="checkbox" name="use_ner" value="true"> Thử dùng NER (nếu model đã cài)</label><button id="submit">Xử lý tệp</button></form>
<p class="muted">Tối đa 50 MB mỗi tệp. Kết quả phát hiện phụ thuộc dữ liệu và cấu hình; hãy rà soát trước khi sử dụng tệp đã xử lý.</p><section id="result" aria-live="polite"></section>
</main><script>
const form=document.querySelector('#upload-form'), result=document.querySelector('#result'), button=document.querySelector('#submit');
function el(tag,text,cls){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x;}
form.addEventListener('submit',async e=>{e.preventDefault();result.replaceChildren(el('p','Đang xử lý...'));button.disabled=true;
try{const fd=new FormData(form);const r=await fetch('/api/process',{method:'POST',body:fd});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Không xử lý được tệp');
const box=el('div');box.append(el('h2','Kết quả xử lý','success'));box.append(el('p',`Batch ${d.batch_id}: ${d.rows_in} dòng đầu vào, ${d.rows_out} dòng sau làm sạch; đã loại ${d.duplicates_removed} dòng trùng.`));
box.append(el('h3','Các trường được phát hiện'));const table=el('table'),thead=el('tr');['Cột','Loại PII','Phương pháp','Điểm'].forEach(v=>thead.append(el('th',v)));table.append(thead);
d.findings.forEach(f=>{const tr=el('tr');[f.column,f.type,f.method,String(f.score)].forEach(v=>tr.append(el('td',v)));table.append(tr)});if(!d.findings.length)box.append(el('p','Không phát hiện trường PII theo cấu hình hiện tại.'));else box.append(table);
box.append(el('h3','Tải kết quả'));const ul=el('ul');Object.entries(d.downloads).forEach(([label,url])=>{const li=el('li'),a=el('a',label);a.href=url;li.append(a);ul.append(li)});box.append(ul);
box.append(el('p','Lưu ý: kết quả tự động có thể bỏ sót dữ liệu nhạy cảm. Hãy kiểm tra cột và tệp đầu ra trước khi chia sẻ.','muted'));result.replaceChildren(box)
}catch(err){result.replaceChildren(el('p',err.message,'error'))}finally{button.disabled=false}});
</script></body></html>"""


def _run_pipeline(path: Path, use_ner: bool) -> dict:
    raw, batch_dir = ingest_file(str(path))
    silver, cleaning_report = clean_and_standardize(raw)
    silver_dir = save_silver(silver, cleaning_report, batch_dir)
    findings = detect(silver, load_ner() if use_ner else None)
    key = os.environ.get("PII_TOKEN_KEY", "").encode() or secrets.token_bytes(32)
    anonymized, applied = anonymize(silver, findings, key)

    output_dir = OUTPUT / batch_dir.name
    output_dir.mkdir(parents=True, exist_ok=False)
    anonymized.to_csv(output_dir / "anonymized.csv", index=False, encoding="utf-8-sig")
    report = {"batch_id": batch_dir.name, "detection": findings,
              "anonymization": applied, "cleaning": cleaning_report}
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "detection.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "anonymization.json").write_text(
        json.dumps(applied, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "batch_id": batch_dir.name,
        "rows_in": len(raw),
        "rows_out": len(anonymized),
        "duplicates_removed": cleaning_report["duplicates_removed"],
        "findings": findings,
        "downloads": {
            "CSV đã ẩn danh": f"/download/{batch_dir.name}/anonymized.csv",
            "Báo cáo tổng hợp JSON": f"/download/{batch_dir.name}/report.json",
            "Kết quả phát hiện JSON": f"/download/{batch_dir.name}/detection.json",
            "Kết quả ẩn danh hóa JSON": f"/download/{batch_dir.name}/anonymization.json",
        },
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.post("/api/process")
async def process_upload(file: UploadFile = File(...), use_ner: bool = Form(False)):
    filename = (file.filename or "upload").replace("\\", "/").split("/")[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
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
            return await run_in_threadpool(_run_pipeline, source, use_ner)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Pipeline gặp lỗi: {type(exc).__name__}: {exc}") from exc
    finally:
        await file.close()


@app.get("/download/{batch_id}/{name}")
def download(batch_id: str, name: str):
    if not re.fullmatch(r"\d{8}_\d{6}(?:_\d{6})?", batch_id):
        raise HTTPException(404, "Không tìm thấy kết quả.")
    allowed = {"anonymized.csv", "report.json", "detection.json", "anonymization.json"}
    if name not in allowed:
        raise HTTPException(404, "Không tìm thấy kết quả.")
    path = OUTPUT / batch_id / name
    if not path.is_file():
        raise HTTPException(404, "Không tìm thấy kết quả.")
    return FileResponse(path, filename=name)
