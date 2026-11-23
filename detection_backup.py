import re

import pandas as pd

from config import NER_MIN_UNIQUE, NER_THRESHOLD, REGEX_FULL_THRESHOLD, REGEX_INLINE_THRESHOLD, SAMPLE_SIZE

PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?84|0)[35789](?:[\s.\-]?\d){8}(?!\d)"),
    "CCCD": re.compile(r"(?<!\d)0\d{11}(?!\d)"),
    "CMND": re.compile(r"(?<!\d)\d{9}(?!\d)"),
}

NAME_HINTS = {
    "EMAIL": r"(^|_)(email|e_mail|mail)($|_)",
    "PHONE": r"(^|_)(phone|mobile|telephone|sdt|so_dien_thoai)($|_)",
    "CCCD": r"(^|_)(cccd|citizen_id|national_id|identity_number)($|_)",
    "CMND": r"(^|_)(cmnd|id_number)($|_)",
    "PERSON": r"(^|_)(ho_ten|ho_va_ten|hoten|full_?name|name|surname|last_?name|first_?name|ten_khach_hang|ten_kh|ten)($|_)",
    "ID": r"(^|_)(customer_?id|patient_id|employee_?id|user_?id|ma_kh|ma_bn|ma_nv)($|_)",
    "ADDRESS": r"(^|_)(dia_chi|address|addr|noi_o|que_quan|thuong_tru)($|_)",
    "DOB": r"(ngay_sinh|dob|birth)",
    "AGE": r"(^|_)(tuoi|age)($|_)",
    "SALARY": r"(luong|salary|income|thu_nhap)",
}

NER_TO_TYPE = {"PERSON": "PERSON", "LOCATION": "ADDRESS"}


class NerDetector:
    def __init__(self, model: str = "xx_ent_wiki_sm"):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        conf = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "xx", "model_name": model}],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": {
                    "PER": "PERSON",
                    "LOC": "LOCATION",
                    "ORG": "ORGANIZATION",
                },
                "labels_to_ignore": ["MISC"],
            },
        }
        nlp = NlpEngineProvider(nlp_configuration=conf).create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["xx"])

    def entities(self, text: str) -> set[str]:
        res = self.analyzer.analyze(text=text, language="xx", entities=["PERSON", "LOCATION"])
        return {r.entity_type for r in res}


def load_ner():
    try:
        return NerDetector()
    except Exception as e:
        print(f"[detection] NER không khả dụng ({type(e).__name__}); chỉ dùng Regex + gợi ý tên cột")
        return None


def _regex_scores(values: pd.Series):
    cands = []
    for label, pat in PATTERNS.items():
        full = values.map(lambda v: bool(pat.fullmatch(v))).mean()
        any_ = values.map(lambda v: bool(pat.search(v))).mean()
        if full >= REGEX_FULL_THRESHOLD:
            cands.append((label, "column", full))
        elif any_ >= REGEX_INLINE_THRESHOLD:
            cands.append((label, "inline", any_))
    return cands


def _hint(col: str):
    for label, pat in NAME_HINTS.items():
        if re.search(pat, col):
            return label
    return None


def detect(df: pd.DataFrame, ner: NerDetector | None = None) -> list[dict]:
    findings = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        sample = s.sample(min(len(s), SAMPLE_SIZE), random_state=0)
        is_text = pd.api.types.is_string_dtype(s) or s.dtype == object
        hits = []

        if is_text:
            cands = _regex_scores(sample.astype(str))
            full = [c for c in cands if c[1] == "column"]
            use = [max(full, key=lambda c: c[2])] if full else cands
            hits = [{"type": c[0], "scope": c[1], "score": round(float(c[2]), 3), "method": "regex"} for c in use]

        if not hits:
            label = _hint(col)
            if label:
                hits = [{"type": label, "scope": "column", "score": 1.0, "method": "column_name"}]

        if not hits and is_text and ner is not None and s.nunique() / len(s) >= NER_MIN_UNIQUE:
            vals = sample.astype(str).head(50)
            counts = {}
            for v in vals:
                for e in ner.entities(v):
                    counts[e] = counts.get(e, 0) + 1
            if counts:
                ent, c = max(counts.items(), key=lambda kv: kv[1])
                if c / len(vals) >= NER_THRESHOLD:
                    hits = [{
                        "type": NER_TO_TYPE[ent],
                        "scope": "column",
                        "score": round(c / len(vals), 3),
                        "method": "ner",
                    }]

        for h in hits:
            findings.append({"column": col, **h})
    return findings