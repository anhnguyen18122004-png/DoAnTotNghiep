from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
OUTPUT = DATA / "output"

for p in (BRONZE, SILVER, OUTPUT):
    p.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200
REGEX_FULL_THRESHOLD = 0.5
REGEX_INLINE_THRESHOLD = 0.2
NER_THRESHOLD = 0.5
NER_MIN_UNIQUE = 0.3