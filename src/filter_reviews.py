import io
import json
import random

import pandas as pd
import zstandard as zstd

from src.utils import ROOT, load_config

VALUATION_KEYS = ("productValuation", "valuation", "mark", "rating")


def get_valuation(rec):
    for k in VALUATION_KEYS:
        v = rec.get(k)
        if v is not None:
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= v <= 5:
                return v
    return None


def iter_jsonl_zst(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def main():
    cfg = load_config()
    rcfg = cfg["reviews"]
    cap = rcfg["per_class_cap"]
    seed = cfg["random_seed"]

    catalog_path = ROOT / cfg["collection"]["catalog_interim"]
    if not catalog_path.exists():
        return
    catalog = pd.read_parquet(catalog_path)
    if catalog.empty or "nmId" not in catalog.columns:
        return

    nm_to_cat = {}
    for nm, cat in zip(catalog["nmId"], catalog["category"]):
        nm_to_cat[int(nm)] = cat

    feed_dir = ROOT / cfg["data"]["wb_feedbacks_dir"]
    files = sorted(feed_dir.glob("**/*.zst"))
    if not files:
        return

    buckets = {1: [], 2: [], 3: [], 4: [], 5: []}
    for path in files:
        for rec in iter_jsonl_zst(path):
            nm = rec.get("nmId")
            if nm is None:
                continue
            try:
                nm = int(nm)
            except (TypeError, ValueError):
                continue
            if nm not in nm_to_cat:
                continue
            v = get_valuation(rec)
            if v is None:
                continue
            if len(buckets[v]) >= cap:
                continue
            buckets[v].append({
                "nmId": nm,
                "category": nm_to_cat[nm],
                "mark": v,
                "color": rec.get("color"),
                "text": rec.get("text"),
                "answer": rec.get("answer"),
            })

    rows = []
    for v in range(1, 6):
        rows += buckets[v]
    random.seed(seed)
    random.shuffle(rows)
    df = pd.DataFrame(rows)

    out = ROOT / rcfg["reviews_interim"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_json(ROOT / rcfg["reviews_interim_jsonl"], orient="records", lines=True, force_ascii=False)

    print("итого отзывов:", len(df))


if __name__ == "__main__":
    main()
