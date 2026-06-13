import io
import json
import random
import re

import pandas as pd
import zstandard as zstd

from src.utils import ROOT, load_config

NMID_RE = re.compile(rb'"nmId"\s*:\s*(\d+)')
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


def iter_lines_zst(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buffered = io.BufferedReader(reader, buffer_size=1 << 20)
            for line in buffered:
                yield line


def main():
    cfg = load_config()
    rcfg = cfg["reviews"]
    cap = rcfg["per_class_cap"]
    rng = random.Random(cfg["random_seed"])

    catalog = pd.read_parquet(ROOT / cfg["collection"]["catalog_interim"])
    nm_to_cat = {}
    for nm, cat in zip(catalog["nmId"], catalog["category"]):
        nm_to_cat[int(nm)] = cat
    nm_set = set(nm_to_cat)

    feed_dir = ROOT / cfg["data"]["wb_feedbacks_dir"]
    files = sorted(feed_dir.glob("**/*.zst"))

    buckets = {v: [] for v in range(1, 6)}
    seen = {v: 0 for v in range(1, 6)}
    matched = 0
    for path in files:
        for line in iter_lines_zst(path):
            m = NMID_RE.search(line)
            if m is None or int(m.group(1)) not in nm_set:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            nm = rec.get("nmId")
            if nm is None or int(nm) not in nm_to_cat:
                continue
            matched += 1
            v = get_valuation(rec)
            text = (rec.get("text") or "").strip()
            if v is None or not text:
                continue
            row = {
                "nmId": int(nm),
                "category": nm_to_cat[int(nm)],
                "mark": v,
                "color": rec.get("color"),
                "text": text,
                "answer": rec.get("answer"),
            }
            seen[v] += 1
            if len(buckets[v]) < cap:
                buckets[v].append(row)
            else:
                j = rng.randrange(seen[v])
                if j < cap:
                    buckets[v][j] = row

    rows = []
    for v in range(1, 6):
        rows += buckets[v]
    rng.shuffle(rows)
    df = pd.DataFrame(rows)

    out = ROOT / rcfg["reviews_interim"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_json(
        ROOT / rcfg["reviews_interim_jsonl"],
        orient="records",
        lines=True,
        force_ascii=False,
    )

    meta = {
        "matched_by_nmid": matched,
        "with_text_by_mark": seen,
        "sampled_by_mark": {v: len(buckets[v]) for v in range(1, 6)},
        "per_class_cap": cap,
        "sampling": "reservoir per class, seed " + str(cfg["random_seed"]),
        "empty_text_dropped": True,
    }
    (out.with_name("reviews_meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )

    print("итого отзывов:", len(df))


if __name__ == "__main__":
    main()
