import hashlib
import os
from datetime import date

import pandas as pd

from src.utils import ROOT, load_config
from src.wb_client import DailyLimitReached, WBClient, load_json, save_json

CATALOG_COLUMNS = [
    "nmId",
    "name",
    "brand",
    "category",
    "subject_id",
    "price",
    "rating",
    "n_reviews",
    "n_photos",
    "colors",
    "n_colors",
    "supplier",
]

STALE_PAGES_LIMIT = 3


def get_price(p):
    for s in p.get("sizes") or []:
        price = s.get("price") or {}
        for key in ("total", "product", "basic"):
            if price.get(key):
                return round(price[key] / 100, 2)
    for key in ("salePriceU", "priceU"):
        if p.get(key):
            return round(p[key] / 100, 2)
    return None


def product_to_row(p, category):
    colors = []
    for c in p.get("colors") or []:
        if c.get("name"):
            colors.append(c["name"])

    rating = p.get("reviewRating")
    if rating is None:
        rating = p.get("rating")
    if rating is None:
        rating = 0.0

    return {
        "nmId": p.get("id"),
        "name": p.get("name"),
        "brand": p.get("brand"),
        "category": category,
        "subject_id": p.get("subjectId"),
        "price": get_price(p),
        "rating": rating,
        "n_reviews": p.get("feedbacks") or 0,
        "n_photos": p.get("pics") or 0,
        "colors": "; ".join(colors),
        "n_colors": len(colors),
        "supplier": p.get("supplier"),
    }


def collect_category(client, name, meta, target, max_pages, raw_dir_base, sorts):
    raw_dir = ROOT / raw_dir_base / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    queries = meta.get("search_queries") or [meta["search_query"]]

    rows = []
    seen = set()
    reason = "exhausted"
    for query in queries:
        for sort in sorts:
            tag = hashlib.sha1((query + "|" + sort).encode()).hexdigest()[:10]
            qdir = raw_dir / tag
            qdir.mkdir(parents=True, exist_ok=True)
            stale = 0
            for page in range(1, max_pages + 1):
                page_file = qdir / ("page_%03d.json" % page)
                products = None
                if page_file.exists():
                    products = load_json(page_file)
                if products is None:
                    try:
                        products = client.get_search_products(query, page, sort)
                    except DailyLimitReached:
                        return rows[:target], "dailylimit"
                    if products is None:
                        return rows[:target], "nodata"
                    save_json(page_file, products)
                if not products:
                    break
                new = 0
                for p in products:
                    nm = p.get("id")
                    if nm and nm not in seen:
                        seen.add(nm)
                        rows.append(product_to_row(p, name))
                        new += 1
                if len(rows) >= target:
                    return rows[:target], "target"
                if new == 0:
                    stale += 1
                else:
                    stale = 0
                if stale >= STALE_PAGES_LIMIT:
                    break

    return rows[:target], reason


def main():
    cfg = load_config()
    coll = cfg["collection"]
    target = int(os.environ.get("WB_TARGET", cfg["data"]["products_per_category"]))
    max_pages = int(os.environ.get("WB_MAXPAGES", coll["max_pages"]))
    sorts = coll.get("search_sorts") or ["popular"]
    client = WBClient(cfg)

    all_rows = []
    reasons = {}
    complete = True
    for name, meta in cfg["categories_meta"].items():
        rows, reason = collect_category(
            client, name, meta, target, max_pages, coll["raw_dir"], sorts
        )
        all_rows += rows
        reasons[name] = {"reason": reason, "n": len(rows)}
        if reason not in ("target", "empty", "exhausted"):
            complete = False
        if reason == "dailylimit":
            break

    df = pd.DataFrame(all_rows)
    if df.empty:
        df = pd.DataFrame(columns=CATALOG_COLUMNS)
    out = ROOT / coll["catalog_interim"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    meta_out = {
        "collected_at": date.today().isoformat(),
        "complete": complete,
        "n_products": int(len(df)),
        "target_per_category": target,
        "by_category": reasons,
    }
    save_json(out.with_name("catalog_meta.json"), meta_out)

    print("собрано товаров:", len(df))


if __name__ == "__main__":
    main()
