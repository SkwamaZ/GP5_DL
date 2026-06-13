import json
from datetime import date

import pandas as pd

from src.utils import ROOT, load_config


def main():
    cfg = load_config()
    docs = ROOT / cfg["paths"]["docs"]
    docs.mkdir(parents=True, exist_ok=True)

    catalog_path = ROOT / cfg["collection"]["catalog_interim"]
    reviews_path = ROOT / cfg["reviews"]["reviews_interim"]

    snap = {
        "collected_at": date.today().isoformat(),
        "catalog_source": cfg["collection"]["search_host"]
        + " ("
        + cfg["collection"]["search_version"]
        + ")",
        "reviews_source": "wb-feedbacks, фильтр по nmId каталога",
        "link_key": "nmId",
        "categories": list(cfg["categories_meta"].keys()),
    }

    meta_path = catalog_path.with_name("catalog_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        snap["collection_complete"] = meta.get("complete")
        snap["collected_at"] = meta.get("collected_at", snap["collected_at"])

    if catalog_path.exists():
        cat = pd.read_parquet(catalog_path)
        if not cat.empty:
            rated = cat[cat["rating"] > 0]
            rating_mean = None
            if not rated.empty:
                rating_mean = round(float(rated["rating"].mean()), 3)
            price_median = None
            if cat["price"].notna().any():
                price_median = round(float(cat["price"].median()), 2)
            by_cat = {}
            for k, v in cat["category"].value_counts().items():
                by_cat[k] = int(v)
            snap["catalog"] = {
                "n_products": int(len(cat)),
                "by_category": by_cat,
                "rating_mean": rating_mean,
                "price_median": price_median,
            }

    if reviews_path.exists():
        rev = pd.read_parquet(reviews_path)
        if not rev.empty:
            by_mark = {}
            for k, v in rev["mark"].value_counts().sort_index().items():
                by_mark[int(k)] = int(v)
            by_cat = {}
            for k, v in rev["category"].value_counts().items():
                by_cat[k] = int(v)
            snap["reviews"] = {
                "n_reviews": int(len(rev)),
                "by_mark": by_mark,
                "by_category": by_cat,
            }
        rmeta_path = reviews_path.with_name("reviews_meta.json")
        if rmeta_path.exists():
            rmeta = json.loads(rmeta_path.read_text())
            snap["reviews_sampling"] = rmeta

    (docs / "snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2))

    lines = [
        "# Снапшот сбора данных",
        "",
        "- Дата: " + snap["collected_at"],
        "- Каталог: " + snap["catalog_source"],
        "- Отзывы: " + snap["reviews_source"],
        "- Ключ связи: " + snap["link_key"],
    ]
    if "collection_complete" in snap:
        if snap["collection_complete"]:
            lines.append("- Каталог собран полностью: да")
        else:
            lines.append("- Каталог собран полностью: нет")
    lines.append("")
    if "catalog" in snap:
        c = snap["catalog"]
        lines.append("## Каталог: " + str(c["n_products"]))
        lines.append("| категория | товаров |")
        lines.append("|---|---|")
        for k, v in c["by_category"].items():
            lines.append("| " + str(k) + " | " + str(v) + " |")
        lines.append("")
        lines.append("- Средний рейтинг: " + str(c["rating_mean"]))
        lines.append("- Медианная цена: " + str(c["price_median"]))
        lines.append("")
    if "reviews" in snap:
        r = snap["reviews"]
        lines.append("## Отзывы: " + str(r["n_reviews"]))
        lines.append("| оценка | отзывов |")
        lines.append("|---|---|")
        for k in sorted(r["by_mark"]):
            lines.append("| " + str(k) + " | " + str(r["by_mark"][k]) + " |")
        lines.append("")
        lines.append("| категория | отзывов |")
        lines.append("|---|---|")
        for k, v in r["by_category"].items():
            lines.append("| " + str(k) + " | " + str(v) + " |")
        lines.append("")
    if "reviews_sampling" in snap:
        s = snap["reviews_sampling"]
        lines.append("## Выборка отзывов")
        lines.append("- Совпало по nmId каталога: " + str(s["matched_by_nmid"]))
        lines.append("- Лимит на класс: " + str(s["per_class_cap"]))
        lines.append("- Метод: " + s["sampling"])
        lines.append("- Пустые тексты отброшены: да")
    (docs / "snapshot.md").write_text("\n".join(lines) + "\n")

    print(
        "снапшот: "
        + str(snap.get("catalog", {}).get("n_products", 0))
        + " товаров, "
        + str(snap.get("reviews", {}).get("n_reviews", 0))
        + " отзывов"
    )


if __name__ == "__main__":
    main()
