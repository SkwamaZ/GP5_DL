import json
from datetime import date

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import ROOT, load_config


def main():
    cfg = load_config()
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

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
            cat["category"].value_counts().plot(kind="bar", rot=0)
            plt.ylabel("товаров")
            plt.tight_layout()
            plt.savefig(reports / "catalog_by_category.png", dpi=120)
            plt.close()

    if reviews_path.exists():
        rev = pd.read_parquet(reviews_path)
        if not rev.empty:
            dist = rev["mark"].value_counts().sort_index()
            by_mark = {}
            for k, v in dist.items():
                by_mark[int(k)] = int(v)
            by_cat = {}
            for k, v in rev["category"].value_counts().items():
                by_cat[k] = int(v)
            snap["reviews"] = {
                "n_reviews": int(len(rev)),
                "by_mark": by_mark,
                "by_category": by_cat,
            }
            dist.plot(kind="bar", rot=0)
            plt.xlabel("оценка")
            plt.ylabel("отзывов")
            plt.tight_layout()
            plt.savefig(reports / "reviews_by_mark.png", dpi=120)
            plt.close()

    (reports / "snapshot.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2)
    )

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
    (reports / "snapshot.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
