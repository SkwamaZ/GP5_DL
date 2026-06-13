import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ROOT, load_config

KEEP_COLS = [
    "nmId",
    "name",
    "brand",
    "category",
    "price",
    "rating",
    "n_reviews",
    "n_photos",
    "n_colors",
    "title_len",
]


def clean_catalog(df):
    df = df.drop_duplicates(subset=["nmId"]).copy()
    df = df[(df["rating"] > 0) & (df["n_reviews"] > 0)]
    df = df[df["price"] > 0]
    brand = df["brand"].fillna("").str.strip()
    df["brand"] = brand.where(brand != "", "unknown")
    df["title_len"] = df["name"].fillna("").str.len()
    return df[KEEP_COLS].reset_index(drop=True)


def main():
    cfg = load_config()
    t = cfg["tabular"]
    seed = cfg["random_seed"]

    df = pd.read_parquet(ROOT / cfg["collection"]["catalog_interim"])
    df = clean_catalog(df)

    train_val, test = train_test_split(df, test_size=t["test_size"], random_state=seed)
    val_frac = t["val_size"] / (1 - t["test_size"])
    train, val = train_test_split(train_val, test_size=val_frac, random_state=seed)

    out_dir = ROOT / cfg["paths"]["data_processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "tabular_clean.parquet", index=False)
    train.to_parquet(out_dir / "tabular_train.parquet", index=False)
    val.to_parquet(out_dir / "tabular_val.parquet", index=False)
    test.to_parquet(out_dir / "tabular_test.parquet", index=False)

    print(
        "табличный датасет:",
        len(df),
        "| train:",
        len(train),
        "| val:",
        len(val),
        "| test:",
        len(test),
    )


if __name__ == "__main__":
    main()
