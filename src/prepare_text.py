import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.aspects import add_aspect_columns
from src.preprocess import normalize_text
from src.utils import ROOT, load_config


def clean_reviews(df):
    df = df.drop_duplicates(subset=["nmId", "text", "mark"]).copy()
    df["text_norm"] = df["text"].map(normalize_text)
    df = df[df["text_norm"].str.len() > 0]
    df["n_words"] = df["text_norm"].str.split().str.len()
    df = add_aspect_columns(df)
    return df.reset_index(drop=True)


def main():
    cfg = load_config()
    t = cfg["text"]
    seed = cfg["random_seed"]

    df = pd.read_parquet(ROOT / cfg["reviews"]["reviews_interim"])
    df = clean_reviews(df)

    train_val, test = train_test_split(
        df, test_size=t["test_size"], stratify=df["mark"], random_state=seed
    )
    val_frac = t["val_size"] / (1 - t["test_size"])
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=train_val["mark"], random_state=seed
    )

    out_dir = ROOT / cfg["paths"]["data_processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "text_clean.parquet", index=False)
    train.to_parquet(out_dir / "text_train.parquet", index=False)
    val.to_parquet(out_dir / "text_val.parquet", index=False)
    test.to_parquet(out_dir / "text_test.parquet", index=False)

    counts = train["mark"].value_counts()
    weights = {
        int(c): round(len(train) / (t["num_classes"] * int(counts[c])), 4)
        for c in sorted(counts.index)
    }
    payload = {
        "strategy": "веса классов в лоссе (balanced), считаны по train",
        "train_counts": {int(c): int(counts[c]) for c in sorted(counts.index)},
        "weights": weights,
    }
    (out_dir / "class_weights.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )

    print(
        "текстовый датасет:",
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
