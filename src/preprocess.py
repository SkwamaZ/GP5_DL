import re

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
ALLOWED_RE = re.compile(r"[^а-яa-z0-9\s.,!?():;%+\-/]")
PUNCT_REPEAT_RE = re.compile(r"([.,!?:;])\1+")
SPACE_RE = re.compile(r"\s+")


def normalize_text(s):
    s = str(s).lower().replace("ё", "е")
    s = URL_RE.sub(" ", s)
    s = ALLOWED_RE.sub(" ", s)
    s = PUNCT_REPEAT_RE.sub(r"\1", s)
    s = SPACE_RE.sub(" ", s)
    return s.strip()


def build_tabular_preprocessor(cfg):
    t = cfg["tabular"]
    log_scaled = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num_log", log_scaled, t["log_features"]),
            ("num", StandardScaler(), t["plain_num_features"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), t["onehot_features"]),
            (
                "brand",
                TargetEncoder(
                    cv=KFold(n_splits=5, shuffle=True, random_state=cfg["random_seed"])
                ),
                t["target_enc_features"],
            ),
        ]
    )
