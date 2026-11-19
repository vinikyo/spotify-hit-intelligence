"""
Pipeline Completo de Ciência de Dados e Machine Learning
Projeto: Spotify Hit & Genre Intelligence (Predição de Popularidade e Classificação de Gêneros)
Autores: Vinícius Francisco Garcia Sobral (cg3034241) e Arthur de Morais Marques (cg3033856)
Dataset: Spotify Tracks Dataset (Maharshi Pandya - Kaggle)
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "dataset.csv")
FIG_DIR = os.path.join("results", "figures")
METRICS_PATH = os.path.join("results", "metrics.json")

# Estilo visual acadêmico com paleta inspirada no Spotify
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_DARK = "#191414"
SPOTIFY_GRAY = "#535353"
ACCENT_BLUE = "#2E86AB"
ACCENT_ORANGE = "#F18F01"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

AUDIO_FEATURES = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]


def load_and_clean_data(path: str) -> tuple[pd.DataFrame, dict]:
    df_raw = pd.read_csv(path)
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])

    raw_rows, raw_cols = df_raw.shape
    missing_before = int(df_raw.isna().sum().sum())
    unique_tracks_raw = int(df_raw["track_id"].nunique())
    n_genres_raw = int(df_raw["track_genre"].nunique())

    # Limpeza: remover nulos críticos e deduplicar por track_id mantendo maior popularidade
    df = df_raw.dropna(subset=["artists", "album_name", "track_name"]).copy()
    df = df.sort_values("popularity", ascending=False).drop_duplicates(
        subset=["track_id"], keep="first"
    )
    df["explicit"] = df["explicit"].astype(int)
    df["duration_min"] = df["duration_ms"] / 60000.0
    df["is_hit"] = (df["popularity"] >= 50).astype(int)

    stats = {
        "raw_rows": int(raw_rows),
        "raw_cols": int(raw_cols),
        "missing_values_raw": missing_before,
        "unique_tracks_raw": unique_tracks_raw,
        "n_genres": n_genres_raw,
        "clean_rows": int(len(df)),
        "duplicates_removed": int(raw_rows - len(df)),
        "popularity_mean": round(float(df["popularity"].mean()), 2),
        "popularity_std": round(float(df["popularity"].std()), 2),
        "popularity_median": round(float(df["popularity"].median()), 2),
        "popularity_zero_pct": round(float((df["popularity"] == 0).mean() * 100), 2),
        "hit_threshold": 50,
        "hit_count": int(df["is_hit"].sum()),
        "hit_pct": round(float(df["is_hit"].mean() * 100), 2),
        "explicit_pct": round(float(df["explicit"].mean() * 100), 2),
        "mean_duration_min": round(float(df["duration_min"].mean()), 2),
    }

    # Correlações com popularity
    num_cols = ["popularity"] + AUDIO_FEATURES
    corrs = df[num_cols].corr()["popularity"].drop("popularity").sort_values(ascending=False)
    stats["top_positive_corr"] = {k: round(float(v), 4) for k, v in corrs.head(5).items()}
    stats["top_negative_corr"] = {k: round(float(v), 4) for k, v in corrs.tail(5).items()}

    # Top gêneros por popularidade média
    genre_pop = df.groupby("track_genre")["popularity"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    stats["top_5_genres_by_popularity"] = [
        {"genre": idx, "mean_popularity": round(float(row["mean"]), 2), "count": int(row["count"])}
        for idx, row in genre_pop.head(5).iterrows()
    ]
    stats["bottom_5_genres_by_popularity"] = [
        {"genre": idx, "mean_popularity": round(float(row["mean"]), 2), "count": int(row["count"])}
        for idx, row in genre_pop.tail(5).iterrows()
    ]

    return df, stats


def generate_eda_figures(df: pd.DataFrame) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Distribuição da Popularidade
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["popularity"], bins=40, color=SPOTIFY_GREEN, edgecolor="white", alpha=0.88)
    ax.axvline(50, color="#D62828", linestyle="--", linewidth=2, label="Limiar de Hit (Popularidade ≥ 50)")
    ax.axvline(df["popularity"].mean(), color=SPOTIFY_DARK, linestyle=":", linewidth=1.8,
               label=f"Média ({df['popularity'].mean():.1f})")
    ax.set_title("Distribuição do Índice de Popularidade das Faixas (Spotify)")
    ax.set_xlabel("Popularidade (0–100)")
    ax.set_ylabel("Número de Faixas")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_distribuicao_popularidade.png"))
    plt.close()

    # 2. Mapa de Calor de Correlação
    corr_cols = [
        "popularity", "danceability", "energy", "loudness",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "duration_ms", "explicit"
    ]
    corr_matrix = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Matriz de Correlação de Pearson — Atributos de Áudio e Popularidade")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_correlacao_atributos_audio.png"))
    plt.close()

    # 3. Top 15 Gêneros por Popularidade Média
    top_genres = (
        df.groupby("track_genre")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.barh(top_genres.index, top_genres.values, color=SPOTIFY_GREEN, edgecolor=SPOTIFY_DARK, alpha=0.9)
    for bar, val in zip(bars, top_genres.values):
        ax.text(val + 0.6, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=8.5)
    ax.set_xlim(0, top_genres.max() + 8)
    ax.set_title("Top 15 Gêneros Musicais por Popularidade Média")
    ax.set_xlabel("Popularidade Média (0–100)")
    ax.set_ylabel("Gênero Musical")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_top_generos_popularidade.png"))
    plt.close()


def train_regression_models(df: pd.DataFrame) -> dict:
    print("Treinando modelos de Regressão (Target: popularity)...")
    feature_cols = AUDIO_FEATURES + ["track_genre"]
    X = df[feature_cols].copy()
    y = df["popularity"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pre_onehot = ColumnTransformer([
        ("num", StandardScaler(), AUDIO_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["track_genre"]),
    ])

    pre_ordinal = ColumnTransformer([
        ("num", "passthrough", AUDIO_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["track_genre"]),
    ])

    models = {
        "Ridge Regression": Pipeline([
            ("prep", pre_onehot),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ]),
        "Random Forest Regressor": Pipeline([
            ("prep", pre_ordinal),
            ("model", RandomForestRegressor(
                n_estimators=120, max_depth=18, min_samples_leaf=4,
                n_jobs=-1, random_state=RANDOM_STATE
            )),
        ]),
        "HistGradientBoosting Regressor": Pipeline([
            ("prep", pre_ordinal),
            ("model", HistGradientBoostingRegressor(
                max_iter=250, max_depth=10, learning_rate=0.07,
                categorical_features=[len(AUDIO_FEATURES)],
                random_state=RANDOM_STATE
            )),
        ]),
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = np.clip(pipe.predict(X_test), 0, 100)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        results[name] = {
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
            "R2": round(r2, 4),
        }
        print(f"  {name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.4f}")

    return results


def train_hit_classification_models(df: pd.DataFrame) -> dict:
    print("Treinando modelos de Classificação Binária (Target: is_hit, popularity >= 50)...")
    feature_cols = AUDIO_FEATURES + ["track_genre"]
    X = df[feature_cols].copy()
    y = df["is_hit"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pre_onehot = ColumnTransformer([
        ("num", StandardScaler(), AUDIO_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["track_genre"]),
    ])

    pre_ordinal = ColumnTransformer([
        ("num", "passthrough", AUDIO_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["track_genre"]),
    ])

    models = {
        "Regressão Logística": Pipeline([
            ("prep", pre_onehot),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("prep", pre_ordinal),
            ("model", RandomForestClassifier(
                n_estimators=150, max_depth=18, min_samples_leaf=3,
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", pre_ordinal),
            ("model", HistGradientBoostingClassifier(
                max_iter=250, max_depth=10, learning_rate=0.07,
                class_weight="balanced",
                categorical_features=[len(AUDIO_FEATURES)],
                random_state=RANDOM_STATE
            )),
        ]),
    }

    results = {}
    best_auc = -1.0
    best_name = ""
    best_pipe = None
    best_preds = None

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        auc = float(roc_auc_score(y_test, probs))

        results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
        }
        print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}, ROC-AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipe = pipe
            best_preds = preds

    # Figura 4: Comparação dos Modelos de Classificação de Hit
    model_names = list(results.keys())
    metrics_to_plot = ["Accuracy", "F1_Score", "ROC_AUC"]
    x = np.arange(len(model_names))
    width = 0.25
    colors = [SPOTIFY_DARK, ACCENT_BLUE, SPOTIFY_GREEN]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (m, col) in enumerate(zip(metrics_to_plot, colors)):
        vals = [results[mn][m] for mn in model_names]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=m.replace("_", "-"), color=col)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pontuação (0–1)")
    ax.set_title("Comparação de Desempenho — Predição de Sucesso (Hit: Popularidade ≥ 50)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_comparacao_modelos_classificacao.png"))
    plt.close()

    # Figura 5: Importância das Variáveis (Permutation Importance no melhor modelo)
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=min(4000, len(X_test)), replace=False)
    perm = permutation_importance(
        best_pipe, X_test.iloc[sample_idx], y_test[sample_idx],
        n_repeats=5, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1
    )
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(imp_df["feature"], imp_df["importance"], color=SPOTIFY_GREEN, edgecolor=SPOTIFY_DARK)
    ax.set_title(f"Top 10 Variáveis Mais Importantes — {best_name} (ROC-AUC)")
    ax.set_xlabel("Queda Média no ROC-AUC por Permutação")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_feature_importance_hit.png"))
    plt.close()

    # Figura 6: Matriz de Confusão do Melhor Modelo
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
        xticklabels=["Não-Hit (<50)", "Hit (≥50)"],
        yticklabels=["Não-Hit (<50)", "Hit (≥50)"],
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_title(f"Matriz de Confusão — {best_name}")
    ax.set_xlabel("Classe Predita")
    ax.set_ylabel("Classe Real")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06_matriz_confusao_hit.png"))
    plt.close()

    top_features = {
        row["feature"]: round(float(row["importance"]), 4)
        for _, row in imp_df.sort_values("importance", ascending=False).iterrows()
    }
    return {
        "best_model": best_name,
        "models": results,
        "top_features_permutation_roc_auc": top_features,
    }


def train_genre_classification_models(df: pd.DataFrame) -> dict:
    print("Treinando modelos de Classificação Multiclasse (Target: track_genre — 10 gêneros)...")
    selected_genres = [
        "classical", "hip-hop", "metal", "pop", "jazz",
        "electronic", "acoustic", "reggaeton", "country", "rock"
    ]
    available = [g for g in selected_genres if g in df["track_genre"].unique()]
    df_sub = df[df["track_genre"].isin(available)].copy()

    X = df_sub[AUDIO_FEATURES].copy()
    y = df_sub["track_genre"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Regressão Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=18, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=10, learning_rate=0.07, random_state=RANDOM_STATE
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1_macro = float(f1_score(y_test, preds, average="macro"))
        results[name] = {
            "Accuracy": round(acc, 4),
            "F1_Macro": round(f1_macro, 4),
        }
        print(f"  {name}: Acc={acc:.4f}, F1-Macro={f1_macro:.4f}")

    return {
        "genres_evaluated": available,
        "subset_rows": int(len(df_sub)),
        "models": results,
    }


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset não encontrado em {DATA_PATH}. Execute 'python src/download_data.py' primeiro."
        )

    df, eda_stats = load_and_clean_data(DATA_PATH)
    print(f"Dataset carregado: {eda_stats['raw_rows']:,} linhas brutas -> {eda_stats['clean_rows']:,} faixas únicas.")

    generate_eda_figures(df)
    reg_results = train_regression_models(df)
    hit_results = train_hit_classification_models(df)
    genre_results = train_genre_classification_models(df)

    output = {
        "eda": eda_stats,
        "target_1_regression_popularity": reg_results,
        "target_2_classification_is_hit": hit_results,
        "target_3_classification_track_genre": genre_results,
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nPipeline finalizado! Métricas salvas em {METRICS_PATH} e gráficos em {FIG_DIR}/")


if __name__ == "__main__":
    main()
