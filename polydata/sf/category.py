"""SF5: category-conditional microstructure.

Category labels: Gamma's API doesn't return a populated `category` field
for our 34k metadata cache. As an interim categorizer, we keyword-match
the question text into a small bucket scheme: Crypto, Sports, Politics,
Entertainment, Business, Other. Each market gets exactly one label by
first-match. SF5 reports per-(measure, category) median + IQR.

When/if Gamma exposes proper categories we can swap classify_question
for the official label without changing the aggregation downstream.
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

CRYPTO_PAT = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|crypto|sol|solana|doge|nft|token|coin|xrp|ada|avax|memecoin|stablecoin|usdc|usdt)\b",  # noqa: E501
    re.IGNORECASE,
)
SPORTS_PAT = re.compile(
    r"\b(nba|nfl|mlb|nhl|fifa|premier league|champions league|football|basketball|baseball|hockey|cricket|f1|formula|grand prix|tennis|atp|wta|golf|pga|ufc|boxing|olympics|world cup|tournament|champion|playoff|finals|vs\.?\s|over/under|moneyline|o/u)\b",  # noqa: E501
    re.IGNORECASE,
)
POLITICS_PAT = re.compile(
    r"\b(trump|biden|harris|election|president|congress|senate|gov|governor|primary|debate|poll|vote|kamala|desantis|democrat|republican|gop|tariff|impeach|supreme court|cabinet|nominee)\b",  # noqa: E501
    re.IGNORECASE,
)
GEOPOLITICS_PAT = re.compile(
    r"\b(iran|israel|gaza|hamas|hezbollah|lebanon|syria|ukraine|russia|putin|zelensky|netanyahu|khamenei|hormuz|ceasefire|airstrike|nato|nuclear|missile|invasion|annex|sanction|north korea|kim jong|china|taiwan|xi jinping|west bank|houthi|yemen)\b",  # noqa: E501
    re.IGNORECASE,
)
ENTERTAINMENT_PAT = re.compile(
    r"\b(grammy|oscar|emmy|movie|film|tv|series|netflix|hbo|disney|spotify|album|song|artist|actor|actress|streaming|box office|cancelled|reality|reunion|tour)\b",  # noqa: E501
    re.IGNORECASE,
)
BUSINESS_PAT = re.compile(
    r"\b(earnings|revenue|ipo|stock|nasdaq|s&p|s & p|sp500|nyse|fed|interest rate|gdp|inflation|cpi|jobs|unemployment|merger|acquisition|q1|q2|q3|q4|fortune)\b",  # noqa: E501
    re.IGNORECASE,
)


def classify_question(q: str | None) -> str:
    if not q:
        return "Other"
    if CRYPTO_PAT.search(q):
        return "Crypto"
    if POLITICS_PAT.search(q):
        return "Politics"
    if GEOPOLITICS_PAT.search(q):
        return "Geopolitics"
    if SPORTS_PAT.search(q):
        return "Sports"
    if ENTERTAINMENT_PAT.search(q):
        return "Entertainment"
    if BUSINESS_PAT.search(q):
        return "Business"
    return "Other"


def aggregate_by_category(
    joined: pl.DataFrame, value_col: str,
) -> pl.DataFrame:
    return (
        joined.filter(pl.col(value_col).is_not_null())
        .group_by("category")
        .agg(
            pl.col(value_col).median().alias("median"),
            pl.col(value_col).quantile(0.25).alias("p25"),
            pl.col(value_col).quantile(0.75).alias("p75"),
            pl.col("market_id").n_unique().alias("n_markets").cast(pl.UInt32),
        )
        .sort("median", descending=True)
    )


def run_sf5(
    panel_trade: Path = Path("data/panel_trade_measures.parquet"),
    metadata: Path = Path("data/panel_metadata.parquet"),
    out_parquet: Path = Path("artifacts/sf5_category.parquet"),
    out_png: Path = Path("artifacts/sf5_category.png"),
) -> pl.DataFrame:
    import matplotlib.pyplot as plt

    measures = pl.read_parquet(panel_trade)
    meta = pl.read_parquet(metadata).select(["market_id", "question"])
    cat_df = meta.with_columns(
        pl.col("question").map_elements(
            classify_question, return_dtype=pl.Utf8,
        ).alias("category"),
    ).select(["market_id", "category"])
    joined = measures.join(cat_df, on="market_id", how="left").with_columns(
        pl.col("category").fill_null("Other"),
    )
    # n-threshold: collapse Politics, Business, Entertainment into Other
    # (panel n < 10 makes per-category statistics noisy / unreliable).
    joined = joined.with_columns(
        pl.when(pl.col("category").is_in(
            ["Politics", "Business", "Entertainment"]
        ))
        .then(pl.lit("Other"))
        .otherwise(pl.col("category"))
        .alias("category"),
    )

    primary_col = "half_spread_prob_dw"
    eff = joined.filter(pl.col("measure") == "effective_spread")
    out = aggregate_by_category(eff, primary_col)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_parquet)

    fig, ax = plt.subplots(figsize=(8, 5))
    cats = out["category"].to_list()
    y = out["median"].to_numpy()
    err_lo = (out["median"] - out["p25"]).to_numpy()
    err_hi = (out["p75"] - out["median"]).to_numpy()
    ax.barh(cats, y, xerr=[err_lo, err_hi], capsize=4, color="steelblue")
    ax.set_xlabel("median effective half-spread (prob pp)")
    ax.set_title("SF5 — category-conditional effective spread")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out
