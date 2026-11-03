"""Smoke test for pages/mba_analysis.py: the page should load and run a full
analysis on a small transactions CSV without raising."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parent.parent / "pages" / "mba_analysis.py")

SAMPLE_CSV = b"""transaction_id,item,category
1,bread,bakery
1,milk,dairy
2,bread,bakery
2,butter,dairy
3,bread,bakery
3,milk,dairy
3,butter,dairy
4,milk,dairy
5,bread,bakery
5,milk,dairy
"""


def test_loads_without_upload():
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)

    assert not at.exception
    assert at.title[0].value == "🛒 Market Basket Analysis"


def test_full_analysis_on_sample_data():
    at = AppTest.from_file(PAGE)
    at.run(timeout=30)

    at.get("file_uploader")[0].upload("transactions.csv", SAMPLE_CSV, "text/csv")
    at.run(timeout=30)
    assert not at.exception

    analyze_button = next(b for b in at.sidebar.button if b.label == "Analyze")
    analyze_button.click().run(timeout=30)

    assert not at.exception
    headers = [h.value for h in at.header]
    assert any("Analysis Results" in h for h in headers)
    assert any("Explanation of the Strongest Rule" in h for h in headers)


# --- Unit tests for the memory-bounded pipeline ------------------------------
import importlib.util

import numpy as np
import pandas as pd
import pytest
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def _load_module():
    spec = importlib.util.spec_from_file_location("mba_analysis", PAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mba = _load_module()


def _random_transactions(n_tx=400, n_items=30, seed=0):
    rng = np.random.default_rng(seed)
    p = 1 / np.arange(1, n_items + 1); p /= p.sum()
    rows = []
    for t in range(n_tx):
        for i in set(rng.choice(n_items, size=rng.integers(1, 7), p=p)):
            rows.append((str(t), f"item{i}", f"cat{i % 4}"))
    return pd.DataFrame(rows, columns=["transaction_id", "item", "category"])


def _reference_rules(df, column, min_support, min_confidence, min_lift):
    """The original dense apriori + mlxtend association_rules pipeline."""
    tl = df.groupby("transaction_id")[column].unique().apply(list).tolist()
    te = TransactionEncoder()
    enc = pd.DataFrame(te.fit(tl).transform(tl), columns=te.columns_)
    fi = apriori(enc, min_support=min_support, use_colnames=True)
    rules = association_rules(fi, num_itemsets=len(tl), metric="confidence",
                              min_threshold=min_confidence)
    return rules[rules["lift"] >= min_lift]


def _as_set(rules):
    return {
        (frozenset(a), frozenset(c), round(s, 9), round(conf, 9), round(l, 9))
        for a, c, s, conf, l in rules[["antecedents", "consequents", "support",
                                       "confidence", "lift"]].itertuples(index=False)
    }


@pytest.mark.parametrize("column", ["product", "category"])
def test_matches_reference_pipeline(column):
    df, errors, _ = mba.preprocess_data(_random_transactions())
    assert not errors
    ref = _reference_rules(df.astype(str), "item" if column == "product" else column,
                           0.02, 0.1, 1.0)
    rules, _ = mba.market_basket_analysis(df, analysis_level=column, min_support=0.02,
                                          min_confidence=0.1, min_lift=1.0, max_len=10)
    assert len(rules) > 0
    assert _as_set(rules) == _as_set(ref)


def test_itemset_cap_stops_early_instead_of_exploding():
    df, _, _ = mba.preprocess_data(_random_transactions())
    rules, messages = mba.market_basket_analysis(df, min_support=0.001,
                                                 max_len=5, max_itemsets=50)
    assert any("Stopped at itemsets of size" in m for m in messages)


def test_chunked_csv_load_matches_single_read(monkeypatch):
    raw = _random_transactions(n_tx=200).to_csv(index=False, sep=";").encode()
    monkeypatch.setattr(mba, "CSV_CHUNK_ROWS", 37)
    df, errors, has_category = mba.load_transactions.__wrapped__(raw)
    expected = pd.read_csv(pd.io.common.BytesIO(raw), sep=";", dtype=str)
    assert not errors and has_category
    pd.testing.assert_frame_equal(df.astype(str), expected)
