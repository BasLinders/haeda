import csv
import io
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import union_categoricals
from scipy import sparse

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
)

# --- Memory guards -----------------------------------------------------------
# The number of frequent itemsets grows combinatorially as min_support drops,
# and each itemset of size k yields up to 2^k - 2 rules. These caps keep a
# single analysis well inside the memory of a small Streamlit container.
MAX_ITEMSETS = 50_000
# Rows sent to the browser. st.dataframe serialises everything it is given into
# one websocket message (200 MB limit by default), so we never send all rules.
MAX_DISPLAY_RULES = 500
MAX_DOWNLOAD_RULES = 100_000
# Rows read per CSV chunk. Each chunk is reduced to categorical columns before
# the next one is read, so peak memory scales with the chunk, not the file.
CSV_CHUNK_ROWS = 250_000
# Itemsets turned into rules per batch; filtered rules are kept, the rest
# of each batch's candidates are discarded before the next batch starts.
RULE_BATCH_ITEMSETS = 5_000
# Sparse entries selected per support-counting batch (~10-15 bytes each with
# intermediates), the most candidates generated for one itemset size, and
# columns per block when counting item pairs.
CANDIDATE_BATCH_NNZ = 5_000_000
MAX_CANDIDATES = 2_000_000
PAIR_BLOCK_COLUMNS = 512

COLUMN_KEYWORDS = {
    'transaction_id': ['transactie', 'transaction', 'order', 'bonid'],
    'item': ['item', 'product'],
    'category': ['categorie', 'category'],
}


def find_column_mapping(columns):
    """Map standard names ('transaction_id', 'item', 'category') to the first
    matching column name (case-insensitive, whitespace-stripped)."""
    normalized = {col: str(col).strip().lower() for col in columns}
    mapping = {}
    for standard_name, keywords in COLUMN_KEYWORDS.items():
        for keyword in keywords:
            match = next(
                (orig for orig, norm in normalized.items()
                 if keyword in norm and orig not in mapping.values()),
                None,
            )
            if match is not None:
                mapping[standard_name] = match
                break
    return mapping


def preprocess_data(df):
    """
    Preprocesses a transactions dataframe.
    - Renames columns to standard names ('transaction_id', 'item', 'category').
    - Checks for missing required columns and drops rows with null IDs/items.
    - Stores the columns as categoricals (one copy of each distinct string).
    Returns (df, errors, has_category).
    """
    errors = []
    mapping = find_column_mapping(df.columns)
    df = df.rename(columns={orig: std for std, orig in mapping.items()})

    for col in ['transaction_id', 'item']:
        if col not in df.columns:
            errors.append(f"Column '{col}' is required but not found. Please ensure a column named like 'TransactionID' or 'Item' exists.")
    if errors:
        return df, errors, False

    for col in ['transaction_id', 'item']:
        if df[col].isnull().any():
            errors.append(f"Warning: Column '{col}' contains null values. Rows with null values in this column will be removed.")
            df = df.dropna(subset=[col])

    keep = [c for c in ['transaction_id', 'item', 'category'] if c in df.columns]
    df = df[keep].copy()
    for col in keep:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str).astype('category')
        df[col] = df[col].cat.remove_unused_categories()

    return df.reset_index(drop=True), errors, 'category' in df.columns


def _sniff_separator(file_bytes):
    sample = file_bytes[:65536].decode('utf-8', errors='ignore')
    try:
        return csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
    except csv.Error:
        return ','


@st.cache_data(max_entries=1, show_spinner="Reading file...")
def load_transactions(file_bytes):
    """Read the uploaded CSV in chunks, keeping only the mapped columns as
    categoricals. Cached so slider changes don't re-parse the file."""
    sep = _sniff_separator(file_bytes)
    header = pd.read_csv(io.BytesIO(file_bytes), sep=sep, nrows=0).columns
    usecols = list(find_column_mapping(header).values()) or None

    pieces = []
    reader = pd.read_csv(io.BytesIO(file_bytes), sep=sep, usecols=usecols,
                         dtype=str, chunksize=CSV_CHUNK_ROWS)
    for chunk in reader:
        pieces.append(chunk.astype('category'))

    if not pieces:
        return preprocess_data(pd.DataFrame(columns=header))
    if len(pieces) == 1:
        df = pieces[0]
    else:
        df = pd.DataFrame({
            col: union_categoricals([p[col] for p in pieces], ignore_order=True)
            for col in pieces[0].columns
        })
    return preprocess_data(df)


def encode_transactions(df, column, min_support):
    """
    Build a sparse one-hot transaction x item matrix (CSC, int32).

    Items whose own support is below min_support can never be part of a
    frequent itemset, so they are dropped *before* encoding. The matrix keeps
    one row per transaction (including ones left empty), so support values are
    still relative to the full transaction count.
    Returns (matrix, item_names, n_transactions, n_items_total).
    """
    tx = df['transaction_id']
    items = df[column]
    if not isinstance(tx.dtype, pd.CategoricalDtype):
        tx = tx.astype('category')
    if not isinstance(items.dtype, pd.CategoricalDtype):
        items = items.astype('category')
    tx_codes = tx.cat.codes.to_numpy()
    item_codes = items.cat.codes.to_numpy()
    valid = (tx_codes >= 0) & (item_codes >= 0)
    tx_codes, item_codes = tx_codes[valid], item_codes[valid]

    # Duplicate (transaction, item) rows collapse to one entry, set to 1.
    matrix = sparse.csc_matrix(
        (np.ones(len(tx_codes), dtype=np.int32), (tx_codes, item_codes)),
        shape=(len(tx.cat.categories), len(items.cat.categories)),
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1

    n_transactions = int(np.count_nonzero(np.bincount(tx_codes, minlength=matrix.shape[0])))
    item_counts = np.diff(matrix.indptr)
    n_items_total = int(np.count_nonzero(item_counts))
    keep = (item_counts > 0) & (item_counts / max(n_transactions, 1) >= min_support)
    matrix = matrix[:, np.flatnonzero(keep)]
    frequent_items = items.cat.categories[keep]
    return matrix, [str(c) for c in frequent_items], n_transactions, n_items_total


def _count_pairs(X, n_transactions, min_support, block=PAIR_BLOCK_COLUMNS):
    """Support counts for all item pairs via X.T @ X, computed one block of
    columns at a time so the co-occurrence matrix is never held in full."""
    XT = X.T.tocsr()
    rows, cols, counts = [], [], []
    for j0 in range(0, X.shape[1], block):
        co = (XT @ X[:, j0:j0 + block]).tocoo()
        col = co.col + j0
        keep = (co.row < col) & (co.data / n_transactions >= min_support)
        rows.append(co.row[keep]); cols.append(col[keep]); counts.append(co.data[keep])
    if not rows:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    cand = np.column_stack([np.concatenate(rows), np.concatenate(cols)]).astype(np.int64)
    order = np.lexsort(cand.T[::-1])
    return cand[order], np.concatenate(counts)[order].astype(np.int64)


def _generate_candidates(prev, max_candidates):
    """Apriori join: combine (k-1)-itemsets that share their first k-2 items,
    keeping only candidates whose every (k-1)-subset is frequent.
    Returns None if more than max_candidates would be generated."""
    prev_set = set(map(tuple, prev.tolist()))
    groups = {}
    for t in map(tuple, prev.tolist()):
        groups.setdefault(t[:-1], []).append(t[-1])
    out = []
    for prefix, lasts in groups.items():
        lasts.sort()
        for i, a in enumerate(lasts):
            for b in lasts[i + 1:]:
                cand = prefix + (a, b)
                if all(cand[:x] + cand[x + 1:] in prev_set for x in range(len(prefix))):
                    out.append(cand)
            if len(out) > max_candidates:
                return None
    return np.array(out, dtype=np.int64).reshape(-1, prev.shape[1] + 1)


def _count_support(X, candidates, min_count, max_frequent, nnz_budget=CANDIDATE_BATCH_NNZ):
    """Count transactions containing every item of each candidate. Candidates
    are processed in batches sized so the selected sparse columns hold at most
    ~nnz_budget entries, which bounds memory regardless of item popularity.
    Returns None as soon as more than max_frequent candidates reach min_count,
    so a level that would exceed the itemset cap is abandoned early."""
    col_nnz = np.diff(X.indptr)
    cost = col_nnz[candidates].sum(axis=1)
    # Batch boundaries where the running cost crosses a multiple of the budget.
    batch_id = np.cumsum(cost) // max(nnz_budget, 1)
    bounds = np.flatnonzero(np.diff(batch_id)) + 1
    counts = np.empty(len(candidates), dtype=np.int64)
    n_frequent = 0
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(candidates)]):
        c = candidates[s:e]
        m = X[:, c[:, 0]]
        for j in range(1, c.shape[1]):
            m = m.multiply(X[:, c[:, j]])
        counts[s:e] = np.asarray(m.sum(axis=0)).ravel()
        n_frequent += int(np.count_nonzero(counts[s:e] >= min_count))
        if n_frequent > max_frequent:
            return None
    return counts


def mine_itemsets(X, item_names, n_transactions, min_support, max_len,
                  max_itemsets=MAX_ITEMSETS, max_candidates=MAX_CANDIDATES):
    """
    Level-wise (Apriori) frequent-itemset mining on a sparse matrix with
    batched support counting. Stops before a level that would exceed
    max_itemsets frequent sets or max_candidates candidates.
    Returns (itemsets DataFrame[support, itemsets], reached_len, stopped_early).
    """
    counts = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    levels = [(np.arange(X.shape[1], dtype=np.int64)[:, None], counts)]
    total, reached, stopped = len(counts), 1, False
    if total > max_itemsets:
        levels, reached, stopped = [], 0, True

    current = levels[0][0] if levels else None
    for k in range(2, max_len + 1):
        if current is None or len(current) < 2:
            break
        if k == 2:
            cand, cnt = _count_pairs(X, n_transactions, min_support)
        else:
            cand = _generate_candidates(current, max_candidates)
            if cand is None:
                stopped = True
                break
            if len(cand) == 0:
                break
            min_count = int(np.ceil(min_support * n_transactions - 1e-9))
            cnt = _count_support(X, cand, min_count, max_itemsets - total)
            if cnt is None:
                stopped = True
                break
            keep = cnt / n_transactions >= min_support
            cand, cnt = cand[keep], cnt[keep]
        if len(cand) == 0:
            break
        if total + len(cand) > max_itemsets:
            stopped = True
            break
        levels.append((cand, cnt))
        total += len(cand)
        reached = k
        current = cand

    names = np.asarray(item_names, dtype=object)
    itemsets, supports = [], []
    for cand, cnt in levels:
        itemsets.extend(frozenset(names[row]) for row in cand)
        supports.append(cnt / n_transactions)
    support = np.concatenate(supports) if supports else np.empty(0)
    return pd.DataFrame({'support': support, 'itemsets': itemsets}), reached, stopped


RULE_COLUMNS = ['antecedents', 'consequents', 'support', 'confidence', 'lift']


def generate_rules(frequent_itemsets, min_confidence, min_lift,
                   batch_size=RULE_BATCH_ITEMSETS):
    """
    Generate association rules in batches of itemsets, applying the confidence
    and lift filters inside each batch so rejected candidates are never stored.
    Equivalent to mlxtend's association_rules(metric='confidence') followed by
    a lift filter, but without materialising every candidate rule and all 12
    of mlxtend's metrics first.
    Returns (rules DataFrame, number of rules passing the confidence filter).
    """
    support_of = dict(zip(frequent_itemsets['itemsets'], frequent_itemsets['support']))
    multi = [(s, sup) for s, sup in support_of.items() if len(s) >= 2]

    batches = []
    n_confident = 0
    for start in range(0, len(multi), batch_size):
        rows = []
        for itemset, s_ac in multi[start:start + batch_size]:
            for r in range(1, len(itemset)):
                for antecedent in combinations(itemset, r):
                    antecedent = frozenset(antecedent)
                    confidence = s_ac / support_of[antecedent]
                    if confidence < min_confidence:
                        continue
                    n_confident += 1
                    consequent = itemset - antecedent
                    lift = confidence / support_of[consequent]
                    if lift >= min_lift:
                        rows.append((antecedent, consequent, s_ac, confidence, lift))
        if rows:
            batches.append(pd.DataFrame(rows, columns=RULE_COLUMNS))

    if not batches:
        return pd.DataFrame(columns=RULE_COLUMNS), n_confident
    return pd.concat(batches, ignore_index=True), n_confident


def market_basket_analysis(df, analysis_level='product', min_support=0.01,
                           min_confidence=0.1, min_lift=1.0, max_len=3,
                           max_itemsets=MAX_ITEMSETS):
    """
    Performs market basket analysis (Apriori on a sparse encoding with
    batched support counting).
    """
    messages = []

    if analysis_level == 'category':
        if 'category' not in df.columns:
            messages.append("Error: Analysis on category level selected, but no 'category' column was found.")
            return pd.DataFrame(), messages
        column = 'category'
    else:
        column = 'item'

    if df.empty:
        messages.append("No transactions found to analyze.")
        return pd.DataFrame(), messages

    X, item_names, n_transactions, n_items_total = encode_transactions(df, column, min_support)
    messages.append(
        f"{n_transactions:,} transactions, {n_items_total:,} distinct {analysis_level}s "
        f"({len(item_names):,} meet the minimum support on their own)."
    )
    if not item_names:
        messages.append("Note: No single item reaches the 'Minimum support'. Try lowering it.")
        return pd.DataFrame(), messages

    frequent_itemsets, reached_len, stopped_early = mine_itemsets(
        X, item_names, n_transactions, min_support, max_len, max_itemsets
    )
    del X

    messages.append(f"Number of frequent sets found at {analysis_level} level: {len(frequent_itemsets)}")
    if stopped_early:
        messages.append(
            f"Warning: Stopped at itemsets of size {reached_len} to stay within memory "
            f"(the next size would exceed {max_itemsets:,} frequent sets). Raise the 'Minimum support' "
            f"to include larger combinations."
        )
    if reached_len < 2:
        messages.append("Note: No rules could be defined because no frequent sets of two or more items were found. Try lowering the 'Minimum support'.")
        return pd.DataFrame(), messages

    rules, n_confident = generate_rules(frequent_itemsets, min_confidence, min_lift)
    del frequent_itemsets
    messages.append(f"Number of rules found (before lift filter): {n_confident}")
    if n_confident == 0:
        messages.append("Note: Frequent sets were found, but no rules match the chosen 'Minimum confidence'. Try lowering it.")
        return rules, messages
    messages.append(f"Number of rules found after lift filter: {len(rules)}")

    rules = rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])
    return rules.reset_index(drop=True), messages


def _format_rules(rules):
    out = rules.copy()
    out['antecedents'] = out['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    out['consequents'] = out['consequents'].apply(lambda x: ', '.join(sorted(x)))
    out.insert(0, 'Rank', np.arange(1, len(out) + 1))
    return out


def run():
    st.title("🛒 Market Basket Analysis")
    st.markdown("""
                This tool helps you uncover hidden patterns in your sales data.
                By analyzing transactions, it identifies products or categories that are frequently purchased together.
                Use the results to analyze the effectiveness of promotions, cross-selling opportunities, and to find customer preferences.

                The resulting table shows the most significant associations, ranked by lift, confidence, and support.
    """)
    with st.expander("How to interpret the results", expanded=False):
        st.markdown("""
            To interpret the results, think of each rule as an "if-then" statement: If a customer buys the Antecedent, then they are also likely to buy the Consequent.

            Explanation of the terms used:
            - **Antecedents**: The "if..." part of the rule. This is the item or set of items found in a basket.
            - **Consequents**: The "...then" part of the rule. This is the item or set of items also found in the basket.
            - **Support** (popularity): How frequently the combination appears in all transactions.
            - **Confidence** (reliability): The probability that, if a customer buys the antecedent, the consequent will also be purchased.
            - **Lift** (strength): Measures how much more often A and B occur together than expected if they were statistically independent.
        """)
    st.markdown("""
                Upload a CSV file with your transaction data to discover association rules.
                The file must contain at least columns for **transaction ID** and **item/product**.
                A **category** column is optional.
    """)
    st.write("")

    # Provide a correct template for download
    template_df = pd.DataFrame({
        "transaction_id": ["Can be any string or number", "1-NDINV00", "10052456", "10254BE"],
        "item": ["Product name or SKU as a string", "1-bc054", "Men's Jacket Size M", "Butter"],
        "category": ["Can be any string", "Video games", "Men's wear", "Dairy"]
    })

    st.download_button(
        label="Download CSV Template",
        data=template_df.to_csv(index=False).encode('utf-8'),
        file_name="mba_template.csv",
        mime="text/csv"
    )

    data = st.file_uploader("Choose a CSV file", type="csv")

    if data is not None:
        try:
            df, errors, has_category = load_transactions(data.getvalue())
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
            return

        for error in errors:
            st.warning(error)

        # Check if preprocessing resulted in critical errors
        if 'transaction_id' not in df.columns or 'item' not in df.columns:
            st.error("Preprocessing failed. Please ensure your file contains the required columns.")
            return
        if not has_category:
            st.info("Note: No 'category' column found. Analysis at the category level is not possible.")

        st.success(f"File successfully loaded and preprocessed ({len(df):,} rows).")
        st.write("### A random sample of your data:")
        st.dataframe(df.sample(min(10, len(df))))

        # Sidebar for controls
        st.sidebar.header("Analysis Settings")

        level_options = ['product']
        if has_category:
            level_options.append('category')

        level = st.sidebar.selectbox("Analyze at level:", level_options)
        min_support = st.sidebar.slider("Minimum support:",
                                        0.001,
                                        0.5,
                                        0.01,
                                        step=0.001,
                                        format="%.3f",
                                        help="How frequently an itemset (e.g., {Game console, Video game X}) appears in all transactions. A minimum support of 0.02 means the itemset must appear in at least 2% of all transactions.")
        min_confidence = st.sidebar.slider("Minimum confidence:",
                                           0.01,
                                           1.0,
                                           0.1,
                                           step=0.01,
                                           help="The conditional probability. For a rule 'If A then B', confidence is P(B|A). A minimum confidence of 0.6 means that in 60% of the transactions containing A, B is also present.")
        min_lift = st.sidebar.slider("Minimum lift:",
                                     1.0,
                                     10.0,
                                     1.0,
                                     step=0.1,
                                     help="Measures how much more often A and B occur together than expected if they were statistically independent. A lift of 1 means A and B are independent. A lift > 1 means they are positively correlated.")
        max_len = st.sidebar.slider("Maximum items per combination:",
                                    2,
                                    5,
                                    3,
                                    help="The largest combination size to search for (antecedent + consequent). Larger values find longer rules but need much more memory, especially with a low minimum support.")

        if st.sidebar.button("Analyze"):
            with st.spinner("Analysis in progress..."):
                try:
                    mba_results, messages = market_basket_analysis(
                        df=df,
                        analysis_level=level,
                        min_support=min_support,
                        min_confidence=min_confidence,
                        min_lift=min_lift,
                        max_len=max_len,
                    )
                except MemoryError:
                    st.error("The analysis ran out of memory. Raise the 'Minimum support' or lower the 'Maximum items per combination' and try again.")
                    return

            st.info("### Analysis Log\n\n- " + "\n- ".join(str(m) for m in messages))

            if not mba_results.empty:
                st.header(f"Analysis Results ({level.capitalize()} level)")

                n_rules = len(mba_results)
                display_results = _format_rules(mba_results.head(MAX_DISPLAY_RULES))
                if n_rules > MAX_DISPLAY_RULES:
                    st.caption(f"Showing the top {MAX_DISPLAY_RULES:,} of {n_rules:,} rules. Use the download button for more.")

                st.dataframe(
                    display_results[[
                        'Rank', 'antecedents', 'consequents', 'support', 'confidence', 'lift'
                    ]],
                    hide_index=True,
                    column_config={
                        "support": st.column_config.NumberColumn(format="%.4f"),
                        "confidence": st.column_config.NumberColumn(format="%.4f"),
                        "lift": st.column_config.NumberColumn(format="%.4f"),
                    }
                )

                download_rules = mba_results.head(MAX_DOWNLOAD_RULES)
                st.download_button(
                    label=(f"Download top {len(download_rules):,} rules as CSV"
                           if n_rules > MAX_DOWNLOAD_RULES else "Download all rules as CSV"),
                    data=_format_rules(download_rules).to_csv(index=False).encode('utf-8'),
                    file_name=f"mba_rules_{level}.csv",
                    mime="text/csv",
                )

                # Conclusion for the top rule
                st.header("Explanation of the Strongest Rule")
                top_rule = display_results.iloc[0]
                antecedents = top_rule['antecedents']
                consequents = top_rule['consequents']
                support = top_rule['support']
                confidence = top_rule['confidence']
                lift = top_rule['lift']

                st.markdown(f"The combination of **'{antecedents}'** and **'{consequents}'** is bought together the most.")
                st.markdown(f"- This combination appears in **{support * 100:.2f}%** of all transactions.")
                st.markdown(f"- If a customer buys '{antecedents}', there is a **{confidence * 100:.2f}%** chance they will also buy '{consequents}'.")
                st.markdown(f"- This happens **{lift:.2f} times more often** than would be expected by chance.")
            else:
                st.warning("\nNo association rules found with the current parameters.")

if __name__ == "__main__":
    run()
