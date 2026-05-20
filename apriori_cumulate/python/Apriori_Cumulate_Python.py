"""
Association-rule mining pipeline.

Usage:
    python pipeline.py <base_path> [options]

In-memory Cumulate-style generalized Apriori
─────────────────────────────────────────────
The literal Cumulate algorithm filters transactions during each database scan,
which was useful for disk-based settings. In our in-memory setting, hierarchy
tokens are expanded once, encoded once as a sparse one-hot matrix, and mined
directly from memory with hierarchy-aware candidate pruning.

Our main algorithm is ``apriori_inmem()`` in ``frequent_patterns/apriori.py``.
The separate ``cumulate_apriori.py`` file is kept only as a slow
literal-Cumulate baseline/reference implementation.
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frequent_patterns.apriori import apriori_inmem
from frequent_patterns.association_rules import association_rules
from frequent_patterns import fpcommon as fpc


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.3f}s"
    minutes, rem = divmod(seconds, 60)
    return f"{int(minutes)}m {rem:.1f}s"


@contextmanager
def timed_step(label: str):
    start = time.perf_counter()
    print(f"[{now_stamp()}] {label} …", flush=True)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        print(f"[{now_stamp()}] {label} failed after {format_elapsed(elapsed)}", flush=True)
        raise
    else:
        elapsed = time.perf_counter() - start
        print(f"[{now_stamp()}] {label} done in {format_elapsed(elapsed)}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Mine association rules from wishlist parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("base", help="Path to folder containing the parquet datasets")
    p.add_argument("--k-levels",     type=int,   required=True, metavar="K")
    p.add_argument("--min-support",  type=float, required=True, metavar="S")
    p.add_argument("--min-conf",     type=float, required=True, metavar="C")
    p.add_argument("--min-lift",     type=float, required=True, metavar="L")
    p.add_argument("--max-ante-len", default=None, metavar="A",
                   help="Max antecedent length, or 'none' for no limit")
    p.add_argument("--max-cons-len", default=None, metavar="B",
                   help="Max consequent length, or 'none' for no limit")
    p.add_argument("--max-len", default=None, metavar="M",
                   help="Max total itemset length, or 'none' for no limit. "
                        "Auto-computed as max-ante-len + max-cons-len when both are set.")
    p.add_argument("--candidate-pruning", default="ancestor-same-path",
                   choices=["none", "ancestor", "same-path", "ancestor-same-path"],
                   help="Hierarchy information used during Apriori candidate generation.")
    p.add_argument("--rule-filtering", default="full",
                   choices=["none", "ancestor", "ancestor-same-path", "full"],
                   help="Hierarchy filters used during association rule generation.")
    p.add_argument("--output", default=None, metavar="FILE",
                   help="Save rules to CSV (prints to stdout if omitted)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

SEP = ">"
TOKEN_RE = re.compile(r"^L(\d+)\|B:([^|]+)\|(.*)$")


def split_path(path: str) -> list[str]:
    return [p.strip() for p in str(path).split(SEP) if p.strip()]


def make_level_tokens(path: str, k_levels: int) -> list[str]:
    parts = split_path(path)
    if k_levels <= 0:
        return []
    window = parts[-k_levels:] if len(parts) >= k_levels else parts
    if not window:
        return []
    branch = window[0]
    max_k = len(window)
    out: list[str] = []
    for lvl in range(0, max_k):
        upto = max_k - lvl
        label = " > ".join(window[:upto])
        out.append(f"L{lvl}|B:{branch}|{label}")
    return out


def parse_token(tok: str) -> tuple[int, str, str]:
    s = str(tok).strip()
    m = TOKEN_RE.match(s)
    if not m:
        return 0, "__UNK__", s
    return int(m.group(1)), m.group(2).strip(), m.group(3).strip()


def level_of(tok: str) -> int:
    return parse_token(tok)[0]


def branch_of(tok: str) -> str:
    return parse_token(tok)[1]


def label_of(tok: str) -> str:
    return parse_token(tok)[2]


def direct_parent_token_of_leaf(leaf_tok: str) -> str | None:
    lvl, branch, lbl = parse_token(leaf_tok)
    if lvl != 0:
        return None
    parts = split_path(lbl)
    if len(parts) <= 1:
        return None
    parent_lbl = " > ".join(parts[:-1])
    return f"L1|B:{branch}|{parent_lbl}"


# ---------------------------------------------------------------------------
# Ancestor map
# ---------------------------------------------------------------------------

def build_ancestors_from_tokens(
    tokens: list[str],
    branch_ancestry=None,
) -> dict[str, set[str]]:
    """Build a full ancestor map in token-name space.

    For each token t = Lℓ|B:branch|label the returned dict contains:

    1. Within-branch ancestors: tokens in the same branch whose label is a
       strict prefix of t's label (intra-branch).

    2. Cross-branch ancestors (intermediate nodes): tokens from any branch
       whose label ends with an intermediate path component of t's label.
       Extended to include the *last* segment, but only tokens with a
       strictly shorter label (fewer path parts) are accepted as ancestors.
       This prevents circular ancestor chains and correctly marks cross-branch
       sibling tokens — e.g. ``L2|B:Clothing|Clothing`` (len=1) is added as
       an ancestor of ``L1|B:Apparel & Accessories|Apparel & Accessories >
       Clothing`` (len=2) so that candidate generation prunes pairs that
       represent the same taxonomy node from different K_LEVELS windows.

    3. Cross-branch top-of-window linking via ``branch_ancestry``: adds the
       root token of a parent branch as ancestor of tokens in a descendant
       branch, catching cross-window within-chain rules that intra-branch and
       intermediate-node checks miss (e.g. branch "Cosmetics" is a child of
       branch "Personal Care").

    Note: ancestor precomputation here corresponds to Cumulate's precomputation
    optimisation (Srikant & Agrawal 1995, Opt. 2).  The actual in-memory
    speedup comes from early candidate pruning in ``Generate_candidates_H``
    and vectorized support counting in ``apriori()``, not from repeated
    transaction filtering.
    """
    tok_set = set(tokens)
    by_key: dict = defaultdict(set)
    leaf_to_tokens: dict = defaultdict(set)

    for t in tokens:
        lvl, branch, lbl = parse_token(t)
        by_key[(branch, lbl)].add(t)
        leaf = split_path(lbl)[-1]
        leaf_to_tokens[leaf].add(t)

    ancestors: dict[str, set[str]] = {t: set() for t in tokens}
    for t in tokens:
        _, branch, lbl = parse_token(t)
        parts = split_path(lbl)

        # 1) Within-branch: strict label prefixes in the same branch.
        for j in range(1, len(parts)):
            pref_lbl = " > ".join(parts[:j])
            for anc_tok in by_key.get((branch, pref_lbl), set()):
                if anc_tok in tok_set and anc_tok != t:
                    ancestors[t].add(anc_tok)

        # 2) Cross-branch ancestors via intermediate (non-leaf) path segments.
        #    K-level windows can represent a broader taxonomy node with the
        #    same number of path components as a descendant window, e.g.
        #    "Toys & Games > Toys" vs. "Toys > Dolls, Playsets & Toy Figures".
        #    Therefore matching intermediate segment names are treated as
        #    ancestor-like tokens without requiring a shorter token label.
        for j in range(len(parts) - 1):
            node_name = parts[j]
            for anc_tok in leaf_to_tokens.get(node_name, set()):
                if anc_tok in tok_set and anc_tok != t:
                    ancestors[t].add(anc_tok)

    # 3) Cross-branch top-of-window linking via branch_ancestry.
    if branch_ancestry is not None:
        branch_to_root_toks: dict = defaultdict(set)
        for t in tokens:
            _, br, lbl = parse_token(t)
            if len(split_path(lbl)) == 1:
                branch_to_root_toks[br].add(t)

        for t2 in tokens:
            _, branch2, _ = parse_token(t2)
            paths2 = branch_ancestry.get(branch2, frozenset())
            if isinstance(paths2, str):
                paths2 = frozenset({paths2})
            if not paths2:
                continue
            for branch1, root_toks1 in branch_to_root_toks.items():
                if branch1 == branch2:
                    continue
                paths1 = branch_ancestry.get(branch1, frozenset())
                if isinstance(paths1, str):
                    paths1 = frozenset({paths1})
                if any(pb.startswith(pa + " > ") for pa in paths1 for pb in paths2):
                    ancestors[t2].update(root_toks1)

    return ancestors


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_data(base: str) -> pd.DataFrame:
    p = Path(base)
    if p.is_dir():
        named = p / "wishlist_data.parquet"
        if named.exists():
            return pd.read_parquet(named)
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {p}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(p)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["wishlist_id", "category_name"])


# ---------------------------------------------------------------------------
# Transactions & encoding
# ---------------------------------------------------------------------------

def build_transactions(df_min: pd.DataFrame, k_levels: int) -> tuple[list[list[str]], list[str]]:
    """Build transactions containing *all* level tokens (L0 … L(k-1)).

    Returns
    -------
    transactions : list of lists
        One list per basket.  Each list contains every token at every level
        generated by ``make_level_tokens`` for that basket's category names.
    all_tokens : list of str
        Sorted vocabulary of all distinct tokens across all transactions.
        Used to build ``ancestors`` and ``path_map``.
    """
    df_items = df_min[["wishlist_id", "category_name"]].copy()
    df_items["tok"] = df_items["category_name"].apply(
        lambda s: make_level_tokens(s, k_levels)
    )
    df_items = df_items.explode("tok")[["wishlist_id", "tok"]].dropna()
    df_items = df_items.drop_duplicates(["wishlist_id", "tok"])

    basket = (
        df_items.groupby("wishlist_id")["tok"]
        .apply(list)
        .reset_index(name="items")
    )
    transactions = [list(dict.fromkeys(tx)) for tx in basket["items"].tolist()]
    all_tokens = sorted({tok for tx in transactions for tok in tx})
    return transactions, all_tokens


def encode_transactions(
    transactions: list[list[str]],
    all_tokens: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot encode expanded transactions into an in-memory sparse matrix."""
    columns = all_tokens if all_tokens is not None else sorted(
        {tok for tx in transactions for tok in tx}
    )
    col_idx = {tok: idx for idx, tok in enumerate(columns)}

    indptr = [0]
    indices = []
    for tx in transactions:
        indices.extend(col_idx[tok] for tok in dict.fromkeys(tx))
        indptr.append(len(indices))

    data = [True] * len(indices)
    X = csr_matrix(
        (data, indices, indptr),
        shape=(len(transactions), len(columns)),
        dtype=bool,
    )
    return pd.DataFrame.sparse.from_spmatrix(X, columns=columns)


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def mine_rules_raw(
    df_encoded: pd.DataFrame,
    min_support: float,
    min_conf: float,
    max_len,
    *,
    ancestors=None,
    max_ante_len=None,
    max_cons_len=None,
    require_single_consequent: bool = False,
    path_map=None,
    branch_ancestry=None,
    candidate_pruning: str = "ancestor-same-path",
    rule_filtering: str = "full",
) -> pd.DataFrame:
    """Mine frequent itemsets and association rules using apriori_inmem.

    The sparse one-hot matrix is already in memory. apriori_inmem reuses it
    across Apriori passes and prunes hierarchy-redundant candidates before
    support counting.
    """
    candidate_ancestors = ancestors if candidate_pruning in {"ancestor", "ancestor-same-path"} else None
    candidate_path_map = path_map if candidate_pruning in {"same-path", "ancestor-same-path"} else None
    rule_ancestors = ancestors if rule_filtering in {"ancestor", "ancestor-same-path", "full"} else None
    rule_path_map = path_map if rule_filtering in {"ancestor-same-path", "full"} else None
    rule_branch_ancestry = branch_ancestry if rule_filtering == "full" else None

    with timed_step("Apriori mining"):
        freq = apriori_inmem(
            df_encoded,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len,
            ancestors=candidate_ancestors,
            path_map=candidate_path_map,
        )
    print(f"Frequent itemsets: {len(freq)}")

    with timed_step("Association rule generation"):
        rules_raw = association_rules(
            freq,
            metric="confidence",
            min_threshold=min_conf,
            ancestors=rule_ancestors,
            path_map=rule_path_map,
            max_ante_len=max_ante_len,
            max_cons_len=max_cons_len,
            require_single_consequent=require_single_consequent,
            branch_ancestry=rule_branch_ancestry,
        )
    print(f"Raw rules: {len(rules_raw)}")
    return rules_raw


# ---------------------------------------------------------------------------
# Scoring & deduplication
# ---------------------------------------------------------------------------

def add_score_and_rank(rules: pd.DataFrame, min_support: float, min_lift: float) -> pd.DataFrame:
    r = rules.copy()
    r = r[(r["support"] >= min_support) & (r["lift"] >= min_lift)].copy()
    r["score"] = r["confidence"] * r["lift"]
    r["_a_key"] = r["antecedents"].apply(lambda s: tuple(sorted(s)))
    r["_b_key"] = r["consequents"].apply(lambda s: tuple(sorted(s)))
    r = (
        r.sort_values(
            ["score", "confidence", "lift", "support", "_a_key", "_b_key"],
            ascending=[False, False, False, False, True, True],
        )
        .drop(columns=["_a_key", "_b_key"])
        .copy()
    )
    return r


def consequent_family_key_one(tok: str) -> str:
    lvl, _, _ = parse_token(tok)
    if lvl == 0:
        p = direct_parent_token_of_leaf(tok)
        return p if p is not None else tok
    return tok


def consequent_family_key_multi(b_toks: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(consequent_family_key_one(tok) for tok in b_toks))


def dedupe_family(rules: pd.DataFrame) -> pd.DataFrame:
    if rules.empty:
        return rules
    r = rules.copy()
    r["b_family"] = r["b_toks"].apply(consequent_family_key_multi)
    r["family_key"] = list(zip(r["a_toks"], r["b_family"]))
    r["_a_key"] = r["a_toks"]
    r["_b_key"] = r["b_toks"]
    r = (
        r.sort_values(
            ["score", "confidence", "lift", "support", "_a_key", "_b_key"],
            ascending=[False, False, False, False, True, True],
        )
         .drop_duplicates(subset=["family_key"], keep="first")
         .drop(columns=["b_family", "family_key", "_a_key", "_b_key"])
         .copy()
    )
    return r


def dedupe_antimirror(rules: pd.DataFrame) -> pd.DataFrame:
    if rules.empty:
        return rules
    r = rules.copy()
    r["pair_key"] = r.apply(
        lambda row: tuple(sorted(set(row["a_toks"]) | set(row["b_toks"]))),
        axis=1,
    )
    r["_a_key"] = r["a_toks"]
    r["_b_key"] = r["b_toks"]
    r = (
        r.sort_values(
            ["score", "confidence", "lift", "support", "_a_key", "_b_key"],
            ascending=[False, False, False, False, True, True],
        )
         .drop_duplicates(subset=["pair_key"], keep="first")
         .drop(columns=["pair_key", "_a_key", "_b_key"])
         .copy()
    )
    return r


def pretty_rules(rules: pd.DataFrame) -> pd.DataFrame:
    r = rules.copy()
    r["rule"] = r["a_lbl"].astype(str) + " → " + r["b_lbl"].astype(str)
    return r[[
        "rule", "a_toks", "b_toks", "a_levels", "b_levels",
        "a_branches", "b_branches", "support", "confidence", "lift", "score",
    ]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_start = time.perf_counter()
    args = parse_args()

    def _parse_optional_int(val):
        if val is None or (isinstance(val, str) and val.lower() == "none"):
            return None
        return int(val)

    args.max_len      = _parse_optional_int(args.max_len)
    args.max_ante_len = _parse_optional_int(args.max_ante_len)
    args.max_cons_len = _parse_optional_int(args.max_cons_len)

    max_len = args.max_len
    if max_len is None and args.max_ante_len is not None and args.max_cons_len is not None:
        max_len = args.max_ante_len + args.max_cons_len

    print(f"[{now_stamp()}] Pipeline started")
    print(f"Base path   : {args.base}")
    print(f"K_LEVELS    : {args.k_levels}")
    print(f"MIN_SUPPORT : {args.min_support}")
    print(f"MIN_CONF    : {args.min_conf}")
    print(f"MIN_LIFT    : {args.min_lift}")
    print(f"max_ante_len: {args.max_ante_len}")
    print(f"max_cons_len: {args.max_cons_len}")
    print(f"MAX_LEN     : {max_len}")
    print(f"candidate_pruning: {args.candidate_pruning}")
    print(f"rule_filtering   : {args.rule_filtering}")
    print()

    with timed_step("Loading data"):
        df = load_data(args.base)

    with timed_step("Cleaning data"):
        df_min = clean_data(df)

    with timed_step("Building transactions"):
        transactions, all_tokens = build_transactions(df_min, args.k_levels)
        path_map = {tok: label_of(tok) for tok in all_tokens}

    with timed_step("Encoding transactions"):
        df_encoded = encode_transactions(transactions, all_tokens)

    with timed_step("Building ancestor map"):
        # Expand leaf paths to all intermediate prefixes so that build_branch_ancestry
        # sees the full taxonomy tree (e.g. "Clothing", "Clothing > Tops", as well as
        # "Clothing > Tops > T-Shirts"), matching the behaviour of pipeline.py in
        # Bachelor_final which passes the complete categories table.
        _all_taxonomy_paths: set[str] = set()
        for path in df_min["category_name"].dropna().unique():
            parts = [p.strip() for p in str(path).split(">") if p.strip()]
            for i in range(1, len(parts) + 1):
                _all_taxonomy_paths.add(" > ".join(parts[:i]))
        branch_ancestry = fpc.build_branch_ancestry(
            pd.DataFrame({"category_name": sorted(_all_taxonomy_paths)}),
            name_col="category_name",
        )
        ancestors = build_ancestors_from_tokens(all_tokens, branch_ancestry=branch_ancestry)

    with timed_step("Mining rules"):
        rules_raw = mine_rules_raw(
            df_encoded,
            args.min_support,
            args.min_conf,
            max_len,
            ancestors=ancestors,
            max_ante_len=args.max_ante_len,
            max_cons_len=args.max_cons_len,
            require_single_consequent=False,
            path_map=path_map,
            branch_ancestry=branch_ancestry,
            candidate_pruning=args.candidate_pruning,
            rule_filtering=args.rule_filtering,
        )

    with timed_step("Scoring & deduplicating"):
        rules_scored = add_score_and_rank(rules_raw, args.min_support, args.min_lift).copy()
        print(f"After score+rank: {len(rules_scored)} rules")

        rules_scored["a_toks"] = rules_scored["antecedents"].apply(lambda s: tuple(sorted(s)))
        rules_scored["b_toks"] = rules_scored["consequents"].apply(lambda s: tuple(sorted(s)))
        rules_scored["a_lbl"]  = rules_scored["a_toks"].apply(lambda toks: " + ".join(label_of(t) for t in toks))
        rules_scored["b_lbl"]  = rules_scored["b_toks"].apply(lambda toks: " + ".join(label_of(t) for t in toks))
        rules_scored["a_levels"]   = rules_scored["a_toks"].apply(lambda toks: tuple(level_of(t) for t in toks))
        rules_scored["b_levels"]   = rules_scored["b_toks"].apply(lambda toks: tuple(level_of(t) for t in toks))
        rules_scored["a_branches"] = rules_scored["a_toks"].apply(lambda toks: tuple(branch_of(t) for t in toks))
        rules_scored["b_branches"] = rules_scored["b_toks"].apply(lambda toks: tuple(branch_of(t) for t in toks))

        rules_family  = dedupe_family(rules_scored)
        print(f"After family dedupe: {len(rules_family)} rules")
        rules_deduped = dedupe_antimirror(rules_family)
        print(f"After antimirror dedupe: {len(rules_deduped)} rules")
        rules_out     = pretty_rules(rules_deduped)

    print(f"\nFound {len(rules_out)} rules.\n")

    if args.output:
        with timed_step("Saving rules"):
            rules_out.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        print(rules_out.to_string(index=False))

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[{now_stamp()}] Pipeline finished in {format_elapsed(total_elapsed)}")


if __name__ == "__main__":
    main()
