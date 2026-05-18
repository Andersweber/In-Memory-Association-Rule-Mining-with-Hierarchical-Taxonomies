# Sebastian Raschka 2014-2026
# myxtend Machine Learning Library Extensions
# Based on Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from . import fpcommon as fpc


def _resolve_path_map_to_colidx(df, path_map, use_colnames):
    """Normalize path_map to be keyed by column indices."""
    if path_map is None:
        return None
    if all(isinstance(k, (int, np.integer)) for k in path_map.keys()):
        return {int(k): v for k, v in path_map.items()}
    if not use_colnames:
        raise ValueError("path_map is keyed by names, but use_colnames=False.")
    col_to_idx = {c: i for i, c in enumerate(df.columns)}
    return {col_to_idx[k]: v for k, v in path_map.items() if k in col_to_idx}


def _resolve_ancestors_to_colidx(df, ancestors, use_colnames):
    """Normalize ancestors to be keyed by column indices."""
    if ancestors is None:
        return None
    if all(isinstance(k, (int, np.integer)) for k in ancestors.keys()):
        return {int(k): set(int(x) for x in v) for k, v in ancestors.items()}
    if not use_colnames:
        raise ValueError("ancestors is keyed by names, but use_colnames=False.")
    col_to_idx = {c: i for i, c in enumerate(df.columns)}
    anc_idx = {}
    for child_name, ancs in ancestors.items():
        if child_name not in col_to_idx:
            continue
        child_idx = col_to_idx[child_name]
        anc_idx[child_idx] = {col_to_idx[a] for a in ancs if a in col_to_idx}
    return anc_idx


def _same_canonical_path(a, b, path_map):
    """Return True if two item indices represent the same taxonomy path."""
    if path_map is None:
        return False
    pa = path_map.get(int(a))
    pb = path_map.get(int(b))
    return pa is not None and pa == pb


def Generate_candidates_H(
    L,
    k,
    anc_idx=None,
    max_itemset_len=None,
    path_map=None,
):
    """Hierarchy-aware Apriori join step.

    Candidates containing ancestor-descendant pairs or duplicate canonical
    taxonomy paths are rejected before support counting.
    """
    if L is None or len(L) == 0:
        return np.empty((0, k), dtype=int)
    if max_itemset_len is not None and k > max_itemset_len:
        return np.empty((0, k), dtype=int)

    L = np.asarray(L, dtype=int)
    prev_set = {tuple(row) for row in L}

    def _sub_is_frequent(sub):
        return tuple(sorted(int(x) for x in sub)) in prev_set

    if k > 2:
        prefix_groups = defaultdict(list)
        for row in L:
            prefix_groups[tuple(row[:-1])].append(row)
        groups = prefix_groups.values()
    else:
        groups = [L]

    candidates = []
    for group in groups:
        group = list(group)
        for i, p in enumerate(group):
            for j in range(i + 1, len(group)):
                q = group[j]
                a = int(p[-1])
                b = int(q[-1])

                if anc_idx is not None:
                    if (b in anc_idx and a in anc_idx[b]) or (a in anc_idx and b in anc_idx[a]):
                        continue

                if path_map is not None:
                    if _same_canonical_path(a, b, path_map):
                        continue
                    if k > 2:
                        prefix_items = [int(x) for x in p[:-1]]
                        if any(
                            _same_canonical_path(new_item, px, path_map)
                            for new_item in (a, b)
                            for px in prefix_items
                        ):
                            continue

                prefix = [int(x) for x in p[:-1]]
                cand = tuple(prefix + ([a, b] if a < b else [b, a]))
                if max_itemset_len is not None and len(cand) > max_itemset_len:
                    continue
                if k == 2 or all(_sub_is_frequent(sub) for sub in itertools.combinations(cand, k - 1)):
                    candidates.append(cand)

    if not candidates:
        return np.empty((0, k), dtype=int)
    return np.array(sorted(set(candidates)), dtype=int)


def _matrix_from_df(df):
    if hasattr(df, "sparse"):
        return (df.values if df.size == 0 else df.sparse.to_coo().tocsc()), True
    return df.values, False


def _single_item_support(X, rows_count, is_sparse):
    if is_sparse:
        return np.array(X.sum(axis=0)).reshape(-1) / rows_count
    return np.sum(X, axis=0).reshape(-1) / rows_count


def _candidate_support_count(X, cand, is_sparse):
    if is_sparse:
        return int(X[:, cand].toarray().all(axis=1).sum())
    return int(X[:, cand].all(axis=1).sum())


def apriori_inmem(
    df,
    min_support=0.5,
    use_colnames=False,
    max_len=None,
    verbose=0,
    ancestors=None,
    path_map=None,
):
    """In-memory generalized Apriori for hierarchy-expanded transactions.

    The caller passes a hierarchy-expanded one-hot matrix that is already in
    memory. The matrix is reused across all Apriori passes. Candidate
    generation is hierarchy-aware, and support is counted directly over the
        in-memory sparse matrix without per-pass matrix rebuilds.
    """
    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive number within the interval `(0, 1]`. "
            f"Got {min_support}."
        )
    fpc.valid_input_check(df)

    X, is_sparse = _matrix_from_df(df)
    rows_count = float(X.shape[0])
    threshold = min_support * rows_count

    anc_idx = _resolve_ancestors_to_colidx(df, ancestors, use_colnames)
    path_map_idx = _resolve_path_map_to_colidx(df, path_map, use_colnames)

    support = _single_item_support(X, rows_count, is_sparse)
    ary_col_idx = np.arange(X.shape[1])
    frequent_1 = support >= min_support

    support_dict = {1: support[frequent_1]}
    itemset_dict = {1: ary_col_idx[frequent_1].reshape(-1, 1)}
    max_itemset = 1

    while max_itemset and max_itemset < (max_len or float("inf")):
        next_k = max_itemset + 1
        candidates = Generate_candidates_H(
            itemset_dict[max_itemset],
            next_k,
            anc_idx=anc_idx,
            max_itemset_len=max_len,
            path_map=path_map_idx,
        )
        if candidates.size == 0:
            break

        if verbose:
            print(
                "\rProcessing %d candidates | itemset size %d"
                % (len(candidates), next_k),
                end="",
            )

        kept_candidates = []
        kept_supports = []
        for cand in candidates:
            support_count = _candidate_support_count(X, cand, is_sparse)
            if support_count >= threshold:
                kept_candidates.append(cand)
                kept_supports.append(support_count / rows_count)

        if not kept_candidates:
            break

        itemset_dict[next_k] = np.array(kept_candidates, dtype=int)
        support_dict[next_k] = np.array(kept_supports, dtype=float)
        max_itemset = next_k

    all_res = []
    for k in sorted(itemset_dict):
        support_series = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")
        all_res.append(pd.concat((support_series, itemsets), axis=1))

    res_df = pd.concat(all_res)
    res_df.columns = ["support", "itemsets"]
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([mapping[i] for i in x])
        )

    if verbose:
        print()
    return res_df.reset_index(drop=True)