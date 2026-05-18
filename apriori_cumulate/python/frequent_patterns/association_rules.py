# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# Function for generating association rules
#
# Author: Joshua Goerner <https://github.com/JoshuaGoerner>
#         Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from . import fpcommon as fpc

# ÆNDRING (Hierarki & Regelfiltrering)
# Valgfri hierarki-aware filtrering af associationsregler
# samt nye parametre til at styre regel-længder.
#
# Oversigt over ændringer:
# - Ny parameter `branch_ancestry` i `association_rules()`:
#     Accepterer en dictionary returneret af `fpc.build_branch_ancestry(taxonomy_df)`.
#     Filtrerer regler hvor en antecedent-tokens *branch-label* (B:xxx-segmentet)
#     er ancestor eller descendant af en consequent-tokens branch-label i den
#     rigtige taksonomi.  Dette fanger within-chain regler som
#     "Cosmetics > Makeup → Personal Care", der passerer den eksisterende
#     within-window ancestor-tjek fordi "Cosmetics" og "Personal Care" er
#     i separate K_LEVELS-vinduer, men stadig er direkte ancestor/descendant
#     i den underliggende taksonomi.
#
# - Ny parameter `ancestors` i `association_rules()`:
#     Accepterer en dictionary med ancestor-relationer.
#     Hvis givet, udelukkes regler hvor antecedent og consequent
#     indeholder et ancestor–descendant-par.
#
# - Ny parameter `max_ante_len` i `association_rules()`:
#     Sætter en øvre grænse for antecedentens længde.
#
# - Ny parameter `max_cons_len` i `association_rules()`:
#     Sætter en øvre grænse for consequentens længde.
#
# - Ny parameter `require_single_consequent` i `association_rules()`:
#     Hvis True, kræves det at consequenten kun indeholder ét element.
#
# - Ny hjælpefunktion `_rule_violates_hierarchy(antecedent, consequent)`:
#     Delegerer til `fpc.h_rule_violates_hierarchy()` og returnerer True,
#     hvis reglen krænker hierarkiet (ancestor–descendant-par på tværs
#     af antecedent og consequent).
#
# - Ny hjælpefunktion `_antecedent_sizes(m)`:
#     Generator der beregner gyldige antecedent-størrelser baseret på
#     `max_ante_len`, `max_cons_len` og `require_single_consequent`.
#     Erstatter den tidligere inline `range(len(k) - 1, 0, -1)`-løkke.
#
# - Minimumstjek på itemset-størrelse i hovedløkken:
#     Itemsets med færre end 2 elementer springes over med `if len(k) < 2`.
#
# - `rule_supports`-udpakning ændret fra eksplicitte variabeltildelinger
#     til tuple-udpakning med én linje.

_metrics = [
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "representativity",
    "leverage",
    "conviction",
    "zhangs_metric",
    "jaccard",
    "certainty",
    "kulczynski",
]


def association_rules(
    df: pd.DataFrame,
    num_itemsets: Optional[int] = 1,
    df_orig: Optional[pd.DataFrame] = None,
    null_values=False,
    metric="confidence",
    min_threshold=0.8,
    support_only=False,
    return_metrics: list = _metrics,
    ancestors: Optional[dict] = None,
    path_map: Optional[dict] = None,
    branch_ancestry: Optional[dict] = None,
    max_ante_len: Optional[int] = None,
    max_cons_len: Optional[int] = None,
    require_single_consequent: bool = False,
) -> pd.DataFrame:
    """Generates a DataFrame of association rules including the
    metrics 'score', 'confidence', and 'lift'
    """
    if null_values and df_orig is None:
        raise TypeError("If null values exist, df_orig must be provided.")

    if null_values and num_itemsets == 1:
        raise TypeError("If null values exist, num_itemsets must be provided.")

    fpc.valid_input_check(df_orig, null_values)

    if not df.shape[0]:
        raise ValueError(
            "The input DataFrame `df` containing the frequent itemsets is empty."
        )

    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError("Dataframe needs to contain the columns 'support' and 'itemsets'")

    def kulczynski_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        conf_AC = sAC * (num_itemsets - disAC) / (sA * (num_itemsets - disA) - dis_int)
        conf_CA = sAC * (num_itemsets - disAC) / (sC * (num_itemsets - disC) - dis_int_)
        kulczynski = (conf_AC + conf_CA) / 2
        return kulczynski

    def conviction_helper(conf, sC):
        conviction = np.empty(conf.shape, dtype=float)
        if not len(conviction.shape):
            conviction = conviction[np.newaxis]
            conf = conf[np.newaxis]
            sC = sC[np.newaxis]
        conviction[:] = np.inf
        conviction[conf < 1.0] = (1.0 - sC[conf < 1.0]) / (1.0 - conf[conf < 1.0])
        return conviction

    def zhangs_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        denominator = np.maximum(sAC * (1 - sA), sA * (sC - sAC))
        numerator = metric_dict["leverage"](
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            zhangs_metric = np.where(denominator == 0, 0, numerator / denominator)
        return zhangs_metric

    def jaccard_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        numerator = metric_dict["support"](
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        )
        denominator = sA + sC - numerator
        jaccard_metric = numerator / denominator
        return jaccard_metric

    def certainty_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        certainty_num = (
            metric_dict["confidence"](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)
            - sC
        )
        certainty_denom = 1 - sC
        cert_metric = np.where(certainty_denom == 0, 0, certainty_num / certainty_denom)
        return cert_metric

    metric_dict = {
        "antecedent support": lambda _, sA, ___, ____, _____, ______, _______, ________: sA,
        "consequent support": lambda _, __, sC, ____, _____, ______, _______, ________: sC,
        "support": lambda sAC, _, __, ___, ____, _____, ______, _______: sAC,
        "confidence": lambda sAC, sA, _, disAC, disA, __, dis_int, ___: (
            sAC * (num_itemsets - disAC)
        )
        / (sA * (num_itemsets - disA) - dis_int),
        "lift": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: metric_dict[
            "confidence"
        ](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)
        / sC,
        "representativity": lambda _, __, ___, disAC, ____, ______, _______, ________: (
            num_itemsets - disAC
        )
        / num_itemsets,
        "leverage": lambda sAC, sA, sC, _, __, ____, _____, ______: metric_dict[
            "support"
        ](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)
        - sA * sC,
        "conviction": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: conviction_helper(
            metric_dict["confidence"](
                sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
            ),
            sC,
        ),
        "zhangs_metric": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: zhangs_metric_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
        "jaccard": lambda sAC, sA, sC, _, __, ____, _____, ______: jaccard_metric_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
        "certainty": lambda sAC, sA, sC, _, __, ____, _____, ______: certainty_metric_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
        "kulczynski": lambda sAC, sA, sC, _, __, ____, _____, ______: kulczynski_helper(
            sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
        ),
    }

    if support_only:
        metric = "support"
    else:
        if metric not in metric_dict.keys():
            raise ValueError(
                "Metric must be one of {}, got '{}'".format(list(metric_dict.keys()), metric)
            )

    keys = df["itemsets"].values
    values = df["support"].values
    frozenset_vect = np.vectorize(
        lambda x: frozenset(
            int(item) if isinstance(item, np.generic) else item for item in x
        )
    )
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    if max_ante_len is not None and max_ante_len < 1:
        return pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)
    if max_cons_len is not None and max_cons_len < 1:
        return pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)

    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    if null_values:
        first_itemset = next(iter(frequent_items_dict.keys()))
        df_orig = df_orig.copy()
        disabled = df_orig.copy()
        disabled = np.where(pd.isna(disabled), 1, np.nan) + np.where(
            (disabled == 0) | (disabled == 1), np.nan, 0
        )
        disabled = pd.DataFrame(disabled)
        if all(isinstance(key, str) for key in first_itemset):
            disabled.columns = df_orig.columns

        if all(isinstance(key, (np.integer, int)) for key in first_itemset):
            cols = np.arange(0, len(df_orig.columns), 1)
            disabled.columns = cols
            df_orig = df_orig.rename(columns=dict(zip(df_orig.columns, cols)))

    def _rule_violates_hierarchy(antecedent: frozenset, consequent: frozenset) -> bool:
        # Ancestor–descendant cross-boundary check (antecedent × consequent)
        if fpc.h_rule_violates_hierarchy(antecedent, consequent, ancestors=ancestors):
            return True
        # Same canonical-path check (safety net — should already be pruned by apriori):
        # A rule like "Clothing (L0) → Clothing (L3)" is tautological.
        if path_map is not None:
            for a in antecedent:
                pa = path_map.get(a)
                if pa is None:
                    continue
                for c in consequent:
                    if path_map.get(c) == pa:
                        return True
        # Branch-level ancestry check:
        # Token branch labels (the B:xxx segment) can themselves be
        # ancestor/descendant in the real taxonomy even when the within-window
        # ancestor check passes.  E.g. branch "Cosmetics" is a child of branch
        # "Personal Care" — rules between tokens from these branches are
        # within-chain and filtered here.
        if branch_ancestry is not None:
            a_tokens = [a for a in antecedent if isinstance(a, str)]
            c_tokens = [c for c in consequent if isinstance(c, str)]
            if a_tokens and c_tokens:
                if fpc.h_rule_violates_branch_ancestry(a_tokens, c_tokens, branch_ancestry):
                    return True
        # Consequent-intern ancestor check:
        # Reglen {X} → {a, b} hvor a anc b er redundant med {X} → {b}
        # (samme confidence, da support({a,b}) = support(b)).
        # Filtrer sådanne regler her så de ikke fylder i output.
        if ancestors is not None and len(consequent) > 1:
            for c1 in consequent:
                anc_c1 = ancestors.get(c1, set())
                for c2 in consequent:
                    if c2 != c1 and c2 in anc_c1:
                        return True  # c1 er ancestor af c2 inden i consequenten
        # Antecedent-intern ancestor check (parallel to consequent check above):
        # {a, b} → {X} where a anc b is redundant with {b} → {X}.
        if ancestors is not None and len(antecedent) > 1:
            for a1 in antecedent:
                anc_a1 = ancestors.get(a1, set())
                for a2 in antecedent:
                    if a2 != a1 and a2 in anc_a1:
                        return True  # a1 er ancestor af a2 inden i antecedenten
        return False

    def _antecedent_sizes(m: int):
        Amax = (m - 1) if max_ante_len is None else min(max_ante_len, m - 1)
        Amin = 1
        if max_cons_len is not None:
            Amin = max(Amin, m - max_cons_len)
        if require_single_consequent:
            Amin = max(Amin, m - 1)
            Amax = min(Amax, m - 1)
        if Amin > Amax:
            return
        for r in range(Amin, Amax + 1):
            yield r

    for k in frequent_items_dict.keys():
        if len(k) < 2:
            continue
        sAC = frequent_items_dict[k]
        m = len(k)
        for idx in _antecedent_sizes(m) or []:
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)

                if _rule_violates_hierarchy(antecedent, consequent):
                    continue

                if support_only:
                    sA = None
                    sC = None
                    disAC, disA, disC, dis_int, dis_int_ = 0, 0, 0, 0, 0
                else:
                    try:
                        sA = frequent_items_dict[antecedent]
                        sC = frequent_items_dict[consequent]
                        if not null_values:
                            disAC, disA, disC, dis_int, dis_int_ = 0, 0, 0, 0, 0
                        else:
                            an = list(antecedent)
                            con = list(consequent)
                            an.extend(con)
                            dec = disabled.loc[:, an]
                            _dec = disabled.loc[:, list(antecedent)]
                            __dec = disabled.loc[:, list(consequent)]
                            dec_ = df_orig.loc[:, list(antecedent)]
                            dec__ = df_orig.loc[:, list(consequent)]
                            disAC, disA, disC, dis_int, dis_int_ = 0, 0, 0, 0, 0
                            for i in range(len(dec.index)):
                                item_comb = list(dec.iloc[i, :])
                                item_dis_an = list(_dec.iloc[i, :])
                                item_dis_con = list(__dec.iloc[i, :])
                                item_or_an = list(dec_.iloc[i, :])
                                item_or_con = list(dec__.iloc[i, :])
                                if 1 in set(item_comb):
                                    disAC += 1
                                if 1 in set(item_dis_an):
                                    disA += 1
                                if 1 in item_dis_con:
                                    disC += 1
                                if (1 in item_dis_con) and all(j == 1 for j in item_or_an):
                                    dis_int += 1
                                if (1 in item_dis_an) and all(j == 1 for j in item_or_con):
                                    dis_int_ += 1
                    except KeyError as e:
                        s = (
                            str(e)
                            + "You are likely getting this error because the DataFrame is missing antecedent and/or consequent information. You can try using the `support_only=True` option"
                        )
                        raise KeyError(s)

                conf = metric_dict["confidence"](
                    sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
                )
                if conf >= min_threshold:
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append(
                        [sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_]
                    )

    if not rule_supports:
        return pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)

    rule_supports = np.array(rule_supports).T.astype(float)
    df_res = pd.DataFrame(
        data=list(zip(rule_antecedents, rule_consequents)),
        columns=["antecedents", "consequents"],
    )

    if support_only:
        sAC = rule_supports[0]
        for mname in return_metrics:
            df_res[mname] = np.nan
        df_res["support"] = sAC
    else:
        sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_ = rule_supports
        for mname in return_metrics:
            df_res[mname] = metric_dict[mname](
                sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_
            )

    return df_res