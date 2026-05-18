#pragma once
#include "fpcommon.hpp"
#include "apriori.hpp"   // for FrequentItemset, _same_canonical_path

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

// =========================================================
// association_rules.hpp — mirrors association_rules.py
// =========================================================

// ── Result row ────────────────────────────────────────────────────────────
struct AssociationRule {
    std::vector<ItemIdx> antecedent;
    std::vector<ItemIdx> consequent;

    double antecedent_support = 0.0;
    double consequent_support = 0.0;
    double support            = 0.0;
    double confidence         = 0.0;
    double lift               = 0.0;
    double leverage           = 0.0;
    double conviction         = 0.0;
    double zhangs_metric      = 0.0;
    double jaccard            = 0.0;
    double certainty          = 0.0;
    double kulczynski         = 0.0;
};

// ── Frequent-itemset lookup map ────────────────────────────────────────────
using FreqMap = std::map<std::vector<ItemIdx>, double>;

// ── _rule_violates_hierarchy ───────────────────────────────────────────────
// Mirrors _rule_violates_hierarchy(antecedent, consequent) in association_rules.py.
// Returns true if the rule should be filtered out.
inline bool _rule_violates_hierarchy(
    const std::vector<ItemIdx>& antecedent,
    const std::vector<ItemIdx>& consequent,
    const AncestorMap*          ancestors,
    const PathMap*              path_map,
    const BranchLabelMap*       branch_label_map = nullptr,
    const BranchAncestry*       branch_ancestry  = nullptr)
{
    // 1. Cross-boundary ancestor–descendant check (antecedent × consequent).
    //    Mirrors: if fpc.h_rule_violates_hierarchy(antecedent, consequent, ancestors=ancestors)
    if (h_rule_violates_hierarchy(antecedent, consequent, ancestors))
        return true;

    // 2. Same canonical-path check (safety net).
    //    Mirrors: if path_map.get(c) == pa: return True
    if (path_map) {
        for (ItemIdx a : antecedent)
            for (ItemIdx c : consequent)
                if (_same_canonical_path(a, c, path_map)) return true;
    }

    // 3. Consequent-internal ancestor: {X}→{a,b} where a is ancestor of b is
    //    redundant with {X}→{b} (same confidence).
    //    Mirrors: if c2 in anc_c1: return True
    if (ancestors && consequent.size() > 1) {
        for (ItemIdx c1 : consequent) {
            auto it = ancestors->find(c1);
            if (it == ancestors->end()) continue;
            const auto& anc_c1 = it->second;
            for (ItemIdx c2 : consequent)
                if (c2 != c1 && anc_c1.count(c2)) return true;
        }
    }

    // 4. Antecedent-internal ancestor: {a,b}→{X} where a is ancestor of b is
    //    redundant with {b}→{X} (same confidence).
    //    Mirrors: if a2 in anc_a1: return True
    if (ancestors && antecedent.size() > 1) {
        for (ItemIdx a1 : antecedent) {
            auto it = ancestors->find(a1);
            if (it == ancestors->end()) continue;
            const auto& anc_a1 = it->second;
            for (ItemIdx a2 : antecedent)
                if (a2 != a1 && anc_a1.count(a2)) return true;
        }
    }

    // 5. Branch-level ancestry check.
    //    Mirrors: if fpc.h_rule_violates_branch_ancestry(a_tokens, c_tokens, branch_ancestry)
    if (branch_label_map && branch_ancestry) {
        if (h_rule_violates_branch_ancestry(antecedent, consequent,
                                            *branch_label_map, *branch_ancestry))
            return true;
    }

    return false;
}

// ── _antecedent_sizes ─────────────────────────────────────────────────────
// Yields valid antecedent sizes r for a k-item itemset, respecting
// max_ante_len, max_cons_len, and require_single_consequent.
// Mirrors _antecedent_sizes(m) generator in association_rules.py.
inline std::vector<int> _antecedent_sizes(
    int m,
    std::optional<int> max_ante_len,
    std::optional<int> max_cons_len,
    bool require_single_consequent)
{
    int Amax = max_ante_len.has_value() ? std::min(*max_ante_len, m - 1) : m - 1;
    int Amin = 1;
    if (max_cons_len.has_value()) Amin = std::max(Amin, m - *max_cons_len);
    if (require_single_consequent) {
        Amin = std::max(Amin, m - 1);
        Amax = std::min(Amax, m - 1);
    }
    std::vector<int> out;
    for (int r = Amin; r <= Amax; ++r) out.push_back(r);
    return out;
}

// ── Combinations helper ────────────────────────────────────────────────────
// Generates all r-combinations of `items`.
// Mirrors itertools.combinations(k, r=idx) in association_rules.py.
// callback receives (chosen, rest) where chosen + rest = items.
inline void combinations_with_complement(
    const std::vector<ItemIdx>& items,
    int r,
    std::function<void(const std::vector<ItemIdx>&, const std::vector<ItemIdx>&)> callback)
{
    int n = static_cast<int>(items.size());
    std::vector<int> chosen_idx;
    chosen_idx.reserve(r);

    std::function<void(int, int)> gen = [&](int start, int remaining) {
        if (remaining == 0) {
            std::vector<ItemIdx> chosen, rest;
            std::set<int> chosen_set(chosen_idx.begin(), chosen_idx.end());
            for (int i = 0; i < n; ++i) {
                if (chosen_set.count(i)) chosen.push_back(items[i]);
                else                    rest.push_back(items[i]);
            }
            callback(chosen, rest);
            return;
        }
        for (int i = start; i <= n - remaining; ++i) {
            chosen_idx.push_back(i);
            gen(i + 1, remaining - 1);
            chosen_idx.pop_back();
        }
    };
    gen(0, r);
}

// ── association_rules ──────────────────────────────────────────────────────
// Mirrors association_rules() in association_rules.py.
inline std::vector<AssociationRule> association_rules(
    const FreqMap&         frequent_items,
    double                 min_confidence            = 0.8,
    const AncestorMap*     ancestors                 = nullptr,
    const PathMap*         path_map                  = nullptr,
    std::optional<int>     max_ante_len              = std::nullopt,
    std::optional<int>     max_cons_len              = std::nullopt,
    bool                   require_single_consequent = false,
    const BranchLabelMap*  branch_label_map          = nullptr,
    const BranchAncestry*  branch_ancestry           = nullptr)
{
    if (frequent_items.empty())
        return {};

    if (max_ante_len.has_value() && *max_ante_len < 1) return {};
    if (max_cons_len.has_value() && *max_cons_len < 1) return {};

    std::vector<AssociationRule> rules;

    // Mirrors: for k in frequent_items_dict.keys(): if len(k) < 2: continue
    for (const auto& [itemset_vec, sAC] : frequent_items) {
        int m = static_cast<int>(itemset_vec.size());
        if (m < 2) continue;

        const double sAC_val = sAC;

        // Mirrors: for idx in _antecedent_sizes(m): for c in combinations(k, r=idx)
        for (int r : _antecedent_sizes(m, max_ante_len, max_cons_len, require_single_consequent)) {
            combinations_with_complement(itemset_vec, r,
                [&](const std::vector<ItemIdx>& ante, const std::vector<ItemIdx>& cons)
            {
                // Mirrors: if _rule_violates_hierarchy(antecedent, consequent): continue
                if (_rule_violates_hierarchy(ante, cons, ancestors, path_map,
                                             branch_label_map, branch_ancestry))
                    return;

                // Look up sA and sC — mirrors: sA = frequent_items_dict[antecedent]
                auto it_a = frequent_items.find(ante);
                auto it_c = frequent_items.find(cons);
                if (it_a == frequent_items.end() || it_c == frequent_items.end())
                    return;

                double sA = it_a->second;
                double sC = it_c->second;

                if (sA <= 0.0) return;

                double confidence = sAC_val / sA;
                if (confidence < min_confidence) return;

                // ── Compute all metrics ────────────────────────────────────
                // Each mirrors the corresponding lambda in metric_dict.
                AssociationRule rule;
                rule.antecedent          = ante;
                rule.consequent          = cons;
                rule.antecedent_support  = sA;
                rule.consequent_support  = sC;
                rule.support             = sAC_val;
                rule.confidence          = confidence;
                rule.lift                = (sC > 0.0) ? confidence / sC : std::numeric_limits<double>::infinity();
                rule.leverage            = sAC_val - sA * sC;

                // conviction: (1-sC)/(1-conf); inf when conf==1; 0 when sC==1
                if (confidence >= 1.0)
                    rule.conviction = std::numeric_limits<double>::infinity();
                else if (sC >= 1.0)
                    rule.conviction = 0.0;
                else
                    rule.conviction = (1.0 - sC) / (1.0 - confidence);

                // Zhang's metric
                double denom_z = std::max(sAC_val * (1.0 - sA), sA * (sC - sAC_val));
                rule.zhangs_metric = (denom_z == 0.0) ? 0.0 : rule.leverage / denom_z;

                // Jaccard
                double denom_j = sA + sC - sAC_val;
                rule.jaccard = (denom_j == 0.0) ? 0.0 : sAC_val / denom_j;

                // Certainty
                double denom_cert = 1.0 - sC;
                rule.certainty = (denom_cert == 0.0) ? 0.0 : (confidence - sC) / denom_cert;

                // Kulczynski
                double conf_AC = confidence;
                double conf_CA = (sC > 0.0) ? sAC_val / sC : 0.0;
                rule.kulczynski = (conf_AC + conf_CA) / 2.0;

                rules.push_back(std::move(rule));
            });
        }
    }

    return rules;
}

// ── Build FreqMap from apriori() output ───────────────────────────────────
// Converts std::vector<FrequentItemset> to the FreqMap used above.
inline FreqMap build_freq_map(const std::vector<FrequentItemset>& freq_itemsets) {
    FreqMap m;
    for (const auto& fi : freq_itemsets)
        m[fi.items] = fi.support;
    return m;
}
