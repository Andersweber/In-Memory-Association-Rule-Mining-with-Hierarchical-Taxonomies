#pragma once
#include "fpcommon.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>
#include <optional>
#include <set>
#include <map>
#include <stdexcept>
#include <tuple>
#include <iostream>
#include <iterator>

// =========================================================
// apriori.hpp — mirrors apriori_2.py
// =========================================================

// ── Types ──────────────────────────────────────────────────────────────────
using Itemset   = std::vector<ItemIdx>;      // always kept sorted
using Row       = std::vector<bool>;
using Matrix    = std::vector<Row>;          // shape: [n_transactions][n_items]

// ── Helper: _same_canonical_path ───────────────────────────────────────────
// Mirrors _same_canonical_path(a, b, path_map) in apriori_2.py.
// Returns True if items a and b share the same canonical category-path string.
inline bool _same_canonical_path(ItemIdx a, ItemIdx b, const PathMap* path_map) {
    if (!path_map) return false;
    auto it_a = path_map->find(a);
    auto it_b = path_map->find(b);
    if (it_a == path_map->end() || it_b == path_map->end()) return false;
    return it_a->second == it_b->second;
}

// ── Support counting ───────────────────────────────────────────────────────
// Returns fraction of transactions that contain ALL items in `itemset`.
inline double compute_support(const Matrix& X, const Itemset& itemset) {
    int count = 0;
    for (const auto& row : X) {
        bool ok = true;
        for (ItemIdx idx : itemset) {
            if (!row[idx]) { ok = false; break; }
        }
        if (ok) ++count;
    }
    return static_cast<double>(count) / static_cast<double>(X.size());
}

inline int compute_support_count(const Matrix& X, const Itemset& itemset) {
    int count = 0;
    for (const auto& row : X) {
        bool ok = true;
        for (ItemIdx idx : itemset) {
            if (!row[idx]) { ok = false; break; }
        }
        if (ok) ++count;
    }
    return count;
}

// =========================================================
// Generate_candidates_H
//   Hierarchy-aware Apriori join step.
//   Mirrors Generate_candidates_H() in apriori_2.py.
//
//   L   : sorted frequent (k-1)-itemsets (each row already sorted ascending)
//   k   : target candidate size
// =========================================================
inline std::vector<Itemset> Generate_candidates_H(
    const std::vector<Itemset>& L,
    int k,
    const AncestorMap* anc_idx              = nullptr,
    std::optional<int> max_itemset_len      = std::nullopt,
    const PathMap*     path_map             = nullptr)
{

    if (L.empty()) return {};
    if (max_itemset_len.has_value() && k > *max_itemset_len) return {};

    // Build set of (k-1)-itemsets for subset-frequency checks.
    // Mirrors: prev_set = {tuple(row) for row in L}
    std::set<Itemset> prev_set(L.begin(), L.end());

    // Subset-frequency check: all (k-1)-subsets must be in prev_set.
    // Mirrors: def _sub_is_frequent(sub): return tuple(sorted(...)) in prev_set
    auto sub_is_frequent = [&](const Itemset& sub) -> bool {
        return prev_set.count(sub) > 0;
    };

    // ── Prefix-grouped join ────────────────────────────────────────────────
    // Group L by the (k-2)-element prefix so only rows sharing a prefix are
    // paired. For k==2 prefix length is 0 → single group.
    // Mirrors: defaultdict(list) prefix_groups / groups = [L_join]
    using PrefixKey = std::vector<ItemIdx>;
    std::map<PrefixKey, std::vector<const Itemset*>> prefix_groups;

    if (k > 2) {
        for (const auto& row : L) {
            PrefixKey pfx(row.begin(), row.begin() + (int)row.size() - 1);
            prefix_groups[pfx].push_back(&row);
        }
    } else {
        PrefixKey empty;
        for (const auto& row : L)
            prefix_groups[empty].push_back(&row);
    }

    std::set<Itemset> C_set; // set for deduplication, mirrors sorted(set(C))

    for (const auto& [pfx, group] : prefix_groups) {
        for (int i = 0; i < (int)group.size(); ++i) {
            for (int j = i + 1; j < (int)group.size(); ++j) { // j > i: each pair once
                const Itemset& p = *group[i];
                const Itemset& q = *group[j];

                ItemIdx a = p.back();
                ItemIdx b = q.back();

                // ── Early abandon: ancestor–descendant check for ANY k ─────
                // Mirrors: if (b in anc_idx and a in anc_idx[b]) or (a in anc_idx and b in anc_idx[a])
                // Any itemset with an ancestor–descendant pair produces rules
                // redundant with its ancestor-free subset — same support and confidence.
                if (anc_idx) {
                    bool b_anc_a = (anc_idx->count(a) && anc_idx->at(a).count(b));
                    bool a_anc_b = (anc_idx->count(b) && anc_idx->at(b).count(a));
                    if (a_anc_b || b_anc_a) continue;
                }

                // ── Same-path early abandon ────────────────────────────────
                // Mirrors: if _same_canonical_path(a, b, path_map): continue
                // For k>2 also check new items against all prefix elements.
                if (path_map) {
                    if (_same_canonical_path(a, b, path_map)) continue;
                    if (k > 2) {
                        bool bad = false;
                        for (ItemIdx px : pfx) {
                            if (_same_canonical_path(a, px, path_map) ||
                                _same_canonical_path(b, px, path_map)) {
                                bad = true; break;
                            }
                        }
                        if (bad) continue;
                    }
                }

                // ── Build candidate c = prefix + {a, b} sorted ────────────
                // Mirrors: c = tuple(prefix + ([a, b] if a < b else [b, a]))
                Itemset c = pfx;
                if (a < b) { c.push_back(a); c.push_back(b); }
                else       { c.push_back(b); c.push_back(a); }

                if (max_itemset_len.has_value() && (int)c.size() > *max_itemset_len)
                    continue;

                // ── Apriori subset-frequency check ─────────────────────────
                // Mirrors: if k == 2 or all(_sub_is_frequent(sub) for sub in combinations(c, k-1))
                // For k==2 singleton subsets are trivially frequent.
                if (k > 2) {
                    bool all_freq = true;
                    for (int om = 0; om < k; ++om) {
                        Itemset sub;
                        sub.reserve(k - 1);
                        for (int s = 0; s < k; ++s)
                            if (s != om) sub.push_back(c[s]);
                        if (!sub_is_frequent(sub)) { all_freq = false; break; }
                    }
                    if (!all_freq) continue;
                }

                C_set.insert(c);
            }
        }
    }

    return std::vector<Itemset>(C_set.begin(), C_set.end());
}

// =========================================================
// FrequentItemset — result entry
// =========================================================
struct FrequentItemset {
    Itemset items;
    double  support;
};

// =========================================================
// apriori
//   Mirrors apriori() in apriori_2.py.
//
//   X            : boolean transaction matrix [n_transactions][n_items]
//   min_support  : minimum support threshold in (0, 1]
//   max_len      : optional cap on itemset length (nullopt = unlimited)
//   ancestors    : optional hierarchy ancestor map (column-index keyed)
//   path_map     : optional same-path map (column-index keyed)
//   verbose      : print progress if true
// =========================================================
inline std::vector<FrequentItemset> apriori(
    const Matrix&           X,
    double                  min_support         = 0.5,
    std::optional<int>      max_len             = std::nullopt,
    const AncestorMap*      ancestors           = nullptr,
    const PathMap*          path_map            = nullptr,
    bool                    verbose             = false)
{
    if (min_support <= 0.0 || min_support > 1.0)
        throw std::invalid_argument("`min_support` must be in (0, 1]");

    const int n_rows  = static_cast<int>(X.size());
    const int n_items = n_rows > 0 ? static_cast<int>(X[0].size()) : 0;

    // ── Frequent 1-itemsets ────────────────────────────────────────────────
    // Mirrors: support = _support(X, X.shape[0], is_sparse)
    //          itemset_dict = {1: ary_col_idx[support >= min_support]}
    std::vector<Itemset>  itemset_dict_1;
    std::vector<double>   support_dict_1;

    for (int col = 0; col < n_items; ++col) {
        int cnt = 0;
        for (int r = 0; r < n_rows; ++r) cnt += X[r][col] ? 1 : 0;
        double sup = static_cast<double>(cnt) / n_rows;
        if (sup >= min_support) {
            itemset_dict_1.push_back({col});
            support_dict_1.push_back(sup);
        }
    }

    std::map<Itemset, double> freq_map;
    for (int i = 0; i < (int)itemset_dict_1.size(); ++i)
        freq_map[itemset_dict_1[i]] = support_dict_1[i];

    std::vector<FrequentItemset> result;
    for (int i = 0; i < (int)itemset_dict_1.size(); ++i)
        result.push_back({itemset_dict_1[i], support_dict_1[i]});

    std::vector<Itemset> current_L = itemset_dict_1;
    int max_itemset = 1;

    // Mirrors: while max_itemset and max_itemset < (max_len or float("inf"))
    while (!current_L.empty() && (!max_len.has_value() || max_itemset < *max_len)) {
        int k = max_itemset + 1;

        // Generate hierarchy-aware candidates from the previous frequent layer.
        std::vector<Itemset> candidates = Generate_candidates_H(
            current_L, k, ancestors, max_len, path_map);

        if (candidates.empty()) break;

        if (verbose)
            std::cout << "\rProcessing " << candidates.size()
                      << " combinations | Sampling itemset size " << k << std::flush;

        // Count support for each candidate from the in-memory matrix.
        std::vector<Itemset> next_L;
        for (const Itemset& cand : candidates) {
            double sup = compute_support(X, cand);
            if (sup >= min_support) {
                next_L.push_back(cand);
                freq_map[cand] = sup;
                result.push_back({cand, sup});
            }
        }

        if (next_L.empty()) break;
        current_L = std::move(next_L);
        max_itemset = k;
    }

    if (verbose) std::cout << '\n';
    return result;
}

// =========================================================
// build_tprime / cumulate_apriori
//   Literal Cumulate-style Apriori matching
//   python/frequent_patterns_2/cumulate_apriori.py.
// =========================================================
inline std::unordered_set<ItemIdx> build_tprime(
    const Itemset& raw_transaction,
    const std::unordered_set<ItemIdx>& candidate_items,
    const AncestorMap& expansion_ancestors)
{
    std::unordered_set<ItemIdx> t_prime;
    for (ItemIdx leaf : raw_transaction) {
        if (candidate_items.count(leaf)) t_prime.insert(leaf);
        auto it = expansion_ancestors.find(leaf);
        if (it == expansion_ancestors.end()) continue;
        for (ItemIdx anc : it->second)
            if (candidate_items.count(anc)) t_prime.insert(anc);
    }
    return t_prime;
}

inline std::vector<FrequentItemset> cumulate_apriori(
    const std::vector<Itemset>& raw_transactions,
    int n_items,
    double min_support,
    const AncestorMap& expansion_ancestors,
    const AncestorMap* pruning_ancestors = nullptr,
    const PathMap* path_map = nullptr,
    std::optional<int> max_len = std::nullopt,
    bool verbose = false)
{
    if (min_support <= 0.0 || min_support > 1.0)
        throw std::invalid_argument("`min_support` must be in (0, 1]");

    const int n_rows = static_cast<int>(raw_transactions.size());
    if (n_rows == 0 || n_items <= 0) return {};

    const double threshold = min_support * static_cast<double>(n_rows);
    const AncestorMap* candidate_ancestors =
        pruning_ancestors ? pruning_ancestors : &expansion_ancestors;

    std::vector<Itemset> current_L;
    std::vector<FrequentItemset> result;

    std::unordered_set<ItemIdx> candidate_items_1;
    candidate_items_1.reserve(n_items);
    for (ItemIdx i = 0; i < n_items; ++i) candidate_items_1.insert(i);

    std::vector<int> item_counts(n_items, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (const Itemset& tx : raw_transactions) {
        auto t_prime = build_tprime(tx, candidate_items_1, expansion_ancestors);
        for (ItemIdx item : t_prime) {
            if (item >= 0 && item < n_items) ++item_counts[item];
        }
    }

    for (ItemIdx idx = 0; idx < n_items; ++idx) {
        if (static_cast<double>(item_counts[idx]) >= threshold) {
            current_L.push_back({idx});
            result.push_back({{idx}, static_cast<double>(item_counts[idx]) / n_rows});
        }
    }

    if (verbose) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::cout << "Pass 1: " << n_items << " items scanned | "
                  << current_L.size() << " frequent | " << ms << " ms\n";
    }

    int max_itemset = 1;
    while (!current_L.empty() && (!max_len.has_value() || max_itemset < *max_len)) {
        int k = max_itemset + 1;
        t0 = std::chrono::high_resolution_clock::now();

        std::vector<Itemset> candidates = Generate_candidates_H(
            current_L, k, candidate_ancestors, max_len, path_map);
        if (candidates.empty()) break;

        std::unordered_set<ItemIdx> candidate_items_k;
        for (const Itemset& cand : candidates)
            for (ItemIdx item : cand)
                candidate_items_k.insert(item);

        std::vector<int> counts(candidates.size(), 0);
        std::size_t tprime_len_sum = 0;
        std::size_t tprime_len_max = 0;

        for (const Itemset& tx : raw_transactions) {
            auto t_prime = build_tprime(tx, candidate_items_k, expansion_ancestors);
            tprime_len_sum += t_prime.size();
            tprime_len_max = std::max(tprime_len_max, t_prime.size());
            if (t_prime.empty()) continue;

            for (std::size_t i = 0; i < candidates.size(); ++i) {
                bool subset = true;
                for (ItemIdx item : candidates[i]) {
                    if (!t_prime.count(item)) {
                        subset = false;
                        break;
                    }
                }
                if (subset) ++counts[i];
            }
        }

        std::vector<Itemset> next_L;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (static_cast<double>(counts[i]) >= threshold) {
                next_L.push_back(candidates[i]);
                result.push_back({candidates[i], static_cast<double>(counts[i]) / n_rows});
            }
        }

        if (verbose) {
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            double avg_tprime = raw_transactions.empty()
                ? 0.0
                : static_cast<double>(tprime_len_sum) / raw_transactions.size();
            std::cout << "Pass " << k << ": " << candidates.size()
                      << " candidates | " << candidate_items_k.size()
                      << " candidate items | avg |t'| " << avg_tprime
                      << " | max |t'| " << tprime_len_max
                      << " | " << ms << " ms\n";
        }

        if (next_L.empty()) break;
        current_L = std::move(next_L);
        max_itemset = k;
    }

    return result;
}
