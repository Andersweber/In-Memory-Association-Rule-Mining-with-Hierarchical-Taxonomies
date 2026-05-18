#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>
#include <string>
#include <optional>
#include <sstream>

// =========================================================
// fpcommon.hpp — mirrors fpcommon.py
// =========================================================

using ItemIdx      = int;
using AncestorMap  = std::unordered_map<ItemIdx, std::unordered_set<ItemIdx>>;
using PathMap      = std::unordered_map<ItemIdx, std::string>;
using BranchLabelMap = std::unordered_map<ItemIdx, std::string>;                       // item_idx  -> branch label string
using BranchAncestry = std::unordered_map<std::string, std::unordered_set<std::string>>; // short_label -> set of full taxonomy paths

// ---------------------------------------------------------------------------
// precompute_ancestors
//   P: parent function (returns -1 for root / no parent)
// ---------------------------------------------------------------------------
inline AncestorMap precompute_ancestors(
    const std::vector<ItemIdx>& universe,
    std::function<ItemIdx(ItemIdx)> parent_fn)
{
    AncestorMap anc;
    for (ItemIdx x : universe) {
        std::unordered_set<ItemIdx> seen;
        ItemIdx y = parent_fn(x);
        while (y != -1 && seen.find(y) == seen.end()) {
            seen.insert(y);
            y = parent_fn(y);
        }
        anc[x] = std::move(seen);
    }
    return anc;
}

// ---------------------------------------------------------------------------
// h_rule_violates_hierarchy
//   Returns true iff any cross-boundary ancestor-descendant pair exists
//   between antecedent A and consequent B.
// ---------------------------------------------------------------------------
inline bool h_rule_violates_hierarchy(
    const std::vector<ItemIdx>& A,
    const std::vector<ItemIdx>& B,
    const AncestorMap* ancestors)
{
    if (!ancestors) return false;

    for (ItemIdx a : A) {
        auto it_a = ancestors->find(a);
        for (ItemIdx b : B) {
            auto it_b = ancestors->find(b);
            if (it_b != ancestors->end() && it_b->second.count(a)) return true; // a is ancestor of b
            if (it_a != ancestors->end() && it_a->second.count(b))   return true; // b is ancestor of a
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// build_branch_ancestry
//   Builds a map of short_label -> set of full taxonomy paths.
//   Input: any collection of full '>' separated category path strings
//          (e.g. "Health & Beauty > Personal Care > Cosmetics").
//   Mirrors fpc.build_branch_ancestry() in fpcommon.py.
// ---------------------------------------------------------------------------
inline BranchAncestry build_branch_ancestry(const std::vector<std::string>& category_paths) {
    BranchAncestry mapping;
    for (const auto& name : category_paths) {
        if (name.empty()) continue;
        // Split by '>' and trim each segment; the last non-empty segment is the short label.
        std::string last_seg;
        std::istringstream ss(name);
        std::string seg;
        while (std::getline(ss, seg, '>')) {
            auto l = seg.find_first_not_of(" \t\r\n");
            auto r = seg.find_last_not_of(" \t\r\n");
            if (l != std::string::npos)
                last_seg = seg.substr(l, r - l + 1);
        }
        if (!last_seg.empty())
            mapping[last_seg].insert(name);
    }
    return mapping;
}

// ---------------------------------------------------------------------------
// h_branches_are_related
//   Returns true if branch_a and branch_b are in an ancestor/descendant
//   relationship in the real taxonomy.
//   Uses a ' > '-terminated prefix check to avoid false positives such as
//   "Clothing" matching "Clothing Accessories".
//   Mirrors fpc.h_branches_are_related() in fpcommon.py.
// ---------------------------------------------------------------------------
inline bool h_branches_are_related(
    const std::string& branch_a,
    const std::string& branch_b,
    const BranchAncestry& label_to_paths)
{
    if (branch_a == branch_b) return false;

    auto it_a = label_to_paths.find(branch_a);
    auto it_b = label_to_paths.find(branch_b);
    if (it_a == label_to_paths.end() || it_b == label_to_paths.end()) return false;
    for (const auto& pa : it_a->second) {
        std::string pa_norm = pa + " > ";
        for (const auto& pb : it_b->second) {
            std::string pb_norm = pb + " > ";
            // pa_norm.startswith(pb_norm) or pb_norm.startswith(pa_norm)
            if (pa_norm.size() >= pb_norm.size() &&
                pa_norm.compare(0, pb_norm.size(), pb_norm) == 0) return true;
            if (pb_norm.size() >= pa_norm.size() &&
                pb_norm.compare(0, pa_norm.size(), pa_norm) == 0) return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// h_rule_violates_branch_ancestry
//   Returns true if any token in A has a branch label that is
//   ancestor/descendant of any branch label of a token in B.
//   Uses BranchLabelMap (ItemIdx -> branch label) instead of extracting
//   branch labels from raw token strings, because the C++ pipeline works
//   entirely with integer column indices rather than string tokens.
//   Mirrors fpc.h_rule_violates_branch_ancestry() in fpcommon.py.
// ---------------------------------------------------------------------------
inline bool h_rule_violates_branch_ancestry(
    const std::vector<ItemIdx>& A,
    const std::vector<ItemIdx>& B,
    const BranchLabelMap&       branch_label_map,
    const BranchAncestry&       branch_ancestry)
{
    for (ItemIdx a : A) {
        auto it_a = branch_label_map.find(a);
        if (it_a == branch_label_map.end()) continue;
        for (ItemIdx b : B) {
            auto it_b = branch_label_map.find(b);
            if (it_b == branch_label_map.end()) continue;
            if (h_branches_are_related(it_a->second, it_b->second, branch_ancestry))
                return true;
        }
    }
    return false;
}
