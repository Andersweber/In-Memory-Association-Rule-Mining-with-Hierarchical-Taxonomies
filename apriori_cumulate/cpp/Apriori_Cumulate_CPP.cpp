// ==========================================================================
// main_bitset_merged.cpp — C++ pipeline reading pre-merged wishlist_categories_merged.parquet
//
// Mirrors (in order):
//   Cell 1  : Config constants
//   Cell 3  : Data loading & cleaning (CSV instead of merged DataFrame)
//   Cell 4  : Tokenization — make_level_tokens, parse_token, build_ancestors_from_tokens
//   Cell 5  : build_transactions, encode_transactions -> bool matrix
//   Cell 6  : mine_rules_raw  (apriori + association_rules)
//   Cell 7  : add_score_and_rank, dedupe_family, dedupe_antimirror, pretty_rules
//   Cell 8  : Pipeline orchestration + CSV output
// ==========================================================================

#include "association_rules.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <filesystem>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

// ── Trim helper ────────────────────────────────────────────────────────────
static std::string trim(std::string s) {
    auto l = s.find_first_not_of(" \t\r\n\"");
    auto r = s.find_last_not_of(" \t\r\n\"");
    return (l == std::string::npos) ? "" : s.substr(l, r - l + 1);
}

// =========================================================
// ── CELL 4 : Tokenization ─────────────────────────────────
// =========================================================

static const char SEP = '>';

// split_path: split taxonomy path by '>'
static std::vector<std::string> split_path(const std::string& path) {
    std::vector<std::string> parts;
    std::istringstream ss(path);
    std::string tok;
    while (std::getline(ss, tok, SEP)) {
        std::string t = trim(tok);
        if (!t.empty()) parts.push_back(t);
    }
    return parts;
}

// make_level_tokens: mirrors make_level_tokens(path, k_levels)
// L0 = leaf (most specific), L(k-1) = top ancestor in window
static std::vector<std::string> make_level_tokens(const std::string& path, int k_levels) {
    if (k_levels <= 0) return {};
    auto parts = split_path(path);
    if (parts.empty()) return {};

    // window = last k_levels elements
    int start = (int)parts.size() > k_levels ? (int)parts.size() - k_levels : 0;
    std::vector<std::string> window(parts.begin() + start, parts.end());
    if (window.empty()) return {};

    std::string branch = window[0];
    int max_k = (int)window.size();

    std::vector<std::string> out;
    for (int lvl = 0; lvl < max_k; ++lvl) {
        int upto = max_k - lvl;
        std::string label;
        for (int i = 0; i < upto; ++i) {
            if (i) label += " > ";
            label += window[i];
        }
        out.push_back("L" + std::to_string(lvl) + "|B:" + branch + "|" + label);
    }
    return out;
}

struct ParsedToken { int level; std::string branch; std::string label; };

// parse_token: mirrors parse_token(tok)
static ParsedToken parse_token(const std::string& tok) {
    // Pattern: ^L(\d+)\|B:([^|]+)\|(.*)$
    static const std::regex TOKEN_RE(R"(^L(\d+)\|B:([^|]+)\|(.*)$)");
    std::smatch m;
    if (std::regex_match(tok, m, TOKEN_RE))
        return { std::stoi(m[1]), trim(m[2]), trim(m[3]) };
    return { 0, "__UNK__", tok };
}

// direct_parent_token_of_leaf
static std::string direct_parent_token_of_leaf(const std::string& tok) {
    auto pt = parse_token(tok);
    if (pt.level != 0) return "";
    auto parts = split_path(pt.label);
    if (parts.size() <= 1) return "";
    std::string parent_lbl;
    for (int i = 0; i < (int)parts.size() - 1; ++i) {
        if (i) parent_lbl += " > ";
        parent_lbl += parts[i];
    }
    return "L1|B:" + pt.branch + "|" + parent_lbl;
}

// build_ancestors_from_tokens: mirrors build_ancestors_from_tokens(tokens)
static std::unordered_map<std::string, std::unordered_set<std::string>>
build_ancestors_from_tokens(
    const std::vector<std::string>& tokens,
    const BranchAncestry* branch_ancestry = nullptr) {
    std::unordered_set<std::string> tok_set(tokens.begin(), tokens.end());

    // (branch, label) -> set of tokens  (multiple levels can share same branch+label)
    std::map<std::pair<std::string,std::string>, std::unordered_set<std::string>> by_key;
    // leaf node name -> set of tokens  (for cross-branch lookup)
    std::unordered_map<std::string, std::unordered_set<std::string>> leaf_to_tokens;

    for (const auto& t : tokens) {
        auto pt = parse_token(t);
        by_key[{pt.branch, pt.label}].insert(t);
        auto parts = split_path(pt.label);
        if (!parts.empty())
            leaf_to_tokens[parts.back()].insert(t);
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> ancestors;
    for (const auto& t : tokens) ancestors[t] = {};

    for (const auto& t : tokens) {
        auto pt = parse_token(t);
        auto parts = split_path(pt.label);

        // 1) Intra-branch: strict label prefixes within same branch
        for (int j = 1; j < (int)parts.size(); ++j) {
            std::string pref_lbl;
            for (int i = 0; i < j; ++i) {
                if (i) pref_lbl += " > ";
                pref_lbl += parts[i];
            }
            auto it = by_key.find({pt.branch, pref_lbl});
            if (it == by_key.end()) continue;
            for (const auto& anc_tok : it->second)
                if (tok_set.count(anc_tok) && anc_tok != t)
                    ancestors[t].insert(anc_tok);
        }

        // 2) Cross-branch: intermediate path components matching tokens in other branches.
        // K-level windows can represent a broader taxonomy node with the same
        // number of path components as a descendant window, e.g.
        // "Toys & Games > Toys" vs. "Toys > Dolls, Playsets & Toy Figures".
        // Therefore matching intermediate segment names are treated as
        // ancestor-like tokens without requiring a shorter token label.
        for (int j = 0; j < (int)parts.size() - 1; ++j) {
            auto it = leaf_to_tokens.find(parts[j]);
            if (it == leaf_to_tokens.end()) continue;
            for (const auto& anc_tok : it->second) {
                if (!tok_set.count(anc_tok) || anc_tok == t) continue;
                ancestors[t].insert(anc_tok);
            }
        }
    }

    // 3) Cross-branch top-of-window linking via branch_ancestry.
    // Mirrors Python's build_ancestors_from_tokens(..., branch_ancestry=...).
    if (branch_ancestry) {
        std::unordered_map<std::string, std::unordered_set<std::string>> branch_to_root_toks;
        for (const auto& t : tokens) {
            auto pt = parse_token(t);
            if (split_path(pt.label).size() == 1)
                branch_to_root_toks[pt.branch].insert(t);
        }

        for (const auto& t2 : tokens) {
            auto pt2 = parse_token(t2);
            auto paths2_it = branch_ancestry->find(pt2.branch);
            if (paths2_it == branch_ancestry->end() || paths2_it->second.empty())
                continue;

            for (const auto& [branch1, root_toks1] : branch_to_root_toks) {
                if (branch1 == pt2.branch) continue;
                auto paths1_it = branch_ancestry->find(branch1);
                if (paths1_it == branch_ancestry->end() || paths1_it->second.empty())
                    continue;

                bool related_parent = false;
                for (const auto& pa : paths1_it->second) {
                    const std::string prefix = pa + " > ";
                    for (const auto& pb : paths2_it->second) {
                        if (pb.rfind(prefix, 0) == 0) {
                            related_parent = true;
                            break;
                        }
                    }
                    if (related_parent) break;
                }

                if (related_parent)
                    ancestors[t2].insert(root_toks1.begin(), root_toks1.end());
            }
        }
    }
    return ancestors;
}

// Expansion ancestors for Cumulate t_prime building: only within-branch,
// within-window prefixes. Cross-branch links are still used for pruning via
// build_ancestors_from_tokens(), but not injected into temporary transactions.
static std::unordered_map<std::string, std::unordered_set<std::string>>
build_expansion_ancestors_from_tokens(const std::vector<std::string>& tokens) {
    std::unordered_set<std::string> tok_set(tokens.begin(), tokens.end());
    std::map<std::pair<std::string,std::string>, std::unordered_set<std::string>> by_key;

    for (const auto& t : tokens) {
        auto pt = parse_token(t);
        by_key[{pt.branch, pt.label}].insert(t);
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> ancestors;
    for (const auto& t : tokens) ancestors[t] = {};

    for (const auto& t : tokens) {
        auto pt = parse_token(t);
        auto parts = split_path(pt.label);
        for (int j = 1; j < (int)parts.size(); ++j) {
            std::string pref_lbl;
            for (int i = 0; i < j; ++i) {
                if (i) pref_lbl += " > ";
                pref_lbl += parts[i];
            }
            auto it = by_key.find({pt.branch, pref_lbl});
            if (it == by_key.end()) continue;
            for (const auto& anc_tok : it->second)
                if (tok_set.count(anc_tok) && anc_tok != t)
                    ancestors[t].insert(anc_tok);
        }
    }
    return ancestors;
}

// =========================================================
// ── CELL 5 : Transaction building & encoding ───────────────
// =========================================================

struct RawRow { std::string wishlist_id; std::string category_name; };

static std::vector<std::vector<std::string>>
build_transactions(const std::vector<RawRow>& rows, int k_levels) {
    std::map<std::string, std::vector<std::string>> basket_map;
    std::map<std::string, std::unordered_set<std::string>> seen;

    for (const auto& row : rows) {
        if (row.wishlist_id.empty() || row.category_name.empty()) continue;
        auto toks = make_level_tokens(row.category_name, k_levels);
        for (const auto& tok : toks) {
            if (seen[row.wishlist_id].insert(tok).second)
                basket_map[row.wishlist_id].push_back(tok);
        }
    }

    std::vector<std::vector<std::string>> transactions;
    transactions.reserve(basket_map.size());
    for (auto& [wid, items] : basket_map)
        transactions.push_back(std::move(items));
    return transactions;
}

struct CumulateTransactions {
    std::vector<std::vector<std::string>> raw_leaf_transactions;
    std::vector<std::string> all_tokens;
};

struct EncodedData {
    Matrix X;
    std::vector<std::string> col_names;
    std::unordered_map<std::string, int> col_idx;
};

struct BitsetEncodedData {
    std::vector<std::vector<std::uint64_t>> item_bits; // item -> transaction bitset
    std::vector<int> item_counts;
    std::vector<std::string> col_names;
    std::unordered_map<std::string, int> col_idx;
    int n_rows = 0;
    int n_blocks = 0;
};

static CumulateTransactions
build_cumulate_transactions(const std::vector<RawRow>& rows, int k_levels) {
    std::map<std::string, std::vector<std::string>> basket_map;
    std::map<std::string, std::unordered_set<std::string>> seen_leaf;
    std::set<std::string> all_tokens;

    for (const auto& row : rows) {
        if (row.wishlist_id.empty() || row.category_name.empty()) continue;
        auto toks = make_level_tokens(row.category_name, k_levels);
        if (toks.empty()) continue;

        for (const auto& tok : toks) all_tokens.insert(tok);

        const std::string& leaf = toks.front();
        if (seen_leaf[row.wishlist_id].insert(leaf).second)
            basket_map[row.wishlist_id].push_back(leaf);
    }

    std::vector<std::vector<std::string>> transactions;
    transactions.reserve(basket_map.size());
    for (auto& [wid, items] : basket_map)
        transactions.push_back(std::move(items));

    return { std::move(transactions),
             std::vector<std::string>(all_tokens.begin(), all_tokens.end()) };
}

static EncodedData encode_cumulate_vocabulary(const std::vector<std::string>& all_tokens) {
    std::vector<std::string> col_names = all_tokens;
    std::sort(col_names.begin(), col_names.end());
    col_names.erase(std::unique(col_names.begin(), col_names.end()), col_names.end());

    std::unordered_map<std::string, int> col_idx;
    for (int i = 0; i < (int)col_names.size(); ++i)
        col_idx[col_names[i]] = i;

    return { Matrix{}, std::move(col_names), std::move(col_idx) };
}

static EncodedData encode_transactions(const std::vector<std::vector<std::string>>& transactions) {
    // Collect unique tokens then sort alphabetically, mirroring the Python encoder.
    std::set<std::string> tok_set;
    for (const auto& tx : transactions)
        for (const auto& tok : tx)
            tok_set.insert(tok);

    std::vector<std::string> col_names(tok_set.begin(), tok_set.end());
    std::unordered_map<std::string, int> col_idx;
    for (int i = 0; i < (int)col_names.size(); ++i)
        col_idx[col_names[i]] = i;

    int n_cols = (int)col_names.size();
    Matrix X(transactions.size(), Row(n_cols, false));
    for (int r = 0; r < (int)transactions.size(); ++r)
        for (const auto& tok : transactions[r])
            X[r][col_idx.at(tok)] = true;

    return { std::move(X), std::move(col_names), std::move(col_idx) };
}

static BitsetEncodedData encode_transactions_bitset(
    const std::vector<std::vector<std::string>>& transactions) {
    // Same vocabulary ordering as encode_transactions() and the Python encoder.
    std::set<std::string> tok_set;
    for (const auto& tx : transactions)
        for (const auto& tok : tx)
            tok_set.insert(tok);

    std::vector<std::string> col_names(tok_set.begin(), tok_set.end());
    std::unordered_map<std::string, int> col_idx;
    for (int i = 0; i < (int)col_names.size(); ++i)
        col_idx[col_names[i]] = i;

    const int n_rows = (int)transactions.size();
    const int n_cols = (int)col_names.size();
    const int n_blocks = (n_rows + 63) / 64;

    std::vector<std::vector<std::uint64_t>> item_bits(
        n_cols, std::vector<std::uint64_t>(n_blocks, 0));
    std::vector<int> item_counts(n_cols, 0);

    for (int r = 0; r < n_rows; ++r) {
        const int block = r >> 6;
        const int offset = r & 63;
        const std::uint64_t mask = 1ULL << offset;
        for (const auto& tok : transactions[r]) {
            int c = col_idx.at(tok);
            if ((item_bits[c][block] & mask) == 0) {
                item_bits[c][block] |= mask;
                ++item_counts[c];
            }
        }
    }

    return {
        std::move(item_bits),
        std::move(item_counts),
        std::move(col_names),
        std::move(col_idx),
        n_rows,
        n_blocks,
    };
}

static inline int popcount64(std::uint64_t x) {
    return __builtin_popcountll((unsigned long long)x);
}

static int support_count_bitset(
    const std::vector<std::vector<std::uint64_t>>& item_bits,
    const Itemset& cand,
    std::vector<std::uint64_t>& scratch) {
    if (cand.empty()) return 0;

    const auto& first = item_bits[cand[0]];
    scratch.assign(first.begin(), first.end());

    for (std::size_t j = 1; j < cand.size(); ++j) {
        const auto& bits = item_bits[cand[j]];
        for (std::size_t b = 0; b < scratch.size(); ++b)
            scratch[b] &= bits[b];
    }

    int count = 0;
    for (std::uint64_t word : scratch)
        count += popcount64(word);
    return count;
}

static std::vector<FrequentItemset> apriori_bitset(
    const BitsetEncodedData& enc,
    double min_support,
    std::optional<int> max_len = std::nullopt,
    const AncestorMap* ancestors = nullptr,
    const PathMap* path_map = nullptr,
    bool verbose = false) {
    if (min_support <= 0.0 || min_support > 1.0)
        throw std::invalid_argument("`min_support` must be in (0, 1]");

    const int n_rows = enc.n_rows;
    const int n_items = (int)enc.col_names.size();
    if (n_rows == 0 || n_items == 0) return {};

    const double threshold = min_support * (double)n_rows;

    std::vector<Itemset> current_L;
    std::vector<FrequentItemset> result;
    current_L.reserve(n_items);
    result.reserve(n_items);

    for (ItemIdx idx = 0; idx < n_items; ++idx) {
        if ((double)enc.item_counts[idx] >= threshold) {
            current_L.push_back({idx});
            result.push_back({{idx}, (double)enc.item_counts[idx] / n_rows});
        }
    }
    if (verbose) {
        std::cout << "Pass 1: " << n_items << " items scanned | "
                  << current_L.size() << " frequent | 0 ms\n";
    }

    int max_itemset = 1;
    std::vector<std::uint64_t> scratch;
    scratch.reserve(enc.n_blocks);

    while (!current_L.empty() && (!max_len.has_value() || max_itemset < *max_len)) {
        int k = max_itemset + 1;
        auto candidates = Generate_candidates_H(
            current_L, k, ancestors, max_len, path_map);
        if (candidates.empty()) break;

        auto pass_t0 = std::chrono::high_resolution_clock::now();

        std::vector<Itemset> next_L;
        next_L.reserve(candidates.size());

        for (const auto& cand : candidates) {
            int cnt = support_count_bitset(enc.item_bits, cand, scratch);
            if ((double)cnt >= threshold) {
                next_L.push_back(cand);
                result.push_back({cand, (double)cnt / n_rows});
            }
        }

        if (verbose) {
            double pass_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - pass_t0).count();
            std::cout << "Pass " << k << ": " << candidates.size()
                      << " candidates | " << next_L.size()
                      << " frequent | " << pass_ms << " ms\n";
        }

        if (next_L.empty()) break;
        current_L = std::move(next_L);
        max_itemset = k;
    }

    if (verbose) std::cout << '\n';
    return result;
}

// =========================================================
// ── Scored rule ────────────────────────────────────────────
// =========================================================
struct ScoredRule {
    std::vector<std::string> a_toks, b_toks;
    std::vector<int>         a_levels, b_levels;
    std::vector<std::string> a_branches, b_branches;
    std::string              a_lbl, b_lbl, rule;
    double support = 0, confidence = 0, lift = 0, score = 0;
};

static bool scored_rule_better(const ScoredRule& a, const ScoredRule& b) {
    if (a.score != b.score)           return a.score > b.score;
    if (a.confidence != b.confidence) return a.confidence > b.confidence;
    if (a.lift != b.lift)             return a.lift > b.lift;
    if (a.support != b.support)       return a.support > b.support;
    if (a.a_toks != b.a_toks)         return a.a_toks < b.a_toks;
    return a.b_toks < b.b_toks;
}

// =========================================================
// ── CELL 7 : Post-processing ──────────────────────────────
// =========================================================

static std::string join_labels(const std::vector<std::string>& toks) {
    std::string s;
    for (int i = 0; i < (int)toks.size(); ++i) {
        if (i) s += " + ";
        s += parse_token(toks[i]).label;
    }
    return s;
}

static std::string consequent_family_key_one(const std::string& tok) {
    auto pt = parse_token(tok);
    if (pt.level == 0) {
        std::string p = direct_parent_token_of_leaf(tok);
        return p.empty() ? tok : p;
    }
    return tok;
}

// add_score_and_rank: score = confidence * lift; filter; sort desc
static std::vector<ScoredRule> add_score_and_rank(
    std::vector<ScoredRule> rules, double min_support, double min_lift)
{
    std::vector<ScoredRule> out;
    for (auto& r : rules) {
        if (r.support >= min_support && r.lift >= min_lift) {
            r.score = r.confidence * r.lift;
            out.push_back(std::move(r));
        }
    }
    std::sort(out.begin(), out.end(), scored_rule_better);
    return out;
}

// dedupe_family: keep best rule per (antecedent, consequent-family)
static std::vector<ScoredRule> dedupe_family(std::vector<ScoredRule> rules) {
    std::sort(rules.begin(), rules.end(), scored_rule_better);
    std::set<std::pair<std::vector<std::string>, std::vector<std::string>>> seen;
    std::vector<ScoredRule> out;
    for (auto& r : rules) {
        std::vector<std::string> b_fam;
        for (const auto& t : r.b_toks) b_fam.push_back(consequent_family_key_one(t));
        std::sort(b_fam.begin(), b_fam.end());
        if (seen.insert({r.a_toks, b_fam}).second)
            out.push_back(std::move(r));
    }
    return out;
}

// dedupe_antimirror: keep best of A->B vs B->A
static std::vector<ScoredRule> dedupe_antimirror(std::vector<ScoredRule> rules) {
    std::sort(rules.begin(), rules.end(), scored_rule_better);
    std::set<std::vector<std::string>> seen;
    std::vector<ScoredRule> out;
    for (auto& r : rules) {
        std::vector<std::string> pair_items;
        for (const auto& t : r.a_toks) pair_items.push_back(t);
        for (const auto& t : r.b_toks) pair_items.push_back(t);
        std::sort(pair_items.begin(), pair_items.end());
        pair_items.erase(std::unique(pair_items.begin(), pair_items.end()), pair_items.end());
        if (seen.insert(pair_items).second) out.push_back(std::move(r));
    }
    return out;
}

// =========================================================
// ── Parquet I/O ───────────────────────────────────────────
// =========================================================

// Read all .parquet files in a directory and concatenate into one table.
static std::shared_ptr<arrow::Table> read_parquet_dir(const std::string& dir) {
    // Collect .parquet files using std::filesystem (works on Windows and Linux)
    std::vector<std::string> files;
    {
        std::filesystem::path dir_path(dir);
        if (!std::filesystem::exists(dir_path))
            throw std::runtime_error("Directory not found: " + dir);
        for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (entry.path().extension() != ".parquet") continue;
            if (entry.path().filename().string().rfind("._", 0) == 0) continue; // skip macOS metadata files
            files.push_back(entry.path().string());
        }
        std::sort(files.begin(), files.end());
    }
    if (files.empty()) throw std::runtime_error("No parquet files in: " + dir);

    std::vector<std::shared_ptr<arrow::Table>> tables;
    for (const auto& path : files) {
        auto infile_res = arrow::io::ReadableFile::Open(path);
        if (!infile_res.ok()) throw std::runtime_error("Cannot open: " + path + " – " + infile_res.status().ToString());
        std::unique_ptr<parquet::arrow::FileReader> reader;
        auto st_open = parquet::arrow::OpenFile(*infile_res, arrow::default_memory_pool(), &reader);
        if (!st_open.ok()) throw std::runtime_error("Cannot open parquet: " + path + " – " + st_open.ToString());
        std::shared_ptr<arrow::Table> tbl;
        auto st = reader->ReadTable(&tbl);
        if (!st.ok()) throw std::runtime_error("Cannot read parquet: " + path + " – " + st.ToString());
        tables.push_back(tbl);
    }
    auto result = arrow::ConcatenateTables(tables);
    if (!result.ok()) throw std::runtime_error("Cannot concat tables: " + result.status().ToString());
    return *result;
}

// Extract a string column (handles both StringArray and LargeStringArray) from a chunked array.
static std::vector<std::string> col_to_strings(const std::shared_ptr<arrow::ChunkedArray>& col) {
    std::vector<std::string> out;
    out.reserve(col->length());
    for (const auto& chunk : col->chunks()) {
        if (chunk->type_id() == arrow::Type::LARGE_STRING) {
            auto arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            for (int64_t i = 0; i < arr->length(); ++i)
                out.push_back(arr->IsNull(i) ? "" : arr->GetString(i));
        } else {
            auto arr = std::static_pointer_cast<arrow::StringArray>(chunk);
            for (int64_t i = 0; i < arr->length(); ++i)
                out.push_back(arr->IsNull(i) ? "" : arr->GetString(i));
        }
    }
    return out;
}

struct LoadResult {
    std::vector<RawRow>      rows;
    std::vector<std::string> cat_paths; // unique category path strings for branch ancestry
};

static std::vector<std::string>
expand_category_path_prefixes(const std::vector<std::string>& category_paths) {
    std::set<std::string> prefixes;
    for (const auto& path : category_paths) {
        auto parts = split_path(path);
        for (int upto = 1; upto <= (int)parts.size(); ++upto) {
            std::string prefix;
            for (int i = 0; i < upto; ++i) {
                if (i) prefix += " > ";
                prefix += parts[i];
            }
            prefixes.insert(prefix);
        }
    }
    return std::vector<std::string>(prefixes.begin(), prefixes.end());
}

// Mirrors notebook join:
//   wish_events JOIN products ON product_id
//              JOIN categories ON mongo_product_id = categories.id
static LoadResult load_parquet(const std::string& base) {
    // Read all parquet files in the given directory (columns: wishlist_id, category_name)
    auto tbl = read_parquet_dir(base);

    auto wish_ids  = col_to_strings(tbl->GetColumnByName("wishlist_id"));
    auto cat_names = col_to_strings(tbl->GetColumnByName("category_name"));

    std::unordered_set<std::string> seen_wish_cat; // dedup (wishlist_id, category_name)
    std::unordered_set<std::string> unique_cat_paths;
    std::vector<RawRow> rows;
    rows.reserve(wish_ids.size());

    for (size_t i = 0; i < wish_ids.size(); ++i) {
        if (wish_ids[i].empty() || cat_names[i].empty()) continue;
        std::string wc_key = wish_ids[i] + '\0' + cat_names[i];
        if (!seen_wish_cat.insert(wc_key).second) continue;
        unique_cat_paths.insert(cat_names[i]);
        rows.push_back({wish_ids[i], cat_names[i]});
    }
    return { std::move(rows),
             std::vector<std::string>(unique_cat_paths.begin(), unique_cat_paths.end()) };
}

static void write_rules_csv(const std::string& path,
                             const std::vector<ScoredRule>& rules) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot write: " + path);
    f << "rule,a_toks,b_toks,a_levels,b_levels,a_branches,b_branches,"
         "support,confidence,lift,score\n";

    auto fmt_str_vec = [](const std::vector<std::string>& v) {
        std::string s = "(";
        for (int i = 0; i < (int)v.size(); ++i) {
            if (i) s += ", ";
            s += v[i];
        }
        return s + (v.size() == 1 ? ",)" : ")");
    };
    auto fmt_int_vec = [](const std::vector<int>& v) {
        std::string s = "(";
        for (int i = 0; i < (int)v.size(); ++i) {
            if (i) s += ", ";
            s += std::to_string(v[i]);
        }
        return s + (v.size() == 1 ? ",)" : ")");
    };
    auto d6 = [](double x) {
        std::ostringstream o; o << std::fixed << std::setprecision(6) << x; return o.str();
    };

    for (const auto& r : rules) {
        f << '"' << r.rule          << '"' << ','
          << '"' << fmt_str_vec(r.a_toks)    << '"' << ','
          << '"' << fmt_str_vec(r.b_toks)    << '"' << ','
          << '"' << fmt_int_vec(r.a_levels)  << '"' << ','
          << '"' << fmt_int_vec(r.b_levels)  << '"' << ','
          << '"' << fmt_str_vec(r.a_branches)<< '"' << ','
          << '"' << fmt_str_vec(r.b_branches)<< '"' << ','
          << d6(r.support)    << ','
          << d6(r.confidence) << ','
          << d6(r.lift)       << ','
          << d6(r.score)      << '\n';
    }
}

// =========================================================
// ── main ──────────────────────────────────────────────────
// Usage:
//   ./apriori_pipeline <csv> [K_LEVELS] [MIN_SUPPORT] [MIN_CONF] [MIN_LIFT] [MAX_ANTE] [MAX_CONS] [OUTPUT_CSV]
//
// Defaults match the notebook:
//   K_LEVELS=4  MIN_SUPPORT=0.02  MIN_CONF=0.60  MIN_LIFT=1.5  MAX_ANTE=3  MAX_CONS=2
//
// Input CSV must have at minimum two columns: wishlist_id, category_name
// =========================================================
int main(int argc, char* argv[]) {
    std::string base_path;
    if (argc > 1) {
        base_path = argv[1];
    } else {
        static const char* candidates[] = {
            "/Users/anderschristensen/Desktop/Bachelor Project/anders_kevin/anders_kevin",
            "/Users/kevinkinsella/Desktop/Apriori_hjemmeside/anders_kevin/anders_kevin",
            "D:/Bachelor",
        };
        for (const char* c : candidates) {
            if (std::filesystem::is_directory(c)) { base_path = c; break; }
        }
        if (base_path.empty())
            throw std::runtime_error("Could not find base path. Pass it as the first argument.");
    }
    int    K_LEVELS       = (argc > 2) ? std::stoi(argv[2]) : 5;
    double MIN_SUPPORT    = (argc > 3) ? std::stod(argv[3]) : 0.02;
    double MIN_CONF       = (argc > 4) ? std::stod(argv[4]) : 0.60;
    double MIN_LIFT       = (argc > 5) ? std::stod(argv[5]) : 1.5;
    int    max_ante_v     = (argc > 6) ? std::stoi(argv[6]) : 3;
    int    max_cons_v     = (argc > 7) ? std::stoi(argv[7]) : 2;
    std::string output_csv = (argc > 8) ? argv[8] : "rules_out_cpp_bitset_merged.csv";

    std::optional<int> max_ante = (max_ante_v > 0) ? std::make_optional(max_ante_v) : std::nullopt;
    std::optional<int> max_cons = (max_cons_v > 0) ? std::make_optional(max_cons_v) : std::nullopt;
    int max_len_v = (max_ante_v > 0 && max_cons_v > 0) ? max_ante_v + max_cons_v : 0;
    std::optional<int> max_len = (max_len_v > 0) ? std::make_optional(max_len_v) : std::nullopt;

    std::cout << "=== C++ Apriori Pipeline (notebook-equivalent) ===\n"
              << "  DATA       : " << base_path << '\n'
              << "  K_LEVELS   : " << K_LEVELS  << '\n'
              << "  MIN_SUPPORT: " << MIN_SUPPORT << '\n'
              << "  MIN_CONF   : " << MIN_CONF  << '\n'
              << "  MIN_LIFT   : " << MIN_LIFT  << '\n'
              << "  MAX_ANTE   : " << (max_ante ? std::to_string(*max_ante) : "unlimited") << '\n'
              << "  MAX_CONS   : " << (max_cons ? std::to_string(*max_cons) : "unlimited") << '\n'
              << "  BACKEND    : vertical bitset + popcount\n"
              << '\n';

    auto t_total = std::chrono::high_resolution_clock::now();

    // [1] Load raw rows ---------------------------------------------------
    LoadResult load_res;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        load_res = load_parquet(base_path);
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[1] Loaded " << load_res.rows.size() << " rows in " << ms << " ms\n";
    }
    std::vector<RawRow>& raw_rows = load_res.rows;

    // [2] Build full expanded transactions ---------------------------------
    // Mirrors Python build_transactions(): each basket contains all generated
    // L0..L(k-1) tokens before one-hot encoding.
    std::vector<std::vector<std::string>> transactions;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        transactions = build_transactions(raw_rows, K_LEVELS);
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[2] Built " << transactions.size()
                  << " transactions in " << ms << " ms\n";
    }

    // [3] Vertical-bitset encode transactions ------------------------------
    BitsetEncodedData enc;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        enc = encode_transactions_bitset(transactions);
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[3] Encoded vertical bitsets " << enc.n_rows << " x "
                  << enc.col_names.size() << " (" << enc.n_blocks
                  << " uint64 blocks/item) in " << ms << " ms\n";
    }

    // [4] Branch ancestry ---------------------------------------------------
    BranchAncestry branch_ancestry;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto taxonomy_paths = expand_category_path_prefixes(load_res.cat_paths);
        branch_ancestry = build_branch_ancestry(taxonomy_paths);
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[4] Built branch ancestry for " << branch_ancestry.size()
                  << " labels in " << ms << " ms\n";
    }

    // [4b] Build ancestor map + path_map ------------------------------------
    AncestorMap ancestors_idx;
    std::unordered_map<int,std::string> path_map;
    BranchLabelMap branch_label_map;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto ancestors_str = build_ancestors_from_tokens(enc.col_names, &branch_ancestry);

        for (const auto& [child, ancs] : ancestors_str) {
            auto it = enc.col_idx.find(child);
            if (it == enc.col_idx.end()) continue;
            for (const auto& anc : ancs) {
                auto jt = enc.col_idx.find(anc);
                if (jt != enc.col_idx.end())
                    ancestors_idx[it->second].insert(jt->second);
            }
        }

        // Mirrors Python path_map and branch_label_map.
        for (const auto& [tok, idx] : enc.col_idx) {
            auto pt = parse_token(tok);
            path_map[idx] = pt.label;
            branch_label_map[idx] = pt.branch;
        }

        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[4b] Built ancestors for " << ancestors_idx.size()
                  << " items in " << ms << " ms\n";
    }

    // [5] Apriori -----------------------------------------------------------
    std::vector<FrequentItemset> freq_itemsets;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        freq_itemsets = apriori_bitset(
            enc,
            MIN_SUPPORT,
            max_len,
            &ancestors_idx,
            &path_map,
            /*verbose=*/true);
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[5] Apriori: " << freq_itemsets.size()
                  << " itemsets in " << ms << " ms\n";
    }

    // [6] Association rules ------------------------------------------------
    FreqMap freq_map = build_freq_map(freq_itemsets);
    std::vector<AssociationRule> raw_rules;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        raw_rules = association_rules(
            freq_map, MIN_CONF,
            &ancestors_idx, &path_map,
            max_ante, max_cons,
            /*require_single_consequent=*/false,
            &branch_label_map,
            &branch_ancestry);
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[6] Rules (raw): " << raw_rules.size() << " in " << ms << " ms\n";
    }

    // Convert AssociationRule -> ScoredRule + annotate --------------------
    std::vector<ScoredRule> scored;
    scored.reserve(raw_rules.size());
    for (const auto& r : raw_rules) {
        ScoredRule sr;
        for (ItemIdx i : r.antecedent) sr.a_toks.push_back(enc.col_names[i]);
        for (ItemIdx i : r.consequent) sr.b_toks.push_back(enc.col_names[i]);
        std::sort(sr.a_toks.begin(), sr.a_toks.end());
        std::sort(sr.b_toks.begin(), sr.b_toks.end());
        sr.support    = r.support;
        sr.confidence = r.confidence;
        sr.lift       = r.lift;
        for (const auto& t : sr.a_toks) {
            auto pt = parse_token(t);
            sr.a_levels.push_back(pt.level);
            sr.a_branches.push_back(pt.branch);
        }
        for (const auto& t : sr.b_toks) {
            auto pt = parse_token(t);
            sr.b_levels.push_back(pt.level);
            sr.b_branches.push_back(pt.branch);
        }
        sr.a_lbl = join_labels(sr.a_toks);
        sr.b_lbl = join_labels(sr.b_toks);
        sr.rule  = sr.a_lbl + " \u2192 " + sr.b_lbl;
        scored.push_back(std::move(sr));
    }

    // [7] Score + rank -----------------------------------------------------
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        scored = add_score_and_rank(std::move(scored), MIN_SUPPORT, MIN_LIFT);
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[7] After score+rank:       " << scored.size() << " rules in " << ms << " ms\n";
    }

    // [8] Family dedupe ----------------------------------------------------
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        scored = dedupe_family(std::move(scored));
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[8] After family dedupe:    " << scored.size() << " rules in " << ms << " ms\n";
    }

    // [9] Antimirror dedupe ------------------------------------------------
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        scored = dedupe_antimirror(std::move(scored));
        double ms = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "[9] After antimirror dedupe:" << scored.size() << " rules in " << ms << " ms\n";
    }

    // Write CSV output -----------------------------------------------------
    write_rules_csv(output_csv, scored);

    double total_ms = std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now()-t_total).count();
    std::cout << "\nTotal pipeline: " << total_ms << " ms\n";
    std::cout << "Output: " << output_csv << "\n\n";

    // Preview top 5 --------------------------------------------------------
    std::cout << "Top 5 rules:\n";
    for (int i = 0; i < std::min(5, (int)scored.size()); ++i) {
        const auto& r = scored[i];
        std::cout << std::fixed << std::setprecision(6)
                  << "  [" << i+1 << "] " << r.rule << '\n'
                  << "       sup=" << r.support
                  << "  conf=" << r.confidence
                  << "  lift=" << r.lift
                  << "  score=" << r.score << '\n';
    }

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "PEAK_RSS_MB: " << usage.ru_maxrss / 1024.0 << "\n";

    return 0;
}
