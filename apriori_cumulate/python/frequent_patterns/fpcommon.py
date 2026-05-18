import collections
import warnings

import numpy as np
import pandas as pd
from pandas import __version__ as pandas_version

warnings.simplefilter("always", DeprecationWarning)

# =========================================================
# === CHANGE (Hierarchy) ==================================
# Utility to precompute ancestor sets from a parent map P.
#
# P can be provided as:
#   - dict(child -> parent) with parent = None at root, OR
#   - callable P(x) -> parent (or None)
#
# Returns:
#   Anc[x] = set of all ancestors of x (parent, grandparent, ...)
# =========================================================

def precompute_ancestors(P, universe):
    """Precompute ancestor sets for all items in `universe`."""
    def parent(x):
        if callable(P):
            return P(x)
        return P.get(x)

    Anc = {}
    for x in universe:
        seen = set()
        y = parent(x)
        while y is not None and y not in seen:
            seen.add(y)
            y = parent(y)
        Anc[x] = seen
    return Anc


# =========================================================
# === CHANGE (Branch-level ancestry check) ================
# Utility to detect when two token *branch labels* are
# themselves ancestor/descendant in the real taxonomy.
#
# Background:
#   With K_LEVELS windowing, the L(K-1) node of the window
#   becomes the "branch" label stored in the token as B:xxx.
#   These branch labels can themselves form an ancestor chain
#   in the real taxonomy (e.g. branch "Cosmetics" is a child
#   of branch "Personal Care" in the Health & Beauty tree).
#   The existing within-window ancestor check in
#   h_rule_violates_hierarchy() is blind to this, because it
#   only compares items within the same K_LEVELS window, not
#   items whose windows are rooted at related branches.
#
# Ambiguity handling:
#   Real taxonomies contain non-unique short labels — e.g.
#   "Joggers" exists under Activewear Pants, Maternity Pants,
#   Baby & Toddler Bottoms, Loungewear Bottoms, and Pants.
#   At higher K_LEVELS these can appear as branch labels, so
#   each label is mapped to the FULL SET of taxonomy paths
#   sharing that short name.  Two labels are treated as
#   related if ANY pair of their paths is ancestor/descendant.
#   This is the conservative choice: if any interpretation
#   could be within-chain, the rule is filtered.
#
# build_branch_ancestry(taxonomy_df):
#   Accepts a taxonomy DataFrame with a 'name' column
#   containing full '>' separated category paths.
#   Returns a dict mapping each short label (last segment)
#   to a frozenset of full path strings — used by
#   h_branches_are_related() and h_rule_violates_branch_ancestry().
#
# h_branches_are_related(branch_a, branch_b, label_to_paths):
#   Returns True if ANY (path_a, path_b) pair drawn from the
#   path-sets of branch_a and branch_b has one path as prefix
#   of the other.  Uses a ' > '-terminated prefix check to
#   prevent false positives like "Clothing" matching
#   "Clothing Accessories".  Accepts legacy single-string
#   mappings too for backwards compatibility.
#
# h_rule_violates_branch_ancestry(A, B, label_to_paths):
#   Returns True if ANY token in antecedent A has a branch
#   label that is an ancestor/descendant of ANY token in
#   consequent B.  Token branch label is extracted from the
#   'B:xxx' segment of the standard token format
#   'Lx|B:BranchLabel|Full > Category > Path'.
#   Called from _rule_violates_hierarchy() in
#   association_rules.py when branch_ancestry is provided.
# =========================================================

def build_branch_ancestry(taxonomy_df, name_col: str = "name"):
    """Build a label->set-of-full-paths mapping from a taxonomy DataFrame.

    The taxonomy contains labels that are not globally unique (e.g. "Joggers"
    exists under Activewear Pants, Maternity Pants, Baby & Toddler Bottoms,
    Sleepwear Loungewear Bottoms, and regular Pants).  With K_LEVELS windowing,
    any such label can become a branch label when it happens to sit at the
    window root.  To handle this safely at any K_LEVELS, we track ALL full
    paths that share a given short label.

    Parameters
    ----------
    taxonomy_df : pd.DataFrame
        Must contain a 'name' column with full '>' separated paths,
        e.g. 'Health & Beauty > Personal Care > Cosmetics'.

    Returns
    -------
    dict mapping short label (last path segment) -> frozenset of full path strings.
    """
    mapping = {}
    for name in taxonomy_df[name_col]:
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name:
            continue
        short = name.split(">")[-1].strip()
        if short in mapping:
            mapping[short] = mapping[short] | {name}
        else:
            mapping[short] = {name}
    # Freeze for immutability / safe reuse across calls.
    return {k: frozenset(v) for k, v in mapping.items()}


def h_branches_are_related(branch_a, branch_b, label_to_paths):
    """Return True if branch_a and branch_b are ancestor/descendant in the real taxonomy.

    Handles ambiguous short labels that map to multiple full paths by checking
    whether ANY pair of (path_for_a, path_for_b) is in an ancestor/descendant
    relationship.  This is the conservative choice: if any interpretation of
    the label pair could be within-chain, treat the pair as related so the
    within-chain rule is filtered.

    Uses a ' > '-terminated prefix check to avoid false positives such as
    'Clothing' matching 'Clothing Accessories'.
    Returns False conservatively when either label is unknown.

    Parameters
    ----------
    branch_a, branch_b : str
        Short branch labels (last segment of a taxonomy path).
    label_to_paths : dict
        Mapping returned by build_branch_ancestry(): label -> frozenset of full paths.
        For backwards compatibility, also accepts label -> str (single path).
    """
    if branch_a == branch_b:
        return False

    paths_a = label_to_paths.get(branch_a)
    paths_b = label_to_paths.get(branch_b)
    if not paths_a or not paths_b:
        return False
    # Back-compat: accept single string as well as a set/frozenset of strings.
    if isinstance(paths_a, str):
        paths_a = (paths_a,)
    if isinstance(paths_b, str):
        paths_b = (paths_b,)
    for pa in paths_a:
        pa_norm = pa + " > "
        for pb in paths_b:
            pb_norm = pb + " > "
            if pa_norm.startswith(pb_norm) or pb_norm.startswith(pa_norm):
                return True
    return False


def h_rule_violates_branch_ancestry(A, B, label_to_paths):
    """Return True if any antecedent token's branch is related to any consequent token's branch.

    Token format expected: 'Lx|B:BranchLabel|Full > Category > Path'
    The branch label is the segment between 'B:' and the second '|'.

    Uses conservative ambiguity handling: when a branch label maps to multiple
    taxonomy paths (e.g. "Joggers" exists under several Pants subtrees), the
    rule is flagged if ANY path interpretation would be within-chain.

    Parameters
    ----------
    A, B : iterable of token strings
        Antecedent and consequent token lists.
    label_to_paths : dict
        Mapping returned by build_branch_ancestry(): label -> frozenset of paths.
    """
    def _branch(token):
        # 'L2|B:Cosmetics|Cosmetics > Makeup > Face Makeup' -> 'Cosmetics'
        parts = token.split("|")
        if len(parts) >= 2:
            return parts[1].replace("B:", "", 1)
        return token  # fallback: treat whole token as branch label

    for a_tok in A:
        ba = _branch(a_tok)
        for b_tok in B:
            if h_branches_are_related(ba, _branch(b_tok), label_to_paths):
                return True
    return False


def h_rule_violates_hierarchy(A, B, P=None, ancestors=None):
    """Return True iff a cross-boundary ancestor pair exists between A and B.

    Parameters
    ----------
    A, B : iterable
        Antecedent and consequent items.
    P : dict | callable | None
        Optional parent map / parent function, used if `ancestors` is not given.
    ancestors : dict | None
        Optional dict mapping item -> set of ancestors.
    """
    if ancestors is None:
        universe = set(A) | set(B)
        if P is None:
            return False
        ancestors = precompute_ancestors(P, universe)

    for a in A:
        anc_a = ancestors.get(a, set())
        for b in B:
            anc_b = ancestors.get(b, set())
            if (a in anc_b) or (b in anc_a):
                return True
    return False


def setup_fptree(df, min_support, null_values=False):
    num_itemsets = len(df.index)  # number of itemsets in the database

    is_sparse = False
    if hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            itemsets = df.values
        else:
            itemsets = df.sparse.to_coo().tocsr()
            is_sparse = True
    else:
        # dense DataFrame
        itemsets = df.values

    # support of each individual item
    # if itemsets is sparse, np.sum returns an np.matrix of shape (1, N)
    disabled = None
    if null_values:
        disabled = df.copy()
        disabled = np.where(pd.isna(disabled), 1, np.nan) + np.where(
            (disabled == 0) | (disabled == 1), np.nan, 0
        )
        item_support = np.array(
            np.nansum(df.values, axis=0)
            / (float(num_itemsets) - np.nansum(disabled, axis=0))
        )
    else:
        item_support = np.array(np.sum(df.values, axis=0) / float(num_itemsets))
    item_support = item_support.reshape(-1)
    items = np.nonzero(item_support >= min_support)[0]

    # Define ordering on items for inserting into FPTree
    indices = item_support[items].argsort()
    rank = {item: i for i, item in enumerate(items[indices])}

    if is_sparse:
        # Ensure that there are no zeros in sparse DataFrame
        itemsets.eliminate_zeros()

    # Building tree by inserting itemsets in sorted order
    # Heuristic for reducing tree size is inserting in order
    #   of most frequent to least frequent
    tree = FPTree(rank)
    for i in range(num_itemsets):
        if is_sparse:
            # itemsets has been converted to CSR format to speed-up the line
            # below.  It has 3 attributes:
            #  - itemsets.data contains non null values, shape(#nnz,)
            #  - itemsets.indices contains the column number of non null
            #    elements, shape(#nnz,)
            #  - itemsets.indptr[i] contains the offset in itemset.indices of
            #    the first non null element in row i, shape(1+#nrows,)
            nonnull = itemsets.indices[itemsets.indptr[i] : itemsets.indptr[i + 1]]
        else:
            nonnull = np.where(itemsets[i, :])[0]
        itemset = [item for item in nonnull if item in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_itemset(itemset)

    return tree, disabled, rank


def generate_itemsets(
    generator, df, disabled, min_support, num_itemsets, colname_map, null_values=False
):
    itemsets = []
    supports = []
    if not null_values or disabled is None:
        for sup, iset in generator:
            support = sup / float(num_itemsets)
            if support >= min_support:
                itemsets.append(frozenset(iset))
                supports.append(support)
    else:
        for sup, iset in generator:
            itemsets.append(frozenset(iset))
            # select data of iset from disabled dataset
            dec = disabled[:, iset]
            # select data of iset from original dataset
            _dec = df.values[:, iset]

            # case if iset only has one element
            if len(iset) == 1:
                supports.append(
                    (sup - np.nansum(dec)) / (num_itemsets - np.nansum(dec))
                )

            # case if iset has multiple elements
            elif len(iset) > 1:
                denom = 0
                num = 0
                for i in range(dec.shape[0]):
                    # select the i-th iset from disabled dataset
                    item_dsbl = list(dec[i, :])
                    # select the i-th iset from original dataset
                    item_orig = list(_dec[i, :])

                    # check and keep count if there is a null value in iset of disabled
                    if 1 in set(item_dsbl):
                        denom += 1

                        # check and keep count if item doesn't exist OR all values are null in iset of original
                        if (0 not in set(item_orig)) or (
                            all(np.isnan(x) for x in item_orig)
                        ):
                            num -= 1

                if num_itemsets - denom == 0:
                    supports.append(0)
                else:
                    supports.append((sup + num) / (num_itemsets - denom))

    res_df = pd.DataFrame({"support": supports, "itemsets": itemsets})
    res_df = res_df[res_df["support"] >= min_support]

    if colname_map is not None:
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([colname_map[i] for i in x])
        )

    return res_df


def valid_input_check(df, null_values=False):
    # Return early if df is None
    if df is None:
        return

    if f"{type(df)}" == "<class 'pandas.core.frame.SparseDataFrame'>":
        msg = (
            "SparseDataFrame support has been deprecated in pandas 1.0,"
            " and is no longer supported in mlxtend. "
            " Please"
            " see the pandas migration guide at"
            " https://pandas.pydata.org/pandas-docs/"
            "stable/user_guide/sparse.html#sparse-data-structures"
            " for supporting sparse data in DataFrames."
        )
        raise TypeError(msg)

    if df.size == 0:
        return
    if hasattr(df, "sparse"):
        if not isinstance(df.columns[0], str) and df.columns[0] != 0:
            raise ValueError(
                "Due to current limitations in Pandas, "
                "if the sparse format has integer column names,"
                "names, please make sure they either start "
                "with `0` or cast them as string column names: "
                "`df.columns = [str(i) for i in df.columns`]."
            )

    # Fast path: if all columns are boolean, there is nothing to checks
    if null_values:
        all_bools = (
            df.apply(lambda col: col.apply(lambda x: pd.isna(x) or isinstance(x, bool)))
            .all()
            .all()
        )
    else:
        all_bools = df.dtypes.apply(pd.api.types.is_bool_dtype).all()

    if not all_bools:
        warnings.warn(
            "DataFrames with non-bool types result in worse computational"
            "performance and their support might be discontinued in the future."
            "Please use a DataFrame with bool type",
            DeprecationWarning,
        )

        # If null_values is True but no NaNs are found, raise an error
        has_nans = pd.isna(df).any().any()
        if null_values and not has_nans:
            warnings.warn(
                "null_values=True is inefficient when there are no NaN values in the DataFrame."
                "Set null_values=False for faster output."
            )
        # If null_values is False but NaNs are found, raise an error
        if not null_values and has_nans:
            raise ValueError(
                "NaN values are not permitted in the DataFrame when null_values=False."
            )

        # Pandas is much slower than numpy, so use np.where on Numpy arrays
        if hasattr(df, "sparse"):
            if df.size == 0:
                values = df.values
            else:
                values = df.sparse.to_coo().tocoo().data
        else:
            values = df.values

        # Ignore NaNs if null_values is True
        if null_values:
            idxs = np.where((values != 1) & (values != 0) & (~np.isnan(values)))
        else:
            idxs = np.where((values != 1) & (values != 0))

        if len(idxs[0]) > 0:
            # idxs has 1 dimension with sparse data and 2 with dense data
            val = values[tuple(loc[0] for loc in idxs)]
            s = (
                "The allowed values for a DataFrame"
                " are True, False, 0, 1. Found value %s" % (val)
            )

            if null_values:
                s = (
                    "The allowed values for a DataFrame"
                    " are True, False, 0, 1, NaN. Found value %s" % (val)
                )
            raise ValueError(s)


class FPTree(object):
    def __init__(self, rank=None):
        self.root = FPNode(None)
        self.nodes = collections.defaultdict(list)
        self.cond_items = []
        self.rank = rank

    def conditional_tree(self, cond_item, minsup):
        """
        Creates and returns the subtree of self conditioned on cond_item.

        Parameters
        ----------
        cond_item : int | str
            Item that the tree (self) will be conditioned on.
        minsup : int
            Minimum support threshold.

        Returns
        -------
        cond_tree : FPtree
        """
        # Find all path from root node to nodes for item
        branches = []
        count = collections.defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count

        # Define new ordering or deep trees may have combinatorially explosion
        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}

        # Create conditional tree
        cond_tree = FPTree(rank)
        for idx, branch in enumerate(branches):
            branch = sorted(
                [i for i in branch if i in rank], key=rank.get, reverse=True
            )
            cond_tree.insert_itemset(branch, self.nodes[cond_item][idx].count)
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

    def insert_itemset(self, itemset, count=1):
        """
        Inserts a list of items into the tree.

        Parameters
        ----------
        itemset : list
            Items that will be inserted into the tree.
        count : int
            The number of occurrences of the itemset.
        """
        self.root.count += count

        if len(itemset) == 0:
            return

        # Follow existing path in tree as long as possible
        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
                index += 1
            else:
                break

        # Insert any remaining items
        for item in itemset[index:]:
            child_node = FPNode(item, count, node)
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True

    def print_status(self, count, colnames):
        cond_items = [str(i) for i in self.cond_items]
        if colnames:
            cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ", ".join(cond_items)
        print(
            "\r%d itemset(s) from tree conditioned on items (%s)" % (count, cond_items),
            end="\n",
        )


class FPNode(object):
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = collections.defaultdict(FPNode)

        if parent is not None:
            parent.children[item] = self

    def itempath_from_root(self):
        """Returns the top-down sequence of items from self to
        (but not including) the root node."""
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path

