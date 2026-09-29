from collections import defaultdict, deque, Counter
from typing import Iterable
from rdflib import Graph, RDFS
import numpy as np
import torch
import random
import graphviz

OBO = "http://purl.obolibrary.org/obo/"
IS_A = RDFS.subClassOf
SPECIAL = {"unknown", "na", None}


FORBIDDEN = {
    "CL:0000000",
    "CL:0000548",
    "CL:0000255",
}


MACRO_ANCHORS = {
    "CL:0000540": "neuron",
    "CL:0000125": "glial cell",
    "CL:0000738": "immune cell",
    "CL:0000115": "endothelial cell",
    "CL:0000057": "fibroblast",
    "CL:0000183": "contractile cell",
    "CL:0000066": "epithelial cell",
    "CL:0001064": "malignant cell",
    "CL:0000034": "stem/progenitor",
    "CL:0009004": "retinal cell",
    "CL:1000497": "kidney cell",
    "CL:0000746": "cardiac muscle cell",
    "CL:0000031": "neuroblast",
    "CL:0008034": "mural cell",
}

NEURAL_SUB = {
    "CL:0000679": "glutamatergic neuron",
    "CL:0000617": "GABAergic neuron",
    "CL:0000127": "astrocyte",
    "CL:0000128": "oligodendrocyte",
    "CL:0002453": "OPC",
    "CL:0000129": "microglia",
}

IMMUNE_SUB = {
    "CL:0000084": "T cell",
    "CL:0000236": "B cell",
    "CL:0000786": "plasma cell",
    "CL:0000623": "NK cell",
    "CL:0000763": "myeloid cell",
}


def print_random_mappings(
    leaf_ids,
    coarse_ids,
    cl_labels,
    n: int = 20,
    seed: int = 0,
):
    rng = random.Random(seed)
    idxs = rng.sample(range(len(leaf_ids)), min(n, len(leaf_ids)))

    print("\n### Random leaf → anchor mappings ###")
    for i in idxs:
        leaf = leaf_ids[i]
        anchor = coarse_ids[i]
        leaf_name = cl_labels.get(leaf, leaf)
        anchor_name = cl_labels.get(anchor, anchor)
        print(f"{leaf:12s} ({leaf_name})  →  {anchor:12s} ({anchor_name})")


# ---------- graph utilities ----------


def invert_parents_to_children(parents: dict[str, set[str]]) -> dict[str, set[str]]:
    children: dict[str, set[str]] = defaultdict(set)
    for ch, ps in parents.items():
        for p in ps:
            children[p].add(ch)
    return children


def compute_min_depths(parents: dict[str, set[str]]) -> dict[str, int]:
    """
    Depth = minimum distance to any root (node with no parents in this graph).
    Root depth = 0.
    Works on a DAG; if there are disconnected parts, they get depth 0.
    """
    all_nodes = set(parents.keys()) | {p for ps in parents.values() for p in ps}
    children = invert_parents_to_children(parents)

    indeg = {n: 0 for n in all_nodes}
    for ch, ps in parents.items():
        indeg[ch] += len(ps)

    q = deque([n for n in all_nodes if indeg.get(n, 0) == 0])
    depth = {n: 0 for n in q}

    while q:
        n = q.popleft()
        for c in children.get(n, ()):
            nd = depth[n] + 1
            if c not in depth or nd < depth[c]:
                depth[c] = nd
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)

    for n in all_nodes:
        depth.setdefault(n, 0)
    return depth


def all_ancestors(
    node: str, parents: dict[str, set[str]], memo: dict[str, set[str]]
) -> set[str]:
    if node in memo:
        return memo[node]
    out: set[str] = set()
    for p in sorted(parents.get(node, ())):
        out.add(p)
        out |= all_ancestors(p, parents, memo)
    memo[node] = out
    return out


def propagate_leaf_counts(
    leaf_counts: Counter, parents: dict[str, set[str]]
) -> Counter:
    """
    Each leaf contributes its count to itself + all ancestors.
    """
    memo: dict[str, set[str]] = {}
    prop = Counter()
    for leaf, c in leaf_counts.items():
        if leaf in SPECIAL:
            continue
        prop[leaf] += c
        for a in all_ancestors(leaf, parents, memo):
            prop[a] += c
    return prop


# ---------- SAME-DEPTH CUT (level set) + min_cells fallback ----------


def map_leaf_deepest_qualifying_with_coverage(
    leaf_id: str,
    parents: dict[str, set[str]],
    depth: dict[str, int],
    propagated: Counter,
    *,
    min_cells: int,
    min_depth: int,
    forbidden: set[str] | None = None,
    other_label: str = "other",
) -> str:
    """
    1) Prefer the deepest ancestor that satisfies:
         depth >= min_depth, propagated >= min_cells, not forbidden
    2) If none exists, fall back to the deepest ancestor that satisfies:
         depth >= min_depth, not forbidden
       (ignores min_cells to ensure coverage)
    """
    if leaf_id in SPECIAL:
        return str(leaf_id)

    forbidden = forbidden or set()

    q = deque([leaf_id])
    seen = {leaf_id}

    best = None
    best_d, best_p = -1, -1

    best_cov = None
    best_cov_d, best_cov_p = -1, -1

    while q:
        x = q.popleft()
        dx = depth.get(x, 0)
        px = propagated.get(x, 0)

        if x not in forbidden and dx >= min_depth:
            # coverage candidate (ignoring min_cells)
            if (
                (dx > best_cov_d)
                or (dx == best_cov_d and px > best_cov_p)
                or (
                    dx == best_cov_d
                    and px == best_cov_p
                    and best_cov is not None
                    and x < best_cov
                )
                or (best_cov is None)
            ):
                best_cov = x
                best_cov_d, best_cov_p = dx, px

            # qualifying candidate
            if px >= min_cells:
                if (
                    (dx > best_d)
                    or (dx == best_d and px > best_p)
                    or (dx == best_d and px == best_p and best is not None and x < best)
                    or (best is None)
                ):
                    best = x
                    best_d, best_p = dx, px

        # deterministic traversal
        for p in sorted(parents.get(x, ())):
            if p not in seen:
                seen.add(p)
                q.append(p)

    if best is not None:
        return best
    if best_cov is not None:
        return best_cov
    return other_label


def map_leaf_deepest_qualifying(
    leaf_id: str,
    parents: dict[str, set[str]],
    depth: dict[str, int],
    propagated: Counter,
    *,
    min_cells: int,
    min_depth: int,
    forbidden: set[str] | None = None,
    other_label: str = "other",
) -> str:
    """
    Pick the deepest (max depth) ancestor of `leaf_id` that satisfies:
      - propagated_count >= min_cells
      - depth >= min_depth
      - not forbidden

    Tie-breaks deterministically by:
      1) greater depth
      2) (optional) greater propagated count
      3) lexicographically smaller CL id
    """
    if leaf_id in SPECIAL:
        return str(leaf_id)

    forbidden = forbidden or set()

    q = deque([leaf_id])
    seen = {leaf_id}

    best = None
    best_d = -1
    best_prop = -1

    while q:
        x = q.popleft()

        dx = depth.get(x, 0)
        px = propagated.get(x, 0)

        if x not in forbidden and dx >= min_depth and px >= min_cells:
            if (
                dx > best_d
                or (dx == best_d and px > best_prop)
                or (dx == best_d and px == best_prop and best is not None and x < best)
                or (best is None)
            ):
                best = x
                best_d = dx
                best_prop = px

        # IMPORTANT: deterministic traversal in DAG
        for p in sorted(parents.get(x, ())):
            if p not in seen:
                seen.add(p)
                q.append(p)

    return best if best is not None else other_label


def nearest_node_at_depth(
    leaf_id: str,
    parents: dict[str, set[str]],
    depth: dict[str, int],
    target_depth: int,
    forbidden: set[str],
) -> str | None:
    """
    BFS upward from leaf to find the first ancestor (including itself) with depth==target_depth,
    skipping forbidden nodes. Returns None if none exists.
    """
    q = deque([leaf_id])
    seen = {leaf_id}
    while q:
        x = q.popleft()
        if x not in forbidden and depth.get(x, 0) == target_depth:
            return x
        for p in sorted(parents.get(x, ())):
            if p not in seen:
                seen.add(p)
                q.append(p)
    return None


def climb_up_until_min_cells(
    start: str,
    parents: dict[str, set[str]],
    propagated: Counter,
    min_cells: int,
    forbidden: set[str],
) -> str | None:
    """
    Starting from `start`, move upward until you find a node with propagated_count>=min_cells
    and not forbidden. Returns None if none found.
    """
    q = deque([start])
    seen = {start}
    while q:
        x = q.popleft()
        if x not in forbidden and propagated.get(x, 0) >= min_cells:
            return x
        for p in sorted(parents.get(x, ())):
            if p not in seen:
                seen.add(p)
                q.append(p)
    return None


def map_leaf_same_depth_with_fallback(
    leaf_id: str,
    parents: dict[str, set[str]],
    depth: dict[str, int],
    propagated: Counter,
    *,
    target_depth: int,
    min_cells: int,
    forbidden: set[str] | None = None,
    other_label: str = "other",
) -> str:
    """
    Strategy:
      1) Find nearest ancestor at exactly `target_depth` (level set cut)
      2) If its propagated count < min_cells, climb upward until meeting min_cells
      3) If nothing found, return `other_label`

    This yields a "cut-like" mapping: every leaf maps to exactly one anchor (or "other").
    """
    if leaf_id in SPECIAL:
        return str(leaf_id)

    forbidden = forbidden or set()

    at_d = nearest_node_at_depth(leaf_id, parents, depth, target_depth, forbidden)
    if at_d is None:
        return other_label

    # fallback: if too small at depth d, climb up
    anchor = climb_up_until_min_cells(at_d, parents, propagated, min_cells, forbidden)
    return anchor if anchor is not None else other_label


# def build_level_set_cut(
#     leaf_ids: list[str],
#     parents: dict[str, set[str]],
#     *,
#     target_depth: int = 4,
#     min_cells: int = 500,
#     forbidden: set[str] | None = None,
#     other_label: str = "other",
# ) -> tuple[list[str], set[str], Counter, dict[str, int]]:
#     """
#     End-to-end:
#       - compute leaf counts
#       - propagate to ancestors
#       - compute min-depths
#       - map each leaf via "same depth cut + min_cells fallback"
#     Returns:
#       coarse_ids: per-cell mapped anchors
#       anchors: unique anchors used
#       propagated: propagated counts for inspection
#       depth: depth map for inspection
#     """
#     forbidden = forbidden or set()

#     leaf_counts = Counter([l for l in leaf_ids if l not in SPECIAL])
#     propagated = propagate_leaf_counts(leaf_counts, parents)
#     depth = compute_min_depths(parents)

#     coarse_ids = [
#         map_leaf_same_depth_with_fallback(
#             l,
#             parents,
#             depth,
#             propagated,
#             target_depth=target_depth,
#             min_cells=min_cells,
#             forbidden=forbidden,
#             other_label=other_label,
#         )
#         for l in leaf_ids
#     ]
#     anchors = set(coarse_ids) - {other_label} - {str(x) for x in SPECIAL}
#     return coarse_ids, anchors, propagated, depth


def build_level_set_cut(
    leaf_ids: list[str],
    parents: dict[str, set[str]],
    *,
    min_cells: int = 500,
    min_depth: int = 3,
    forbidden: set[str] | None = None,
    other_label: str = "other",
) -> tuple[list[str], set[str], Counter, dict[str, int]]:
    """
    End-to-end using "deepest qualifying ancestor" mapping:
      - compute leaf counts
      - propagate to ancestors
      - compute min-depths
      - map each leaf to deepest qualifying ancestor

    Returns:
      coarse_ids: per-cell mapped anchors
      anchors: unique anchors used (excluding other/special)
      propagated: propagated counts
      depth: depth map
    """
    forbidden = forbidden or set()

    leaf_counts = Counter([l for l in leaf_ids if l not in SPECIAL])
    propagated = propagate_leaf_counts(leaf_counts, parents)
    depth = compute_min_depths(parents)

    coarse_ids = [
        map_leaf_deepest_qualifying(
            l,
            parents,
            depth,
            propagated,
            min_cells=min_cells,
            min_depth=min_depth,
            forbidden=forbidden,
            other_label=other_label,
        )
        for l in leaf_ids
    ]
    anchors = set(coarse_ids) - {other_label} - {str(x) for x in SPECIAL}
    return coarse_ids, anchors, propagated, depth


def _is_cl_iri(x) -> bool:
    return str(x).startswith(OBO + "CL_")


def _iri_to_curie(iri: str) -> str:
    local = str(iri).rsplit("/", 1)[-1]
    return local.replace("_", ":")


def build_parents_from_cl_owl(cl_owl_path: str):
    g = Graph()
    g.parse(cl_owl_path)
    parents = defaultdict(set)
    labels = {}

    for s, _, o in g.triples((None, RDFS.label, None)):
        if _is_cl_iri(s):
            labels[_iri_to_curie(s)] = str(o)

    for child, _, parent in g.triples((None, IS_A, None)):
        if _is_cl_iri(child) and _is_cl_iri(parent):
            parents[_iri_to_curie(child)].add(_iri_to_curie(parent))

    return parents, labels


def nearest_anchor(leaf_id, parents, anchor_set):
    if leaf_id in SPECIAL:
        return str(leaf_id)
    q = deque([leaf_id])
    seen = {leaf_id}
    while q:
        x = q.popleft()
        if x in anchor_set:
            return x
        for p in sorted(parents.get(x, ())):
            if p not in seen:
                seen.add(p)
                q.append(p)
    return "other"


def build_anchor_mapper(leaf_ids, parents, anchors_dict):
    anchor_set = set(anchors_dict.keys())
    return {leaf: nearest_anchor(leaf, parents, anchor_set) for leaf in leaf_ids}


def map_with_refinement(leaf_id, parents, macro_mapper, neural_sub, immune_sub):
    if leaf_id in SPECIAL:
        return str(leaf_id)
    macro = macro_mapper.get(leaf_id, "other")
    if macro == "CL:0000540":  # neuron
        return nearest_anchor(leaf_id, parents, set(neural_sub.keys()))
    if macro == "CL:0000738":  # immune
        return nearest_anchor(leaf_id, parents, set(immune_sub.keys()))
    return macro


def encode_string_labels_to_int(label_list):
    uniq = sorted(set(label_list))
    vocab = {lab: i for i, lab in enumerate(uniq)}
    y = np.array([vocab[x] for x in label_list], dtype=np.int64)
    return torch.from_numpy(y), vocab


def build_pruned_tree(
    leaf_ids: list[str],
    parents: dict[str, set[str]],
    k: int = 20,
    forbidden: set[str] | None = None,
) -> tuple[list[str], set[str]]:
    """
    Collapses a cell type ontology tree into exactly k clusters using a 
    bottom-up greedy 'Ascending' strategy.

    The algorithm iteratively moves the 'least valuable' population in the 
    current frontier up to its most relevant parent until only k nodes remain.

    Selection Logic for Pruning:
        1. Least Cells: Targets the node representing the smallest cell population.
        2. Deepest Level: If counts are tied, targets the node furthest from the root.
        3. Deterministic: If still tied, uses lexicographical ID order.

    Parent Selection Logic (The Merge):
        Cells are moved to a parent based on:
        1. Presence in current frontier (direct merge with an existing cluster).
        2. Lineage affinity (parent has other descendants already in the frontier).
        3. Global significance (highest propagated count in the original data).

    Args:
        leaf_ids: Original list of specific Cell Ontology IDs for every cell.
        parents: Dictionary mapping child IDs to a set of parent IDs (the DAG).
        k: The target number of terminal leaf nodes (clusters) to retain.
        forbidden: Set of IDs that should never be used as anchors or parents.

    Returns:
        coarse_ids: A list of the same length as leaf_ids, where each specific 
                    ID is replaced by its assigned cluster anchor.
        frontier: The set of k anchor IDs that represent the final clusters.
    """
    forbidden = forbidden or set()
    leaf_counts = Counter([l for l in leaf_ids if l not in SPECIAL])
    depth_map = compute_min_depths(parents)
    # We need the global propagation once to judge 'importance' of parents
    global_prop = propagate_leaf_counts(leaf_counts, parents)
    
    # Each original leaf starts as its own 'Local' cluster
    current_map = {l: l for l in leaf_counts}
    frontier = set(leaf_counts.keys())

    while len(frontier) > k:
        # STEP A: Calculate LOCAL counts only.
        # If 100 cells are mapped to 'B cell', they ONLY count toward 'B cell', 
        # NOT toward its parent 'Leukocyte'.
        local_counts = Counter()
        for leaf, count in leaf_counts.items():
            anchor = current_map[leaf]
            local_counts[anchor] += count

        moveable = [n for n in frontier if parents.get(n) and n not in forbidden]
        if not moveable: break
            
        # STEP B: Target the node with the fewest LOCAL cells
        target_node = min(
            moveable, 
            key=lambda x: (local_counts[x], -depth_map.get(x, 0), x)
        )
        
        # STEP C: Smart Parent Selection (The 'Where to send' logic)
        all_ps = list(parents[target_node] - forbidden)
        if not all_ps:
            frontier.remove(target_node)
            continue
            
        def parent_priority(p):
            # Priority 0: Parent is already an anchor (instant merge)
            if p in frontier:
                return (0, -global_prop[p], p)
            
            # Priority 1: Parent has other descendants in the frontier
            # (Check if any current anchor has p as an ancestor)
            has_descendant = any(p in all_ancestors(a, parents, {}) for a in frontier if a != target_node)
            if has_descendant:
                return (1, -global_prop[p], p)
            
            # Priority 2: Generic merge (pick the most 'significant' parent)
            return (2, -global_prop[p], p)
        
        chosen_parent = min(all_ps, key=parent_priority)
        
        # STEP D: Move the cells.
        # Now those 'local' cells from target_node belong to chosen_parent.
        frontier.remove(target_node)
        frontier.add(chosen_parent)
        for leaf in current_map:
            if current_map[leaf] == target_node:
                current_map[leaf] = chosen_parent

    coarse_ids = [current_map.get(l, l) for l in leaf_ids]
    return coarse_ids, frontier, current_map

def visualize_cell_ontology(
    target_nodes: set[str],
    parents: dict[str, set[str]],
    labels: dict[str, str],
    propagated_counts: Counter,
    output_path: str = "cell_tree",
    highlight_color: str = "#AED6F1",
    aspect_ratio: float = 0.5625,  # Default to 9/16 for 16:9 aspect ratio
    dpi: int = 2200  # <--- Added for high-resolution output
):
    """
    A unified function to visualize any subset of the ontology.
    - target_nodes: The 'leaves' of your current view (either original leaves or pruned anchors).
    - parents: The ontology structure.
    """
    dot = graphviz.Digraph(format='png', strict=True)
    
    # Core Attributes for Widescreen High-Res
    dot.attr(
        rankdir='TB', 
        nodesep='0.2',    
        ranksep='0.5',    
        #ratio=str(aspect_ratio), 
        size='16,9',     
        dpi=str(dpi),     # <--- Crucial: Scales the pixel density
        splines='polyline',
        overlap='false',   # Prevents nodes from sitting on top of each other
        concentrate='true',
        fontsize='24'
    )

    def safe_id(cl_id: str) -> str:
        return str(cl_id).replace(":", "_")

    # 1. Trace paths to root
    visual_nodes = set(target_nodes)
    memo = {}
    for node in target_nodes:
        visual_nodes |= all_ancestors(node, parents, memo)
        
    # 2. Add Nodes
    for node in visual_nodes:
        name = labels.get(node, node)
        count = propagated_counts.get(node, 0)
        display_label = f"{name}\n({node})\nCells: {count:,}"
        
        if node in target_nodes:
            # Highlighted terminal nodes (Squares)
            dot.node(safe_id(node), display_label, style='filled', 
                     fillcolor=highlight_color, shape='box', penwidth='1.5')
        else:
            # Scaffolding (Circles)
            dot.node(safe_id(node), display_label, shape='ellipse', 
                     color='#808080', fontcolor='#444444', fontsize='10')

    # 3. Add Edges
    for node in visual_nodes:
        for p in parents.get(node, []):
            if p in visual_nodes:
                dot.edge(safe_id(p), safe_id(node), color='#AAAAAA')

    return dot.render(output_path, cleanup=True)