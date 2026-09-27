"""Candidate-side field-subset selection.

We choose the candidate template's field subset G_c* that maximizes retrieval
effectiveness on a held-out validation set:

    G_c* = argmax_{G_c subseteq G}  nDCG@k(G_c; V)

Because |G| = 6 the power set has 64 elements, so exhaustive search is cheap.
We also provide forward greedy selection (start from {provenance}, add the
field with the largest marginal nDCG gain until the gain drops below eps)
to trace individual field contributions.

Both routines take a ``scorer`` callable that evaluates a field subset on the
validation set and returns nDCG@k; this keeps selection decoupled from the
encoder / index implementation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterable, Optional, Sequence


Scorer = Callable[[frozenset], float]

# fields eligible for selection (provenance is always retained: it is required
# for the online provenance filter and is never rendered into the embed text).
DEFAULT_FIELDS: tuple[str, ...] = ("phase", "summary", "targets", "error", "resolution")
ALWAYS_ON: frozenset = frozenset({"provenance"})


def exhaustive_selection(
    scorer: Scorer,
    fields: Sequence[str] = DEFAULT_FIELDS,
    always_on: Iterable[str] = ALWAYS_ON,
) -> tuple[frozenset, float, list[tuple[frozenset, float]]]:
    """Evaluate all 2^|fields| subsets of the renderable fields; return (best_subset, best_score, trace)."""
    base = frozenset(always_on)
    trace: list[tuple[frozenset, float]] = []
    best_subset, best_score = base, float("-inf")
    for r in range(len(fields) + 1):
        for combo in combinations(fields, r):
            subset = base | frozenset(combo)
            score = scorer(subset)
            trace.append((subset, score))
            if score > best_score:
                best_subset, best_score = subset, score
    trace.sort(key=lambda x: x[1], reverse=True)
    return best_subset, best_score, trace


def greedy_selection(
    scorer: Scorer,
    fields: Sequence[str] = DEFAULT_FIELDS,
    always_on: Iterable[str] = ALWAYS_ON,
    eps: float = 1e-3,
) -> tuple[frozenset, float, list[tuple[str, float]]]:
    """Forward greedy selection.

    Returns (selected_subset, score, path) where ``path`` lists the
    (added_field, nDCG_after_adding) decisions in order.
    """
    selected = frozenset(always_on)
    current = scorer(selected)
    remaining = list(fields)
    path: list[tuple[str, float]] = []
    while remaining:
        best_field: Optional[str] = None
        best_gain = eps
        best_after = current
        for f in remaining:
            after = scorer(selected | {f})
            gain = after - current
            if gain > best_gain:
                best_gain, best_field, best_after = gain, f, after
        if best_field is None:
            break
        selected = selected | {best_field}
        current = best_after
        remaining.remove(best_field)
        path.append((best_field, current))
    return selected, current, path


def validation_split(queries, val_frac=0.2, stratify_key=None, seed=0):
    """Hold out ``val_frac`` of ``queries`` for candidate-side field selection.

    Without ``stratify_key`` the tail of the list is held out (stable, no
    shuffle). With ``stratify_key`` (e.g. ``"task_type"``) the hold-out is drawn
    per stratum with a fixed ``seed``, so the validation set mirrors the stratum
    mix of the full query set, giving a stratified 80/20 split. Queries lacking
    the key form their own stratum, and every stratum contributes at least one
    query.
    Returns ``(test, val)`` preserving the original order within each list.
    """
    import random
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    queries = list(queries)
    if stratify_key is None:
        n_val = max(1, int(len(queries) * val_frac))
        return queries[:-n_val], queries[-n_val:]
    strata = {}
    for i, q in enumerate(queries):
        strata.setdefault(q.get(stratify_key), []).append(i)
    rng = random.Random(seed)
    val_idx = set()
    for key in sorted(strata, key=lambda k: (k is None, str(k))):
        idx = list(strata[key])
        rng.shuffle(idx)
        val_idx.update(idx[:max(1, int(round(len(idx) * val_frac)))])
    train = [q for i, q in enumerate(queries) if i not in val_idx]
    val = [q for i, q in enumerate(queries) if i in val_idx]
    return train, val
