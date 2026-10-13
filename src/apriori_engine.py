"""
apriori_engine.py - Educational Apriori implementation with full intermediate state tracking.

This module implements the Apriori algorithm step-by-step, exposing all
intermediate states (C_k candidates, pruning decisions, L_k frequent itemsets)
for educational transparency.
"""

import time
from itertools import combinations
from collections import defaultdict
import pandas as pd
import numpy as np


def df_to_transactions(df_long):
    """Convert long-format DataFrame to list of frozensets."""
    return (
        df_long.groupby("transaction_id")["item"]
        .apply(lambda x: frozenset(x))
        .tolist()
    )


def count_support(transactions, itemsets):
    """
    Count the support (number of transactions containing each itemset).
    
    Parameters
    ----------
    transactions : list of frozenset
    itemsets : iterable of frozenset
    
    Returns
    -------
    dict : {frozenset: int} support counts
    """
    counts = defaultdict(int)
    itemsets_list = list(itemsets)
    for transaction in transactions:
        for itemset in itemsets_list:
            if itemset.issubset(transaction):
                counts[itemset] += 1
    return dict(counts)


def generate_c1(transactions):
    """
    Generate C1 - all candidate 1-itemsets with their support counts.
    
    Returns
    -------
    list of dict : Each dict has 'itemset', 'support_count', 'items'
    """
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1

    c1 = []
    for itemset, count in sorted(item_counts.items(),
                                  key=lambda x: (-x[1], str(sorted(x[0])))):
        c1.append({
            "itemset": itemset,
            "support_count": count,
            "items": sorted(itemset),
        })
    return c1


def generate_candidates(l_prev, k):
    """
    Generate candidate k-itemsets from frequent (k-1)-itemsets using the
    Apriori candidate generation (join step).
    
    For k=2: combine all pairs of frequent 1-itemsets.
    For k>2: join L_{k-1} items that share k-2 items.
    
    Parameters
    ----------
    l_prev : list of frozenset - frequent (k-1)-itemsets
    k : int - target itemset size
    
    Returns
    -------
    set of frozenset : candidate k-itemsets
    """
    candidates = set()

    if k == 2:
        # Simple pairwise combination for 2-itemsets
        for i in range(len(l_prev)):
            for j in range(i + 1, len(l_prev)):
                candidate = l_prev[i] | l_prev[j]
                if len(candidate) == k:
                    candidates.add(candidate)
    else:
        # Join step: pairs of (k-1)-itemsets sharing (k-2) items
        l_prev_sorted = sorted(l_prev, key=lambda x: sorted(x))
        for i in range(len(l_prev_sorted)):
            for j in range(i + 1, len(l_prev_sorted)):
                union = l_prev_sorted[i] | l_prev_sorted[j]
                if len(union) == k:
                    candidates.add(union)

    return candidates


def apriori_prune(candidates, l_prev, k):
    """
    Apply the Apriori property: prune candidates whose (k-1)-subsets
    are not all in L_{k-1}.
    
    Parameters
    ----------
    candidates : set of frozenset
    l_prev : set of frozenset - frequent (k-1)-itemsets
    k : int
    
    Returns
    -------
    tuple : (surviving_candidates, pruning_details)
        pruning_details is a list of dicts describing each pruning decision
    """
    l_prev_set = set(l_prev)
    survived = set()
    pruning_details = []

    for candidate in candidates:
        pruned = False
        infrequent_subset = None

        # Check all (k-1)-subsets
        for subset in combinations(candidate, k - 1):
            subset_fs = frozenset(subset)
            if subset_fs not in l_prev_set:
                pruned = True
                infrequent_subset = subset_fs
                break

        if pruned:
            pruning_details.append({
                "candidate": candidate,
                "items": sorted(candidate),
                "reason": f"Subset {{{', '.join(sorted(infrequent_subset))}}} is not frequent",
                "infrequent_subset": sorted(infrequent_subset),
            })
        else:
            survived.add(candidate)

    return survived, pruning_details


def run_apriori(transactions, min_support, min_confidence=0.5,
                max_k=None, progress_callback=None):
    """
    Run the complete Apriori algorithm with full intermediate state tracking.
    
    Parameters
    ----------
    transactions : list of frozenset
        Transaction database
    min_support : float
        Minimum support threshold (0.0 to 1.0, as a fraction)
    min_confidence : float
        Minimum confidence for rule generation (0.0 to 1.0)
    max_k : int or None
        Maximum itemset size. None means no limit.
    progress_callback : callable or None
        Called with (step_description, progress_fraction) for UI feedback
    
    Returns
    -------
    dict : Complete execution result with all intermediate states
    """
    start_time = time.time()
    if min_support <= 0.0 or min_support > 1.0:
        raise ValueError(f"Invalid min_support: {min_support}. Must be strictly greater than 0.0 and <= 1.0.")
    if min_confidence <= 0.0 or min_confidence > 1.0:
        raise ValueError(f"Invalid min_confidence: {min_confidence}. Must be strictly greater than 0.0 and <= 1.0.")
    n_transactions = len(transactions)
    if n_transactions == 0:
        return {
            "n_transactions": 0,
            "min_support": min_support,
            "min_support_count": 0,
            "min_confidence": min_confidence,
            "max_k": max_k,
            "iterations": [],
            "all_frequent_itemsets": [],
            "total_candidates": 0,
            "total_pruned": 0,
            "total_db_scans": 0,
            "termination_reason": "Database contains 0 transactions.",
            "execution_time": 0,
            "max_k_reached": 0,
            "limited_by_config": False,
        }

    min_support_count = max(1, int(np.ceil(min_support * n_transactions)))

    result = {
        "n_transactions": n_transactions,
        "min_support": min_support,
        "min_support_count": min_support_count,
        "min_confidence": min_confidence,
        "max_k": max_k,
        "iterations": [],
        "all_frequent_itemsets": [],
        "total_candidates": 0,
        "total_pruned": 0,
        "total_db_scans": 0,
        "termination_reason": "",
        "execution_time": 0,
        "max_k_reached": 0,
        "limited_by_config": False,
    }

    if progress_callback:
        progress_callback("Generating C1...", 0.05)

    # --- Step 1: Generate C1 ---
    c1 = generate_c1(transactions)
    result["total_db_scans"] += 1

    c1_details = []
    for entry in c1:
        sup_pct = entry["support_count"] / n_transactions
        c1_details.append({
            "itemset": entry["itemset"],
            "items": entry["items"],
            "support_count": entry["support_count"],
            "support_pct": sup_pct,
            "is_frequent": entry["support_count"] >= min_support_count,
        })

    # --- Step 2: Generate L1 ---
    l1 = [e["itemset"] for e in c1_details if e["is_frequent"]]

    iteration_1 = {
        "k": 1,
        "candidates": c1_details,
        "candidate_count": len(c1_details),
        "pruned_details": [],
        "pruned_count": 0,
        "frequent_itemsets": [
            e for e in c1_details if e["is_frequent"]
        ],
        "frequent_count": len(l1),
        "removed_items": [
            e for e in c1_details if not e["is_frequent"]
        ],
    }
    result["iterations"].append(iteration_1)
    result["total_candidates"] += len(c1_details)
    result["all_frequent_itemsets"].extend(l1)
    result["max_k_reached"] = 1

    if not l1:
        result["termination_reason"] = (
            "No frequent 1-itemsets found. All items have support below "
            f"the minimum threshold ({min_support:.1%})."
        )
        result["execution_time"] = time.time() - start_time
        return result

    # --- Steps 3+: Iterative candidate generation ---
    l_prev = l1
    k = 2

    while True:
        if max_k and k > max_k:
            result["termination_reason"] = (
                f"Maximum itemset size k={max_k} reached. "
                "Larger frequent itemsets may exist but were not searched."
            )
            result["limited_by_config"] = True
            break

        if progress_callback:
            progress_callback(f"Generating C{k}...", min(0.9, 0.1 + 0.8 * (k / 10)))

        # Generate candidates
        candidates = generate_candidates(l_prev, k)
        n_candidates_before_prune = len(candidates)

        if not candidates:
            result["termination_reason"] = (
                f"No candidate {k}-itemsets could be generated from "
                f"the {len(l_prev)} frequent {k-1}-itemset(s). "
                "Algorithm terminated naturally."
            )
            break

        # Apriori pruning
        survived, pruning_details = apriori_prune(candidates, l_prev, k)
        n_pruned = len(pruning_details)

        if not survived:
            # All candidates pruned
            iteration_k = {
                "k": k,
                "candidates": [
                    {"itemset": c, "items": sorted(c),
                     "support_count": 0, "support_pct": 0.0,
                     "is_frequent": False}
                    for c in candidates
                ],
                "candidate_count": n_candidates_before_prune,
                "pruned_details": pruning_details,
                "pruned_count": n_pruned,
                "frequent_itemsets": [],
                "frequent_count": 0,
                "removed_items": [],
            }
            result["iterations"].append(iteration_k)
            result["total_candidates"] += n_candidates_before_prune
            result["total_pruned"] += n_pruned
            result["termination_reason"] = (
                f"All candidate {k}-itemsets were pruned by the Apriori property. "
                "Algorithm terminated naturally."
            )
            break

        # Count support for surviving candidates
        support_counts = count_support(transactions, survived)
        result["total_db_scans"] += 1

        # Build candidate details
        candidate_details = []
        for candidate in sorted(survived, key=lambda x: sorted(x)):
            count = support_counts.get(candidate, 0)
            sup_pct = count / n_transactions
            candidate_details.append({
                "itemset": candidate,
                "items": sorted(candidate),
                "support_count": count,
                "support_pct": sup_pct,
                "is_frequent": count >= min_support_count,
            })

        # Frequent k-itemsets
        l_k = [e["itemset"] for e in candidate_details if e["is_frequent"]]

        iteration_k = {
            "k": k,
            "candidates": candidate_details,
            "candidate_count": n_candidates_before_prune,
            "pruned_details": pruning_details,
            "pruned_count": n_pruned,
            "frequent_itemsets": [
                e for e in candidate_details if e["is_frequent"]
            ],
            "frequent_count": len(l_k),
            "removed_items": [
                e for e in candidate_details if not e["is_frequent"]
            ],
        }
        result["iterations"].append(iteration_k)
        result["total_candidates"] += n_candidates_before_prune
        result["total_pruned"] += n_pruned
        result["all_frequent_itemsets"].extend(l_k)
        result["max_k_reached"] = k

        if not l_k:
            result["termination_reason"] = (
                f"No frequent {k}-itemsets found after support counting. "
                "Algorithm terminated naturally."
            )
            break

        l_prev = l_k
        k += 1

    result["execution_time"] = time.time() - start_time

    if progress_callback:
        progress_callback("Complete", 1.0)

    return result


def generate_rules(apriori_result, transactions, min_confidence=None):
    """
    Generate association rules from frequent itemsets.
    
    For every frequent itemset Z with |Z| >= 2, enumerate all non-empty
    proper subsets X as antecedents with Y = Z - X as consequent.
    
    Parameters
    ----------
    apriori_result : dict
        Result from run_apriori()
    transactions : list of frozenset
    min_confidence : float or None
        Override min_confidence from apriori_result
    
    Returns
    -------
    list of dict : All candidate rules with metrics
    """
    if min_confidence is None:
        min_confidence = apriori_result["min_confidence"]

    if min_confidence <= 0.0 or min_confidence > 1.0:
        raise ValueError(f"Invalid min_confidence: {min_confidence}. Must be strictly greater than 0.0 and <= 1.0.")

    n = apriori_result.get("n_transactions", 0)
    frequent = apriori_result.get("all_frequent_itemsets", [])
    if n == 0 or not frequent:
        return []

    # Pre-compute support for all frequent itemsets
    support_map = {}
    for iteration in apriori_result["iterations"]:
        for entry in iteration.get("frequent_itemsets", []):
            support_map[entry["itemset"]] = entry["support_count"]

    # Also need support for individual items as consequents
    # Compute any missing supports
    all_items_in_frequent = set()
    for fs in frequent:
        all_items_in_frequent.update(fs)

    for item in all_items_in_frequent:
        single = frozenset([item])
        if single not in support_map:
            count = sum(1 for t in transactions if single.issubset(t))
            support_map[single] = count

    rules = []
    for itemset in frequent:
        if len(itemset) < 2:
            continue

        # Enumerate all non-empty proper subsets as antecedents
        items = sorted(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                if not consequent:
                    continue

                # Get supports
                sup_union = support_map.get(itemset, 0)
                sup_ante = support_map.get(antecedent, 0)
                sup_cons = support_map.get(consequent, 0)

                # Compute supports for subsets if not already tracked
                if sup_ante == 0 and antecedent not in support_map:
                    sup_ante = sum(
                        1 for t in transactions if antecedent.issubset(t)
                    )
                    support_map[antecedent] = sup_ante

                if sup_cons == 0 and consequent not in support_map:
                    sup_cons = sum(
                        1 for t in transactions if consequent.issubset(t)
                    )
                    support_map[consequent] = sup_cons

                # Calculate metrics
                confidence = sup_union / sup_ante if sup_ante > 0 else 0
                lift = (confidence / (sup_cons / n)) if sup_cons > 0 else 0

                rule = {
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "antecedent_items": sorted(antecedent),
                    "consequent_items": sorted(consequent),
                    "support": sup_union / n,
                    "support_count": sup_union,
                    "support_pct": sup_union / n,
                    "confidence": confidence,
                    "lift": lift,
                    "antecedent_support": sup_ante / n,
                    "consequent_support": sup_cons / n,
                    "accepted": confidence >= min_confidence,
                    "lift_interpretation": (
                        "Statistical independence"
                        if abs(lift - 1.0) < 1e-5
                        else ("Positive association" if lift > 1.0 else "Negative association")
                    ),
                }
                rules.append(rule)

    # Sort by confidence descending, then by support descending
    rules.sort(key=lambda r: (-r["confidence"], -r["support_pct"]))
    return rules


def get_recommendations(cart_items, rules, n_recommendations=10,
                        sort_by="confidence"):
    """
    Given a shopping cart, find applicable rules and recommend products.
    
    Parameters
    ----------
    cart_items : set of str
    rules : list of dict (from generate_rules, accepted rules only)
    n_recommendations : int
    sort_by : str - 'confidence', 'lift', or 'support'
    
    Returns
    -------
    list of dict : Recommended products with supporting rule info
    """
    if not cart_items or not rules:
        return []

    cart_set = set(cart_items)
    applicable = []

    for rule in rules:
        if not rule["accepted"]:
            continue
        # Check if antecedent is a subset of the cart
        if rule["antecedent"].issubset(cart_set):
            # Check if consequent items are NOT in the cart
            new_items = rule["consequent"] - cart_set
            if new_items:
                sorted_new = sorted(new_items)
                applicable.append({
                    "item": sorted_new[0],
                    "recommended_items": sorted_new,
                    "antecedent": rule["antecedent_items"],
                    "consequent": rule["consequent_items"],
                    "support": rule["support_pct"],
                    "support_count": rule["support_count"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                    "lift_interpretation": rule["lift_interpretation"],
                    "rule_str": (
                        f"{{{', '.join(rule['antecedent_items'])}}} "
                        f"-> {{{', '.join(rule['consequent_items'])}}}"
                    ),
                })

    # Sort
    sort_key = {
        "confidence": lambda x: (-x["confidence"], -x["lift"]),
        "lift": lambda x: (-x["lift"], -x["confidence"]),
        "support": lambda x: (-x["support"], -x["confidence"]),
    }.get(sort_by, lambda x: (-x["confidence"], -x["lift"]))
    applicable.sort(key=sort_key)

    # Deduplicate by recommended product (keep strongest rule)
    seen_products = set()
    unique_recs = []
    for rec in applicable:
        item_key = rec.get("item") or (rec["recommended_items"][0] if rec["recommended_items"] else None)
        if item_key and item_key not in seen_products:
            seen_products.add(item_key)
            unique_recs.append(rec)
        if len(unique_recs) >= n_recommendations:
            break

    return unique_recs
