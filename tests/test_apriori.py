"""
test_apriori.py - Unit tests for the SmartCart Apriori engine.

Tests mathematical correctness of support, confidence, lift calculations,
candidate generation, Apriori pruning, and rule generation using a small,
manually verifiable dataset. Also validates consistency with mlxtend.

Run with:  python -m pytest test_apriori.py -v
"""

import pytest
import numpy as np
import pandas as pd
from collections import Counter

from src.apriori_engine import (
    df_to_transactions,
    count_support,
    generate_c1,
    generate_candidates,
    apriori_prune,
    run_apriori,
    generate_rules,
    get_recommendations,
)
from src.datasets import (
    load_dataset,
    get_dataset_names,
    compute_dataset_stats,
    parse_uploaded_csv,
    generate_dataset_a,
)


# ---------------------------------------------------------------------------
# Tiny manually-verifiable dataset
# ---------------------------------------------------------------------------
TINY_TRANSACTIONS_RAW = [
    ["A", "B", "C"],
    ["A", "B"],
    ["A", "C"],
    ["B", "C"],
    ["A", "B", "C"],
]
TINY_TRANSACTIONS = [frozenset(t) for t in TINY_TRANSACTIONS_RAW]
TINY_N = 5  # total transactions


def tiny_df():
    """Build a long-format DataFrame for the tiny dataset."""
    rows = []
    for i, items in enumerate(TINY_TRANSACTIONS_RAW):
        for item in items:
            rows.append({"transaction_id": f"T{i+1}", "item": item})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Support count tests
# ---------------------------------------------------------------------------
class TestSupportCounting:
    def test_single_item_support(self):
        # A appears in T1, T2, T3, T5 => support = 4
        counts = count_support(TINY_TRANSACTIONS, [frozenset(["A"])])
        assert counts[frozenset(["A"])] == 4

    def test_single_item_B(self):
        # B appears in T1, T2, T4, T5 => support = 4
        counts = count_support(TINY_TRANSACTIONS, [frozenset(["B"])])
        assert counts[frozenset(["B"])] == 4

    def test_single_item_C(self):
        # C appears in T1, T3, T4, T5 => support = 4
        counts = count_support(TINY_TRANSACTIONS, [frozenset(["C"])])
        assert counts[frozenset(["C"])] == 4

    def test_pair_AB(self):
        # {A,B} in T1, T2, T5 => support = 3
        counts = count_support(TINY_TRANSACTIONS, [frozenset(["A", "B"])])
        assert counts[frozenset(["A", "B"])] == 3

    def test_pair_AC(self):
        # {A,C} in T1, T3, T5 => support = 3
        counts = count_support(TINY_TRANSACTIONS, [frozenset(["A", "C"])])
        assert counts[frozenset(["A", "C"])] == 3

    def test_pair_BC(self):
        # {B,C} in T1, T4, T5 => support = 3
        counts = count_support(TINY_TRANSACTIONS, [frozenset(["B", "C"])])
        assert counts[frozenset(["B", "C"])] == 3

    def test_triple_ABC(self):
        # {A,B,C} in T1, T5 => support = 2
        counts = count_support(TINY_TRANSACTIONS, [frozenset(["A", "B", "C"])])
        assert counts[frozenset(["A", "B", "C"])] == 2

    def test_empty_itemset(self):
        counts = count_support(TINY_TRANSACTIONS, [frozenset()])
        assert counts[frozenset()] == TINY_N

    def test_nonexistent_item(self):
        counts = count_support(TINY_TRANSACTIONS, [frozenset(["Z"])])
        assert counts.get(frozenset(["Z"]), 0) == 0


# ---------------------------------------------------------------------------
# C1 generation tests
# ---------------------------------------------------------------------------
class TestC1Generation:
    def test_c1_counts(self):
        c1 = generate_c1(TINY_TRANSACTIONS)
        items = {frozenset(e["items"]): e["support_count"] for e in c1}
        assert items[frozenset(["A"])] == 4
        assert items[frozenset(["B"])] == 4
        assert items[frozenset(["C"])] == 4

    def test_c1_length(self):
        c1 = generate_c1(TINY_TRANSACTIONS)
        assert len(c1) == 3  # A, B, C


# ---------------------------------------------------------------------------
# Candidate generation tests
# ---------------------------------------------------------------------------
class TestCandidateGeneration:
    def test_c2_from_l1(self):
        l1 = [frozenset(["A"]), frozenset(["B"]), frozenset(["C"])]
        c2 = generate_candidates(l1, 2)
        expected = {frozenset(["A", "B"]), frozenset(["A", "C"]), frozenset(["B", "C"])}
        assert c2 == expected

    def test_c3_from_l2(self):
        l2 = [frozenset(["A", "B"]), frozenset(["A", "C"]), frozenset(["B", "C"])]
        c3 = generate_candidates(l2, 3)
        assert frozenset(["A", "B", "C"]) in c3

    def test_no_candidates_when_impossible(self):
        # Only one L2 itemset, can't form C3
        l2 = [frozenset(["A", "B"])]
        c3 = generate_candidates(l2, 3)
        assert len(c3) == 0


# ---------------------------------------------------------------------------
# Apriori pruning tests
# ---------------------------------------------------------------------------
class TestAprioriPruning:
    def test_no_pruning_when_all_subsets_frequent(self):
        l1 = {frozenset(["A"]), frozenset(["B"]), frozenset(["C"])}
        candidates = {frozenset(["A", "B"]), frozenset(["A", "C"])}
        survived, pruned = apriori_prune(candidates, l1, 2)
        assert survived == candidates
        assert len(pruned) == 0

    def test_pruning_with_infrequent_subset(self):
        # B is not in L1 => {A, B} should be pruned
        l1 = {frozenset(["A"]), frozenset(["C"])}
        candidates = {frozenset(["A", "B"]), frozenset(["A", "C"])}
        survived, pruned = apriori_prune(candidates, l1, 2)
        assert frozenset(["A", "C"]) in survived
        assert frozenset(["A", "B"]) not in survived
        assert len(pruned) == 1

    def test_pruning_c3(self):
        # {A,B} is infrequent => {A,B,C} should be pruned
        l2 = {frozenset(["A", "C"]), frozenset(["B", "C"])}
        candidates = {frozenset(["A", "B", "C"])}
        survived, pruned = apriori_prune(candidates, l2, 3)
        assert len(survived) == 0
        assert len(pruned) == 1


# ---------------------------------------------------------------------------
# Full Apriori execution tests
# ---------------------------------------------------------------------------
class TestAprioriExecution:
    def test_tiny_dataset_min_sup_40pct(self):
        # min_sup = 0.4 => min_count = ceil(0.4 * 5) = 2
        result = run_apriori(TINY_TRANSACTIONS, min_support=0.4)
        freq = result["all_frequent_itemsets"]

        # All single items have support 4/5 = 0.8 >= 0.4 ✓
        assert frozenset(["A"]) in freq
        assert frozenset(["B"]) in freq
        assert frozenset(["C"]) in freq

        # All pairs have support 3/5 = 0.6 >= 0.4 ✓
        assert frozenset(["A", "B"]) in freq
        assert frozenset(["A", "C"]) in freq
        assert frozenset(["B", "C"]) in freq

        # Triple has support 2/5 = 0.4 >= 0.4 ✓
        assert frozenset(["A", "B", "C"]) in freq

    def test_tiny_dataset_min_sup_70pct(self):
        # min_sup = 0.7 => min_count = ceil(0.7 * 5) = 4
        result = run_apriori(TINY_TRANSACTIONS, min_support=0.7)
        freq = result["all_frequent_itemsets"]

        # All single items have support 4 >= 4 ✓
        assert frozenset(["A"]) in freq
        assert frozenset(["B"]) in freq
        assert frozenset(["C"]) in freq

        # Pairs have support 3 < 4 ✗
        assert frozenset(["A", "B"]) not in freq

    def test_min_support_boundary(self):
        # min_sup = 3/5 = 0.6 => min_count = ceil(0.6 * 5) = 3
        # Pairs have count 3 >= 3 ✓
        result = run_apriori(TINY_TRANSACTIONS, min_support=0.6)
        freq = result["all_frequent_itemsets"]
        assert frozenset(["A", "B"]) in freq
        assert frozenset(["A", "C"]) in freq
        assert frozenset(["B", "C"]) in freq

    def test_max_k_limits_search(self):
        result = run_apriori(TINY_TRANSACTIONS, min_support=0.4, max_k=2)
        freq = result["all_frequent_itemsets"]

        # Should have 1-itemsets and 2-itemsets but NOT 3-itemsets
        assert frozenset(["A", "B"]) in freq
        assert frozenset(["A", "B", "C"]) not in freq
        assert result["limited_by_config"] is True

    def test_no_frequent_itemsets(self):
        # min_sup = 1.0 => min_count = 5
        # Only items appearing in all 5 transactions would qualify
        # But no item appears in all 5... wait, A appears in 4, so none at 100%
        result = run_apriori(TINY_TRANSACTIONS, min_support=1.0)
        # With min_count = 5, no items have support >= 5
        assert len(result["all_frequent_itemsets"]) == 0

    def test_iteration_counts_are_consistent(self):
        result = run_apriori(TINY_TRANSACTIONS, min_support=0.4)
        total_freq = sum(it["frequent_count"] for it in result["iterations"])
        assert total_freq == len(result["all_frequent_itemsets"])


# ---------------------------------------------------------------------------
# Rule generation tests
# ---------------------------------------------------------------------------
class TestRuleGeneration:
    def setup_method(self):
        self.result = run_apriori(TINY_TRANSACTIONS, min_support=0.4,
                                   min_confidence=0.5)
        self.rules = generate_rules(self.result, TINY_TRANSACTIONS,
                                     min_confidence=0.5)

    def test_rules_generated(self):
        assert len(self.rules) > 0

    def test_antecedent_consequent_disjoint(self):
        for rule in self.rules:
            assert rule["antecedent"] & rule["consequent"] == frozenset()

    def test_antecedent_consequent_nonempty(self):
        for rule in self.rules:
            assert len(rule["antecedent"]) > 0
            assert len(rule["consequent"]) > 0

    def test_confidence_calculation(self):
        # {A} -> {B}: conf = support({A,B}) / support({A}) = 3/4 = 0.75
        a_to_b = [r for r in self.rules
                   if r["antecedent"] == frozenset(["A"])
                   and r["consequent"] == frozenset(["B"])]
        assert len(a_to_b) == 1
        assert abs(a_to_b[0]["confidence"] - 0.75) < 1e-6

    def test_lift_calculation(self):
        # {A} -> {B}: lift = conf / support(B) = 0.75 / (4/5) = 0.9375
        a_to_b = [r for r in self.rules
                   if r["antecedent"] == frozenset(["A"])
                   and r["consequent"] == frozenset(["B"])]
        assert len(a_to_b) == 1
        assert abs(a_to_b[0]["lift"] - 0.9375) < 1e-6

    def test_support_calculation(self):
        # {A} -> {B}: support = support({A,B}) = 3/5 = 0.6
        a_to_b = [r for r in self.rules
                   if r["antecedent"] == frozenset(["A"])
                   and r["consequent"] == frozenset(["B"])]
        assert abs(a_to_b[0]["support_pct"] - 0.6) < 1e-6

    def test_min_confidence_filtering(self):
        rules_strict = generate_rules(self.result, TINY_TRANSACTIONS,
                                       min_confidence=0.8)
        for rule in rules_strict:
            if rule["accepted"]:
                assert rule["confidence"] >= 0.8

    def test_rules_from_frequent_itemsets_only(self):
        freq_set = set(frozenset(f) for f in self.result["all_frequent_itemsets"])
        for rule in self.rules:
            union = rule["antecedent"] | rule["consequent"]
            assert union in freq_set


# ---------------------------------------------------------------------------
# Recommendation tests
# ---------------------------------------------------------------------------
class TestRecommendations:
    def setup_method(self):
        result = run_apriori(TINY_TRANSACTIONS, min_support=0.4,
                              min_confidence=0.5)
        self.rules = generate_rules(result, TINY_TRANSACTIONS,
                                     min_confidence=0.5)
        self.accepted = [r for r in self.rules if r["accepted"]]

    def test_recommendation_excludes_cart_items(self):
        recs = get_recommendations({"A"}, self.accepted)
        for rec in recs:
            assert "A" not in rec["recommended_items"]

    def test_recommendation_with_empty_cart(self):
        recs = get_recommendations(set(), self.accepted)
        assert len(recs) == 0

    def test_recommendation_all_items_in_cart(self):
        recs = get_recommendations({"A", "B", "C"}, self.accepted)
        assert len(recs) == 0


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------
class TestDatasets:
    def test_all_datasets_load(self):
        for name in get_dataset_names():
            df = load_dataset(name)
            assert len(df) > 0
            assert "transaction_id" in df.columns
            assert "item" in df.columns

    def test_dataset_a_small(self):
        df = load_dataset("Dataset A: Classroom Example")
        stats = compute_dataset_stats(df)
        assert stats["n_transactions"] <= 20
        assert stats["n_unique_items"] <= 20

    def test_dataset_stats_computation(self):
        df = load_dataset("Dataset A: Classroom Example")
        stats = compute_dataset_stats(df)
        assert "n_transactions" in stats
        assert "n_unique_items" in stats
        assert "avg_basket_size" in stats
        assert stats["min_basket_size"] >= 1
        assert stats["max_basket_size"] >= stats["min_basket_size"]


# ---------------------------------------------------------------------------
# CSV parsing tests
# ---------------------------------------------------------------------------
class TestCSVParsing:
    def test_long_format(self):
        csv_data = "transaction_id,item\nT1,A\nT1,B\nT2,C\n"
        df = parse_uploaded_csv(pd.io.common.StringIO(csv_data))
        assert len(df) == 3
        assert set(df["transaction_id"]) == {"T1", "T2"}

    def test_duplicate_removal(self):
        csv_data = "transaction_id,item\nT1,A\nT1,A\nT1,B\n"
        df = parse_uploaded_csv(pd.io.common.StringIO(csv_data))
        assert len(df) == 2  # duplicate T1,A removed

    def test_wide_format(self):
        csv_data = "tid,A,B,C\nT1,1,1,0\nT2,0,1,1\n"
        df = parse_uploaded_csv(pd.io.common.StringIO(csv_data))
        assert len(df) == 4  # T1-A, T1-B, T2-B, T2-C


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_single_transaction(self):
        txns = [frozenset(["A", "B"])]
        result = run_apriori(txns, min_support=0.5)
        assert frozenset(["A"]) in result["all_frequent_itemsets"]

    def test_all_same_items(self):
        txns = [frozenset(["A"])] * 10
        result = run_apriori(txns, min_support=0.5)
        assert frozenset(["A"]) in result["all_frequent_itemsets"]
        # No 2-itemsets possible
        assert result["max_k_reached"] == 1

    def test_confidence_at_threshold(self):
        # Build scenario where confidence equals threshold exactly
        result = run_apriori(TINY_TRANSACTIONS, min_support=0.4,
                              min_confidence=0.75)
        rules = generate_rules(result, TINY_TRANSACTIONS,
                                min_confidence=0.75)
        # {A} -> {B} has confidence = 0.75, should be accepted
        a_to_b = [r for r in rules
                   if r["antecedent"] == frozenset(["A"])
                   and r["consequent"] == frozenset(["B"])]
        assert len(a_to_b) == 1
        assert a_to_b[0]["accepted"] is True

    def test_invalid_min_support_raises_error(self):
        with pytest.raises(ValueError, match="Invalid min_support"):
            run_apriori(TINY_TRANSACTIONS, min_support=0.0)
        with pytest.raises(ValueError, match="Invalid min_support"):
            run_apriori(TINY_TRANSACTIONS, min_support=-0.1)
        with pytest.raises(ValueError, match="Invalid min_support"):
            run_apriori(TINY_TRANSACTIONS, min_support=1.5)

    def test_invalid_min_confidence_raises_error(self):
        with pytest.raises(ValueError, match="Invalid min_confidence"):
            run_apriori(TINY_TRANSACTIONS, min_support=0.4, min_confidence=0.0)
        with pytest.raises(ValueError, match="Invalid min_confidence"):
            run_apriori(TINY_TRANSACTIONS, min_support=0.4, min_confidence=-0.5)
        with pytest.raises(ValueError, match="Invalid min_confidence"):
            run_apriori(TINY_TRANSACTIONS, min_support=0.4, min_confidence=1.2)


# ---------------------------------------------------------------------------
# mlxtend consistency test
# ---------------------------------------------------------------------------
class TestMlxtendConsistency:
    def test_consistency_with_mlxtend(self):
        """Verify our implementation matches mlxtend on the tiny dataset."""
        try:
            from mlxtend.frequent_patterns import apriori as mlx_apriori
            from mlxtend.frequent_patterns import association_rules as mlx_rules
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError:
            pytest.skip("mlxtend not installed")

        # Convert to mlxtend format
        te = TransactionEncoder()
        te_ary = te.fit_transform(TINY_TRANSACTIONS_RAW)
        df_mlx = pd.DataFrame(te_ary, columns=te.columns_)

        # Run mlxtend
        freq_mlx = mlx_apriori(df_mlx, min_support=0.4, use_colnames=True)

        # Run our implementation
        our_result = run_apriori(TINY_TRANSACTIONS, min_support=0.4)

        # Compare frequent itemsets
        mlx_itemsets = set(
            frozenset(row["itemsets"])
            for _, row in freq_mlx.iterrows()
        )
        our_itemsets = set(our_result["all_frequent_itemsets"])

        assert mlx_itemsets == our_itemsets, (
            f"Mismatch!\nmlxtend: {mlx_itemsets}\nOurs: {our_itemsets}"
        )

        # Compare rules
        if len(freq_mlx) > 0:
            rules_mlx = mlx_rules(
                freq_mlx, metric="confidence", min_threshold=0.5,
                num_itemsets=len(TINY_TRANSACTIONS)
            )

            our_rules = generate_rules(
                our_result, TINY_TRANSACTIONS, min_confidence=0.5
            )
            our_accepted = [r for r in our_rules if r["accepted"]]

            # Check that every mlxtend rule exists in our results
            for _, mlx_rule in rules_mlx.iterrows():
                ante = frozenset(mlx_rule["antecedents"])
                cons = frozenset(mlx_rule["consequents"])
                matching = [
                    r for r in our_accepted
                    if r["antecedent"] == ante and r["consequent"] == cons
                ]
                assert len(matching) == 1, (
                    f"Rule {ante} -> {cons} from mlxtend not found in ours"
                )
                # Compare confidence
                assert abs(matching[0]["confidence"] - mlx_rule["confidence"]) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
