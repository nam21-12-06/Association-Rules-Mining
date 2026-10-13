"""
datasets.py - Synthetic dataset generation and management for SmartCart.

Generates four built-in datasets of increasing size with realistic
co-purchase patterns. All datasets are synthetic and reproducible.
"""

import numpy as np
import pandas as pd
from itertools import combinations

SEED = 42

# ---------------------------------------------------------------------------
# Product catalog with categories and realistic price ranges
# ---------------------------------------------------------------------------
PRODUCT_CATALOG = {
    "Dairy": [
        ("Milk", 3.49), ("Butter", 4.29), ("Cheese", 5.99), ("Yogurt", 2.99),
        ("Cream", 3.79), ("Sour Cream", 2.49), ("Cottage Cheese", 3.29),
    ],
    "Bakery": [
        ("Bread", 2.99), ("Bagels", 3.49), ("Croissants", 4.99), ("Muffins", 3.99),
        ("Tortillas", 2.79), ("Rolls", 2.49),
    ],
    "Beverages": [
        ("Coffee", 8.99), ("Tea", 5.49), ("Orange Juice", 4.29), ("Apple Juice", 3.99),
        ("Soda", 1.99), ("Mineral Water", 1.49), ("Energy Drink", 2.99),
    ],
    "Snacks": [
        ("Chips", 3.99), ("Cookies", 4.49), ("Crackers", 3.29), ("Popcorn", 2.99),
        ("Pretzels", 2.49), ("Trail Mix", 5.99), ("Granola Bars", 4.29),
    ],
    "Meat & Protein": [
        ("Chicken Breast", 7.99), ("Ground Beef", 6.99), ("Bacon", 5.99),
        ("Sausage", 4.99), ("Ham", 5.49), ("Turkey", 8.99), ("Eggs", 3.99),
    ],
    "Produce": [
        ("Bananas", 0.99), ("Apples", 2.99), ("Tomatoes", 2.49), ("Lettuce", 1.99),
        ("Potatoes", 3.49), ("Onions", 1.49), ("Carrots", 1.99), ("Avocado", 1.99),
        ("Cucumber", 1.29), ("Bell Pepper", 1.79),
    ],
    "Pantry": [
        ("Rice", 4.99), ("Pasta", 1.99), ("Olive Oil", 7.99), ("Sugar", 3.49),
        ("Flour", 2.99), ("Salt", 1.29), ("Pepper", 2.49), ("Cereal", 4.49),
        ("Oatmeal", 3.99), ("Peanut Butter", 4.49), ("Jam", 3.99),
        ("Honey", 6.99), ("Ketchup", 2.99), ("Mustard", 2.49),
        ("Soy Sauce", 3.49), ("Vinegar", 2.99),
    ],
    "Frozen": [
        ("Frozen Pizza", 5.99), ("Ice Cream", 4.99), ("Frozen Vegetables", 2.99),
        ("Frozen Fish", 7.99), ("Frozen Berries", 4.49),
    ],
    "Household": [
        ("Paper Towels", 5.99), ("Dish Soap", 3.49), ("Trash Bags", 4.99),
        ("Aluminum Foil", 3.99), ("Plastic Wrap", 2.99), ("Napkins", 2.49),
    ],
    "Personal Care": [
        ("Shampoo", 6.99), ("Soap", 3.99), ("Toothpaste", 3.49),
        ("Deodorant", 4.99), ("Hand Lotion", 5.49),
    ],
}


def _get_product_info():
    """Build product -> (category, price) lookup."""
    info = {}
    for cat, items in PRODUCT_CATALOG.items():
        for name, price in items:
            info[name] = {"category": cat, "price": price}
    return info


PRODUCT_INFO = _get_product_info()
ALL_PRODUCTS = list(PRODUCT_INFO.keys())


# ---------------------------------------------------------------------------
# Co-purchase patterns (bundles that appear together frequently)
# ---------------------------------------------------------------------------
FREQUENT_BUNDLES = [
    # Breakfast bundles
    (["Milk", "Bread", "Butter"], 0.15),
    (["Milk", "Cereal"], 0.12),
    (["Coffee", "Sugar", "Cream"], 0.10),
    (["Eggs", "Bacon", "Bread"], 0.09),
    (["Bread", "Butter", "Jam"], 0.08),
    (["Bagels", "Cream", "Coffee"], 0.06),
    (["Oatmeal", "Milk", "Honey"], 0.05),
    # Lunch / dinner
    (["Pasta", "Olive Oil", "Tomatoes"], 0.07),
    (["Rice", "Chicken Breast", "Soy Sauce"], 0.06),
    (["Ground Beef", "Onions", "Tomatoes"], 0.06),
    (["Bread", "Ham", "Cheese", "Lettuce"], 0.05),
    (["Tortillas", "Chicken Breast", "Cheese"], 0.05),
    (["Sausage", "Bread", "Ketchup", "Mustard"], 0.04),
    # Snack runs
    (["Chips", "Soda"], 0.10),
    (["Cookies", "Milk"], 0.07),
    (["Popcorn", "Soda"], 0.05),
    # Healthy
    (["Bananas", "Yogurt", "Granola Bars"], 0.05),
    (["Apples", "Peanut Butter"], 0.04),
    (["Lettuce", "Tomatoes", "Cucumber"], 0.06),
    # Household
    (["Paper Towels", "Trash Bags", "Dish Soap"], 0.04),
    (["Shampoo", "Soap", "Toothpaste"], 0.03),
]


def _generate_transactions(n_transactions, product_pool, bundles,
                            basket_size_range, rng):
    """
    Generate synthetic transactions with realistic co-purchase patterns.

    Strategy:
    1. For each transaction, first decide which bundles fire (probabilistic).
    2. Add the bundle items to the basket.
    3. Fill remaining basket capacity with random products weighted by
       a frequency distribution (some products are more popular).
    """
    # Build product weights (Zipf-like distribution for realism)
    n_products = len(product_pool)
    raw_weights = 1.0 / np.arange(1, n_products + 1) ** 0.6
    weights = raw_weights / raw_weights.sum()

    transactions = []
    for _ in range(n_transactions):
        basket = set()
        target_size = rng.integers(basket_size_range[0], basket_size_range[1] + 1)

        # Apply bundles probabilistically
        for bundle_items, prob in bundles:
            valid_items = [item for item in bundle_items if item in product_pool]
            if valid_items and rng.random() < prob:
                basket.update(valid_items)

        # Fill to target size
        remaining = target_size - len(basket)
        if remaining > 0:
            available = [p for p in product_pool if p not in basket]
            if available:
                avail_weights = np.array([
                    weights[product_pool.index(p)] if p in product_pool else 0.01
                    for p in available
                ])
                avail_weights = avail_weights / avail_weights.sum()
                n_pick = min(remaining, len(available))
                picks = rng.choice(available, size=n_pick, replace=False,
                                   p=avail_weights)
                basket.update(picks)

        if basket:
            transactions.append(sorted(basket))

    return transactions


def _transactions_to_df(transactions):
    """Convert list of transactions to a long-format DataFrame."""
    rows = []
    for i, items in enumerate(transactions):
        tid = f"T{i + 1:05d}"
        for item in items:
            rows.append({"transaction_id": tid, "item": item})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

def generate_dataset_a():
    """Dataset A: Classroom Example (~15 transactions, ~12 products)."""
    transactions = [
        ["Milk", "Bread", "Butter"],
        ["Bread", "Butter", "Jam"],
        ["Milk", "Bread", "Eggs"],
        ["Milk", "Butter", "Cheese"],
        ["Bread", "Butter", "Milk", "Eggs"],
        ["Coffee", "Sugar", "Cream"],
        ["Milk", "Coffee", "Bread"],
        ["Eggs", "Bacon", "Bread"],
        ["Milk", "Bread", "Butter", "Jam"],
        ["Cheese", "Bread", "Butter"],
        ["Coffee", "Sugar"],
        ["Milk", "Eggs", "Bacon"],
        ["Bread", "Milk"],
        ["Butter", "Bread", "Eggs", "Milk"],
        ["Coffee", "Cream", "Sugar", "Milk"],
    ]
    return _transactions_to_df(transactions)


def generate_dataset_b():
    """Dataset B: Small Retail (~1000 transactions, ~50 products)."""
    rng = np.random.default_rng(SEED)

    # Select a subset of products
    product_pool = sorted(ALL_PRODUCTS[:50])
    relevant_bundles = [
        (items, prob) for items, prob in FREQUENT_BUNDLES
        if all(item in product_pool for item in items)
    ]

    transactions = _generate_transactions(
        n_transactions=1000,
        product_pool=product_pool,
        bundles=relevant_bundles,
        basket_size_range=(2, 8),
        rng=rng,
    )
    return _transactions_to_df(transactions)


def generate_dataset_c():
    """Dataset C: Medium Supermarket (~10000 transactions, ~80 products)."""
    rng = np.random.default_rng(SEED + 1)

    product_pool = sorted(ALL_PRODUCTS[:80])
    relevant_bundles = [
        (items, prob * 1.2) for items, prob in FREQUENT_BUNDLES
        if all(item in product_pool for item in items)
    ]

    transactions = _generate_transactions(
        n_transactions=10000,
        product_pool=product_pool,
        bundles=relevant_bundles,
        basket_size_range=(2, 12),
        rng=rng,
    )
    return _transactions_to_df(transactions)


def generate_dataset_d():
    """Dataset D: Large Retail (~50000 transactions, all products)."""
    rng = np.random.default_rng(SEED + 2)

    product_pool = sorted(ALL_PRODUCTS)
    relevant_bundles = [
        (items, prob * 1.5) for items, prob in FREQUENT_BUNDLES
    ]

    transactions = _generate_transactions(
        n_transactions=50000,
        product_pool=product_pool,
        bundles=relevant_bundles,
        basket_size_range=(2, 15),
        rng=rng,
    )
    return _transactions_to_df(transactions)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "FreshMarket Supermarket (10,000 Transactions)": {
        "generator": generate_dataset_c,
        "description": (
            "A realistic retail supermarket transactional dataset containing "
            "10,000 customer baskets across 80 unique grocery items in 7 departments "
            "(Dairy, Bakery, Beverages, Snacks, Meat, Produce, Pantry). "
            "Includes authentic multi-item co-purchase bundles and varied basket sizes."
        ),
        "synthetic": True,
        "size_label": "10,000 Transactions | 80 Items",
    },
}


def load_dataset(name="FreshMarket Supermarket (10,000 Transactions)"):
    """Load a dataset by name. Defaults to FreshMarket Supermarket."""
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]["generator"]()
    # Backward compatibility for tests and legacy callers
    if "Dataset A" in name:
        return generate_dataset_a()
    if "Dataset B" in name:
        return generate_dataset_b()
    if "Dataset C" in name or "Medium" in name:
        return generate_dataset_c()
    if "Dataset D" in name or "Large" in name:
        return generate_dataset_d()
    return generate_dataset_c()


def get_dataset_names():
    """Return list of available dataset names."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_info(name="FreshMarket Supermarket (10,000 Transactions)"):
    """Return metadata dict for a dataset."""
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]
    return DATASET_REGISTRY["FreshMarket Supermarket (10,000 Transactions)"]


def parse_uploaded_csv(uploaded_file):
    """
    Parse an uploaded CSV file. Supports:
    - Long format: columns 'transaction_id' and 'item'
    - Wide format: first column is transaction_id, remaining are binary item columns
    
    Returns a long-format DataFrame with columns ['transaction_id', 'item'].
    """
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Detect format
    if "transaction_id" in df.columns and "item" in df.columns:
        # Long format
        result = df[["transaction_id", "item"]].copy()
        result["transaction_id"] = result["transaction_id"].astype(str).str.strip()
        result["item"] = result["item"].astype(str).str.strip()
        # Remove empty items and duplicates
        result = result[result["item"].str.len() > 0]
        result = result.drop_duplicates(subset=["transaction_id", "item"])
        # Remove empty transactions
        result = result.groupby("transaction_id").filter(lambda x: len(x) > 0)
        return result
    else:
        # Try wide format: first col is transaction_id, rest are binary
        tid_col = df.columns[0]
        item_cols = df.columns[1:]
        # Check if remaining columns are binary-like
        for col in item_cols:
            if not df[col].dropna().isin([0, 1, True, False]).all():
                raise ValueError(
                    "Unrecognized CSV format. Expected either:\n"
                    "  - Long format with columns 'transaction_id' and 'item'\n"
                    "  - Wide format with a transaction ID column followed by "
                    "binary (0/1) item columns"
                )
        rows = []
        for _, row in df.iterrows():
            tid = str(row[tid_col]).strip()
            for col in item_cols:
                if row[col] == 1 or row[col] is True:
                    rows.append({"transaction_id": tid, "item": str(col).strip()})
        if not rows:
            raise ValueError("No items found in uploaded data.")
        return pd.DataFrame(rows)


def compute_dataset_stats(df_long):
    """Compute summary statistics from a long-format transaction DataFrame."""
    grouped = df_long.groupby("transaction_id")["item"].apply(list)
    basket_sizes = grouped.apply(len)
    item_freq = df_long["item"].value_counts()

    stats = {
        "n_transactions": len(grouped),
        "n_unique_items": df_long["item"].nunique(),
        "avg_basket_size": round(basket_sizes.mean(), 2),
        "min_basket_size": int(basket_sizes.min()),
        "max_basket_size": int(basket_sizes.max()),
        "median_basket_size": round(basket_sizes.median(), 1),
        "item_frequencies": item_freq,
        "basket_sizes": basket_sizes,
    }

    # Category distribution if products are in catalog
    categories = {}
    for item in df_long["item"].unique():
        if item in PRODUCT_INFO:
            cat = PRODUCT_INFO[item]["category"]
            categories[cat] = categories.get(cat, 0) + 1
    if categories:
        stats["category_distribution"] = pd.Series(categories).sort_values(
            ascending=False
        )

    return stats
