"""
visualizations.py - Chart generation for the SmartCart application.

Uses Plotly for interactive, publication-quality charts with consistent
styling. All charts use real computed data.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Consistent theme settings
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#2563eb",
    "primary_light": "#60a5fa",
    "secondary": "#7c3aed",
    "success": "#059669",
    "warning": "#d97706",
    "danger": "#dc2626",
    "info": "#0891b2",
    "neutral": "#64748b",
    "bg": "#ffffff",
    "text": "#1e293b",
}

COLOR_SEQUENCE = [
    "#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626",
    "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
]

CHART_LAYOUT = dict(
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(
        font_size=13,
        font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    ),
)


def _apply_layout(fig, **kwargs):
    """Apply consistent layout to a plotly figure with responsive margins."""
    layout = {**CHART_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    fig.update_xaxes(
        automargin=True,
        gridcolor="rgba(148,163,184,0.2)",
        zerolinecolor="rgba(148,163,184,0.25)",
    )
    fig.update_yaxes(
        automargin=True,
        gridcolor="rgba(148,163,184,0.2)",
        zerolinecolor="rgba(148,163,184,0.25)",
    )
    return fig


# ---------------------------------------------------------------------------
# Dataset exploration charts
# ---------------------------------------------------------------------------

def plot_item_frequency(item_freq, top_n=25):
    """Bar chart of product frequencies."""
    if item_freq is None or len(item_freq) == 0:
        return None
    data = item_freq.head(top_n).reset_index()
    data.columns = ["Product", "Frequency"]

    fig = px.bar(
        data, x="Frequency", y="Product",
        orientation="h",
        title=f"Top {min(top_n, len(data))} Products by Frequency",
        color="Frequency",
        color_continuous_scale=["#bfdbfe", "#2563eb"],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_coloraxes(showscale=False)
    _apply_layout(fig, height=max(400, min(top_n, len(data)) * 28 + 100))
    return fig


def plot_basket_size_distribution(basket_sizes):
    """Histogram of transaction basket sizes."""
    if basket_sizes is None or len(basket_sizes) == 0:
        return None
    data = pd.DataFrame({"Basket Size": basket_sizes.values})

    fig = px.histogram(
        data, x="Basket Size",
        title="Transaction Basket Size Distribution",
        nbins=max(5, int(data["Basket Size"].max() - data["Basket Size"].min() + 1)),
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig.update_layout(
        xaxis_title="Number of Items in Transaction",
        yaxis_title="Number of Transactions",
    )
    _apply_layout(fig, height=400)
    return fig


def plot_category_distribution(category_dist, product_info=None):
    """Pie chart of product category distribution."""
    if category_dist is None:
        return None
    if isinstance(category_dist, pd.Series):
        data = category_dist.reset_index()
        data.columns = ["Category", "Count"]
    elif isinstance(category_dist, dict):
        data = pd.DataFrame(list(category_dist.items()), columns=["Category", "Count"])
    else:
        return None

    fig = px.pie(
        data, names="Category", values="Count",
        title="Product Category Distribution",
        color_discrete_sequence=COLOR_SEQUENCE,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        margin=dict(l=30, r=30, t=50, b=65),
    )
    _apply_layout(fig, height=450)
    return fig


def plot_transaction_item_matrix(df_long, max_transactions=25):
    """
    Heatmap showing the transaction-item matrix.
    Only suitable for small datasets.
    """
    grouped = df_long.groupby("transaction_id")["item"].apply(set)
    items = sorted(df_long["item"].unique())
    tids = sorted(grouped.index)[:max_transactions]

    matrix = []
    for tid in tids:
        row = [1 if item in grouped[tid] else 0 for item in items]
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=items,
        y=tids,
        colorscale=[[0, "#f1f5f9"], [1, "#2563eb"]],
        showscale=False,
        hovertemplate="Transaction: %{y}<br>Item: %{x}<br>Present: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Transaction-Item Matrix",
        xaxis_title="Items",
        yaxis_title="Transaction ID",
        xaxis=dict(tickangle=45),
        margin=dict(l=50, r=20, t=50, b=80),
    )
    _apply_layout(fig, height=max(400, len(tids) * 25 + 150))
    return fig


# ---------------------------------------------------------------------------
# Apriori iteration charts
# ---------------------------------------------------------------------------

def plot_iteration_summary(iterations):
    """Bar chart comparing candidates vs frequent itemsets per iteration."""
    if not iterations:
        return None
    data = []
    for it in iterations:
        data.append({
            "Iteration": f"k = {it['k']}",
            "Candidate Count": it["candidate_count"],
            "Pruned Count": it["pruned_count"],
            "Frequent Count": it["frequent_count"],
        })

    df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Candidates Generated",
        x=df["Iteration"], y=df["Candidate Count"],
        marker_color=COLORS["primary_light"],
    ))
    fig.add_trace(go.Bar(
        name="Pruned (Apriori Property)",
        x=df["Iteration"], y=df["Pruned Count"],
        marker_color=COLORS["warning"],
    ))
    fig.add_trace(go.Bar(
        name="Frequent Itemsets",
        x=df["Iteration"], y=df["Frequent Count"],
        marker_color=COLORS["success"],
    ))

    fig.update_layout(
        title=dict(
            text="Apriori Iterations: Candidates vs Frequent Itemsets",
            y=0.96,
            x=0.0,
            xanchor="left",
        ),
        xaxis_title="Iteration",
        yaxis_title="Count",
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=45, r=20, t=55, b=85),
    )
    _apply_layout(fig, height=430)
    return fig


def plot_frequent_itemset_by_size(iterations):
    """Bar chart of frequent itemset counts grouped by itemset size."""
    data = []
    for it in iterations:
        if it["frequent_count"] > 0:
            data.append({
                "Itemset Size": f"{it['k']}-itemset",
                "Count": it["frequent_count"],
            })

    if not data:
        return None

    df = pd.DataFrame(data)
    fig = px.bar(
        df, x="Itemset Size", y="Count",
        title="Frequent Itemsets by Size",
        color="Count",
        color_continuous_scale=["#bbf7d0", "#059669"],
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        xaxis_title="Itemset Size",
        yaxis_title="Number of Frequent Itemsets",
    )
    _apply_layout(fig, height=400)
    return fig


# ---------------------------------------------------------------------------
# Association Rule charts
# ---------------------------------------------------------------------------

def _normalize_rules_df(rules_input):
    """Ensure rules input is a DataFrame with standardized columns."""
    if rules_input is None:
        return pd.DataFrame()
    if isinstance(rules_input, pd.DataFrame):
        df = rules_input.copy()
    elif isinstance(rules_input, list):
        if not rules_input:
            return pd.DataFrame()
        rows = []
        for r in rules_input:
            ante_items = sorted(list(r["antecedent"])) if isinstance(r.get("antecedent"), (set, frozenset, list)) else []
            cons_items = sorted(list(r["consequent"])) if isinstance(r.get("consequent"), (set, frozenset, list)) else []
            rows.append({
                "antecedent": r.get("antecedent"),
                "consequent": r.get("consequent"),
                "antecedent_items": ante_items,
                "consequent_items": cons_items,
                "support": r.get("support", 0.0),
                "support_pct": r.get("support", 0.0),
                "confidence": r.get("confidence", 0.0),
                "lift": r.get("lift", 1.0),
                "accepted": r.get("accepted", True),
            })
        df = pd.DataFrame(rows)
    else:
        return pd.DataFrame()

    if "antecedent_items" not in df.columns and "antecedent" in df.columns:
        df["antecedent_items"] = df["antecedent"].apply(lambda x: sorted(list(x)) if isinstance(x, (set, frozenset, list)) else [])
    if "consequent_items" not in df.columns and "consequent" in df.columns:
        df["consequent_items"] = df["consequent"].apply(lambda x: sorted(list(x)) if isinstance(x, (set, frozenset, list)) else [])
    if "support_pct" not in df.columns and "support" in df.columns:
        df["support_pct"] = df["support"]

    return df


def plot_support_confidence_scatter(rules_input, max_points=500):
    """
    Scatter plot of Support vs Confidence with Lift as color.
    """
    df = _normalize_rules_df(rules_input)
    if df.empty:
        return None

    df = df.head(max_points).copy()
    df["Rule"] = df.apply(
        lambda r: f"{{{', '.join(r['antecedent_items'])}}} -> {{{', '.join(r['consequent_items'])}}}",
        axis=1,
    )
    df["Lift Category"] = df["lift"].apply(
        lambda l: "Positive (Lift > 1)" if l > 1
        else ("Independent (Lift = 1)" if abs(l - 1) < 0.01 else "Negative (Lift < 1)")
    )

    fig = px.scatter(
        df,
        x="support_pct",
        y="confidence",
        color="lift",
        size=df["lift"].clip(lower=0.1),
        hover_data={
            "Rule": True,
            "support_pct": ":.4f",
            "confidence": ":.4f",
            "lift": ":.4f",
        },
        title="Support-Confidence Scatter Plot (color = Lift)",
        color_continuous_scale="RdYlGn",
        labels={
            "support_pct": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
        },
        size_max=18,
    )
    fig.update_layout(
        xaxis_title="Support",
        yaxis_title="Confidence",
    )
    _apply_layout(fig, height=500)
    return fig


def plot_top_rules_bar(rules_df, metric="lift", top_n=10):
    """Horizontal bar chart of top rules by a given metric."""
    df = _normalize_rules_df(rules_df)
    if df.empty:
        return None

    # Secondary sort to break ties meaningfully (e.g. when confidence == 1.0)
    secondary_col = "lift" if metric != "lift" else "confidence"
    if secondary_col in df.columns:
        df = df.sort_values(by=[metric, secondary_col], ascending=[False, False]).head(top_n).copy()
    else:
        df = df.sort_values(by=metric, ascending=False).head(top_n).copy()

    df["Rule"] = df.apply(
        lambda r: f"{{{', '.join(r['antecedent_items'])}}} -> {{{', '.join(r['consequent_items'])}}}",
        axis=1,
    )

    color_map = {
        "confidence": COLORS["primary"],
        "lift": COLORS["secondary"],
        "support_pct": COLORS["success"],
    }
    metric_labels = {
        "confidence": "Confidence",
        "lift": "Lift",
        "support_pct": "Support",
    }

    # Format on-bar value labels
    if metric == "lift":
        df["text_label"] = df[metric].apply(lambda v: f"{v:.2f}x")
    elif metric == "confidence":
        df["text_label"] = df[metric].apply(lambda v: f"{v*100:.1f}%")
    else:
        df["text_label"] = df[metric].apply(lambda v: f"{v*100:.2f}%")

    fig = px.bar(
        df, x=metric, y="Rule",
        text="text_label",
        orientation="h",
        title=f"Top {min(top_n, len(df))} Rules by {metric_labels.get(metric, metric)}",
        color_discrete_sequence=[color_map.get(metric, COLORS["secondary"])],
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed", automargin=True),
        xaxis=dict(automargin=True),
        xaxis_title=metric_labels.get(metric, metric),
        yaxis_title="Association Rule",
        margin=dict(l=60, r=30, t=50, b=50),
    )
    _apply_layout(fig, height=max(360, min(top_n, len(df)) * 34 + 100))
    return fig


def plot_rule_network(rules_df, max_rules=30):
    """
    Network graph showing relationships between items via association rules.
    Uses a spring layout approach.
    """
    df = _normalize_rules_df(rules_df)
    if df.empty:
        return None

    df = df.head(max_rules)

    # Collect unique items
    items = set()
    edges = []
    for _, rule in df.iterrows():
        ante = rule["antecedent_items"]
        cons = rule["consequent_items"]
        items.update(ante)
        items.update(cons)
        for a in ante:
            for c in cons:
                edges.append((a, c, rule["confidence"], rule["lift"]))

    items = sorted(items)
    n = len(items)
    if n == 0:
        return None

    # Simple circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {item: (np.cos(a), np.sin(a)) for item, a in zip(items, angles)}

    # Draw edges
    edge_traces = []
    for src, dst, conf, lift in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        width = max(1, min(5, conf * 5))
        color = COLORS["success"] if lift > 1 else (
            COLORS["neutral"] if abs(lift - 1) < 0.01 else COLORS["danger"]
        )
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="none",
            showlegend=False,
        ))

    # Draw nodes
    node_x = [pos[item][0] for item in items]
    node_y = [pos[item][1] for item in items]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=18, color=COLORS["primary"], line=dict(width=2, color="white")),
        text=items,
        textposition="top center",
        textfont=dict(size=11),
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=f"Rule Relationship Network (Top {min(max_rules, len(df))} Rules)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    _apply_layout(fig, height=550)
    return fig
