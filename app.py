"""
app.py - SmartCart: Interactive Association Rule Mining Demo

Designed for university-level students in the "Intelligent System" course.
Demonstrates Market Basket Analysis and the Apriori algorithm with an intuitive,
accessible web interface.

Course: Intelligent System (He thong thong minh)
Semester: Fall 2026
Institution: HCMUT (Ho Chi Minh City University of Technology)
"""

import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
import time
import io

from src.datasets import (
    load_dataset, get_dataset_names, get_dataset_info,
    compute_dataset_stats, parse_uploaded_csv, PRODUCT_INFO, PRODUCT_CATALOG,
)
from src.apriori_engine import (
    df_to_transactions, run_apriori, generate_rules, get_recommendations,
)
from src.visualizations import (
    plot_item_frequency, plot_basket_size_distribution,
    plot_category_distribution, plot_transaction_item_matrix,
    plot_iteration_summary, plot_frequent_itemset_by_size,
    plot_support_confidence_scatter, plot_top_rules_bar,
    plot_rule_network, COLORS,
)

# ---------------------------------------------------------------------------
# Page configuration - Clean wide layout, collapsed sidebar
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartCart - Association Rule Mining Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Modern, Adaptive Design System (CSS)
# Zero hardcoded themes or fixed sizes - fully adapts to any client machine/screen/theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* -----------------------------------------------------------------------
       Adaptive Dynamic Design System
       Inherits Streamlit theme tokens, system themes, and viewports
       ----------------------------------------------------------------------- */
    :root,
    .stApp,
    [data-testid="stAppViewContainer"] {
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --primary-active: #1e40af;
        --primary-light: rgba(37, 99, 235, 0.08);
        --primary-border: rgba(37, 99, 235, 0.25);
        --bg-main: var(--background-color, #f8fafc);
        --card-bg: var(--secondary-background-color, #ffffff);
        --card-hover-border: #60a5fa;
        --text-main: var(--text-color, #0f172a);
        --text-muted: #64748b;
        --border-color: rgba(148, 163, 184, 0.25);
        --chip-bg: rgba(241, 245, 249, 0.85);
        --chip-text: #334155;
        --input-bg: #ffffff;
        --input-text: #0f172a;
        --input-border: #cbd5e1;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 14px -2px rgba(37, 99, 235, 0.08);
        --shadow-hover: 0 8px 24px -4px rgba(37, 99, 235, 0.16);
    }

    /* Auto Dark Mode Adaptation (Operating System or Streamlit Dark Theme) */
    @media (prefers-color-scheme: dark) {
        :root,
        .stApp,
        [data-testid="stAppViewContainer"] {
            --bg-main: var(--background-color, #0b0f19);
            --card-bg: var(--secondary-background-color, #131c2e);
            --card-hover-border: #3b82f6;
            --text-main: var(--text-color, #f8fafc);
            --text-muted: #94a3b8;
            --border-color: rgba(148, 163, 184, 0.2);
            --chip-bg: rgba(30, 41, 59, 0.85);
            --chip-text: #cbd5e1;
            --input-bg: #1e293b;
            --input-text: #f8fafc;
            --input-border: #334155;
            --primary-light: rgba(37, 99, 235, 0.2);
            --primary-border: rgba(96, 165, 250, 0.35);
        }
    }

    /* Streamlit in-app Dark Theme overrides */
    [data-theme="dark"],
    [data-testid="stAppViewContainer"][data-theme="dark"],
    .stApp[data-theme="dark"] {
        --bg-main: var(--background-color, #0b0f19);
        --card-bg: var(--secondary-background-color, #131c2e);
        --card-hover-border: #3b82f6;
        --text-main: var(--text-color, #f8fafc);
        --text-muted: #94a3b8;
        --border-color: rgba(148, 163, 184, 0.2);
        --chip-bg: rgba(30, 41, 59, 0.85);
        --chip-text: #cbd5e1;
        --input-bg: #1e293b;
        --input-text: #f8fafc;
        --input-border: #334155;
        --primary-light: rgba(37, 99, 235, 0.2);
        --primary-border: rgba(96, 165, 250, 0.35);
    }

    /* Plotly Chart High-Contrast Theme Adaptation (Light & Dark Mode) */
    .js-plotly-plot .plotly svg text.gtitle,
    .js-plotly-plot .plotly svg .infolayer text.gtitle,
    .js-plotly-plot .plotly svg .infolayer .g-gtitle text,
    .js-plotly-plot .plotly svg .infolayer .legend text,
    .js-plotly-plot .plotly svg text.legendtext,
    .js-plotly-plot .plotly svg text.xtitle,
    .js-plotly-plot .plotly svg text.ytitle,
    .js-plotly-plot .plotly svg .xtick text,
    .js-plotly-plot .plotly svg .ytick text,
    .js-plotly-plot .plotly svg .cbaxis text,
    .js-plotly-plot .plotly svg .textpoint text {
        fill: var(--text-main) !important;
        color: var(--text-main) !important;
    }
    .js-plotly-plot .plotly svg .gridpath {
        stroke: var(--border-color) !important;
    }
    .js-plotly-plot .plotly svg .zeroline {
        stroke: var(--border-color) !important;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-main);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Universal Dynamic High-Contrast Widget Labels & Form Typography */
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] span,
    .stApp [data-testid="stWidgetLabel"] label,
    .stApp label,
    .stApp label p,
    .stApp label span {
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }

    /* Tooltip icons */
    .stApp [data-testid="stTooltipIcon"] svg,
    .stApp [data-testid="stTooltipHoverTarget"] svg {
        fill: var(--text-muted) !important;
        color: var(--text-muted) !important;
    }
    .stApp [data-testid="stTooltipIcon"]:hover svg,
    .stApp [data-testid="stTooltipHoverTarget"]:hover svg {
        fill: var(--primary) !important;
        color: var(--primary) !important;
    }

    /* Sliders High Contrast */
    .stApp [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }
    .stApp [data-testid="stSlider"] [data-testid="stThumbValue"],
    .stApp [data-testid="stSlider"] [data-testid="stTickBarMin"],
    .stApp [data-testid="stSlider"] [data-testid="stTickBarMax"],
    .stApp [data-testid="stSlider"] div {
        color: var(--text-main) !important;
    }

    /* Form Inputs (NumberInput, TextInput, SelectBox) */
    .stApp input,
    .stApp textarea {
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        caret-color: var(--primary) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 8px !important;
    }
    .stApp input:focus,
    .stApp textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px var(--primary-light) !important;
    }
    .stApp input::placeholder,
    .stApp textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* Selectbox base */
    .stApp [data-baseweb="select"] > div {
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 8px !important;
    }
    .stApp [data-baseweb="select"] * {
        color: var(--input-text) !important;
    }
    .stApp [data-baseweb="popover"],
    .stApp [data-baseweb="menu"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] ul {
        background-color: var(--card-bg) !important;
        color: var(--input-text) !important;
        border: 1px solid var(--border-color) !important;
    }
    div[data-baseweb="menu"] li {
        color: var(--input-text) !important;
    }
    div[data-baseweb="menu"] li:hover {
        background-color: var(--primary-light) !important;
        color: var(--primary) !important;
    }

    /* NumberInput +/- step buttons */
    .stApp [data-testid="stNumberInput"] button {
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border: 1px solid var(--input-border) !important;
    }
    .stApp [data-testid="stNumberInput"] button:hover {
        background-color: var(--primary-light) !important;
        color: var(--primary) !important;
        border-color: var(--primary) !important;
    }

    /* Markdown Text, Headings & Captions */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: var(--text-main) !important;
    }
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li {
        color: var(--text-main);
    }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp small {
        color: var(--text-muted) !important;
    }

    /* Expanders */
    .stApp [data-testid="stExpander"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }
    .stApp [data-testid="stExpander"] details summary,
    .stApp [data-testid="stExpander"] details summary p,
    .stApp [data-testid="stExpander"] details summary span {
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }

    /* Top Header Space Optimization */
    header[data-testid="stHeader"] {
        height: 0px !important;
        min-height: 0px !important;
        background: transparent !important;
        z-index: 100 !important;
    }
    [data-testid="stToolbar"] {
        right: 1.2rem !important;
        top: 0.4rem !important;
    }
    .main > .block-container,
    div[data-testid="stAppViewBlockContainer"],
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 2.2rem !important;
        padding-left: clamp(0.5rem, 1.2vw, 1.4rem) !important;
        padding-right: clamp(0.5rem, 1.2vw, 1.4rem) !important;
        max-width: 100% !important;
    }

    /* Hide sidebar collapse button padding */
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Subtle Card Entry Animation */
    @keyframes cardFadeIn {
        from {
            opacity: 0;
            transform: translateY(6px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Website Top Bar Brand Header - Extended Fluid Dimensions */
    .app-topbar-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: clamp(10px, 1.5vw, 20px);
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #1d4ed8 100%);
        border-radius: 12px;
        padding: clamp(14px, 1.8vw, 22px) clamp(16px, 2.2vw, 28px);
        min-height: clamp(56px, 6vw, 76px);
        width: 100%;
        margin-top: 0 !important;
        margin-bottom: clamp(12px, 1.4vw, 18px);
        color: #ffffff !important;
        box-shadow: 0 4px 18px -2px rgba(30, 58, 138, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: box-shadow 0.25s ease;
    }
    .app-topbar-header:hover {
        box-shadow: 0 8px 26px -2px rgba(30, 58, 138, 0.35);
    }
    .app-brand-title {
        font-size: clamp(1.2rem, 1.7vw, 1.55rem);
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .app-brand-tag {
        font-size: clamp(0.7rem, 0.8vw, 0.76rem);
        font-weight: 600;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.35);
        border-radius: 20px;
        padding: 3px 10px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #ffffff !important;
        transition: all 0.2s ease;
    }
    .app-brand-tag:hover {
        background: rgba(255, 255, 255, 0.32);
        transform: scale(1.04);
    }
    .app-brand-sub {
        font-size: clamp(0.82rem, 0.98vw, 0.92rem);
        opacity: 0.94;
        color: #eff6ff !important;
        margin-top: 4px;
    }
    .dataset-status-pill {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.32);
        border-radius: 8px;
        padding: clamp(6px, 0.8vw, 10px) clamp(12px, 1.2vw, 16px);
        font-size: clamp(0.76rem, 0.88vw, 0.85rem);
        font-weight: 500;
        color: #ffffff !important;
        white-space: normal;
        word-break: break-word;
        transition: all 0.2s ease;
    }
    .dataset-status-pill:hover {
        background: rgba(255, 255, 255, 0.28);
        border-color: rgba(255, 255, 255, 0.5);
        transform: translateY(-1px);
    }

    /* Highly Interactive Buttons (Navigation Tabs & Action Buttons) */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--card-bg) !important;
        color: var(--text-main) !important;
        padding: clamp(6px, 0.8vw, 10px) clamp(8px, 1vw, 16px) !important;
        font-size: clamp(0.76rem, 0.9vw, 0.92rem) !important;
        min-width: 0 !important;
        white-space: normal !important;
        word-break: break-word !important;
        text-align: center !important;
        line-height: 1.3 !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:active:not(:disabled) {
        transform: translateY(1px) scale(0.97) !important;
    }
    .stButton > button[kind="secondary"]:not(:disabled),
    .stButton > button[data-testid="baseButton-secondary"]:not(:disabled) {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-main) !important;
    }
    .stButton > button[kind="secondary"]:hover:not(:disabled),
    .stButton > button[data-testid="baseButton-secondary"]:hover:not(:disabled),
    .stButton > button:hover:not(:disabled) {
        background-color: var(--primary-light) !important;
        border-color: var(--card-hover-border) !important;
        color: var(--primary) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.14) !important;
    }
    .stButton > button[kind="primary"]:not(:disabled),
    .stButton > button[data-testid="baseButton-primary"]:not(:disabled) {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: 1px solid #1d4ed8 !important;
        color: #ffffff !important;
        box-shadow: 0 3px 8px rgba(37, 99, 235, 0.25) !important;
    }
    .stButton > button[kind="primary"]:not(:disabled) *,
    .stButton > button[data-testid="baseButton-primary"]:not(:disabled) * {
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover:not(:disabled),
    .stButton > button[data-testid="baseButton-primary"]:hover:not(:disabled) {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        border-color: #1e40af !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.38) !important;
    }

    /* Disabled Button State - Explicit Academic Greyed Out */
    .stButton > button:disabled,
    .stButton > button[disabled] {
        background: var(--chip-bg) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-muted) !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
        transform: none !important;
        opacity: 0.55 !important;
        pointer-events: none !important;
    }
    .stButton > button:disabled *,
    .stButton > button[disabled] * {
        color: var(--text-muted) !important;
    }
    .stButton > button:disabled:hover,
    .stButton > button[disabled]:hover {
        background: var(--chip-bg) !important;
        border-color: var(--border-color) !important;
        color: var(--text-muted) !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Soft Modern Cards with Hover Depth */
    .modern-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        color: var(--text-main);
        border-radius: 12px;
        padding: clamp(12px, 1.8vw, 20px) clamp(14px, 2vw, 24px);
        margin-bottom: clamp(12px, 1.5vw, 18px);
        box-shadow: var(--shadow-sm);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
        animation: cardFadeIn 0.25s ease-out forwards;
    }
    .modern-card:hover {
        border-color: var(--card-hover-border);
        box-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.08);
    }
    .card-title {
        font-size: clamp(0.95rem, 1.2vw, 1.08rem);
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-subtitle {
        font-size: clamp(0.78rem, 0.9vw, 0.88rem);
        color: var(--text-muted);
        margin-bottom: clamp(10px, 1.2vw, 16px);
    }

    /* Streamlit Bordered Container Hover */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--card-hover-border) !important;
        box-shadow: 0 4px 16px -2px rgba(37, 99, 235, 0.08) !important;
    }

    /* Fluid Metric Cards with Interactive Hover Elevation */
    [data-testid="stMetric"] {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-left: 3.5px solid var(--primary) !important;
        border-radius: 10px !important;
        padding: clamp(8px, 1.2vw, 14px) clamp(10px, 1.4vw, 18px) !important;
        box-shadow: var(--shadow-sm) !important;
        min-width: 0 !important;
        overflow: hidden !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: default;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px) !important;
        border-color: var(--card-hover-border) !important;
        border-left-color: var(--primary) !important;
        box-shadow: var(--shadow-hover) !important;
    }
    [data-testid="stMetricLabel"] * {
        color: var(--text-muted) !important;
        font-size: clamp(0.7rem, 0.8vw, 0.8rem) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        white-space: normal !important;
        word-break: break-word !important;
        transition: color 0.2s ease !important;
    }
    [data-testid="stMetric"]:hover [data-testid="stMetricLabel"] * {
        color: var(--primary) !important;
    }
    [data-testid="stMetricValue"] * {
        color: var(--text-main) !important;
        font-weight: 700 !important;
        font-size: clamp(1.1rem, 1.6vw, 1.5rem) !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
    }

    /* Pill Badges with Micro-interactions */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: clamp(0.72rem, 0.8vw, 0.78rem);
        font-weight: 600;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        cursor: default;
    }
    .badge:hover {
        transform: scale(1.05);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.07);
    }
    .badge-frequent {
        background: rgba(16, 185, 129, 0.12);
        color: #059669 !important;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-pruned {
        background: rgba(245, 158, 11, 0.12);
        color: #d97706 !important;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-infrequent {
        background: rgba(239, 68, 68, 0.12);
        color: #dc2626 !important;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-blue {
        background: var(--primary-light);
        color: var(--primary) !important;
        border: 1px solid var(--primary-border);
    }

    /* Academic Typography Badges & Term Highlights (Zero Icons) */
    .academic-badge {
        display: inline-block;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: clamp(0.68rem, 0.76vw, 0.74rem);
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--primary);
        background: var(--chip-bg);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 2px 7px;
        line-height: 1.25;
        vertical-align: middle;
    }
    .def-badge {
        display: inline-block;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        background: var(--chip-bg);
        border-radius: 3px;
        padding: 1px 5px;
        margin-right: 5px;
        vertical-align: middle;
    }
    .term-highlight {
        display: inline-block;
        background: var(--primary-light);
        color: var(--primary);
        font-weight: 600;
        padding: 1px 6px;
        border-radius: 4px;
        border: 1px solid var(--primary-border);
        font-size: 0.88em;
        line-height: 1.35;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .term-highlight:hover {
        background: rgba(37, 99, 235, 0.18);
        border-color: var(--primary);
    }
    .academic-subbar {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        color: var(--text-muted);
        font-size: clamp(0.82rem, 0.92vw, 0.9rem);
        margin-bottom: 14px;
    }
    .academic-callout {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: clamp(12px, 1.4vw, 16px) clamp(14px, 1.8vw, 20px);
        margin-bottom: clamp(12px, 1.5vw, 18px);
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        transition: all 0.25s ease;
    }
    .academic-callout:hover {
        border-color: var(--card-hover-border);
        border-left-color: var(--primary-hover);
        box-shadow: 0 4px 14px -2px rgba(37, 99, 235, 0.08);
    }
    .callout-header {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 8px;
    }
    .callout-title {
        font-size: clamp(0.92rem, 1.1vw, 1.02rem);
        font-weight: 700;
        color: var(--primary);
        letter-spacing: -0.01em;
    }
    .academic-callout.callout-pruning {
        border-left-color: #f59e0b !important;
    }
    .academic-callout.callout-pruning .callout-title {
        color: #d97706 !important;
    }
    .academic-callout.callout-pruning .academic-badge {
        color: #d97706 !important;
        border-color: rgba(245, 158, 11, 0.4) !important;
        background: rgba(245, 158, 11, 0.12) !important;
    }
    .academic-callout.callout-principle {
        border-left-color: var(--text-muted) !important;
    }
    .academic-callout.callout-principle .callout-title {
        color: var(--text-main) !important;
    }
    .academic-callout.callout-principle .academic-badge {
        color: var(--text-muted) !important;
        border-color: var(--border-color) !important;
        background: var(--chip-bg) !important;
    }
    .callout-body {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .callout-statement,
    .callout-consequence {
        font-size: clamp(0.82rem, 0.92vw, 0.9rem);
        color: var(--text-main);
        line-height: 1.55;
    }
    .callout-statement code,
    .callout-consequence code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.85em;
        background: var(--chip-bg);
        color: var(--text-main);
        padding: 2px 5px;
        border-radius: 4px;
        border: 1px solid var(--border-color);
    }

    /* Explainer Callout Box */
    .explainer-box {
        background: var(--primary-light);
        border: 1px solid var(--primary-border);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: clamp(10px, 1.4vw, 16px) clamp(12px, 1.6vw, 20px);
        margin-bottom: clamp(12px, 1.5vw, 18px);
        color: var(--text-main);
        transition: box-shadow 0.2s ease;
    }
    .explainer-box:hover {
        box-shadow: 0 3px 10px rgba(37, 99, 235, 0.08);
    }
    .explainer-box h4 {
        margin: 0 0 6px 0 !important;
        color: var(--primary) !important;
        font-size: clamp(0.92rem, 1.1vw, 1rem) !important;
        font-weight: 700 !important;
    }
    .explainer-box p {
        margin: 0 !important;
        color: var(--text-main) !important;
        font-size: clamp(0.82rem, 0.95vw, 0.92rem) !important;
        line-height: 1.5 !important;
    }
    .metric-desc-item {
        font-size: clamp(0.82rem, 0.9vw, 0.88rem);
        color: var(--text-main);
        line-height: 1.5;
        margin-top: 6px;
        margin-bottom: 10px;
    }

    /* Recommendation Product Card with Interactive Lift & Metric Highlights */
    .recom-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: clamp(12px, 1.5vw, 16px) clamp(14px, 1.8vw, 20px);
        margin-bottom: clamp(10px, 1.2vw, 14px);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: default;
        animation: cardFadeIn 0.25s ease-out forwards;
    }
    .recom-card:hover {
        border-color: var(--card-hover-border);
        transform: translateY(-3px);
        box-shadow: var(--shadow-hover);
    }
    .recom-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 6px;
    }
    .recom-item-name {
        font-size: clamp(0.95rem, 1.2vw, 1.1rem);
        font-weight: 700;
        color: var(--text-main);
        transition: color 0.2s ease;
    }
    .recom-card:hover .recom-item-name {
        color: var(--primary);
    }
    .recom-reason {
        font-size: clamp(0.8rem, 0.9vw, 0.88rem);
        color: var(--text-muted);
        margin-bottom: 10px;
    }
    .recom-metrics {
        display: flex;
        gap: clamp(6px, 1vw, 10px);
        flex-wrap: wrap;
    }
    .metric-chip {
        font-size: clamp(0.72rem, 0.8vw, 0.8rem);
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
        background: var(--chip-bg);
        border: 1px solid var(--border-color);
        color: var(--chip-text);
        transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .metric-chip:hover {
        background: var(--primary-light);
        color: var(--primary);
        border-color: var(--primary-border);
        transform: translateY(-1px);
    }

    /* Basket Cart Item Tag */
    .cart-tag {
        display: inline-flex;
        align-items: center;
        background: var(--primary-light);
        border: 1px solid var(--primary-border);
        color: var(--primary);
        border-radius: 20px;
        padding: 5px 12px;
        font-size: clamp(0.78rem, 0.9vw, 0.88rem);
        font-weight: 600;
        margin: 4px;
        transition: all 0.18s ease;
    }
    .cart-tag:hover {
        background: rgba(37, 99, 235, 0.18);
        border-color: var(--primary);
        transform: scale(1.03);
    }

    /* Clean Dynamic Tabs with Hover State */
    .stTabs [data-baseweb="tab-list"] {
        gap: clamp(4px, 0.6vw, 8px) !important;
        background-color: var(--chip-bg) !important;
        border-radius: 8px !important;
        padding: 4px !important;
        flex-wrap: wrap !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        padding: clamp(4px, 0.8vw, 8px) clamp(8px, 1.2vw, 16px) !important;
        font-size: clamp(0.76rem, 0.9vw, 0.9rem) !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        border: none !important;
        min-width: 0 !important;
        white-space: normal !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary) !important;
        background-color: var(--primary-light) !important;
        transform: translateY(-1px) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--card-bg) !important;
        color: var(--primary) !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.12) !important;
        font-weight: 700 !important;
    }

    /* System Architecture / Real-World Overview Card */
    .system-overview-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: clamp(14px, 2vw, 24px) clamp(16px, 2.5vw, 26px);
        margin-bottom: clamp(14px, 1.8vw, 22px);
        box-shadow: var(--shadow-sm);
        transition: all 0.25s ease;
    }
    .system-overview-card:hover {
        border-color: var(--primary-border);
        box-shadow: 0 6px 18px -3px rgba(37, 99, 235, 0.08);
    }
    .system-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: clamp(0.68rem, 0.78vw, 0.75rem);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        background: var(--primary-light);
        color: var(--primary);
        border: 1px solid var(--primary-border);
        margin-bottom: 0px;
    }
    .system-header-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 8px;
    }
    .system-title {
        font-size: clamp(1.1rem, 1.6vw, 1.35rem);
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }
    .system-lead-text {
        font-size: clamp(0.85rem, 1vw, 0.94rem);
        color: var(--text-main);
        line-height: 1.55;
        margin-bottom: 14px;
    }
    .system-desc {
        font-size: clamp(0.82rem, 1vw, 0.92rem);
        color: var(--text-muted);
        line-height: 1.55;
        margin-bottom: clamp(12px, 1.5vw, 18px);
    }
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(clamp(180px, 20vw, 260px), 1fr));
        gap: clamp(10px, 1.2vw, 16px);
        margin-top: 14px;
    }
    .pipeline-step-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-top: 3px solid var(--primary);
        border-radius: 8px;
        padding: clamp(12px, 1.3vw, 16px) clamp(14px, 1.5vw, 18px);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-top-color 0.2s ease;
        cursor: default;
    }
    .pipeline-step-card:hover {
        border-top-color: var(--primary-hover);
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
    }
    .step-number {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--primary);
        margin-bottom: 6px;
    }
    .step-title {
        font-size: clamp(0.9rem, 1.05vw, 0.98rem);
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 6px;
    }
    .step-desc {
        font-size: clamp(0.78rem, 0.88vw, 0.85rem);
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* Responsive Dataframe and Plotly Containers */
    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"],
    .stPlotlyChart {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Prominent Shopping Basket & Recommender Styling */
    .basket-showcase-card {
        background: var(--card-bg);
        border: 2px solid var(--primary);
        border-radius: 12px;
        padding: clamp(14px, 1.8vw, 20px);
        margin-bottom: 16px;
        box-shadow: var(--shadow-md);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .basket-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-color);
    }
    .basket-title-text {
        font-size: clamp(1.05rem, 1.3vw, 1.2rem);
        font-weight: 700;
        color: var(--text-main);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .basket-count-pill {
        background: var(--primary);
        color: #ffffff !important;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.02em;
        transition: transform 0.2s ease;
    }
    .basket-count-pill:hover {
        transform: scale(1.08);
    }
    .basket-empty-box {
        background: var(--chip-bg);
        border: 2px dashed var(--border-color);
        border-radius: 10px;
        padding: clamp(20px, 3vw, 28px) 16px;
        text-align: center;
        margin-bottom: 14px;
        transition: border-color 0.25s ease;
    }
    .basket-empty-box:hover {
        border-color: var(--card-hover-border);
    }
    .basket-empty-title {
        font-size: clamp(0.92rem, 1.1vw, 1.02rem);
        font-weight: 700;
        color: var(--text-muted);
        margin-bottom: 4px;
    }
    .basket-empty-desc {
        font-size: clamp(0.78rem, 0.9vw, 0.85rem);
        color: var(--text-muted);
        max-width: min(100%, 380px);
        margin: 0 auto;
        line-height: 1.45;
    }
    .basket-items-list {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-bottom: 12px;
        max-height: clamp(200px, 32vh, 300px);
        overflow-y: auto;
        padding-right: 4px;
    }
    .basket-item-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--chip-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 8px 12px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: default;
    }
    .basket-item-row:hover {
        border-color: var(--card-hover-border);
        background: var(--primary-light);
        transform: translateX(4px);
        box-shadow: 0 3px 10px rgba(37, 99, 235, 0.08);
    }
    .basket-item-info {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .basket-item-name {
        font-weight: 700;
        color: var(--text-main);
        font-size: 0.9rem;
    }
    .basket-item-dept {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--primary);
        background: var(--primary-light);
        border: 1px solid var(--primary-border);
        border-radius: 4px;
        padding: 1px 6px;
        transition: all 0.15s ease;
    }
    .basket-item-row:hover .basket-item-dept {
        background: rgba(37, 99, 235, 0.18);
        border-color: var(--primary);
    }
    .basket-item-price {
        font-weight: 700;
        color: var(--text-main);
        font-size: 0.88rem;
    }
    .basket-summary-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--primary-light);
        border: 1px solid var(--primary-border);
        border-radius: 8px;
        padding: 9px 14px;
        margin-top: 10px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }
    .basket-summary-footer:hover {
        border-color: var(--primary);
    }
    .basket-total-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--primary);
    }
    .basket-total-value {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--primary);
    }

    /* Right Column Empty State */
    .recom-empty-box {
        background: var(--card-bg);
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: clamp(28px, 4.5vw, 44px) 20px;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        transition: border-color 0.25s ease;
    }
    .recom-empty-box:hover {
        border-color: var(--card-hover-border);
    }
    .recom-empty-title {
        font-size: clamp(1rem, 1.25vw, 1.15rem);
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 8px;
    }
    .recom-empty-desc {
        font-size: clamp(0.82rem, 0.95vw, 0.9rem);
        color: var(--text-muted);
        max-width: min(100%, 440px);
        margin: 0 auto;
        line-height: 1.55;
    }

    /* Input Focus Rings with Soft Glow */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        transition: all 0.2s ease !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: var(--card-hover-border) !important;
    }
    div[data-baseweb="select"]:focus-within > div,
    div[data-baseweb="input"]:focus-within > div {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.16) !important;
    }

    /* Responsive Mobile / Narrow Screen Breakpoint */
    @media (max-width: 768px) {
        .main > .block-container,
        div[data-testid="stAppViewBlockContainer"],
        .block-container {
            padding-left: 0.4rem !important;
            padding-right: 0.4rem !important;
        }
        .app-topbar-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
            padding: 12px 14px;
        }
        .app-brand-right {
            width: 100%;
        }
        .dataset-status-pill {
            display: block;
            text-align: center;
            width: 100%;
        }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "dataset_name": "FreshMarket Supermarket (10,000 Transactions)",
        "dataset_df": None,
        "dataset_stats": None,
        "transactions": None,
        "baskets_df": None,
        "basket_search_input": "",
        "min_support": 0.02,       # 2.0%
        "min_confidence": 0.30,    # 30.0%
        "max_k": 3,
        "apriori_result": None,
        "rules": None,
        "last_run_params": None,
        "cart_items": set(),
        "cart_version": 0,
        "workbench_version": 0,
        "active_page": "Transaction Overview",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Migrate legacy page name if needed
    if st.session_state.get("active_page") in ["Dataset Explorer", "Transaction Management"]:
        st.session_state.active_page = "Transaction Overview"

    # Sanitize existing cart_items against catalog
    if "cart_items" in st.session_state and isinstance(st.session_state.cart_items, set):
        valid = {it for it in st.session_state.cart_items if it in PRODUCT_INFO}
        st.session_state.cart_items = valid


init_session()


# ---------------------------------------------------------------------------
# Data Loading Helpers (Cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_dataset(name):
    return load_dataset(name)


def ensure_dataset_loaded():
    """Ensure dataset, transactions, and grouped baskets are in session state."""
    if st.session_state.dataset_df is None:
        st.session_state.dataset_df = get_cached_dataset(st.session_state.dataset_name)
        st.session_state.dataset_stats = compute_dataset_stats(st.session_state.dataset_df)
        st.session_state.transactions = df_to_transactions(st.session_state.dataset_df)

    if st.session_state.baskets_df is None:
        grouped = (
            st.session_state.dataset_df.groupby("transaction_id")["item"]
            .apply(lambda items: sorted(list(items)))
            .reset_index(name="item_list")
        )
        grouped["item_count"] = grouped["item_list"].apply(len)
        grouped["basket_display"] = grouped["item_list"].apply(lambda items: "{" + ", ".join(items) + "}")
        st.session_state.baskets_df = grouped


def ensure_default_rules():
    """Rules are explicitly generated when the user clicks 'Run Apriori' in the workbench."""
    pass


ensure_dataset_loaded()


# ---------------------------------------------------------------------------
# Website Top Navigation Bar (Header + Clickable Category Tabs)
# ---------------------------------------------------------------------------
NAV_PAGES = [
    "Transaction Overview",
    "Apriori Workbench",
    "Association Rules",
    "Product Recommender",
    "Theory & Reference",
]

# Top Brand Header
st.markdown(
    """
    <div class="app-topbar-header">
        <div class="app-brand-left">
            <div class="app-brand-title">SmartCart <span class="app-brand-tag">Intelligent System · Fall 2026 · HCMUT</span></div>
            <div class="app-brand-sub">Market Basket Analysis & Apriori Association Rule Mining</div>
        </div>
        <div class="app-brand-right">
            <span class="dataset-status-pill">FreshMarket POS System (10,000 Transactions | 80 Products)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Clickable Top Bar Categories (Direct click, zero radio circles)
nav_cols = st.columns(len(NAV_PAGES))
for col, page_name in zip(nav_cols, NAV_PAGES):
    with col:
        is_active = (st.session_state.active_page == page_name)
        if st.button(
            page_name,
            key=f"nav_cat_{page_name}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            if not is_active:
                st.session_state.active_page = page_name
                st.rerun()

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TAB 1: Apriori Workbench (Primary Working Interface)
# ---------------------------------------------------------------------------
def render_apriori_workbench():
    st.markdown("### Apriori Algorithm Workbench")
    st.markdown(
        """
        <div class="academic-subbar">
            <span class="academic-badge">[ALGORITHM ENGINE]</span>
            <span>Candidate Generation (<code>C<sub>k</sub></code>), Downward-Closure Pruning, and Frequent Itemsets (<code>L<sub>k</sub></code>).</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Parameter Configuration Card
    with st.container():
        st.markdown(
            """
            <div class="modern-card">
                <div class="card-title">Mining Configuration & Parameter Constraints</div>
                <div class="card-subtitle">
                    Constrained exploration on <span class="term-highlight">10,000 POS transactions</span>. Minimum Support defines frequency threshold; Minimum Confidence guarantees conditional certainty.
                </div>
            """,
            unsafe_allow_html=True,
        )

        wb_v = st.session_state.get("workbench_version", 0)
        col1, col2, col3 = st.columns(3)

        with col1:
            sup_pct = st.number_input(
                "Minimum Support (%)",
                min_value=None,
                max_value=None,
                value=float(st.session_state.min_support * 100),
                step=0.5,
                format="%.1f",
                key=f"input_min_sup_{wb_v}",
                help="Minimum fraction of transactions that must contain the itemset.",
            )

        with col2:
            conf_pct = st.number_input(
                "Minimum Confidence (%)",
                min_value=None,
                max_value=None,
                value=float(st.session_state.min_confidence * 100),
                step=5.0,
                format="%.1f",
                key=f"input_min_conf_{wb_v}",
                help="Minimum conditional probability P(Y | X) required to accept rule X -> Y.",
            )

        with col3:
            max_k = st.selectbox(
                "Maximum Itemset Size (k)",
                options=[2, 3, 4, 5],
                index=[2, 3, 4, 5].index(st.session_state.max_k) if st.session_state.max_k in [2, 3, 4, 5] else 1,
                key=f"input_max_k_{wb_v}",
                help="Halts search after k items to safeguard memory and laptop responsiveness.",
            )
            st.caption(f"Explores itemsets of sizes 1, 2, ... up to {max_k}.")

        # Strict Input Constraint Validation & Dynamic Captions
        has_blocking_error = False
        blocking_errors = []

        # Validate Support
        if sup_pct is None or sup_pct <= 0.0:
            has_blocking_error = True
            blocking_errors.append("Minimum Support must be strictly greater than 0.0% (recommended minimum: 0.1%). A support threshold of 0% would treat every combination as frequent, causing exponential combinatorial memory explosion.")
            min_sup = 0.0
            with col1:
                st.caption("Required occurrence: Invalid threshold (must be strictly > 0%).")
        elif sup_pct > 30.0:
            has_blocking_error = True
            blocking_errors.append("Minimum Support cannot exceed 30.0%. Supermarket transactions are sparse.")
            min_sup = sup_pct / 100.0
            with col1:
                st.caption("Required occurrence: Exceeds allowed upper bound (max 30%).")
        else:
            min_sup = sup_pct / 100.0
            abs_count = int(np.ceil(min_sup * len(st.session_state.transactions)))
            with col1:
                st.caption(f"Required occurrence: at least {abs_count:,} out of {len(st.session_state.transactions):,} baskets.")

        # Validate Confidence
        if conf_pct is None or conf_pct <= 0.0:
            has_blocking_error = True
            blocking_errors.append("Minimum Confidence must be strictly greater than 0.0%.")
            min_conf = 0.0
            with col2:
                st.caption("Rule requirement: Invalid threshold (must be strictly > 0%).")
        elif conf_pct > 100.0:
            has_blocking_error = True
            blocking_errors.append("Minimum Confidence cannot exceed 100.0%.")
            min_conf = 1.0
            with col2:
                st.caption("Rule requirement: Invalid threshold (cannot exceed 100%).")
        else:
            min_conf = conf_pct / 100.0
            with col2:
                st.caption(f"Rule requirement: X -> Y must hold true in >= {conf_pct:.0f}% of cases.")

        # Display Constraint Violation Errors or Operational Warnings
        if has_blocking_error:
            for err in blocking_errors:
                st.error(f"Constraint Violation: {err}")
        else:
            if min_sup < 0.005:
                st.warning("Low Support Warning (< 0.5%): May produce an excessively large candidate tree and take longer to execute.")
            elif min_sup > 0.15:
                st.info("High Support Notice (> 15%): Supermarket baskets are typically sparse; few multi-item combinations may meet this threshold.")

        btn_col1, btn_col2, _ = st.columns([1.3, 1.3, 3.4])
        with btn_col1:
            run_clicked = st.button(
                "Run Apriori",
                type="primary",
                disabled=has_blocking_error,
                use_container_width=True,
                help="Execution disabled: resolve input constraint violations above first." if has_blocking_error else "Execute Apriori algorithm with candidate pruning.",
            )
        with btn_col2:
            reset_clicked = st.button("Reset Defaults", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Reset Handling
    if reset_clicked:
        st.session_state.min_support = 0.02
        st.session_state.min_confidence = 0.30
        st.session_state.max_k = 3
        st.session_state.apriori_result = None
        st.session_state.rules = None
        st.session_state.last_run_params = None
        st.session_state.workbench_version = st.session_state.get("workbench_version", 0) + 1
        ensure_dataset_loaded()
        st.rerun()

    # Mining Execution with Exception Handling
    if run_clicked and not has_blocking_error:
        st.session_state.min_support = min_sup
        st.session_state.min_confidence = min_conf
        st.session_state.max_k = max_k

        try:
            with st.spinner("Executing Apriori algorithm with candidate pruning..."):
                t0 = time.time()
                result = run_apriori(
                    st.session_state.transactions,
                    min_support=min_sup,
                    max_k=max_k,
                    min_confidence=min_conf,
                )
                rules = generate_rules(
                    result,
                    st.session_state.transactions,
                    min_confidence=min_conf,
                )
                st.session_state.apriori_result = result
                st.session_state.rules = rules
                st.session_state.last_run_params = {
                    "min_support": min_sup,
                    "min_confidence": min_conf,
                    "max_k": max_k,
                }
        except Exception as e:
            st.error(f"Runtime Execution Error: {str(e)}")
            st.session_state.apriori_result = None
            st.session_state.rules = None
            st.session_state.last_run_params = None
            return

    # If there are active constraint violations, halt and do not show stale results
    if has_blocking_error:
        st.warning("Execution Suspended: Correct the input constraint violations above to run the Apriori algorithm and view results.")
        return

    result = st.session_state.apriori_result
    rules = st.session_state.rules

    if result is None:
        st.info("The Apriori algorithm has not been executed yet. Configure the parameters above and click 'Run Apriori' to view frequent itemsets and candidate pruning metrics.")
        return

    # Stale Parameter Notification
    last_params = st.session_state.get("last_run_params")
    current_params = {"min_support": min_sup, "min_confidence": min_conf, "max_k": max_k}
    if last_params is not None and last_params != current_params:
        st.info("Configuration Changed: Parameters above have been modified since the last execution. Results displayed below reflect the previous run. Click 'Run Apriori' to recompute.")

    # Execution KPI Summary
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Frequent Itemsets", f"{len(result['all_frequent_itemsets']):,}")
    with m2:
        st.metric("Candidates Tested", f"{result['total_candidates']:,}")
    with m3:
        st.metric("Candidates Pruned", f"{result['total_pruned']:,}")
    with m4:
        st.metric("Max Size (k)", f"k = {result['max_k_reached']}")
    with m5:
        st.metric("Execution Time", f"{result['execution_time'] * 1000:.1f} ms")

    st.markdown("---")

    # Step-by-Step Algorithm Inspection
    step_tab1, step_tab2, step_tab3, step_tab4 = st.tabs([
        "Step 1: Frequent 1-Itemsets (L1)",
        "Step 2: Candidate Generation & Pruning",
        "Step 3: All Frequent Itemsets",
        "Algorithm Iteration History",
    ])

    with step_tab1:
        st.markdown("#### Step 1: Count Frequencies of Individual Products (C1 -> L1)")
        st.markdown(
            f"""
            <div class="academic-subbar">
                <span class="academic-badge">[CANDIDATE SCAN]</span>
                <span>Single-item frequency count (<code>C<sub>1</sub></code>). Products satisfying <span class="term-highlight">Support &ge; {st.session_state.min_support * 100:.1f}%</span> ({result['min_support_count']:,} receipts) form frequent 1-itemsets (<span class="term-highlight">L<sub>1</sub></span>).</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if len(result["iterations"]) > 0:
            iter1 = result["iterations"][0]
            c1_rows = []
            for entry in iter1["candidates"]:
                item_name = entry["items"][0] if entry.get("items") else next(iter(entry["itemset"]))
                c1_rows.append({
                    "Product": item_name,
                    "Support Count": entry["support_count"],
                    "Support (%)": f"{entry['support_pct'] * 100:.2f}%",
                    "Status": "Frequent (Kept in L1)" if entry["is_frequent"] else "Infrequent (Discarded)",
                    "_count": entry["support_count"],
                })
            df_c1 = pd.DataFrame(c1_rows).sort_values("_count", ascending=False).drop(columns=["_count"])
            st.dataframe(df_c1, hide_index=True, use_container_width=True)

    with step_tab2:
        st.markdown("#### Step 2: Apriori Candidate Generation & Downward-Closure Pruning")
        st.markdown(
            """
            <div class="academic-callout">
                <div class="callout-header">
                    <span class="academic-badge">[THEOREM]</span>
                    <span class="callout-title">The Apriori Monotonicity Property (Agrawal & Srikant, 1994)</span>
                </div>
                <div class="callout-body">
                    <div class="callout-statement">
                        <b>Formal Axiom:</b> All non-empty subsets of a frequent itemset must also be frequent:
                        <code>&forall; X &sube; Y &implies; Support(Y) &le; Support(X)</code>
                    </div>
                    <div class="callout-consequence">
                        <b>Pruning Consequence:</b> If any (k&minus;1)-subset of candidate <i>C<sub>k</sub></i> is infrequent (not in <i>L<sub>k&minus;1</sub></i>), 
                        then <i>C<sub>k</sub></i> is <b>guaranteed infrequent</b> and pruned <b>immediately without scanning raw transaction storage</b>.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        multi_iters = [it for it in result["iterations"] if it["k"] >= 2]
        if not multi_iters:
            st.info("No candidate iterations beyond k=1 were generated at this threshold.")
        else:
            selected_k = st.selectbox(
                "Select Iteration k:",
                options=[it["k"] for it in multi_iters],
                format_func=lambda k: f"Iteration k = {k}",
            )
            curr_iter = next(it for it in multi_iters if it["k"] == selected_k)

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown(f"**Pruned Candidates (Failed Apriori Property):** {curr_iter['pruned_count']:,}")
                if curr_iter["pruned_details"]:
                    pruned_data = [
                        {"Candidate": " + ".join(p["items"]), "Pruning Reason": p["reason"]}
                        for p in curr_iter["pruned_details"]
                    ]
                    st.dataframe(pd.DataFrame(pruned_data), hide_index=True, use_container_width=True)
                else:
                    st.write("No candidates were pruned by the Apriori property at this step.")

            with p_col2:
                st.markdown(f"**Frequent {selected_k}-Itemsets (L{selected_k}):** {curr_iter['frequent_count']:,}")
                if curr_iter["frequent_itemsets"]:
                    freq_data = [
                        {
                            "Itemset": " + ".join(f["items"]),
                            "Count": f["support_count"],
                            "Support (%)": f"{f['support_pct'] * 100:.2f}%",
                        }
                        for f in curr_iter["frequent_itemsets"]
                    ]
                    st.dataframe(pd.DataFrame(freq_data).sort_values("Count", ascending=False), hide_index=True, use_container_width=True)
                else:
                    st.write("No candidates satisfied the minimum support threshold.")

    with step_tab3:
        st.markdown("#### Complete List of Discovered Frequent Itemsets")
        st.markdown(
            """
            <div class="academic-subbar">
                <span class="academic-badge">[OUTPUT REPOSITORY]</span>
                <span>Full enumeration of itemsets (<code>L<sub>k</sub></code>) satisfying minimum support threshold across sizes 1 to k.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        all_freq_entries = []
        for it in result["iterations"]:
            for f in it.get("frequent_itemsets", []):
                all_freq_entries.append({
                    "Itemset": " + ".join(f["items"]),
                    "Size (k)": it["k"],
                    "Support Count": f["support_count"],
                    "Support (%)": f"{f['support_pct'] * 100:.2f}%",
                    "_count": f["support_count"],
                })
        if not all_freq_entries:
            st.warning("No frequent itemsets found. Try lowering Minimum Support.")
        else:
            df_all_freq = pd.DataFrame(all_freq_entries).sort_values(["Size (k)", "_count"], ascending=[True, False]).drop(columns=["_count"])
            st.dataframe(df_all_freq, hide_index=True, use_container_width=True)

    with step_tab4:
        st.markdown("#### Iteration Summary Table & Chart")
        st.markdown(
            """
            <div class="academic-subbar">
                <span class="academic-badge">[ALGORITHM TRACE]</span>
                <span>Computational convergence per pass: Candidate Generation (<code>C<sub>k</sub></code>), Pruning Count, and Frequent Yield (<code>L<sub>k</sub></code>).</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        iter_records = []
        for it in result["iterations"]:
            iter_records.append({
                "Iteration (k)": f"k = {it['k']}",
                "Candidates Generated": it["candidate_count"],
                "Pruned by Apriori": it["pruned_count"],
                "Frequent Itemsets": it["frequent_count"],
            })
        st.dataframe(pd.DataFrame(iter_records), hide_index=True, use_container_width=True)

        fig_iter = plot_iteration_summary(result["iterations"])
        if fig_iter:
            st.plotly_chart(fig_iter, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 2: Association Rules
# ---------------------------------------------------------------------------
def render_association_rules():
    ensure_default_rules()
    st.markdown("### Association Rules Explorer")
    st.markdown(
        """
        <div class="academic-subbar">
            <span class="academic-badge">[RULE DISCOVERY]</span>
            <span>Implication rules (<code>X &rarr; Y</code>) evaluated across <span class="term-highlight">Support</span>, <span class="term-highlight">Confidence</span>, and <span class="term-highlight">Lift</span>.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rules = st.session_state.rules
    if rules is None:
        st.info("No rules available. Run the Apriori algorithm from the 'Apriori Workbench' tab first.")
        return

    if len(rules) == 0:
        st.warning(
            "No candidate association rules could be generated with itemset size >= 2. "
            "Try lowering Minimum Support in the Apriori Workbench."
        )
        return

    accepted_rules = [r for r in rules if r["accepted"]]
    if not accepted_rules:
        st.warning(
            f"None of the {len(rules):,} candidate rules met the Minimum Confidence threshold "
            f"({st.session_state.min_confidence * 100:.1f}%). Try lowering Minimum Confidence in the Apriori Workbench."
        )
        return

    # Filter Bar
    with st.container():
        st.markdown(
            """
            <div class="modern-card">
                <div class="card-title">Filter & Search Rules</div>
            """,
            unsafe_allow_html=True,
        )
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            filter_conf = st.slider("Min Confidence Filter", 0.0, 1.0, float(st.session_state.min_confidence), 0.05)
        with f_col2:
            filter_lift = st.slider("Min Lift Filter", 0.5, 5.0, 1.0, 0.1)
        with f_col3:
            search_item = st.text_input("Filter by Product Name", placeholder="e.g. Milk, Bread")
        with f_col4:
            sort_by = st.selectbox("Sort Rules By", ["Lift (Highest First)", "Confidence (Highest First)", "Support (Highest First)"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    filtered = [
        r for r in accepted_rules
        if r["confidence"] >= filter_conf and r["lift"] >= filter_lift
    ]
    if search_item.strip():
        q_tokens = [tok.strip().lower() for tok in search_item.replace(";", ",").split(",") if tok.strip()]
        if q_tokens:
            def rule_matches_query(r):
                rule_items = [it.lower() for it in r.get("antecedent_items", []) + r.get("consequent_items", [])]
                return all(any(tok in it for it in rule_items) for tok in q_tokens)
            filtered = [r for r in filtered if rule_matches_query(r)]

    # Sort
    if "Lift" in sort_by:
        filtered.sort(key=lambda r: r["lift"], reverse=True)
    elif "Confidence" in sort_by:
        filtered.sort(key=lambda r: r["confidence"], reverse=True)
    else:
        filtered.sort(key=lambda r: r.get("support", r.get("support_pct", 0.0)), reverse=True)

    # Summary KPI
    rf1, rf2, rf3, rf4 = st.columns(4)
    with rf1:
        st.metric("Total Rules Found", len(filtered))
    with rf2:
        avg_conf = np.mean([r["confidence"] for r in filtered]) if filtered else 0
        st.metric("Average Confidence", f"{avg_conf * 100:.1f}%")
    with rf3:
        avg_lift = np.mean([r["lift"] for r in filtered]) if filtered else 0
        st.metric("Average Lift", f"{avg_lift:.2f}x")
    with rf4:
        max_lift = max([r["lift"] for r in filtered]) if filtered else 0
        st.metric("Maximum Lift", f"{max_lift:.2f}x")

    st.markdown("---")

    # Visualizations
    v_col1, v_col2 = st.columns([1.2, 1])
    with v_col1:
        st.markdown("#### Support vs. Confidence (Colored by Lift)")
        if filtered:
            fig_scatter = plot_support_confidence_scatter(filtered)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No rules match the current filter criteria to display in the scatter plot.")

    with v_col2:
        top_bar_h, top_bar_sel = st.columns([1.4, 1])
        with top_bar_h:
            st.markdown("#### Top Association Rules")
        with top_bar_sel:
            ranking_metric = st.selectbox(
                "Rank by metric",
                ["Lift", "Confidence", "Support"],
                index=0,
                key="chart_rank_metric",
                label_visibility="collapsed",
            )
        metric_key = {"Lift": "lift", "Confidence": "confidence", "Support": "support_pct"}[ranking_metric]
        if filtered:
            fig_bar = plot_top_rules_bar(filtered, metric=metric_key, top_n=10)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No rules match the current filter criteria to display in the ranking chart.")

    # Rule Association Network Graph
    with st.expander("Interactive Item Association Network Graph", expanded=False):
        if filtered:
            fig_net = plot_rule_network(filtered, max_rules=30)
            if fig_net:
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.info("No network edges could be formed from the filtered rules.")
        else:
            st.info("Filter returned 0 rules. Adjust filters above to inspect the association network graph.")

    # Rule Table
    st.markdown("#### Discovered Rules Table")
    if not filtered:
        st.warning("No rules match the current filter criteria. Try lowering the Lift or Confidence filter.")
    else:
        table_rows = []
        for r in filtered:
            ante_str = ", ".join(sorted(r["antecedent"]))
            cons_str = ", ".join(sorted(r["consequent"]))
            lift_val = r["lift"]
            lift_eval = "Positive (> 1.0)" if lift_val > 1.05 else ("Negative (< 1.0)" if lift_val < 0.95 else "Neutral (~ 1.0)")
            sup_val = r.get("support", r.get("support_pct", 0.0))
            table_rows.append({
                "Rule": f"{{{ante_str}}} -> {{{cons_str}}}",
                "Antecedent": ante_str,
                "Consequent": cons_str,
                "Support (%)": f"{sup_val * 100:.2f}%",
                "Confidence (%)": f"{r['confidence'] * 100:.1f}%",
                "Lift": f"{r['lift']:.2f}x",
                "Association Type": lift_eval,
                "_lift": r["lift"],
            })
        st.dataframe(pd.DataFrame(table_rows).drop(columns=["_lift"]), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB: Transaction Overview & Management (FreshMarket Supermarket)
# ---------------------------------------------------------------------------
def render_transaction_overview():
    # -----------------------------------------------------------------------
    # Real-World Operational Context & System Architecture
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div class="system-overview-card">
            <div class="system-header-row">
                <span class="academic-badge">[OPERATIONAL PIPELINE]</span>
                <span class="system-badge">Enterprise Retail Intelligence Platform</span>
            </div>
            <div class="system-title">FreshMarket Point-of-Sale (POS) & Transaction Management</div>
            <div class="system-lead-text">
                Enterprise retail intelligence platform bridging high-throughput checkout streams with Apriori pattern mining to automate cross-selling recommendations and store merchandise layouts.
            </div>
            <div class="pipeline-grid">
                <div class="pipeline-step-card">
                    <div class="step-number">Phase 01 · Ingestion</div>
                    <div class="step-title">POS Transaction Stream</div>
                    <div class="step-desc">Captures live customer checkout receipts logging <b>10,000 transactions</b> across <b>80 retail products</b>.</div>
                </div>
                <div class="pipeline-step-card">
                    <div class="step-number">Phase 02 · Mining Core</div>
                    <div class="step-title">Apriori Pruning Engine</div>
                    <div class="step-desc">Eliminates infrequent candidate sets via <b>downward-closure pruning</b> without scanning raw storage.</div>
                </div>
                <div class="pipeline-step-card">
                    <div class="step-number">Phase 03 · Analytics</div>
                    <div class="step-title">Association Rules</div>
                    <div class="step-desc">Evaluates statistical significance using <b>Support</b>, <b>Confidence</b>, and <b>Lift &gt; 1.0</b>.</div>
                </div>
                <div class="pipeline-step-card">
                    <div class="step-number">Phase 04 · Commercial Action</div>
                    <div class="step-title">Product Recommender</div>
                    <div class="step-desc">Deploys real-time <b>cross-selling suggestions</b> at checkout kiosks and guides aisle shelf placement.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = st.session_state.dataset_df
    stats = st.session_state.dataset_stats

    # Real-World Operational KPIs
    st.markdown("#### Operational Store Metrics & Checkout Volume")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Checkout Receipts", f"{stats['n_transactions']:,}")
    with k2:
        st.metric("Active Catalog Products", f"{stats['n_unique_items']:,}")
    with k3:
        st.metric("Average Basket Volume", f"{stats['avg_basket_size']:.1f} items")
    with k4:
        st.metric("Basket Size Spread", f"{stats['min_basket_size']} - {stats['max_basket_size']} items")

    st.markdown("---")

    # Retail Sales Analytics & Department Turnover
    st.markdown("#### Retail Demand & Customer Purchasing Habits")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Top 15 Highest-Volume Grocery Products")
        st.caption("High-velocity core staples frequently serving as transaction anchor items.")
        fig_freq = plot_item_frequency(stats["item_frequencies"], top_n=15)
        st.plotly_chart(fig_freq, use_container_width=True)

    with c2:
        st.markdown("##### Customer Basket Size Distribution")
        st.caption("Distribution of item counts per transaction (quick runs vs. weekly grocery hauls).")
        fig_size = plot_basket_size_distribution(stats["basket_sizes"])
        st.plotly_chart(fig_size, use_container_width=True)

    # Category Breakdown
    st.markdown("##### Product Department / Category Breakdown")
    fig_cat = plot_category_distribution(stats.get("category_distribution"), PRODUCT_INFO)
    if fig_cat:
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Interactive Transaction Basket Explorer (Grouped Market Baskets)
    # -----------------------------------------------------------------------
    st.markdown("#### Customer Checkout Receipts & Market Basket Explorer")
    st.caption(
        "Each record below represents a complete customer checkout receipt (market basket). "
        "Filter by basket size, test preset grocery bundles, or look up specific Transaction IDs."
    )

    # Ensure precomputed grouped baskets dataframe exists
    if "baskets_df" not in st.session_state or st.session_state.baskets_df is None:
        grouped = (
            df.groupby("transaction_id")["item"]
            .apply(lambda items: sorted(list(items)))
            .reset_index(name="item_list")
        )
        grouped["item_count"] = grouped["item_list"].apply(len)
        grouped["basket_display"] = grouped["item_list"].apply(lambda items: "{" + ", ".join(items) + "}")
        st.session_state.baskets_df = grouped

    baskets_df = st.session_state.baskets_df

    # Quick Search Presets
    st.markdown("**Quick Preset Filters:**")
    q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)
    with q_col1:
        if st.button("Bread & Butter", key="btn_filter_bb", use_container_width=True):
            st.session_state.basket_search_input = "Bread, Butter"
            st.rerun()
    with q_col2:
        if st.button("Milk & Cookies", key="btn_filter_mc", use_container_width=True):
            st.session_state.basket_search_input = "Milk, Cookies"
            st.rerun()
    with q_col3:
        if st.button("Coffee & Sugar", key="btn_filter_cs", use_container_width=True):
            st.session_state.basket_search_input = "Coffee, Sugar"
            st.rerun()
    with q_col4:
        if st.button("Pasta & Olive Oil", key="btn_filter_po", use_container_width=True):
            st.session_state.basket_search_input = "Pasta, Olive Oil"
            st.rerun()
    with q_col5:
        if st.button("Reset Filter", key="btn_filter_reset", use_container_width=True):
            st.session_state.basket_search_input = ""
            st.rerun()

    if "basket_search_input" not in st.session_state:
        st.session_state.basket_search_input = ""

    s_col1, s_col2 = st.columns([3.5, 1.5])
    with s_col1:
        search_query = st.text_input(
            "Search Baskets (Product names or Transaction ID):",
            value=st.session_state.basket_search_input,
            placeholder="e.g. Bread, Butter or Milk or T00001 or 1",
            help="Comma-separated product names (e.g. 'Milk, Bread') finds baskets containing all entered items. Also matches Transaction IDs.",
        )
        if search_query != st.session_state.basket_search_input:
            st.session_state.basket_search_input = search_query

    with s_col2:
        size_filter = st.selectbox(
            "Basket Size:",
            ["All Sizes", "Small (1-2 items)", "Medium (3-5 items)", "Large (6+ items)"],
        )

    # Filter evaluation
    filtered_baskets = baskets_df

    # Size filtering
    if size_filter == "Small (1-2 items)":
        filtered_baskets = filtered_baskets[filtered_baskets["item_count"] <= 2]
    elif size_filter == "Medium (3-5 items)":
        filtered_baskets = filtered_baskets[(filtered_baskets["item_count"] >= 3) & (filtered_baskets["item_count"] <= 5)]
    elif size_filter == "Large (6+ items)":
        filtered_baskets = filtered_baskets[filtered_baskets["item_count"] >= 6]

    # Search query filtering
    q_str = search_query.strip()
    if q_str:
        import re
        clean_q = q_str.lower()
        # Check if entire query is an explicit Transaction ID (e.g. "T00001", "T1", "1")
        tid_num_match = re.fullmatch(r"t?0*(\d+)", clean_q)
        target_tid = int(tid_num_match.group(1)) if tid_num_match else None

        tokens = [t.strip().lower() for t in q_str.replace(";", ",").split(",") if t.strip()]
        if tokens:
            def matches_basket(row):
                tid_str = row["transaction_id"].lower()
                row_num = int(re.sub(r"\D", "", tid_str) or 0)

                # If user searched an explicit transaction ID number (e.g. '1' -> T00001, '42' -> T00042):
                if target_tid is not None and target_tid > 0:
                    return row_num == target_tid

                # Otherwise check multi-token items
                items_lower = [it.lower() for it in row["item_list"]]
                for t in tokens:
                    # Token matches TID exactly
                    tok_tid = re.fullmatch(r"t?0*(\d+)", t)
                    if tok_tid and int(tok_tid.group(1)) == row_num:
                        continue
                    if t == tid_str:
                        continue
                    # Must match at least one product in basket
                    if not any(t in it for it in items_lower):
                        return False
                return True

            mask = filtered_baskets.apply(matches_basket, axis=1)
            filtered_baskets = filtered_baskets[mask]

    n_matched = len(filtered_baskets)
    if n_matched == 0:
        st.warning(f"No baskets found matching '{search_query}'. Try broader keywords or click 'Reset Filter'.")
    else:
        table_to_show = filtered_baskets[["transaction_id", "basket_display", "item_count"]].copy()
        table_to_show.columns = ["Transaction ID", "Basket Contents", "Total Items"]
        st.dataframe(
            table_to_show.head(200),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Transaction ID": st.column_config.TextColumn("Transaction ID", width="small"),
                "Basket Contents": st.column_config.TextColumn("Basket Contents {Items...}", width="large"),
                "Total Items": st.column_config.NumberColumn("Total Items", format="%d items", width="small"),
            },
        )
        st.caption(f"Showing first {min(200, n_matched):,} baskets out of {n_matched:,} matching transactions (Total: {len(baskets_df):,} baskets in dataset).")

render_dataset_explorer = render_transaction_overview


# ---------------------------------------------------------------------------
# TAB: Real-World Cross-Selling Recommender
# ---------------------------------------------------------------------------
def render_product_recommender():
    ensure_default_rules()
    st.markdown("### Real-World Application: Cross-Selling Recommender")

    st.markdown(
        """
        <div class="academic-callout">
            <div class="callout-header">
                <span class="academic-badge">[OPERATIONAL ARCHITECTURE]</span>
                <span class="callout-title">Point-of-Sale Real-Time Cross-Selling Mechanism</span>
            </div>
            <div class="callout-body">
                <div class="callout-statement">
                    <b>Antecedent Matching:</b> When customer items match rule conditions (<code>Antecedent(X) &sube; Basket</code>), candidate complements are retrieved.
                </div>
                <div class="callout-consequence">
                    <b>Dispatch Priority:</b> Recommendations are ranked by <span class="term-highlight">Lift &gt; 1.0</span> (correlation strength over independence) and <span class="term-highlight">Confidence</span> (probability of purchase).
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rules = st.session_state.rules
    if not rules:
        st.info("No association rules available yet. Please run Apriori from the 'Apriori Workbench' tab first.")
        return

    accepted_rules = [r for r in rules if r["accepted"]]

    col_cart, col_recom = st.columns([1.15, 1.35])

    with col_cart:
        cart_items = st.session_state.cart_items
        n_items = len(cart_items)

        # Highlighted Active Basket Container
        with st.container(border=True):
            st.markdown(
                f'<div class="basket-header-row"><span class="basket-title-text">1. Customer Shopping Basket</span><span class="basket-count-pill">{n_items} {"Item" if n_items == 1 else "Items"}</span></div>',
                unsafe_allow_html=True,
            )

            if not cart_items:
                st.markdown(
                    '<div class="basket-empty-box"><div class="basket-empty-title">Shopping Basket is Empty</div><div class="basket-empty-desc">Select popular items below or search the product catalog to simulate checkout and trigger recommendations.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                total_price = sum(PRODUCT_INFO.get(it, {}).get("price", 3.49) for it in cart_items)
                row_items_html = "".join([
                    f'<div class="basket-item-row"><div class="basket-item-info"><span class="basket-item-name">{item}</span><span class="basket-item-dept">{PRODUCT_INFO.get(item, {}).get("category", "Grocery")}</span></div><span class="basket-item-price">${PRODUCT_INFO.get(item, {}).get("price", 3.49):.2f}</span></div>'
                    for item in sorted(cart_items)
                ])
                st.markdown(f'<div class="basket-items-list">{row_items_html}</div>', unsafe_allow_html=True)

                st.markdown(
                    f'<div class="basket-summary-footer"><span class="basket-total-label">Estimated Basket Subtotal ({n_items} items):</span><span class="basket-total-value">${total_price:.2f}</span></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Clear Basket", key="btn_clear_basket", type="secondary", use_container_width=True):
                    st.session_state.cart_items.clear()
                    st.session_state.cart_version = st.session_state.get("cart_version", 0) + 1
                    st.rerun()

        # Add Items Section
        with st.container():
            st.markdown("**Quick Add Popular Items:**")
            quick_items = ["Bread", "Butter", "Milk", "Coffee", "Sugar", "Eggs", "Cookies", "Pasta"]
            q_cols = st.columns(4)
            for i, q_item in enumerate(quick_items):
                with q_cols[i % 4]:
                    in_cart = q_item in st.session_state.cart_items
                    btn_label = f"{q_item} (In Cart)" if in_cart else f"+ {q_item}"
                    btn_type = "primary" if in_cart else "secondary"
                    if st.button(btn_label, key=f"quick_{q_item}", type=btn_type, use_container_width=True):
                        if in_cart:
                            st.session_state.cart_items.remove(q_item)
                        else:
                            st.session_state.cart_items.add(q_item)
                        st.session_state.cart_version = st.session_state.get("cart_version", 0) + 1
                        st.rerun()

            all_catalog_items = sorted(list(PRODUCT_INFO.keys()))
            valid_defaults = sorted([p for p in st.session_state.cart_items if p in all_catalog_items])
            cart_v = st.session_state.get("cart_version", 0)
            selected_from_picker = st.multiselect(
                "Or search from 80 supermarket products:",
                options=all_catalog_items,
                default=valid_defaults,
                key=f"cart_picker_{cart_v}",
            )
            if set(selected_from_picker) != st.session_state.cart_items:
                st.session_state.cart_items = set(selected_from_picker)
                st.rerun()

    with col_recom:
        st.markdown(
            '<div class="modern-card"><div class="card-title">2. Product Recommendations</div><div class="card-subtitle">Triggered in real time based on active association rules.</div></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.cart_items:
            st.markdown(
                '<div class="recom-empty-box"><div class="recom-empty-title">Recommendations Inactive (Basket Empty)</div><div class="recom-empty-desc">Add one or more products to the customer basket on the left to activate the recommendation engine and discover personalized cross-selling suggestions.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            recs = get_recommendations(st.session_state.cart_items, accepted_rules, sort_by="lift")
            if not recs:
                st.markdown(
                    '<div class="explainer-box"><h4>No Matching Association Rules</h4><p>No discovered rules match this specific item combination. Tips: Add complementary products (e.g. Bread, Butter, Coffee) or lower thresholds in the Apriori Workbench.</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.success(f"Found {len(recs)} recommended complementary product(s)!")
                for idx, r in enumerate(recs):
                    rec_item = r.get("item", (r.get("recommended_items") or ["Product"])[0])
                    ante_str = ", ".join(sorted(r["antecedent"]))
                    cat = PRODUCT_INFO.get(rec_item, {}).get("category", "Grocery")
                    price = PRODUCT_INFO.get(rec_item, {}).get("price", 3.99)
                    sup_val = r.get("support", r.get("support_pct", 0.0))

                    card_html = (
                        f'<div class="recom-card">'
                        f'<div class="recom-header"><span class="recom-item-name">{rec_item} (${price:.2f})</span><span class="badge badge-blue">{cat}</span></div>'
                        f'<div class="recom-reason">Triggered because your basket contains <b>[{ante_str}]</b>.</div>'
                        f'<div class="recom-metrics">'
                        f'<span class="metric-chip">Rule: {{{ante_str}}} &rarr; {{{rec_item}}}</span>'
                        f'<span class="metric-chip">Confidence: {r["confidence"]*100:.1f}%</span>'
                        f'<span class="metric-chip">Lift: {r["lift"]:.2f}x</span>'
                        f'<span class="metric-chip">Support: {sup_val*100:.1f}%</span>'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
                    if st.button(f"+ Add {rec_item} to Basket", key=f"add_{rec_item}_{idx}"):
                        st.session_state.cart_items.add(rec_item)
                        st.session_state.cart_version = st.session_state.get("cart_version", 0) + 1
                        st.rerun()


# ---------------------------------------------------------------------------
# TAB 5: Theory & Reference (Consolidated Academic Knowledge)
# ---------------------------------------------------------------------------
def render_theory_reference():
    st.markdown("### Course Reference: Association Rule Mining Foundations")
    st.markdown(
        """
        <div class="academic-subbar">
            <span class="academic-badge">[FOUNDATIONS]</span>
            <span>Comprehensive reference aligned with <b>Chapter 7: Association Rule Mining</b> in the <i>Intelligent System</i> curriculum (HCMUT).</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_col1, t_col2 = st.columns(2)

    with t_col1:
        with st.container(border=True):
            st.markdown("#### 1. Mathematical Formulations")
            st.caption("Standard definitions from Agrawal et al. (1993, 1994).")
            
            st.markdown("**Support (Frequency of Occurrence):**")
            st.latex(r"\text{Support}(X) = \frac{\sigma(X)}{|D|} = \frac{|\{T \in D \mid X \subseteq T\}|}{|D|}")
            st.markdown(
                '<div class="metric-desc-item"><span class="def-badge">DEFINITION</span> Ratio of transactions containing itemset <span class="term-highlight">X</span> out of total database size <span class="term-highlight">|D|</span>. Threshold: <span class="term-highlight">Support(X) &ge; min_sup</span>.</div>',
                unsafe_allow_html=True,
            )
            st.latex(r"\text{Support}(X \rightarrow Y) = \text{Support}(X \cup Y)")

            st.markdown("**Confidence (Conditional Probability):**")
            st.latex(r"\text{Confidence}(X \rightarrow Y) = P(Y \mid X) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}")
            st.markdown(
                '<div class="metric-desc-item"><span class="def-badge">DEFINITION</span> Conditional probability <span class="term-highlight">P(Y | X)</span> quantifying certainty of finding <span class="term-highlight">Y</span> given that customer has chosen <span class="term-highlight">X</span>. Threshold: <span class="term-highlight">Confidence &ge; min_conf</span>.</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Lift (Strength of Statistical Association):**")
            st.latex(r"\text{Lift}(X \rightarrow Y) = \frac{\text{Confidence}(X \rightarrow Y)}{\text{Support}(Y)} = \frac{P(X \cup Y)}{P(X) \cdot P(Y)}")
            st.markdown(
                """
                <div class="metric-desc-item">
                    <span class="def-badge">INTERPRETATION</span>
                    Ratio of joint occurrence to expected probability under statistical independence:
                    <ul style="margin-top: 6px; padding-left: 20px; line-height: 1.6;">
                        <li><span class="term-highlight">Lift &gt; 1.0</span>: <b>Positive Affinity</b> (co-occurs significantly more than chance).</li>
                        <li><span class="term-highlight">Lift = 1.0</span>: <b>Statistical Independence</b> (events are completely uncorrelated).</li>
                        <li><span class="term-highlight">Lift &lt; 1.0</span>: <b>Negative Substitution</b> (items substitute or repel each other).</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with t_col2:
        with st.container(border=True):
            st.markdown("#### 2. The Apriori Property (Downward-Closure)")
            st.caption("Theoretical cornerstone of candidate space pruning.")

            st.markdown("**Formal Monotonicity Theorem:**")
            st.latex(r"\forall X, Y: (X \subseteq Y) \implies \text{Support}(Y) \le \text{Support}(X)")
            
            st.markdown(
                """
                <div class="academic-callout">
                    <div class="callout-header">
                        <span class="academic-badge">[THEOREM]</span>
                        <span class="callout-title">Downward-Closure Property of Support</span>
                    </div>
                    <div class="callout-body">
                        <div class="callout-statement">
                            <b>Core Axiom:</b> An itemset is frequent if and only if <b>all</b> of its non-empty subsets are frequent.
                        </div>
                        <div class="callout-consequence">
                            <b>Pruning Impact:</b> If any subset of candidate <i>C<sub>k</sub></i> is infrequent, candidate <i>C<sub>k</sub></i> is guaranteed infrequent and discarded prior to counting.
                        </div>
                    </div>
                </div>

                <div class="academic-callout callout-pruning">
                    <div class="callout-header">
                        <span class="academic-badge">[PRUNING SCENARIO]</span>
                        <span class="callout-title">Concrete Pruning Walkthrough</span>
                    </div>
                    <div class="callout-body">
                        <div class="callout-statement">
                            Suppose <b>{Bread, Butter}</b> is infrequent (<code>Support &lt; min_sup</code>).
                        </div>
                        <div class="callout-consequence">
                            Any candidate 3-itemset containing both (e.g., <b>{Bread, Butter, Milk}</b>) is <b>immediately pruned</b> without scanning the database!
                        </div>
                    </div>
                </div>

                <div class="academic-callout callout-principle">
                    <div class="callout-header">
                        <span class="academic-badge">[SCIENTIFIC PRINCIPLE]</span>
                        <span class="callout-title">Correlation vs. Causation</span>
                    </div>
                    <div class="callout-body">
                        <div class="callout-statement">
                            Rule <code>X &rarr; Y</code> documents empirical co-occurrence patterns. It does <b>not</b> prove that purchasing <i>X</i> causally triggers the purchase of <i>Y</i>.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown("#### 3. The 4-Step Mining Workflow")
        st.caption("From raw transaction logs to intelligent operational decisions.")
        
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.markdown("**Step 1: Transaction Data**")
            st.caption("Ingest raw checkout receipts and encode into <span class='term-highlight'>Binary Basket Matrices</span>.", unsafe_allow_html=True)
        with w2:
            st.markdown("**Step 2: Frequent Itemsets**")
            st.caption("Extract high-support itemsets (<code>L<sub>k</sub></code>) via <span class='term-highlight'>Downward-Closure Pruning</span>.", unsafe_allow_html=True)
        with w3:
            st.markdown("**Step 3: Association Rules**")
            st.caption("Generate rules (<code>X &rarr; Y</code>) and filter by <span class='term-highlight'>Confidence</span> and <span class='term-highlight'>Lift</span>.", unsafe_allow_html=True)
        with w4:
            st.markdown("**Step 4: Operational Action**")
            st.caption("Deploy rules to <span class='term-highlight'>POS Cross-Selling</span> and strategic store shelf layouts.", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Router (Category-based Navigation)
# ---------------------------------------------------------------------------
active_tab = st.session_state.active_page

if active_tab in ["Transaction Overview", "Transaction Management", "Dataset Explorer"]:
    render_transaction_overview()
elif active_tab == "Apriori Workbench":
    render_apriori_workbench()
elif active_tab == "Association Rules":
    render_association_rules()
elif active_tab == "Product Recommender":
    render_product_recommender()
elif active_tab == "Theory & Reference":
    render_theory_reference()
else:
    render_transaction_overview()
