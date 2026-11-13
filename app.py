"""Clonetric - Copy-Move Image Forgery Detection Platform.

Commercial SaaS forensic suite providing multi-algorithm copy-move and splicing
forgery localization using classical computer vision and deep learning spatial self-correlation.
"""

import hashlib
import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import ExifTags, Image
import streamlit as st
import torch

from engines import (
    AVAILABLE_ENGINES,
    DeepLearningSelfCorrEngine,
    ELAEngine,
    ORBEngine,
    PatchSIFTEngine,
    SIFTEngine,
    get_engine,
)
from orchestrator import ForgeryOrchestrator


# -----------------------------------------------------------------------------
# Safe HTML Renderer (Prevents CommonMark 4-space indented code block parsing)
# -----------------------------------------------------------------------------
def render_html(html_str: str, target=st):
    """Renders HTML safely by collapsing lines to eliminate leading indentation code block triggers."""
    clean_html = " ".join(line.strip() for line in html_str.splitlines() if line.strip())
    target.markdown(clean_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Clonetric | Copy-Move Image Forgery Detection Platform",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Monoline Vector SVG Icons
# -----------------------------------------------------------------------------
def get_svg_icon(name: str, stroke: str = "#818cf8", width: int = 16, height: int = 16) -> str:
    """Returns a clean, inline monoline vector SVG icon."""
    icons = {
        "layers": (
            f'<svg width="{width}" height="{height}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;">'
            '<polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>'
            '<polyline points="2 17 12 22 22 17"></polyline>'
            '<polyline points="2 12 12 17 22 12"></polyline></svg>'
        ),
        "sliders": (
            f'<svg width="{width}" height="{height}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;">'
            '<line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line>'
            '<line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line>'
            '<line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line>'
            '<line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line>'
            '<line x1="17" y1="16" x2="23" y2="16"></line></svg>'
        ),
        "shield_check": (
            f'<svg width="{width}" height="{height}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -3px; margin-right: 6px;">'
            '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>'
            '<path d="m9 12 2 2 4-4"></path></svg>'
        ),
        "fingerprint": (
            f'<svg width="{width}" height="{height}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;">'
            '<line x1="4" y1="9" x2="20" y2="9"></line><line x1="4" y1="15" x2="20" y2="15"></line>'
            '<line x1="10" y1="3" x2="8" y2="21"></line><line x1="16" y1="3" x2="14" y2="21"></line></svg>'
        ),
        "book": (
            f'<svg width="{width}" height="{height}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;">'
            '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>'
            '<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>'
        ),
        "compass": (
            f'<svg width="{width}" height="{height}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;">'
            '<circle cx="12" cy="12" r="10"></circle>'
            '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"></polygon></svg>'
        ),
        "upload": (
            f'<svg width="{width}" height="{height}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;">'
            '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>'
            '<polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>'
        ),
        "cpu": (
            f'<svg width="{width}" height="{height}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;">'
            '<rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect>'
            '<line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line>'
            '<line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line>'
            '<line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line>'
            '<line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>'
        ),
        "matrix": (
            f'<svg width="{width}" height="{height}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            'style="vertical-align: -2px; margin-right: 6px;">'
            '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>'
            '<line x1="3" y1="9" x2="21" y2="9"></line><line x1="3" y1="15" x2="21" y2="15"></line>'
            '<line x1="9" y1="3" x2="9" y2="21"></line><line x1="15" y1="3" x2="15" y2="21"></line></svg>'
        ),
    }
    return icons.get(name, "")


# -----------------------------------------------------------------------------
# Custom "Clonetric" Vector SVG Logo & Brand Lockup
# -----------------------------------------------------------------------------
def render_brand_header(location: str = "main", hw_text: str = "", target=st):
    """Renders the custom Clonetric Copy-Move displacement vector SVG logo and SaaS title lockup."""
    grad_id = f"cmfdGrad_{location}"

    svg_logo = (
        f'<svg width="34" height="34" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink: 0;">'
        f'<defs><linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="#6366f1" />'
        f'<stop offset="100%" stop-color="#06b6d4" />'
        f'</linearGradient></defs>'
        f'<rect x="4" y="14" width="13" height="13" rx="2.5" stroke="#6366f1" stroke-width="2" fill="rgba(99, 102, 241, 0.2)" />'
        f'<circle cx="10.5" cy="20.5" r="1.5" fill="#818cf8" />'
        f'<rect x="19" y="4" width="13" height="13" rx="2.5" stroke="#06b6d4" stroke-width="2" stroke-dasharray="2 2" fill="rgba(6, 182, 212, 0.2)" />'
        f'<circle cx="25.5" cy="10.5" r="1.5" fill="#22d3ee" />'
        f'<line x1="12" y1="19" x2="23" y2="12" stroke="url(#{grad_id})" stroke-width="2" stroke-linecap="round" />'
        f'<polygon points="24,10 19,12 22,15" fill="#06b6d4" />'
        f'</svg>'
    )

    if location == "sidebar":
        html_str = f"""
        <div class="brand-container sidebar">
            <div class="brand-logo-wrapper">
                {svg_logo}
            </div>
            <div class="brand-info">
                <div class="brand-title-row">
                    <span class="brand-title">Clonetric</span>
                    <span class="brand-badge">CMFD ENGINE v2.4</span>
                </div>
                <div class="brand-subtitle">COPY-MOVE IMAGE FORGERY DETECTION PLATFORM</div>
            </div>
        </div>
        """
    else:
        hw_badge_html = ""
        if hw_text:
            hw_badge_html = f"""
            <div class="hw-status-pill">
                <span class="hw-status-dot"></span>
                <span>{hw_text}</span>
            </div>
            """

        html_str = f"""
        <div class="top-header-bar">
            <div class="brand-container main">
                <div class="brand-logo-wrapper">
                    {svg_logo}
                </div>
                <div class="brand-info">
                    <div class="brand-title-row">
                        <span class="brand-title">Clonetric</span>
                        <span class="brand-badge">CMFD ENGINE v2.4</span>
                    </div>
                    <div class="brand-subtitle">COPY-MOVE IMAGE FORGERY DETECTION PLATFORM</div>
                </div>
            </div>
            {hw_badge_html}
        </div>
        """

    render_html(html_str, target=target)


# -----------------------------------------------------------------------------
# Uniform SaaS Component Design System (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Hide Streamlit default header and footer */
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* Base Typography & Background */
    .stApp {
        background-color: #090d16 !important;
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em;
    }

    code, .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Brand Header Bar */
    .top-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.50) 0%, rgba(15, 23, 42, 0.80) 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 10px;
        margin-bottom: 20px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .top-header-bar:hover {
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 4px 20px -2px rgba(99, 102, 241, 0.12);
    }

    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .brand-info {
        display: flex;
        flex-direction: column;
    }
    .brand-title-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    .brand-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        color: #a5b4fc;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 9999px;
        padding: 2px 8px;
        letter-spacing: 0.04em;
    }
    .brand-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.6875rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* Sidebar Brand Styling */
    .brand-container.sidebar {
        padding: 6px 4px 16px 4px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        margin-bottom: 14px;
    }
    .brand-container.sidebar .brand-title {
        font-size: 1.25rem;
    }
    .brand-container.sidebar .brand-subtitle {
        font-size: 0.58rem;
        letter-spacing: 0.06em;
    }

    /* Hardware Status Pill */
    .hw-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 9999px;
        padding: 5px 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #cbd5e1;
        flex-shrink: 0;
    }
    .hw-status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #3b82f6;
        box-shadow: 0 0 8px #6366f1;
    }

    /* Uniform SaaS Card Surface */
    .saas-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.50) 0%, rgba(15, 23, 42, 0.80) 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        border-radius: 10px !important;
        padding: 18px !important;
        margin-bottom: 14px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .saas-card:hover {
        border-color: rgba(99, 102, 241, 0.45) !important;
        box-shadow: 0 4px 20px -2px rgba(99, 102, 241, 0.12);
    }

    .custom-panel {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.50) 0%, rgba(15, 23, 42, 0.80) 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
        transition: border-color 0.2s ease;
    }
    .custom-panel:hover {
        border-color: rgba(99, 102, 241, 0.45);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.50) 0%, rgba(15, 23, 42, 0.80) 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value-mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 3px;
    }

    /* Modern SaaS Pills with 6px status dots */
    .pill-authentic {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 4px 12px;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .pill-authentic::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10b981;
    }

    .pill-tampered {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 9999px;
        padding: 4px 12px;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .pill-tampered::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #f43f5e;
    }

    .pill-engine {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.28);
        border-radius: 9999px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 600;
    }
    .pill-engine::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #6366f1;
    }

    /* Verdict Panels */
    .verdict-panel-tampered {
        background: linear-gradient(180deg, rgba(127, 29, 29, 0.35) 0%, rgba(15, 23, 42, 0.85) 100%) !important;
        border: 1px solid rgba(244, 63, 94, 0.35) !important;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .verdict-panel-authentic {
        background: linear-gradient(180deg, rgba(6, 78, 59, 0.35) 0%, rgba(15, 23, 42, 0.85) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.35) !important;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .verdict-card-title {
        font-size: 11px;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.06em;
        display: flex;
        align-items: center;
    }
    .verdict-main-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin: 4px 0;
    }
    .verdict-desc-text {
        font-size: 0.8125rem;
        color: #cbd5e1;
        line-height: 1.4;
    }

    /* Primary Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        border: 1px solid rgba(199, 210, 254, 0.2) !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
        border-color: rgba(199, 210, 254, 0.4) !important;
    }

    /* Section Headings */
    .sidebar-heading {
        display: flex;
        align-items: center;
        font-size: 0.82rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 16px 0 8px 0;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .section-title {
        display: flex;
        align-items: center;
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 12px;
    }

    /* Metadata Callouts */
    .metadata-warning {
        background: rgba(120, 53, 15, 0.35);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 6px;
        color: #fde68a;
        padding: 8px 12px;
        font-size: 0.8125rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .metadata-info {
        background: rgba(30, 58, 138, 0.35);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 6px;
        color: #bfdbfe;
        padding: 8px 12px;
        font-size: 0.8125rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .metadata-neutral {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 6px;
        color: #94a3b8;
        padding: 8px 12px;
        font-size: 0.8125rem;
    }

    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #111827;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 6px;
        font-size: 0.8125rem;
        font-weight: 600;
        padding: 6px 14px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.18) !important;
        color: #f8fafc !important;
        border-bottom: 2px solid #6366f1 !important;
    }

    /* Dataframe Container */
    div[data-testid="stDataFrame"] {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.40) 0%, rgba(15, 23, 42, 0.70) 100%);
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        padding: 4px;
    }

    /* Documentation Portal Styling */
    .stage-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        background: rgba(99, 102, 241, 0.18);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 4px;
        padding: 2px 6px;
        margin-right: 6px;
        letter-spacing: 0.05em;
    }
    .specs-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-top: 14px;
    }
    @media (max-width: 768px) {
        .specs-grid {
            grid-template-columns: 1fr;
        }
    }
    .spec-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 8px;
        padding: 14px;
    }
    .spec-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 8px;
    }
    .spec-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }
    .green-dot { background-color: #10b981; }
    .red-dot { background-color: #f43f5e; }
    .spec-card ul {
        margin: 0;
        padding-left: 18px;
        font-size: 0.8125rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    .spec-card li {
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Global Caching & Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_orchestrator(max_dim: int = 800) -> ForgeryOrchestrator:
    """Pre-warms and caches the orchestrator singleton instance."""
    return ForgeryOrchestrator(max_dimension=max_dim, max_workers=5)


@st.cache_resource
def get_cached_mobilenet_extractor(feature_layers_cutoff: int = 3) -> torch.nn.Module:
    """Pre-warms and caches MobileNetV3 feature extractor in memory."""
    from engines.dl_selfcorr_engine import get_cached_feature_extractor

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return get_cached_feature_extractor(dev, feature_layers_cutoff=feature_layers_cutoff)


def get_hardware_info() -> Tuple[str, bool]:
    """Detects active hardware acceleration for the deep learning engine."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA GPU ({name})", True
    return "CPU (Multi-threaded AVX2)", False


def extract_forensic_metadata(raw_bytes: bytes, pil_img: Image.Image, filename: str) -> Dict[str, Any]:
    """Extracts SHA-256 hash, image resolution, color space, and EXIF software tags."""
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    exif_dict: Dict[str, str] = {}
    software_tag: Optional[str] = None

    # 1. Parse EXIF data if present (JPEG, TIFF, WebP)
    try:
        raw_exif = pil_img.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_dict[str(tag_name)] = str(value)
                if str(tag_name).lower() == "software" and not software_tag:
                    software_tag = str(value).strip()
    except Exception:
        pass

    # 2. Parse text chunk metadata (PNG, WebP text blocks)
    if pil_img.info:
        for k, v in pil_img.info.items():
            if k not in exif_dict and isinstance(v, (str, int, float)):
                exif_dict[str(k)] = str(v)
            if "software" in str(k).lower() and not software_tag:
                software_tag = str(v).strip()

    # 3. Detect known photo manipulation / editing software signatures
    suspicious_signatures = [
        "photoshop", "gimp", "canva", "picasa", "lightroom",
        "paint.net", "snapseed", "pixlr", "affinity", "photopea",
        "corel", "vsco", "facetune", "procreate",
    ]
    is_suspicious_software = False
    if software_tag:
        lower_sw = software_tag.lower()
        for sig in suspicious_signatures:
            if sig in lower_sw:
                is_suspicious_software = True
                break

    return {
        "filename": filename,
        "sha256": sha256_hash,
        "width": pil_img.width,
        "height": pil_img.height,
        "mode": pil_img.mode,
        "format": pil_img.format or "RGB Matrix",
        "software": software_tag,
        "is_suspicious_software": is_suspicious_software,
        "exif_dict": exif_dict,
    }


def render_forensic_header(meta: Dict[str, Any]):
    """Renders sleek forensic metadata and cryptographic header above results."""
    with st.expander("Image Integrity & EXIF Metadata", expanded=True):
        c_ex1, c_ex2, c_ex3 = st.columns([1.1, 1.4, 1.5])
        with c_ex1:
            st.markdown(f"**Artifact Name:** `{meta['filename']}`")
            st.markdown(f"**Resolution:** `{meta['width']} x {meta['height']} px`")
            st.markdown(f"**Color Space:** `{meta['mode']}` ({meta['format']})")
        with c_ex2:
            st.markdown("**Cryptographic SHA-256 Hash:**")
            st.code(meta["sha256"], language="text")
        with c_ex3:
            st.markdown("**Metadata Provenance:**")
            if meta["is_suspicious_software"]:
                render_html(f'<div class="metadata-warning">Editing Software Signature: {meta["software"]}</div>')
            elif meta["software"]:
                render_html(f'<div class="metadata-info">Software Header: {meta["software"]}</div>')
            else:
                render_html('<div class="metadata-neutral">Stripped / No EXIF Metadata Found</div>')

            if meta.get("exif_dict") and len(meta["exif_dict"]) > 0:
                with st.popover(f"Detailed EXIF Attributes ({len(meta['exif_dict'])} tags)"):
                    st.json(meta["exif_dict"])


def apply_interactive_tampering(
    image: np.ndarray,
    src_x: int,
    src_y: int,
    patch_w: int,
    patch_h: int,
    dst_x: int,
    dst_y: int,
    rotation: float = 0.0,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clones a patch from (src_x, src_y) to (dst_x, dst_y) with rotation and scale.

    Returns:
        forged_image: The clean tampered RGB image without preview graphics.
        preview_guides: Image showing bounding boxes for source and destination.
    """
    img_h, img_w = image.shape[:2]
    src_x = max(0, min(src_x, img_w - 10))
    src_y = max(0, min(src_y, img_h - 10))
    patch_w = max(10, min(patch_w, img_w - src_x))
    patch_h = max(10, min(patch_h, img_h - src_y))

    patch = image[src_y : src_y + patch_h, src_x : src_x + patch_w].copy()

    if rotation != 0.0 or scale != 1.0:
        center = (patch_w / 2.0, patch_h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle=rotation, scale=scale)
        cos_val = np.abs(rot_mat[0, 0])
        sin_val = np.abs(rot_mat[0, 1])
        new_w = int((patch_h * sin_val) + (patch_w * cos_val))
        new_h = int((patch_h * cos_val) + (patch_w * sin_val))
        rot_mat[0, 2] += (new_w / 2.0) - center[0]
        rot_mat[1, 2] += (new_h / 2.0) - center[1]

        warped_patch = cv2.warpAffine(
            patch,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        alpha_mask = cv2.warpAffine(
            np.ones((patch_h, patch_w), dtype=np.uint8) * 255,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
        )
    else:
        warped_patch = patch
        new_w, new_h = patch_w, patch_h
        alpha_mask = np.ones((new_h, new_w), dtype=np.uint8) * 255

    forged_img = image.copy()
    guide_img = image.copy()

    # Draw Source Box on Guide Image (Cyan #38bdf8)
    cv2.rectangle(
        guide_img,
        (src_x, src_y),
        (src_x + patch_w, src_y + patch_h),
        (56, 189, 248),
        2,
    )
    cv2.putText(
        guide_img,
        "SOURCE",
        (src_x + 4, max(18, src_y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (56, 189, 248),
        1,
        cv2.LINE_AA,
    )

    dst_x = max(0, min(dst_x, img_w - new_w))
    dst_y = max(0, min(dst_y, img_h - new_h))

    target_region = forged_img[dst_y : dst_y + new_h, dst_x : dst_x + new_w]
    mask_norm = (alpha_mask[..., np.newaxis] / 255.0).astype(np.float32)
    blended_target = (1.0 - mask_norm) * target_region.astype(np.float32) + mask_norm * warped_patch.astype(np.float32)
    forged_img[dst_y : dst_y + new_h, dst_x : dst_x + new_w] = np.clip(blended_target, 0, 255).astype(np.uint8)

    # Draw Destination Box on Guide Image (Red #ef4444)
    cv2.rectangle(
        guide_img,
        (dst_x, dst_y),
        (dst_x + new_w, dst_y + new_h),
        (239, 68, 68),
        2,
    )
    cv2.putText(
        guide_img,
        "TARGET",
        (dst_x + 4, max(18, dst_y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (239, 68, 68),
        1,
        cv2.LINE_AA,
    )

    return forged_img, guide_img


# -----------------------------------------------------------------------------
# CMFD Documentation Portal (Algorithm Knowledge Base)
# -----------------------------------------------------------------------------
def render_knowledge_base():
    """Renders structured SaaS documentation of CMFD algorithms and consensus math."""
    render_html(f'<div class="section-title">{get_svg_icon("book")} CMFD Algorithm Knowledge Base</div>')
    st.caption("Technical specification, shift-vector localization principles, and mathematical foundations for Copy-Move Forgery Detection.")

    kb_tabs = st.tabs(
        [
            "Consensus Architecture",
            "SIFT Engine",
            "ORB Engine",
            "Patch-SIFT Engine",
            "Error Level Analysis",
            "DL Self-Correlation",
        ]
    )

    # -------------------------------------------------------------------------
    # Tab 0: Consensus Architecture
    # -------------------------------------------------------------------------
    with kb_tabs[0]:
        render_html(
            """
            <div class="saas-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Multi-Algorithm Consensus Engine</span>
                    <span class="brand-badge">ARCHITECTURE OVERVIEW</span>
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                    Individual copy-move detectors exhibit complementary failure modes: sparse keypoint algorithms fail on smooth clones,
                    dense block matchers struggle with high rotations, and ELA requires consistent JPEG histories.
                    The Consensus Engine executes multiple mathematical formulations in parallel, cross-verifying coherent shift vectors
                    and fusing continuous spatial probability distributions to achieve robust decision boundaries.
                </div>
            </div>
            """
        )

        st.markdown("#### Execution Pipeline")
        st.code(
            """
[Input RGB Artifact]
       |
       +-- SIFT Engine (Scale-space extrema + shift vectors) --> [Mask M1]
       +-- ORB Engine (FAST corners + partial affine)       --> [Mask M2]
       +-- Patch-SIFT (Dense blocks + KD-tree shifts)       --> [Mask M3]
       +-- ELA Engine (JPEG recompression difference)       --> [Mask M4]
       +-- DL Self-Corr (MobileNetV3 spatial correlation)   --> [Mask M5]
       |
       v
[Reliability Weighting: w_sift=0.35, w_patch=0.25, w_orb=0.25, w_dl=0.10, w_ela=0.05]
       |
       v
[Weighted Consensus Map: M_c(x,y) = Sum(w_i * M_i) / Sum(w_i)]
       |
       v
[Spatial IoU Overlap & Quorum Gate] --> Verdict: TAMPERED / AUTHENTIC
            """,
            language="text",
        )

        st.markdown("#### Mathematical Formulation & Decision Gate")
        render_html(
            """
            <div class="saas-card">
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 01</span> <strong>Continuous Spatial Probability Normalization:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Each engine generates a normalized continuous 2D spatial probability distribution map:
                    </div>
                    <div class="font-mono" style="margin: 6px 0; color: #a5b4fc; font-size: 0.85rem; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        M_i(x, y) &isin; [0.0, 1.0], &forall; i &isin; {1, ..., K}
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 02</span> <strong>Empirical Reliability Specificity Weighting:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Weights reflect algorithm specificity against natural repeating textures:
                    </div>
                    <div class="font-mono" style="margin: 6px 0; color: #a5b4fc; font-size: 0.85rem; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        w_SIFT = 0.35 | w_Patch = 0.25 | w_ORB = 0.25 | w_DL = 0.10 | w_ELA = 0.05
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 03</span> <strong>Weighted Spatial Consensus Map Fusion:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Fuses all active probability masks into one unified spatial heat distribution:
                    </div>
                    <div class="font-mono" style="margin: 6px 0; color: #a5b4fc; font-size: 0.85rem; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        M_consensus(x, y) = &Sigma;(w_i &times; M_i(x, y)) / &Sigma;(w_i)
                    </div>
                </div>
                <div>
                    <span class="stage-pill">STAGE 04</span> <strong>Spatial IoU Overlap & Quorum Gate:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Dual-condition verification requiring both continuous confidence and shift-vector geometric agreement:
                    </div>
                    <div class="font-mono" style="margin: 6px 0; color: #a5b4fc; font-size: 0.85rem; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        Verdict = TAMPERED if (Conf &ge; 40.0% AND (Flagged_Engines &ge; 2 OR Shift_Verified)) else AUTHENTIC
                    </div>
                </div>
            </div>
            """
        )

    # -------------------------------------------------------------------------
    # Tab 1: SIFT Engine
    # -------------------------------------------------------------------------
    with kb_tabs[1]:
        render_html(
            """
            <div class="saas-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">SIFT (Scale-Invariant Feature Transform)</span>
                    <span class="pill-engine">CMFD FEATURE DETECTOR</span>
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                    Detects scale-space extrema (blobs and corners) that remain strictly invariant to translation,
                    rotation, continuous scaling, and affine illumination shifts between cloned regions.
                </div>
            </div>
            """
        )

        st.markdown("#### Detection Pipeline & Shift Vectors")
        render_html(
            """
            <div class="saas-card">
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 01</span> <strong>Scale-Space Extrema Detection (DoG):</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Convolves input image with variable-scale Gaussians and calculates Difference-of-Gaussians:
                    </div>
                    <div class="font-mono" style="margin: 6px 0; color: #a5b4fc; font-size: 0.85rem; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        D(x, y, &sigma;) = L(x, y, k&sigma;) - L(x, y, &sigma;)
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 02</span> <strong>128-D Descriptors with Canonical Orientation:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Assigns dominant gradient orientation &theta; for rotation invariance, followed by a 16x16 patch divided into sixteen 4x4 subregions contributing an 8-bin histogram (4 x 4 x 8 = 128 dimensions).
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 03</span> <strong>Self-Descriptor Matching with Lowe's Ratio:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Matches each keypoint against all other keypoints in the image (k-NN, k=3), comparing the first non-self neighbor d1 against background neighbor d2:
                    </div>
                    <div class="font-mono" style="margin: 6px 0; color: #a5b4fc; font-size: 0.85rem; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        d1 / d2 &lt; 0.70
                    </div>
                </div>
                <div>
                    <span class="stage-pill">STAGE 04</span> <strong>Shift-Vector Clustering & RANSAC:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Filters out pairs with spatial Euclidean distance &lt; 5% of image diagonal. Calculates displacement vectors &Delta; = (x2 - x1, y2 - y1); clusters sharing coherent shift vectors confirm copy-move duplication.
                    </div>
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="specs-grid">
                <div class="spec-card">
                    <div class="spec-title"><span class="spec-dot green-dot"></span> Strengths in CMFD</div>
                    <ul>
                        <li>Invariant to arbitrary 2D rotation angles and continuous geometric scaling.</li>
                        <li>High discriminative power with 128-D gradient representations.</li>
                        <li>Robust against post-copy illumination variations and contrast adjustments.</li>
                    </ul>
                </div>
                <div class="spec-card">
                    <div class="spec-title"><span class="spec-dot red-dot"></span> Operational Boundaries</div>
                    <ul>
                        <li>Fails on smooth, low-contrast, or textureless regions (skies, walls).</li>
                        <li>Higher computational latency due to dense floating-point descriptor distances.</li>
                    </ul>
                </div>
            </div>
            """
        )

    # -------------------------------------------------------------------------
    # Tab 2: ORB Engine
    # -------------------------------------------------------------------------
    with kb_tabs[2]:
        render_html(
            """
            <div class="saas-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">ORB (Oriented FAST and Rotated BRIEF)</span>
                    <span class="pill-engine">HIGH-THROUGHPUT CMFD</span>
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                    A computationally efficient binary feature detector combining FAST corners with steered BRIEF descriptors,
                    optimized for real-time edge screening and rapid clone localization.
                </div>
            </div>
            """
        )

        st.markdown("#### Detection Pipeline & Shift Vectors")
        render_html(
            """
            <div class="saas-card">
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 01</span> <strong>FAST Corner Extraction across Pyramids:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Evaluates 16-pixel Bresenham circles around candidate pixels to identify rapid intensity discontinuities across a multi-scale pyramid.
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 02</span> <strong>Intensity Centroid Orientation:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Computes patch moments m_pq = &Sigma; x^p y^q I(x, y) to determine centroid angle:
                    </div>
                    <div class="font-mono" style="margin: 6px 0; color: #a5b4fc; font-size: 0.85rem; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        &theta; = atan2(m_01, m_10)
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 03</span> <strong>Steered BRIEF 256-Bit Binary Descriptors:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Computes pairwise intensity tests steered by orientation angle &theta;. Matches descriptors via hardware bitwise XOR (NORM_HAMMING).
                    </div>
                </div>
                <div>
                    <span class="stage-pill">STAGE 04</span> <strong>Partial Affine RANSAC Verification:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Solves for rigid partial affine transformations between candidate match clusters. Requires &ge; 6 inliers to trigger positive tamper detection.
                    </div>
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="specs-grid">
                <div class="spec-card">
                    <div class="spec-title"><span class="spec-dot green-dot"></span> Strengths in CMFD</div>
                    <ul>
                        <li>Extremely low latency (sub-50ms execution on CPU).</li>
                        <li>Hardware-accelerated bitwise Hamming distance computation.</li>
                        <li>Resilient to patch rotations through intensity centroid steering.</li>
                    </ul>
                </div>
                <div class="spec-card">
                    <div class="spec-title"><span class="spec-dot red-dot"></span> Operational Boundaries</div>
                    <ul>
                        <li>Lower scale tolerance than SIFT on large rescaled clones.</li>
                        <li>Sensitive to lossy JPEG compression blocking artifacts.</li>
                    </ul>
                </div>
            </div>
            """
        )

    # -------------------------------------------------------------------------
    # Tab 3: Patch-SIFT Engine
    # -------------------------------------------------------------------------
    with kb_tabs[3]:
        render_html(
            """
            <div class="saas-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Patch-SIFT (Dense Block-Based Matching)</span>
                    <span class="pill-engine">DENSE BLOCK CMFD</span>
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                    Exhaustive overlapping block matching designed specifically for copy-move cloning detection in smooth,
                    low-contrast, and textureless image regions where sparse keypoint detectors fail.
                </div>
            </div>
            """
        )

        st.markdown("#### Detection Pipeline & Shift Vectors")
        render_html(
            """
            <div class="saas-card">
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 01</span> <strong>Scale Bounding & Block Extraction:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Resizes image to max 400px while maintaining aspect ratio, extracting 16x16 pixel blocks with a stride of 4 pixels.
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 02</span> <strong>Variance Gating:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Prunes flat background blocks where pixel standard deviation &sigma; = std(B_k) &lt; 5.0, eliminating false positives on uniform skies.
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 03</span> <strong>8-D Feature Extraction & KD-Tree Indexing:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Downsamples zero-mean blocks into compact 8-D normalized descriptors, querying a spatial cKDTree for O(N log N) nearest neighbor pairs.
                    </div>
                </div>
                <div>
                    <span class="stage-pill">STAGE 04</span> <strong>Shift-Vector Histogram Verification:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Quantizes rounded displacement vectors &Delta; = (|x2 - x1|, |y2 - y1|). If the dominant displacement vector appears &ge; 6 times, the cloned region is localized.
                    </div>
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="specs-grid">
                <div class="spec-card">
                    <div class="spec-title"><span class="spec-dot green-dot"></span> Strengths in CMFD</div>
                    <ul>
                        <li>100% detection reliability on exact and near-exact copy-move clones.</li>
                        <li>Dense coverage across smooth, uniform, or low-corner areas.</li>
                        <li>Robust spatial displacement quantization via cKDTree nearest neighbors.</li>
                    </ul>
                </div>
                <div class="spec-card">
                    <div class="spec-title"><span class="spec-dot red-dot"></span> Operational Boundaries</div>
                    <ul>
                        <li>Requires scale-bounding to preserve sub-2-second execution bounds.</li>
                        <li>Sensitive to significant patch rotation (> 15 degrees).</li>
                    </ul>
                </div>
            </div>
            """
        )

    # -------------------------------------------------------------------------
    # Tab 4: Error Level Analysis (ELA)
    # -------------------------------------------------------------------------
    with kb_tabs[4]:
        render_html(
            """
            <div class="saas-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Error Level Analysis (ELA)</span>
                    <span class="pill-engine">COMPRESSION DOMAIN</span>
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                    Analyzes discrete cosine transform (DCT) quantization error distributions. Spliced or cloned regions
                    originating from disparate compression generations exhibit divergent error rates upon re-compression.
                </div>
            </div>
            """
        )

        st.markdown("#### Detection Pipeline & Error Equilibrium")
        render_html(
            """
            <div class="saas-card">
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 01</span> <strong>In-Memory Reference Recompression:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Recompresses image at reference quality Q = 90 via an in-memory byte buffer with zero disk I/O.
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 02</span> <strong>Absolute Pixel Difference Map:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Calculates channel-wise absolute pixel error:
                    </div>
                    <div class="font-mono" style="margin: 6px 0; color: #a5b4fc; font-size: 0.85rem; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        D(x, y) = 1/3 &times; &Sigma; |I_orig(x, y, c) - I_recomp(x, y, c)|
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 03</span> <strong>Structural Edge Damping:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Computes Sobel gradient magnitude M_edge to suppress natural high-contrast edges from generating false positives:
                    </div>
                    <div class="font-mono" style="margin: 6px 0; color: #a5b4fc; font-size: 0.85rem; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        E_damped(x, y) = D(x, y) &times; (1.0 - &lambda; &times; M_edge(x, y))
                    </div>
                </div>
                <div>
                    <span class="stage-pill">STAGE 04</span> <strong>Statistical Anomaly Thresholding:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Detects connected spatial clusters exceeding threshold T = &mu;_E + 2.4 &times; &sigma;_E.
                    </div>
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="specs-grid">
                <div class="spec-card">
                    <div class="spec-title"><span class="spec-dot green-dot"></span> Strengths in CMFD</div>
                    <ul>
                        <li>Effective at identifying multi-generation saves and composite boundaries.</li>
                        <li>Independent of keypoint geometry or visual patterns.</li>
                    </ul>
                </div>
                <div class="spec-card">
                    <div class="spec-title"><span class="spec-dot red-dot"></span> Operational Boundaries</div>
                    <ul>
                        <li>Ineffective on uncompressed lossless originals (PNG, TIFF).</li>
                        <li>Uniform re-saving after forgery can equalize quantization error rates.</li>
                    </ul>
                </div>
            </div>
            """
        )

    # -------------------------------------------------------------------------
    # Tab 5: DL Spatial Self-Correlation
    # -------------------------------------------------------------------------
    with kb_tabs[5]:
        render_html(
            """
            <div class="saas-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Deep Learning Spatial Self-Correlation</span>
                    <span class="pill-engine">DEEP FEATURE CMFD</span>
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                    Utilizes an ImageNet-pretrained convolutional neural network (MobileNetV3-Small) to extract high-level semantic
                    feature maps and computes spatial Gram correlation matrices across all pixel neighborhoods.
                </div>
            </div>
            """
        )

        st.markdown("#### Detection Pipeline & Gram Matrix")
        render_html(
            """
            <div class="saas-card">
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 01</span> <strong>Early Convolutional Feature Maps:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Extracts intermediate feature representation F &isin; R^(C &times; H' &times; W') from MobileNetV3-Small.
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 02</span> <strong>Unit Hypersphere Normalization:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Normalizes spatial feature channels onto unit hypersphere:
                    </div>
                    <div class="font-mono" style="margin: 6px 0; color: #a5b4fc; font-size: 0.85rem; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
                        x_hat_i = x_i / (||x_i||_2 + &epsilon;)
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="stage-pill">STAGE 03</span> <strong>Spatial Gram Matrix & Neighborhood Exclusion:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Computes all-pairs cosine similarity S = X_hat &times; X_hat^T, applying an exclusion radius (r &ge; 15% feature dimensions) around identical spatial locations.
                    </div>
                </div>
                <div>
                    <span class="stage-pill">STAGE 04</span> <strong>Z-Score Peak Isolation & Bilinear Upsampling:</strong>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">
                        Isolates duplicate correlation peaks exceeding statistical threshold Z &gt; 2.8 and bilinearly interpolates back to source dimensions.
                    </div>
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="specs-grid">
                <div class="spec-card">
                    <div class="spec-title"><span class="spec-dot green-dot"></span> Strengths in CMFD</div>
                    <ul>
                        <li>Resilient against post-tamper blurring, sensor noise, and color grading.</li>
                        <li>Captures semantic context beyond raw pixel intensity values.</li>
                    </ul>
                </div>
                <div class="spec-card">
                    <div class="spec-title"><span class="spec-dot red-dot"></span> Operational Boundaries</div>
                    <ul>
                        <li>May highlight naturally repeating architectural symmetries (identical windows).</li>
                        <li>Requires GPU acceleration for ultra-high resolution inputs.</li>
                    </ul>
                </div>
            </div>
            """
        )


# -----------------------------------------------------------------------------
# 3-Stage Pipeline Breakdown per Executed Algorithm
# -----------------------------------------------------------------------------
def render_algorithm_pipeline_breakdown(engine_key: str, result: Dict[str, Any], raw_image: np.ndarray):
    """Renders the 3-stage pipeline breakdown and theoretical foundation."""
    render_html(f'<div class="section-title" style="font-size: 1rem;">{get_svg_icon("cpu")} Pipeline Breakdown: {engine_key.upper()}</div>')

    stage_cols = st.columns(3)

    with stage_cols[0]:
        st.markdown("**Stage 1: Preprocessed Input**")
        if engine_key in ["sift", "orb", "patch_sift"]:
            gray_view = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
            st.image(gray_view, use_container_width=True, caption="Grayscale Intensity Domain")
        elif engine_key == "ela":
            st.image(raw_image, use_container_width=True, caption="Original RGB Spatial Domain")
        elif engine_key == "dl_selfcorr":
            st.image(raw_image, use_container_width=True, caption="ImageNet Normalized RGB Tensor")

    with stage_cols[1]:
        st.markdown("**Stage 2: Intermediate Representation**")
        caption_dict = {
            "sift": "SIFT Scale-Space Keypoints",
            "orb": "Oriented FAST Keypoints",
            "patch_sift": "Block Partition Grid & Local Adaptive Features",
            "ela": "Quantization Error Difference Map (Q90)",
            "dl_selfcorr": "Spatial Cosine Gram Matrix (Masked)",
        }
        st.image(result["intermediate_view"], use_container_width=True, caption=caption_dict.get(engine_key, "Intermediate Features"))

    with stage_cols[2]:
        st.markdown("**Stage 3: Localized Detection Map**")
        st.image(result["visual_result"], use_container_width=True, caption="Localization Overlay")

    render_html("<div style='height: 16px;'></div>")
    mask_c1, mask_c2 = st.columns([1, 2])
    with mask_c1:
        st.caption("Localization Mask [0.0, 1.0]")
        mask_u8 = (result["binary_mask"] * 255).astype(np.uint8)
        st.image(mask_u8, use_container_width=True, clamp=True)
    with mask_c2:
        st.markdown("**Algorithm Diagnostics**")
        st.json(result["metrics"])


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    hw_text, is_gpu = get_hardware_info()
    _ = get_cached_mobilenet_extractor()

    # Workspace Top Header Bar
    render_brand_header(location="main", hw_text=hw_text, target=st)

    # -------------------------------------------------------------------------
    # Sidebar Navigation & Controls
    # -------------------------------------------------------------------------
    render_brand_header(location="sidebar", target=st.sidebar)

    render_html(f'<div class="sidebar-heading">{get_svg_icon("compass")} Navigation</div>', target=st.sidebar)
    mode = st.sidebar.radio(
        "Navigation",
        [
            "Ensemble Analysis",
            "Live Forge Sandbox",
            "CMFD Knowledge Base",
        ],
        index=0,
        label_visibility="collapsed",
    )

    if mode == "CMFD Knowledge Base":
        render_knowledge_base()
        return

    # Image Input
    st.sidebar.markdown("---")
    render_html(f'<div class="sidebar-heading">{get_svg_icon("upload")} Image Input</div>', target=st.sidebar)
    uploaded = st.sidebar.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
        label_visibility="collapsed",
    )

    working_image: Optional[np.ndarray] = None
    working_name = "Image"
    raw_file_bytes: Optional[bytes] = None
    working_meta: Optional[Dict[str, Any]] = None

    if uploaded is not None:
        raw_file_bytes = uploaded.getvalue()
        working_pil = Image.open(io.BytesIO(raw_file_bytes))
        working_image = np.array(working_pil.convert("RGB"), dtype=np.uint8)
        working_name = uploaded.name
        working_meta = extract_forensic_metadata(raw_file_bytes, working_pil, working_name)

    # Algorithm Selection & Hyperparameters
    st.sidebar.markdown("---")
    render_html(f'<div class="sidebar-heading">{get_svg_icon("layers")} Active Algorithms</div>', target=st.sidebar)
    st.sidebar.caption("Select CMFD detection engines to include in the ensemble analysis.")

    selected_engines = []
    c_sift = st.sidebar.checkbox("SIFT", value=True)
    if c_sift:
        selected_engines.append("sift")
    c_orb = st.sidebar.checkbox("ORB", value=True)
    if c_orb:
        selected_engines.append("orb")
    c_psift = st.sidebar.checkbox("Patch-SIFT", value=True)
    if c_psift:
        selected_engines.append("patch_sift")
    c_ela = st.sidebar.checkbox("ELA", value=True)
    if c_ela:
        selected_engines.append("ela")
    c_dl = st.sidebar.checkbox("MobileNetV3 Self-Correlation", value=True)
    if c_dl:
        selected_engines.append("dl_selfcorr")

    consensus_weights = {}
    render_html(f'<div class="sidebar-heading">{get_svg_icon("sliders")} Hyperparameters</div>', target=st.sidebar)
    with st.sidebar.expander("Tuning Controls", expanded=False):
        min_inliers = st.slider("RANSAC Inliers Threshold", 4, 20, 6, 1)
        ratio_threshold_sift = st.slider("SIFT Ratio Threshold", 0.50, 0.90, 0.70, 0.02)
        ratio_threshold_orb = st.slider("ORB Ratio Threshold", 0.50, 0.90, 0.75, 0.02)
        ela_quality = st.slider("ELA Recompression Quality", 50, 98, 90, 2)
        st.markdown("**Specificity Weighting:**")
        for k in selected_engines:
            default_w = {"sift": 0.35, "orb": 0.25, "patch_sift": 0.25, "dl_selfcorr": 0.10, "ela": 0.05}.get(k, 0.20)
            consensus_weights[k] = st.slider(f"{k.upper()} Weight", 0.05, 1.0, default_w, 0.05)

    if not selected_engines:
        st.sidebar.warning("Select at least one algorithm to run.")
        st.warning("Please select at least one algorithm in the sidebar to proceed.")
        return

    # If No Image Uploaded Yet
    if working_image is None or working_meta is None:
        render_html(
            """
            <div class="custom-panel" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">No Image Loaded</div>
                <div style="color: #94a3b8; font-size: 0.875rem; max-width: 620px; margin: 0 auto 16px auto; line-height: 1.5;">
                    Upload an artifact (JPG, PNG, WEBP, BMP, TIFF) using the sidebar controls to perform multi-algorithm
                    copy-move forgery detection or test custom clone scenarios in the Live Forge Sandbox.
                </div>
            </div>
            """
        )
        return

    # -------------------------------------------------------------------------
    # Mode 2: Live Forge Sandbox
    # -------------------------------------------------------------------------
    forged_image_clean: Optional[np.ndarray] = None
    active_meta = working_meta

    if mode == "Live Forge Sandbox":
        render_html(
            """
            <div class="saas-card">
                <div style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">Live Forge Sandbox</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px; line-height: 1.4;">
                    Synthesize real-time copy-move tampering by cloning an image patch to an alternate spatial coordinate
                    with optional rotation and scaling to evaluate detector robustness.
                </div>
            </div>
            """
        )

        h_img, w_img = working_image.shape[:2]
        col_ctrl1, col_ctrl2 = st.columns(2)

        with col_ctrl1:
            st.markdown("**Source Coordinates**")
            p_w = st.slider("Patch Width (px)", 20, max(21, w_img // 2), min(120, w_img // 3), step=5)
            p_h = st.slider("Patch Height (px)", 20, max(21, h_img // 2), min(120, h_img // 3), step=5)
            s_x = st.slider("Source X", 0, max(1, w_img - p_w), min(60, max(0, w_img - p_w)), step=5)
            s_y = st.slider("Source Y", 0, max(1, h_img - p_h), min(60, max(0, h_img - p_h)), step=5)

        with col_ctrl2:
            st.markdown("**Target Transformations**")
            d_x = st.slider("Target X", 0, max(1, w_img - p_w), min(300, max(0, w_img - p_w)), step=5)
            d_y = st.slider("Target Y", 0, max(1, h_img - p_h), min(300, max(0, h_img - p_h)), step=5)
            rot = st.slider("Rotation Angle (deg)", -180, 180, 0, 5)
            scl = st.slider("Scale Factor", 0.5, 2.0, 1.0, 0.1)

        forged_image_clean, guide_image = apply_interactive_tampering(
            working_image, s_x, s_y, p_w, p_h, d_x, d_y, rotation=rot, scale=scl
        )

        col_pv1, col_pv2 = st.columns(2)
        with col_pv1:
            st.caption("Source and Target Spatial Bounding Boxes")
            st.image(guide_image, use_container_width=True)
        with col_pv2:
            st.caption("Synthesized Copy-Move Image")
            st.image(forged_image_clean, use_container_width=True)

        st.markdown("---")
        btn_sandbox = st.button("Run Forensic Analysis", type="primary", use_container_width=True)

        buf = io.BytesIO()
        Image.fromarray(forged_image_clean).save(buf, format="PNG")
        forged_bytes = buf.getvalue()
        forged_meta = extract_forensic_metadata(
            forged_bytes,
            Image.fromarray(forged_image_clean),
            f"sandbox_{working_name}",
        )

        if btn_sandbox:
            active_target_image = forged_image_clean
            active_image_name = f"tampered_{working_name}"
            active_meta = forged_meta
            st.session_state["execute_sandbox"] = True
        elif st.session_state.get("execute_sandbox", False):
            active_target_image = forged_image_clean
            active_image_name = f"tampered_{working_name}"
            active_meta = forged_meta
        else:
            return
    else:
        active_target_image = working_image
        active_image_name = working_name
        active_meta = working_meta

    # -------------------------------------------------------------------------
    # Cryptographic & Forensic Header
    # -------------------------------------------------------------------------
    render_forensic_header(active_meta)

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------
    orchestrator = get_cached_orchestrator(max_dim=800)

    with st.spinner("Executing CMFD detection engines in parallel..."):
        pipeline_result = orchestrator.analyze(
            active_target_image,
            selected_engines=selected_engines,
            custom_weights=consensus_weights,
        )

    # -------------------------------------------------------------------------
    # Verdict Display
    # -------------------------------------------------------------------------
    verdict = pipeline_result["verdict"]
    conf = pipeline_result["confidence"]
    spatial_iou = pipeline_result["spatial_iou"]
    latency = pipeline_result["execution_time_ms"]
    agreed_engines = pipeline_result["metrics"]["forged_engines"]
    total_eng = pipeline_result["metrics"]["total_engines"]

    render_html(f'<div class="section-title">{get_svg_icon("shield_check")} Copy-Move Consensus Verdict</div>')

    col_hero1, col_hero2, col_hero3 = st.columns([1.1, 1.1, 1.3])

    with col_hero1:
        st.caption(f"Target Artifact: `{active_image_name}` ({active_target_image.shape[1]}x{active_target_image.shape[0]} px)")
        st.image(active_target_image, use_container_width=True)

    with col_hero2:
        st.caption(f"Fused Consensus Localization Map (Spatial IoU: {spatial_iou:.3f})")
        st.image(pipeline_result["heatmap_overlay"], use_container_width=True)

    with col_hero3:
        if verdict == "TAMPERED":
            render_html(
                f"""
                <div class="verdict-panel-tampered">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span class="verdict-card-title">{get_svg_icon('shield_check')} CONSENSUS DECISION</span>
                        <span class="pill-tampered">COPY-MOVE TAMPERED</span>
                    </div>
                    <div class="verdict-main-value" style="color: #fb7185;">TAMPERED</div>
                    <div class="verdict-desc-text">Statistically significant duplicate regions and coherent displacement vectors detected across multiple engines.</div>
                </div>
                """
            )
        else:
            render_html(
                f"""
                <div class="verdict-panel-authentic">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span class="verdict-card-title">{get_svg_icon('shield_check')} CONSENSUS DECISION</span>
                        <span class="pill-authentic">AUTHENTIC ARTIFACT</span>
                    </div>
                    <div class="verdict-main-value" style="color: #34d399;">AUTHENTIC</div>
                    <div class="verdict-desc-text">No statistically significant copy-move clone clusters or displacement shift vectors identified.</div>
                </div>
                """
            )

        p1, p2 = st.columns(2)
        with p1:
            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Ensemble Confidence</div>
                    <div class="metric-value-mono" style="color: {'#fb7185' if verdict == 'TAMPERED' else '#34d399'};">{conf:.1f}%</div>
                </div>
                """,
                target=p1,
            )
            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Execution Latency</div>
                    <div class="metric-value-mono">{latency:.1f} ms</div>
                </div>
                """,
                target=p1,
            )
        with p2:
            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Spatial IoU Overlap</div>
                    <div class="metric-value-mono">{spatial_iou:.3f}</div>
                </div>
                """,
                target=p2,
            )
            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Engines Flagged</div>
                    <div class="metric-value-mono">{len(agreed_engines)} / {total_eng}</div>
                </div>
                """,
                target=p2,
            )

    render_html("<div style='height: 16px;'></div>")

    # -------------------------------------------------------------------------
    # Evaluation Tabs
    # -------------------------------------------------------------------------
    engine_keys_ordered = list(pipeline_result["engine_results"].keys())
    tab_names = ["Evaluation Matrix", "CMFD Knowledge Base"] + [f"{k.upper()}" for k in engine_keys_ordered]
    tabs = st.tabs(tab_names)

    # Tab 1: Evaluation Matrix & Gallery
    with tabs[0]:
        render_html(f'<div class="section-title" style="font-size: 1rem;">{get_svg_icon("matrix")} Cross-Algorithm Evaluation Matrix</div>')

        rows = []
        for k, out in pipeline_result["engine_results"].items():
            m = out["metrics"]
            features_extracted = m.get("total_keypoints", m.get("feature_shape", m.get("mean_error", "N/A")))

            if k in ["sift", "orb", "patch_sift"]:
                inliers = m.get("ransac_inliers", m.get("cloned_pairs", 0))
                raw_matches = m.get("matched_pairs", 0)
                ransac_summary = f"{inliers} inliers / {raw_matches} matches"
            elif k == "dl_selfcorr":
                cells = m.get("cluster_cells", 0)
                ransac_summary = f"{cells} cells (Z > 2.8s)"
            elif k == "ela":
                px = m.get("cluster_pixels", 0)
                ransac_summary = f"{px} px"
            else:
                ransac_summary = "N/A"

            rows.append({
                "Algorithm": k.upper(),
                "Verdict": "TAMPERED" if out["is_forged"] else "AUTHENTIC",
                "Confidence": f"{out['confidence']:.1f}%",
                "Latency": f"{out['execution_time_ms']:.1f} ms",
                "Features / Dimensions": str(features_extracted),
                "Filtered Inliers / Matches": ransac_summary,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        render_html("<div style='height: 16px;'></div>")
        render_html(f'<div class="section-title" style="font-size: 1rem;">{get_svg_icon("layers")} Algorithm Visual Outputs</div>')
        gallery_cols = st.columns(min(len(engine_keys_ordered), 4))
        for idx, k in enumerate(engine_keys_ordered):
            with gallery_cols[idx % len(gallery_cols)]:
                out = pipeline_result["engine_results"][k]
                pill_class = "pill-tampered" if out["is_forged"] else "pill-authentic"
                status_text = "TAMPERED" if out["is_forged"] else "AUTHENTIC"
                render_html(
                    f"<div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;'>"
                    f"<strong>{k.upper()}</strong> <span class='{pill_class}'>{status_text}</span>"
                    f"</div>"
                )
                st.image(out["visual_result"], use_container_width=True)

    # Tab 2: Knowledge Base
    with tabs[1]:
        render_knowledge_base()

    # Algorithm Deep-Dive Tabs (Tab 2..N)
    for idx, k in enumerate(engine_keys_ordered):
        with tabs[idx + 2]:
            render_algorithm_pipeline_breakdown(k, pipeline_result["engine_results"][k], active_target_image)


if __name__ == "__main__":
    main()
