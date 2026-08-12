import html
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


# =========================================================
# 1. Site configuration
# =========================================================

def read_setting(name: str, default: str = "") -> str:
    """Read an environment variable first, then Streamlit Secrets."""
    environment_value = os.getenv(name)

    if environment_value:
        return environment_value

    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


SITE_NAME = read_setting(
    "SITE_NAME",
    "Briefline",
)

SITE_TAGLINE = read_setting(
    "SITE_TAGLINE",
    "Clear summaries. Connected coverage.",
)

ACCENT_COLOR = read_setting(
    "THEME_ACCENT_COLOR",
    "#4F46E5",
)

DATABASE_URL = read_setting(
    "DATABASE_URL",
    "",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ARTIFACT_DIR = Path(
    read_setting(
        "FRONTEND_ARTIFACT_DIR",
        str(PROJECT_ROOT / "artifacts" / "frontend"),
    )
)

PAGE_SIZE = 20
SEARCH_PAGE_SIZE = 18
HOMEPAGE_ROTATOR_LIMIT = 10
ROTATOR_SECONDS_PER_ARTICLE = 4.5


st.set_page_config(
    page_title=SITE_NAME,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render_html(markup: str) -> None:
    """Render HTML without letting Markdown treat indentation as a code block."""
    cleaned_markup = dedent(markup).strip()

    if hasattr(st, "html"):
        st.html(cleaned_markup)
        return

    compact_markup = " ".join(
        line.strip()
        for line in cleaned_markup.splitlines()
    )

    st.markdown(
        compact_markup,
        unsafe_allow_html=True,
    )


# =========================================================
# 2. Page styles
# =========================================================

render_html(
    f"""
    <style>
        :root {{
            --page-background: #F3F6FB;
            --surface: rgba(255, 255, 255, 0.92);
            --surface-solid: #FFFFFF;
            --surface-soft: #F8FAFC;
            --primary-text: #0F172A;
            --secondary-text: #64748B;
            --muted-text: #94A3B8;
            --border-color: #E2E8F0;
            --accent-color: {ACCENT_COLOR};
            --accent-soft: #EEF2FF;
            --accent-secondary: #06B6D4;
            --shadow-sm: 0 8px 24px rgba(15, 23, 42, 0.06);
            --shadow-md: 0 18px 48px rgba(15, 23, 42, 0.11);
        }}

        html {{
            scroll-behavior: smooth;
        }}

        html,
        body,
        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at 5% 0%, rgba(79, 70, 229, 0.10), transparent 28rem),
                radial-gradient(circle at 100% 8%, rgba(6, 182, 212, 0.08), transparent 26rem),
                var(--page-background);
            color: var(--primary-text);
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        [data-testid="stHeader"] {{
            height: 0;
            background: transparent;
        }}

        [data-testid="stMainBlockContainer"] {{
            max-width: 1540px;
            padding-top: 1.25rem;
            padding-bottom: 5rem;
        }}

        #MainMenu,
        footer,
        [data-testid="stToolbar"] {{
            visibility: hidden;
        }}

        .site-header {{
            position: sticky;
            top: 0.75rem;
            z-index: 50;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 1rem 1.15rem;
            margin-bottom: 1.8rem;
            border: 1px solid rgba(226, 232, 240, 0.82);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.80);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }}

        .brand-group {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            min-width: 0;
        }}

        .brand-mark {{
            display: grid;
            place-items: center;
            width: 42px;
            height: 42px;
            flex: 0 0 auto;
            border-radius: 13px;
            background: linear-gradient(135deg, var(--accent-color), var(--accent-secondary));
            color: white;
            font-size: 1rem;
            font-weight: 850;
            letter-spacing: -0.03em;
            box-shadow: 0 10px 24px rgba(79, 70, 229, 0.26);
        }}

        .site-name {{
            display: flex;
            align-items: center;
            margin: 0;
            color: var(--primary-text);
            font-size: clamp(1.35rem, 2.4vw, 1.8rem);
            font-weight: 820;
            line-height: 1;
            letter-spacing: -0.045em;
        }}


        .archive-search-form {{
            position: relative;
            display: flex;
            align-items: center;
            width: min(390px, 38vw);
            flex: 0 1 390px;
        }}

        .archive-search-input {{
            width: 100%;
            height: 46px;
            padding: 0 3.45rem 0 1.15rem;
            border: 1px solid rgba(203, 213, 225, 0.95);
            border-radius: 18px;
            outline: none;
            background: rgba(248, 250, 252, 0.94);
            color: var(--primary-text);
            font: inherit;
            font-size: 0.84rem;
            font-weight: 560;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.92),
                0 4px 12px rgba(15, 23, 42, 0.025);
            transition:
                border-color 170ms ease,
                box-shadow 170ms ease,
                background-color 170ms ease;
        }}

        .archive-search-input::placeholder {{
            color: #94A3B8;
            font-weight: 500;
        }}

        .archive-search-input:focus {{
            border-color: rgba(79, 70, 229, 0.58);
            background: #FFFFFF;
            box-shadow:
                0 0 0 4px rgba(79, 70, 229, 0.09),
                0 10px 26px rgba(15, 23, 42, 0.07);
        }}

        .archive-search-submit {{
            position: absolute;
            right: 0.34rem;
            display: grid;
            place-items: center;
            width: 37px;
            height: 37px;
            padding: 0;
            border: 1px solid rgba(99, 102, 241, 0.16);
            border-radius: 50%;
            background: rgba(79, 70, 229, 0.075);
            color: var(--accent-color);
            cursor: pointer;
            box-shadow: none;
            transition:
                color 160ms ease,
                background-color 160ms ease,
                border-color 160ms ease,
                transform 160ms ease,
                box-shadow 160ms ease;
        }}

        .archive-search-submit:hover {{
            color: #FFFFFF;
            background: var(--accent-color);
            border-color: var(--accent-color);
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(79, 70, 229, 0.20);
        }}

        .archive-search-submit:active {{
            transform: translateY(0) scale(0.95);
        }}

        .search-pagination {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
            align-items: center;
            gap: 1rem;
            margin-top: 1.45rem;
        }}

        .search-pagination-side {{
            display: flex;
            align-items: center;
        }}

        .search-pagination-side:last-child {{
            justify-content: flex-end;
        }}

        .search-page-link,
        .search-page-disabled {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.55rem;
            min-width: 142px;
            min-height: 46px;
            padding: 0.7rem 1.1rem;
            border: 1px solid rgba(203, 213, 225, 0.96);
            border-radius: 15px;
            font-size: 0.84rem;
            font-weight: 680;
            line-height: 1;
            text-decoration: none;
            transition:
                color 170ms ease,
                background-color 170ms ease,
                border-color 170ms ease,
                transform 170ms ease,
                box-shadow 170ms ease;
        }}

        .search-page-link {{
            color: #334155;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
        }}

        .search-page-link:hover {{
            color: var(--accent-color);
            background: #FFFFFF;
            border-color: rgba(99, 102, 241, 0.34);
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(79, 70, 229, 0.10);
        }}

        .search-page-link svg,
        .search-page-disabled svg {{
            width: 16px;
            height: 16px;
            flex: 0 0 auto;
        }}

        .search-page-disabled {{
            color: #A8B2C2;
            background: rgba(248, 250, 252, 0.62);
            cursor: not-allowed;
            box-shadow: none;
        }}

        .search-page-status {{
            color: #64748B;
            font-size: 0.82rem;
            font-weight: 680;
            text-align: center;
            white-space: nowrap;
        }}

        .search-hero {{
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: end;
            gap: 0.75rem 1.25rem;
            scroll-margin-top: 1.25rem;
            overflow: hidden;
            padding: clamp(1.25rem, 2.2vw, 1.65rem) clamp(1.3rem, 2.5vw, 1.8rem);
            margin-bottom: 0.85rem;
            border: 1px solid rgba(226, 232, 240, 0.96);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.055);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}

        .search-hero::before {{
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: linear-gradient(180deg, var(--accent-color), var(--accent-secondary));
        }}

        .search-hero-label {{
            grid-column: 1 / -1;
            margin: 0;
            color: var(--accent-color);
            font-size: 0.67rem;
            font-weight: 850;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }}

        .search-hero-title {{
            min-width: 0;
            max-width: 920px;
            margin: 0;
            color: var(--primary-text);
            font-size: clamp(1.75rem, 3.6vw, 2.7rem);
            font-weight: 810;
            line-height: 1.06;
            letter-spacing: -0.045em;
            text-wrap: balance;
        }}

        .search-hero-copy {{
            align-self: center;
            margin: 0;
            padding: 0.55rem 0.8rem;
            border: 1px solid rgba(203, 213, 225, 0.88);
            border-radius: 999px;
            background: rgba(248, 250, 252, 0.82);
            color: var(--secondary-text);
            font-size: 0.78rem;
            font-weight: 650;
            line-height: 1;
            white-space: nowrap;
        }}

        .search-results-list {{
            overflow: hidden;
            border-top: 1px solid rgba(203, 213, 225, 0.88);
            border-bottom: 1px solid rgba(203, 213, 225, 0.88);
            background: rgba(255, 255, 255, 0.42);
        }}

        .search-result-link {{
            display: block;
            color: inherit;
            text-decoration: none;
        }}

        .search-result-row {{
            display: grid;
            grid-template-columns: minmax(150px, 0.2fr) minmax(0, 1fr) minmax(108px, 0.13fr);
            align-items: start;
            gap: clamp(1rem, 2.4vw, 2rem);
            padding: clamp(1.15rem, 2.1vw, 1.55rem) 0.75rem;
            border-bottom: 1px solid rgba(226, 232, 240, 0.96);
            transition:
                background-color 170ms ease,
                padding-left 170ms ease,
                padding-right 170ms ease;
        }}

        .search-result-link:last-child .search-result-row {{
            border-bottom: none;
        }}

        .search-result-link:hover .search-result-row {{
            padding-left: 1rem;
            padding-right: 0.5rem;
            background: rgba(238, 242, 255, 0.58);
        }}

        .search-result-taxonomy {{
            min-width: 0;
        }}

        .search-result-broad {{
            color: var(--accent-color);
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.11em;
            line-height: 1.35;
            text-transform: uppercase;
        }}

        .search-result-fine {{
            margin-top: 0.45rem;
            color: #64748B;
            font-size: 0.78rem;
            font-weight: 620;
            line-height: 1.45;
        }}

        .search-result-main {{
            min-width: 0;
        }}

        .search-result-title {{
            margin: 0;
            color: var(--primary-text);
            font-size: clamp(1.12rem, 2vw, 1.45rem);
            font-weight: 780;
            line-height: 1.24;
            letter-spacing: -0.028em;
            text-wrap: balance;
        }}

        .search-result-highlight {{
            display: -webkit-box;
            margin-top: 0.62rem;
            overflow: hidden;
            color: #5F6F86;
            font-size: 0.9rem;
            line-height: 1.58;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
        }}

        .search-result-keywords {{
            margin-top: 0.7rem;
            color: #7A889D;
            font-size: 0.75rem;
            line-height: 1.45;
        }}

        .search-result-side {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            min-height: 100%;
            text-align: right;
        }}

        .search-result-date {{
            color: #64748B;
            font-size: 0.75rem;
            font-weight: 650;
            line-height: 1.4;
            white-space: nowrap;
        }}

        .search-result-open {{
            display: grid;
            place-items: center;
            width: 34px;
            height: 34px;
            border: 1px solid rgba(99, 102, 241, 0.18);
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.76);
            color: var(--accent-color);
            transition:
                color 170ms ease,
                background-color 170ms ease,
                border-color 170ms ease,
                transform 170ms ease;
        }}

        .search-result-open svg {{
            width: 16px;
            height: 16px;
        }}

        .search-result-link:hover .search-result-open {{
            color: #FFFFFF;
            background: var(--accent-color);
            border-color: var(--accent-color);
            transform: translate(2px, -2px);
        }}

        .navigation-heading {{
            margin: 0 0 0.9rem 0.2rem;
            color: var(--muted-text);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }}


        .category-hero {{
            position: relative;
            overflow: hidden;
            padding: 1.65rem 1.75rem;
            margin-bottom: 1.45rem;
            border: 1px solid rgba(226, 232, 240, 0.92);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(238,242,255,0.90));
            box-shadow: var(--shadow-sm);
            animation: fadeUp 420ms ease both;
        }}

        .category-hero::after {{
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            right: -70px;
            top: -100px;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(79,70,229,0.17), rgba(6,182,212,0.10));
            filter: blur(2px);
        }}

        .category-hero-label {{
            position: relative;
            z-index: 1;
            margin-bottom: 0.5rem;
            color: var(--accent-color);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}

        .content-heading {{
            position: relative;
            z-index: 1;
            margin: 0;
            color: var(--primary-text);
            font-size: clamp(2rem, 4vw, 3.2rem);
            font-weight: 820;
            line-height: 1.02;
            letter-spacing: -0.055em;
        }}

        .content-description {{
            position: relative;
            z-index: 1;
            margin-top: 0.65rem;
            color: var(--secondary-text);
            font-size: 0.9rem;
        }}

        .section-header {{
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1.2rem;
            margin: 0.05rem 0 0.58rem;
            padding: 0 0.1rem;
            animation: fadeUp 420ms ease both;
        }}

        .section-heading-group {{
            min-width: 0;
        }}

        .section-eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
            color: var(--accent-color);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }}

        .section-eyebrow::before {{
            content: "";
            width: 18px;
            height: 3px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent-color), var(--accent-secondary));
        }}

        .section-title {{
            margin: 0;
            color: var(--primary-text);
            font-size: clamp(1.45rem, 2.3vw, 1.9rem);
            font-weight: 780;
            line-height: 1.1;
            letter-spacing: -0.035em;
        }}

        .section-subtitle {{
            margin-top: 0.3rem;
            color: var(--secondary-text);
            font-size: 0.82rem;
        }}

        .section-count {{
            flex: 0 0 auto;
            padding: 0.48rem 0.72rem;
            border: 1px solid var(--border-color);
            border-radius: 999px;
            background: rgba(255,255,255,0.82);
            color: var(--secondary-text);
            font-size: 0.72rem;
            font-weight: 700;
            box-shadow: 0 4px 14px rgba(15,23,42,0.04);
        }}

        .featured-article {{
            position: relative;
            overflow: hidden;
            min-height: 270px;
            padding: clamp(1.45rem, 3.5vw, 2.25rem);
            margin-bottom: 0.7rem;
            border: 1px solid rgba(226, 232, 240, 0.94);
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96));
            box-shadow: var(--shadow-sm);
            transition:
                transform 190ms ease,
                box-shadow 190ms ease,
                border-color 190ms ease;
            animation: fadeUp 460ms ease both;
        }}

        .featured-article::before {{
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -110px;
            bottom: -150px;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(79,70,229,0.16), rgba(6,182,212,0.10));
        }}

        .featured-article:hover {{
            transform: translateY(-4px);
            border-color: rgba(79, 70, 229, 0.22);
            box-shadow: var(--shadow-md);
        }}

        .lead-label {{
            position: relative;
            z-index: 1;
            display: inline-flex;
            align-items: center;
            gap: 0.48rem;
            margin-bottom: 1.1rem;
            padding: 0.42rem 0.66rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-color);
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }}

        .lead-label-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--accent-color);
        }}

        .article-category {{
            position: relative;
            z-index: 1;
            margin-bottom: 0.7rem;
            color: var(--accent-color);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }}

        .featured-title {{
            position: relative;
            z-index: 1;
            max-width: 1020px;
            margin: 0;
            color: var(--primary-text);
            font-size: clamp(1.65rem, 2.75vw, 2.55rem);
            font-weight: 800;
            line-height: 1.08;
            letter-spacing: -0.046em;
            text-wrap: balance;
        }}

        .featured-highlight {{
            position: relative;
            z-index: 1;
            max-width: 900px;
            margin-top: 1rem;
            color: #475569;
            font-size: 0.98rem;
            line-height: 1.65;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 4;
            overflow: hidden;
        }}

        .article-meta {{
            position: relative;
            z-index: 1;
            margin-top: 1.15rem;
            color: var(--secondary-text);
            font-size: 0.74rem;
            line-height: 1.45;
        }}

        .story-card {{
            position: relative;
            display: flex;
            flex-direction: column;
            min-height: 250px;
            height: 100%;
            padding: 1.3rem;
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-radius: 19px;
            background: rgba(255, 255, 255, 0.90);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            transition:
                transform 180ms ease,
                box-shadow 180ms ease,
                border-color 180ms ease;
            animation: fadeUp 500ms ease both;
        }}

        .story-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(79, 70, 229, 0.22);
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.10);
        }}

        .story-card::after {{
            content: "↗";
            position: absolute;
            right: 1.15rem;
            top: 1.05rem;
            color: var(--muted-text);
            font-size: 0.95rem;
            transition: transform 180ms ease, color 180ms ease;
        }}

        .story-card:hover::after {{
            transform: translate(2px, -2px);
            color: var(--accent-color);
        }}

        .compact-title {{
            margin: 0;
            padding-right: 1.4rem;
            color: var(--primary-text);
            font-size: clamp(1.05rem, 1.8vw, 1.35rem);
            font-weight: 760;
            line-height: 1.2;
            letter-spacing: -0.03em;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 3;
            overflow: hidden;
        }}

        .compact-highlight {{
            margin-top: 0.7rem;
            color: var(--secondary-text);
            font-size: 0.86rem;
            line-height: 1.55;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 3;
            overflow: hidden;
        }}

        .story-card .article-meta {{
            margin-top: auto;
            padding-top: 1rem;
        }}

        .story-rotator {{
            position: relative;
            width: 100%;
            max-width: 1040px;
            margin: 1.2rem auto 0;
            animation: fadeUp 520ms ease both;
        }}

        .story-rotator-label {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 0 0 0.72rem 0.15rem;
            color: #64748B;
            font-size: 0.7rem;
            font-weight: 780;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }}

        .story-rotator-label::before {{
            content: "";
            width: 14px;
            height: 10px;
            flex: 0 0 auto;
            border: 1.5px solid #6366F1;
            border-radius: 4px;
            background: #EEF2FF;
            box-shadow:
                4px 4px 0 -1px #FFFFFF,
                4px 4px 0 0 #818CF8;
        }}

        .story-rotator-window {{
            --rotator-card-height: 300px;
            --rotator-active-y: 70px;
            --rotator-prev-y: -225px;
            --rotator-next-y: 365px;
            --rotator-hidden-top: -360px;
            --rotator-hidden-bottom: 500px;

            position: relative;
            height: 440px;
            overflow: hidden;
            border-radius: 26px;
            isolation: isolate;
        }}

        .story-rotator-window::before,
        .story-rotator-window::after {{
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            z-index: 20;
            height: 28px;
            pointer-events: none;
        }}

        .story-rotator-window::before {{
            top: 0;
            background: linear-gradient(
                180deg,
                rgba(243, 246, 251, 0.76) 0%,
                rgba(243, 246, 251, 0.26) 48%,
                rgba(243, 246, 251, 0) 100%
            );
        }}

        .story-rotator-window::after {{
            bottom: 0;
            background: linear-gradient(
                0deg,
                rgba(243, 246, 251, 0.76) 0%,
                rgba(243, 246, 251, 0.26) 48%,
                rgba(243, 246, 251, 0) 100%
            );
        }}

        .rotator-slide {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            z-index: 0;
            display: block;
            height: var(--rotator-card-height);
            color: inherit;
            text-decoration: none;
            opacity: 0;
            transform: translateY(var(--rotator-hidden-bottom)) scale(0.94);
            pointer-events: none;
            will-change: opacity, transform;
            animation-fill-mode: both;
        }}

        .rotator-slide:visited,
        .rotator-slide:hover,
        .rotator-slide:active {{
            color: inherit;
            text-decoration: none;
        }}

        .story-rotator:hover .rotator-slide {{
            animation-play-state: paused;
        }}

        .rotator-slide-static {{
            opacity: 1;
            transform: translateY(var(--rotator-active-y)) scale(1);
            pointer-events: auto;
            z-index: 5;
        }}

        .rotator-card {{
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1.12fr) minmax(0, 0.88fr);
            grid-template-areas: "secondary primary";
            width: 100%;
            height: var(--rotator-card-height);
            overflow: hidden;
            border: 1px solid rgba(186, 199, 222, 0.92);
            border-radius: 24px;
            background: #ffffff;
            box-shadow:
                0 18px 44px rgba(15, 23, 42, 0.13),
                0 2px 8px rgba(79, 70, 229, 0.06);
            transition:
                transform 180ms ease,
                box-shadow 180ms ease,
                border-color 180ms ease;
        }}

        .rotator-card-reverse {{
            grid-template-columns: minmax(0, 1.12fr) minmax(0, 0.88fr);
            grid-template-areas: "secondary primary";
        }}

        .rotator-primary,
        .rotator-secondary {{
            min-width: 0;
            padding: clamp(1.35rem, 2.3vw, 1.85rem);
        }}

        .rotator-primary {{
            grid-area: primary;
            display: flex;
            flex-direction: column;
            background:
                radial-gradient(circle at 100% 0%, rgba(79, 70, 229, 0.12), transparent 12rem),
                linear-gradient(160deg, #ffffff 0%, #f4f6ff 100%);
        }}

        .rotator-secondary {{
            grid-area: secondary;
            position: relative;
            display: flex;
            flex-direction: column;
            justify-content: center;
            border-right: 1px solid rgba(186, 199, 222, 0.82);
            background:
                radial-gradient(circle at 0% 100%, rgba(6, 182, 212, 0.15), transparent 16rem),
                linear-gradient(145deg, #eef2ff 0%, #e8f5ff 100%);
        }}

        .rotator-card-reverse .rotator-secondary {{
            border-left: 0;
            border-right: 1px solid rgba(186, 199, 222, 0.82);
        }}

        .rotator-summary-label {{
            margin-bottom: 0.68rem;
            color: #4338ca;
            font-size: 0.67rem;
            font-weight: 850;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}

        .rotator-open-mark {{
            position: absolute;
            right: 1.2rem;
            top: 1.05rem;
            color: var(--accent-color);
            font-size: 1rem;
            font-weight: 800;
            opacity: 0.72;
            transition: transform 180ms ease, opacity 180ms ease;
        }}

        .rotator-slide:hover .rotator-card {{
            transform: translateY(-3px);
            border-color: rgba(79, 70, 229, 0.48);
            box-shadow: 0 24px 54px rgba(15, 23, 42, 0.17);
        }}

        .rotator-slide:hover .rotator-open-mark {{
            transform: translate(2px, -2px);
            opacity: 1;
        }}

        .rotator-title {{
            margin: 0;
            padding-right: 1.2rem;
            color: #0f172a;
            font-size: clamp(1.18rem, 2vw, 1.58rem);
            font-weight: 780;
            line-height: 1.18;
            letter-spacing: -0.035em;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 4;
            overflow: hidden;
        }}

        .rotator-highlight {{
            margin: 0;
            color: #334155;
            font-size: clamp(0.88rem, 1.2vw, 0.98rem);
            line-height: 1.68;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 5;
            overflow: hidden;
        }}

        .rotator-primary .article-meta {{
            margin-top: auto;
            padding-top: 1rem;
        }}

        .section-spacer {{
            height: 0;
        }}

        .section-tightener {{
            height: 0;
            margin: -1.35rem 0 -0.55rem;
        }}

        .empty-message {{
            padding: 2rem;
            border: 1px dashed var(--border-color);
            border-radius: 18px;
            background: rgba(255,255,255,0.70);
            color: var(--secondary-text);
            text-align: center;
        }}

        div[data-testid="stButton"] > button {{
            width: 100%;
            min-height: 3rem;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            padding: 0.7rem 0.72rem;
            border: 1px solid transparent;
            border-radius: 12px;
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 0.8rem;
            font-weight: 650;
            text-align: left;
            line-height: 1.3;
            box-shadow: none;
            transition:
                transform 150ms ease,
                box-shadow 150ms ease,
                background-color 150ms ease,
                border-color 150ms ease;
        }}

        div[data-testid="stButton"] > button p {{
            width: 100%;
            margin: 0;
            text-align: left;
            line-height: 1.3;
        }}

        div[data-testid="stButton"] > button[kind="secondary"] {{
            color: #334155;
            background: rgba(255,255,255,0.50);
            border-color: rgba(226,232,240,0.65);
        }}

        div[data-testid="stButton"] > button[kind="secondary"]:hover {{
            color: var(--primary-text);
            background: var(--surface-solid);
            border-color: #CBD5E1;
            box-shadow: 0 8px 20px rgba(15,23,42,0.07);
            transform: translateY(-1px);
        }}

        div[data-testid="stButton"] > button[kind="primary"] {{
            color: white;
            background: linear-gradient(135deg, var(--accent-color), #6366F1);
            border-color: transparent;
            box-shadow: 0 10px 22px rgba(79,70,229,0.24);
        }}

        div[data-testid="stButton"] > button[kind="primary"]:hover {{
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(79,70,229,0.30);
        }}

        div[data-testid="stButton"] > button:focus {{
            box-shadow: 0 0 0 3px rgba(79,70,229,0.15);
        }}

        div[data-testid="stButton"] > button:disabled {{
            opacity: 0.45;
            transform: none;
            box-shadow: none;
        }}

        .story-link {{
            display: block;
            height: 100%;
            color: inherit;
            text-decoration: none;
        }}

        .story-link:visited,
        .story-link:hover,
        .story-link:active {{
            color: inherit;
            text-decoration: none;
        }}

        .detail-back-row {{
            margin-bottom: 1rem;
        }}

        .detail-hero {{
            position: relative;
            z-index: 1;
            width: min(100%, 1320px);
            overflow: hidden;
            padding: clamp(1.55rem, 3.6vw, 3rem);
            padding-bottom: clamp(5rem, 7.5vw, 6.8rem);
            margin: 0 auto;
            border: 1px solid rgba(226, 232, 240, 0.92);
            border-radius: 28px;
            background:
                radial-gradient(circle at 93% 4%, rgba(6, 182, 212, 0.17), transparent 19rem),
                radial-gradient(circle at 8% 100%, rgba(79, 70, 229, 0.15), transparent 24rem),
                linear-gradient(135deg, rgba(255,255,255,0.98), rgba(238,242,255,0.93));
            box-shadow: var(--shadow-md);
            animation: fadeUp 420ms ease both;
        }}

        .detail-hero::after {{
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -130px;
            bottom: -160px;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(79,70,229,0.20), rgba(6,182,212,0.12));
        }}

        .detail-category-row {{
            position: relative;
            z-index: 1;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 1rem;
        }}

        .detail-broad-category,
        .detail-final-category {{
            display: inline-flex;
            align-items: center;
            min-height: 30px;
            padding: 0.4rem 0.68rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .detail-broad-category {{
            background: var(--accent-soft);
            color: var(--accent-color);
        }}

        .detail-final-category {{
            border: 1px solid rgba(203, 213, 225, 0.88);
            background: rgba(255, 255, 255, 0.72);
            color: var(--secondary-text);
        }}

        .detail-title {{
            position: relative;
            z-index: 1;
            max-width: 1120px;
            margin: 0;
            color: var(--primary-text);
            font-size: clamp(2rem, 3.75vw, 3.35rem);
            font-weight: 810;
            line-height: 1.075;
            letter-spacing: -0.047em;
            text-wrap: balance;
        }}

        .detail-date {{
            position: relative;
            z-index: 1;
            margin-top: 1rem;
            color: var(--secondary-text);
            font-size: 0.8rem;
            font-weight: 650;
        }}

        .detail-reading-panel {{
            position: relative;
            z-index: 3;
            display: grid;
            grid-template-columns: minmax(0, 790px) minmax(230px, 275px);
            justify-content: center;
            gap: clamp(1.5rem, 3vw, 3.25rem);
            width: calc(100% - 2rem);
            max-width: 1240px;
            margin: clamp(-4.5rem, -5.5vw, -3.5rem) auto 3.6rem;
            border: 1px solid rgba(226, 232, 240, 0.92);
            border-radius: 26px;
            background: rgba(255, 255, 255, 0.96);
            box-shadow: 0 24px 58px rgba(15, 23, 42, 0.12);
            animation: fadeUp 500ms ease both;
        }}

        .reading-main {{
            min-width: 0;
            width: 100%;
            padding: clamp(1.55rem, 3.2vw, 2.7rem) 0 clamp(1.8rem, 3.5vw, 3rem);
        }}

        .integrated-brief {{
            width: 100%;
            max-width: 760px;
            padding-bottom: 1.6rem;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 1.7rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .brief-label {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.72rem;
            color: var(--accent-color);
            font-size: 0.67rem;
            font-weight: 850;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }}

        .brief-label::before {{
            content: "";
            width: 18px;
            height: 3px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent-color), var(--accent-secondary));
        }}

        .brief-copy {{
            color: #334155;
            font-size: clamp(1.05rem, 2vw, 1.22rem);
            font-weight: 510;
            line-height: 1.68;
            letter-spacing: -0.012em;
        }}

        .article-body {{
            width: 100%;
            max-width: 760px;
            margin: 0 auto;
            color: #243044;
            font-size: clamp(1.04rem, 1.45vw, 1.16rem);
            line-height: 1.82;
            letter-spacing: -0.008em;
        }}

        .article-body p {{
            margin: 0 0 1.55em;
            color: #243044;
            font-weight: 400;
            text-align: justify;
            text-justify: inter-word;
            text-wrap: pretty;
            hyphens: auto;
            overflow-wrap: break-word;
        }}

        .article-body p:first-child {{
            color: #243044;
            font-weight: 400;
        }}

        .drop-cap {{
            float: left;
            display: inline-block;
            margin: 0.11em 0.15em 0 0;
            color: var(--accent-color);
            background: linear-gradient(145deg, var(--accent-color), var(--accent-secondary));
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.55rem;
            font-weight: 840;
            line-height: 0.79;
            letter-spacing: -0.075em;
            text-transform: none;
        }}

        .story-aside-wrap {{
            min-width: 0;
            padding: clamp(1.55rem, 3.2vw, 2.7rem) 0 1.5rem;
        }}

        .story-aside {{
            position: sticky;
            top: 7.2rem;
            padding: 1.15rem;
            border: 1px solid rgba(226, 232, 240, 0.92);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(238,242,255,0.66));
        }}

        .aside-label {{
            margin-bottom: 0.8rem;
            color: var(--muted-text);
            font-size: 0.66rem;
            font-weight: 850;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }}

        .aside-value {{
            margin-bottom: 0.8rem;
            color: var(--primary-text);
            font-size: 0.84rem;
            font-weight: 680;
            line-height: 1.45;
        }}

        .aside-divider {{
            height: 1px;
            margin: 1rem 0;
            background: var(--border-color);
        }}

        .mobile-topic-row {{
            display: none;
            flex-wrap: wrap;
            gap: 0.48rem;
            margin-top: 1rem;
        }}

        .keyword-pill {{
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.64rem;
            border: 1px solid rgba(203, 213, 225, 0.86);
            border-radius: 999px;
            background: rgba(248, 250, 252, 0.90);
            color: #475569;
            font-size: 0.7rem;
            font-weight: 650;
        }}

        .related-section {{
            margin-top: 0.2rem;
            animation: fadeUp 560ms ease both;
        }}

        .related-heading {{
            margin: 0;
            color: var(--primary-text);
            font-size: clamp(1.8rem, 3vw, 2.5rem);
            font-weight: 820;
            line-height: 1.05;
            letter-spacing: -0.045em;
        }}

        .related-subtitle {{
            margin-top: 0.45rem;
            margin-bottom: 1.15rem;
            color: var(--secondary-text);
            font-size: 0.88rem;
        }}

        .related-carousel {{
            display: grid;
            grid-auto-flow: column;
            grid-auto-columns: clamp(340px, 43vw, 520px);
            gap: 1rem;
            overflow-x: auto;
            padding: 0.2rem 0.1rem 1rem;
            scroll-snap-type: x mandatory;
            scroll-padding-inline: 0.1rem;
            overscroll-behavior-x: contain;
            scrollbar-width: thin;
            scrollbar-color: rgba(79, 70, 229, 0.30) transparent;
            cursor: grab;
            -webkit-overflow-scrolling: touch;
        }}

        .related-carousel:active {{
            cursor: grabbing;
        }}

        .related-carousel::-webkit-scrollbar {{
            height: 8px;
        }}

        .related-carousel::-webkit-scrollbar-track {{
            background: transparent;
        }}

        .related-carousel::-webkit-scrollbar-thumb {{
            border: 2px solid transparent;
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.26);
            background-clip: padding-box;
        }}

        .related-card-link {{
            display: block;
            height: 100%;
            color: inherit;
            text-decoration: none;
            scroll-snap-align: start;
            scroll-snap-stop: normal;
        }}

        .related-card-link:visited,
        .related-card-link:hover,
        .related-card-link:active {{
            color: inherit;
            text-decoration: none;
        }}

        .related-card {{
            position: relative;
            display: flex;
            flex-direction: column;
            min-height: 290px;
            height: 100%;
            padding: 1.35rem;
            overflow: hidden;
            border: 1px solid rgba(226, 232, 240, 0.96);
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: var(--shadow-sm);
            transition:
                transform 180ms ease,
                box-shadow 180ms ease,
                border-color 180ms ease;
        }}

        .related-card::after {{
            content: "↗";
            position: absolute;
            top: 1.15rem;
            right: 1.15rem;
            color: var(--accent-color);
            font-size: 1rem;
            font-weight: 800;
            opacity: 0.72;
        }}

        .related-card-link:hover .related-card {{
            transform: translateY(-3px);
            border-color: rgba(79, 70, 229, 0.30);
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.11);
        }}

        .related-card-category {{
            max-width: calc(100% - 2rem);
            margin-bottom: 0.85rem;
            color: var(--accent-color);
            font-size: 0.7rem;
            font-weight: 820;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .related-card-title {{
            display: -webkit-box;
            margin: 0;
            overflow: hidden;
            color: var(--primary-text);
            font-size: clamp(1.12rem, 2vw, 1.35rem);
            font-weight: 780;
            line-height: 1.24;
            letter-spacing: -0.026em;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 3;
        }}

        .related-card-highlight {{
            display: -webkit-box;
            margin-top: 0.85rem;
            overflow: hidden;
            color: var(--secondary-text);
            font-size: 0.9rem;
            line-height: 1.58;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 3;
        }}

        .related-card-meta {{
            margin-top: auto;
            padding-top: 1.1rem;
            color: #64748B;
            font-size: 0.75rem;
            line-height: 1.45;
        }}

        @keyframes fadeUp {{
            from {{
                opacity: 0;
                transform: translateY(12px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @media (max-width: 900px) {{
            [data-testid="stMainBlockContainer"] {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            .site-header {{
                position: static;
            }}

            .site-header {{
                align-items: stretch;
                flex-wrap: wrap;
            }}

            .archive-search-form {{
                width: 100%;
                flex-basis: 100%;
            }}

            .featured-article {{
                min-height: auto;
            }}
        }}

        @media (max-width: 820px) {{
            .search-hero {{
                grid-template-columns: 1fr;
                align-items: start;
            }}

            .search-hero-label {{
                grid-column: 1;
            }}

            .search-hero-copy {{
                justify-self: start;
            }}

            .search-result-row {{
                grid-template-columns: 1fr;
                gap: 0.7rem;
                padding: 1.05rem 0.35rem;
            }}

            .search-result-link:hover .search-result-row {{
                padding-left: 0.55rem;
                padding-right: 0.15rem;
            }}

            .search-result-fine {{
                margin-top: 0.25rem;
            }}

            .search-result-side {{
                flex-direction: row;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
                min-height: 0;
                text-align: left;
            }}

            .search-result-open {{
                width: 32px;
                height: 32px;
            }}

            .article-body p {{
                text-align: left;
                hyphens: none;
            }}

            .search-pagination {{
                grid-template-columns: 1fr 1fr;
                gap: 0.75rem;
            }}

            .search-page-status {{
                grid-column: 1 / -1;
                grid-row: 1;
                margin-bottom: 0.1rem;
            }}

            .search-pagination-side {{
                grid-row: 2;
            }}

            .search-page-link,
            .search-page-disabled {{
                width: 100%;
                min-width: 0;
            }}

            .detail-hero {{
                padding: 1.45rem;
                padding-bottom: 5rem;
                border-radius: 22px;
            }}

            .detail-title {{
                font-size: clamp(1.8rem, 8.2vw, 2.55rem);
                line-height: 1.08;
            }}

            .drop-cap {{
                margin-right: 0.13em;
                font-size: 2.9rem;
            }}

            .detail-reading-panel {{
                display: block;
                width: calc(100% - 0.5rem);
                margin: -3.3rem auto 3rem;
                border-radius: 22px;
            }}

            .reading-main {{
                padding: 1.35rem;
            }}

            .story-rotator {{
                max-width: 100%;
            }}

            .story-rotator-window {{
                --rotator-card-height: 300px;
                --rotator-active-y: 60px;
                --rotator-prev-y: -250px;
                --rotator-next-y: 360px;
                --rotator-hidden-top: -380px;
                --rotator-hidden-bottom: 480px;
                height: 420px;
                border-radius: 22px;
            }}

            .rotator-card,
            .rotator-card-reverse {{
                grid-template-columns: 1fr;
                grid-template-rows: minmax(0, 1fr) minmax(0, 0.92fr);
                grid-template-areas:
                    "primary"
                    "secondary";
                border-radius: 20px;
            }}

            .rotator-primary,
            .rotator-secondary {{
                padding: 1.05rem 1.15rem;
            }}

            .rotator-secondary,
            .rotator-card-reverse .rotator-secondary {{
                border-left: 0;
                border-right: 0;
                border-top: 1px solid rgba(226, 232, 240, 0.96);
            }}

            .rotator-title {{
                font-size: 1.12rem;
                -webkit-line-clamp: 3;
            }}

            .rotator-highlight {{
                font-size: 0.86rem;
                line-height: 1.55;
                -webkit-line-clamp: 3;
            }}

            .story-aside-wrap {{
                display: none;
            }}

            .mobile-topic-row {{
                display: flex;
            }}

            .related-carousel {{
                grid-auto-columns: min(86vw, 430px);
                gap: 0.85rem;
            }}

            .related-card {{
                min-height: 275px;
            }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                scroll-behavior: auto !important;
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }}

            .rotator-slide {{
                display: none;
                animation: none !important;
            }}

            .rotator-slide:first-child {{
                position: relative;
                display: block;
                opacity: 1;
                transform: none;
                pointer-events: auto;
            }}
        }}
    </style>
    """
)


# =========================================================
# 3. PostgreSQL connection
# =========================================================

@st.cache_resource
def get_database_pool() -> ConnectionPool:
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is missing. Configure it in the environment or "
            "Streamlit Secrets before starting the app."
        )

    return ConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=4,
        timeout=15,
        open=True,
        kwargs={
            "row_factory": dict_row,
        },
    )


# =========================================================
# 4. Data queries
# =========================================================

@st.cache_data(
    ttl=180,
    show_spinner=False,
)
def fetch_categories() -> list[dict[str, Any]]:
    sql = """
        SELECT
            broad_category,
            MIN(broad_category_rank) AS category_rank
        FROM category_broad_mapping
        WHERE broad_category IS NOT NULL
        GROUP BY broad_category
        ORDER BY
            MIN(broad_category_rank),
            broad_category;
    """

    with get_database_pool().connection() as connection:
        return connection.execute(sql).fetchall()


@st.cache_data(
    ttl=180,
    show_spinner=False,
)
def fetch_all_homepage_articles() -> list[dict[str, Any]]:
    sql = """
        WITH category_map AS (
            SELECT DISTINCT ON (source_category)
                source_category,
                broad_category,
                broad_category_rank

            FROM category_broad_mapping

            WHERE
                source_category IS NOT NULL
                AND broad_category IS NOT NULL

            ORDER BY
                source_category,
                broad_category_rank NULLS LAST,
                broad_category
        ),

        ranked_articles AS (
            SELECT
                ra.source_id,
                ra.title,
                ra.published_at,

                jr.final_highlight,
                jr.final_category,
                jr.final_keywords,

                category_map.broad_category,
                category_map.broad_category_rank,

                ROW_NUMBER() OVER (
                    PARTITION BY category_map.broad_category
                    ORDER BY
                        ra.published_at DESC NULLS LAST,
                        ra.source_id
                ) AS article_rank

            FROM raw_articles AS ra

            INNER JOIN judge_results AS jr
                ON jr.source_id = ra.source_id

            INNER JOIN category_map
                ON category_map.source_category = jr.final_category

            WHERE
                jr.final_quality_status IN ('OK', 'REVISED')
                AND COALESCE(jr.any_parse_failed, FALSE) = FALSE
                AND ra.title IS NOT NULL
        )

        SELECT
            source_id,
            title,
            published_at,
            final_highlight,
            final_category,
            final_keywords,
            broad_category,
            broad_category_rank,
            article_rank

        FROM ranked_articles

        WHERE article_rank <= %s

        ORDER BY
            broad_category_rank,
            broad_category,
            article_rank;
    """

    with get_database_pool().connection() as connection:
        return connection.execute(
            sql,
            (HOMEPAGE_ROTATOR_LIMIT + 1,),
        ).fetchall()


@st.cache_data(
    ttl=180,
    show_spinner=False,
)
def fetch_category_articles(
    broad_category: str,
    page_number: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:

    offset = (page_number - 1) * page_size

    count_sql = """
        SELECT COUNT(*) AS article_count

        FROM raw_articles AS ra

        INNER JOIN judge_results AS jr
            ON jr.source_id = ra.source_id

        INNER JOIN category_broad_mapping AS bcm
            ON bcm.source_category = jr.final_category

        WHERE
            bcm.broad_category = %s
            AND jr.final_quality_status IN ('OK', 'REVISED')
            AND COALESCE(jr.any_parse_failed, FALSE) = FALSE
            AND ra.title IS NOT NULL;
    """

    article_sql = """
        SELECT
            ra.source_id,
            ra.title,
            ra.published_at,

            jr.final_highlight,
            jr.final_category,
            jr.final_keywords,

            bcm.broad_category

        FROM raw_articles AS ra

        INNER JOIN judge_results AS jr
            ON jr.source_id = ra.source_id

        INNER JOIN category_broad_mapping AS bcm
            ON bcm.source_category = jr.final_category

        WHERE
            bcm.broad_category = %s
            AND jr.final_quality_status IN ('OK', 'REVISED')
            AND COALESCE(jr.any_parse_failed, FALSE) = FALSE
            AND ra.title IS NOT NULL

        ORDER BY
            ra.published_at DESC NULLS LAST,
            ra.source_id

        LIMIT %s
        OFFSET %s;
    """

    with get_database_pool().connection() as connection:
        total_row = connection.execute(
            count_sql,
            (broad_category,),
        ).fetchone()

        articles = connection.execute(
            article_sql,
            (
                broad_category,
                page_size,
                offset,
            ),
        ).fetchall()

    total_articles = int(total_row["article_count"])

    return articles, total_articles


@st.cache_data(
    ttl=180,
    show_spinner=False,
)
def fetch_article_detail(
    source_id: str,
) -> dict[str, Any] | None:
    sql = """
        WITH category_map AS (
            SELECT DISTINCT ON (source_category)
                source_category,
                broad_category

            FROM category_broad_mapping

            WHERE source_category IS NOT NULL

            ORDER BY
                source_category,
                broad_category_rank NULLS LAST,
                broad_category
        )

        SELECT
            ra.source_id,
            ra.title,
            ra.body_text,
            ra.summary,
            ra.published_at,

            jr.final_highlight,
            jr.final_category,
            jr.final_keywords,

            category_map.broad_category

        FROM raw_articles AS ra

        INNER JOIN judge_results AS jr
            ON jr.source_id = ra.source_id

        LEFT JOIN category_map
            ON category_map.source_category = jr.final_category

        WHERE
            ra.source_id = %s
            AND jr.final_quality_status IN ('OK', 'REVISED')
            AND COALESCE(jr.any_parse_failed, FALSE) = FALSE
            AND ra.title IS NOT NULL

        LIMIT 1;
    """

    with get_database_pool().connection() as connection:
        return connection.execute(
            sql,
            (source_id,),
        ).fetchone()


@st.cache_data(
    ttl=180,
    show_spinner=False,
)
def fetch_recommended_articles(
    source_id: str,
) -> list[dict[str, Any]]:
    sql = """
        WITH category_map AS (
            SELECT DISTINCT ON (source_category)
                source_category,
                broad_category

            FROM category_broad_mapping

            WHERE source_category IS NOT NULL

            ORDER BY
                source_category,
                broad_category_rank NULLS LAST,
                broad_category
        ),

        ordered_recommendations AS (
            SELECT
                recommended.recommended_source_id,
                recommended.recommendation_order

            FROM article_recommendations AS recommendations

            CROSS JOIN LATERAL UNNEST(
                COALESCE(
                    recommendations.recommended_source_ids,
                    ARRAY[]::TEXT[]
                )
            ) WITH ORDINALITY AS recommended(
                recommended_source_id,
                recommendation_order
            )

            WHERE recommendations.source_id = %s
        )

        SELECT
            ra.source_id,
            ra.title,
            ra.published_at,

            jr.final_highlight,
            jr.final_category,
            jr.final_keywords,

            category_map.broad_category,
            ordered.recommendation_order

        FROM ordered_recommendations AS ordered

        INNER JOIN raw_articles AS ra
            ON ra.source_id = ordered.recommended_source_id

        INNER JOIN judge_results AS jr
            ON jr.source_id = ra.source_id

        LEFT JOIN category_map
            ON category_map.source_category = jr.final_category

        WHERE
            ordered.recommended_source_id <> %s
            AND jr.final_quality_status IN ('OK', 'REVISED')
            AND COALESCE(jr.any_parse_failed, FALSE) = FALSE
            AND ra.title IS NOT NULL

        ORDER BY ordered.recommendation_order;
    """

    with get_database_pool().connection() as connection:
        return connection.execute(
            sql,
            (
                source_id,
                source_id,
            ),
        ).fetchall()


@st.cache_data(
    ttl=90,
    show_spinner=False,
)
def search_articles(
    query: str,
    page_number: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    normalized_query = " ".join(str(query or "").split()).strip()

    if not normalized_query:
        return [], 0

    offset = (page_number - 1) * page_size
    pattern = f"%{normalized_query}%"

    base_cte = """
        WITH category_map AS (
            SELECT DISTINCT ON (source_category)
                source_category,
                broad_category,
                broad_category_rank

            FROM category_broad_mapping

            WHERE
                source_category IS NOT NULL
                AND broad_category IS NOT NULL

            ORDER BY
                source_category,
                broad_category_rank NULLS LAST,
                broad_category
        ),

        searchable_articles AS (
            SELECT
                ra.source_id,
                ra.title,
                ra.published_at,
                ra.summary,

                jr.final_highlight,
                jr.final_category,
                jr.final_keywords,

                category_map.broad_category,
                category_map.broad_category_rank,

                CONCAT_WS(
                    ' ',
                    COALESCE(ra.title, ''),
                    COALESCE(ra.summary, ''),
                    COALESCE(jr.final_highlight, ''),
                    COALESCE(jr.final_category, ''),
                    COALESCE(jr.final_keywords::TEXT, ''),
                    COALESCE(category_map.broad_category, '')
                ) AS search_document

            FROM raw_articles AS ra

            INNER JOIN judge_results AS jr
                ON jr.source_id = ra.source_id

            LEFT JOIN category_map
                ON category_map.source_category = jr.final_category

            WHERE
                jr.final_quality_status IN ('OK', 'REVISED')
                AND COALESCE(jr.any_parse_failed, FALSE) = FALSE
                AND ra.title IS NOT NULL
        )
    """

    count_sql = base_cte + """
        SELECT COUNT(*) AS article_count
        FROM searchable_articles
        WHERE search_document ILIKE %s;
    """

    article_sql = base_cte + """
        SELECT
            source_id,
            title,
            published_at,
            final_highlight,
            final_category,
            final_keywords,
            broad_category

        FROM searchable_articles

        WHERE search_document ILIKE %s

        ORDER BY
            CASE
                WHEN title ILIKE %s THEN 0
                WHEN final_category ILIKE %s THEN 1
                ELSE 2
            END,
            published_at DESC NULLS LAST,
            source_id

        LIMIT %s
        OFFSET %s;
    """

    with get_database_pool().connection() as connection:
        total_row = connection.execute(
            count_sql,
            (pattern,),
        ).fetchone()

        articles = connection.execute(
            article_sql,
            (
                pattern,
                pattern,
                pattern,
                page_size,
                offset,
            ),
        ).fetchall()

    return articles, int(total_row["article_count"])


# =========================================================
# 5. Display formatting
# =========================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).split())


def safe_text(value: Any) -> str:
    return html.escape(
        clean_text(value),
        quote=True,
    )


def clean_highlight(value: Any) -> str:
    """Remove model-style labels such as 'highlight:' from the start."""
    highlight = clean_text(value)

    if not highlight:
        return ""

    return re.sub(
        r"^\s*(?:(?:final\s+)?highlight\s*[:：\-–—]\s*)+",
        "",
        highlight,
        flags=re.IGNORECASE,
    ).strip()


def format_date(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, datetime):
        return value.strftime("%b %d, %Y").replace(" 0", " ")

    value_string = str(value)

    try:
        parsed_value = datetime.fromisoformat(
            value_string.replace("Z", "+00:00")
        )

        return parsed_value.strftime(
            "%b %d, %Y"
        ).replace(" 0", " ")

    except ValueError:
        return value_string[:10]


def normalize_keywords(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [
            clean_text(item)
            for item in value
            if clean_text(item)
        ]

    value_string = clean_text(value)

    if not value_string:
        return []

    if value_string.startswith("["):
        try:
            parsed_value = json.loads(value_string)

            if isinstance(parsed_value, list):
                return [
                    clean_text(item)
                    for item in parsed_value
                    if clean_text(item)
                ]

        except json.JSONDecodeError:
            pass

    if (
        value_string.startswith("{")
        and value_string.endswith("}")
    ):
        value_string = value_string[1:-1]

    return [
        part.strip().strip('"')
        for part in value_string.split(",")
        if part.strip().strip('"')
    ]


def build_meta_text(article: dict[str, Any]) -> str:
    metadata_parts: list[str] = []

    keywords = normalize_keywords(
        article.get("final_keywords")
    )

    metadata_parts.extend(keywords[:3])

    published_date = format_date(
        article.get("published_at")
    )

    if published_date:
        metadata_parts.append(published_date)

    return " · ".join(metadata_parts)


def get_query_value(name: str) -> str:
    value = st.query_params.get(name, "")

    if isinstance(value, list):
        return str(value[0]) if value else ""

    return str(value or "")


def article_href(article: dict[str, Any]) -> str:
    source_id = clean_text(article.get("source_id"))
    return f"?article={quote(source_id, safe='')}"


def category_page_href(
    category_name: str,
    page_number: int,
) -> str:
    encoded_category = quote(clean_text(category_name), safe="")
    safe_page = max(1, int(page_number))
    return (
        f"?section={encoded_category}&page={safe_page}"
        "#category-results-top"
    )


def article_absolute_url(article: dict[str, Any]) -> str:
    """Return an absolute app URL for links rendered inside an iframe."""
    relative_url = article_href(article)

    try:
        app_url = clean_text(st.context.url)
    except Exception:
        app_url = ""

    if not app_url:
        return relative_url

    return f"{app_url}{relative_url}"


def build_keyword_pills(value: Any) -> str:
    keywords = normalize_keywords(value)

    return "".join(
        f'<span class="keyword-pill">{safe_text(keyword)}</span>'
        for keyword in keywords
    )


def split_article_paragraphs(value: Any) -> list[str]:
    body = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    if not body:
        return []

    explicit_paragraphs = [
        re.sub(r"[ \t]+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n+", body)
        if paragraph.strip()
    ]

    if len(explicit_paragraphs) > 1:
        return explicit_paragraphs

    line_paragraphs = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in body.split("\n")
        if line.strip()
    ]

    if len(line_paragraphs) > 1:
        return line_paragraphs

    normalized_body = re.sub(r"\s+", " ", body).strip()

    if len(normalized_body) <= 680:
        return [normalized_body]

    sentence_matches = re.findall(
        r".+?(?:[.!?](?:[\"'”’])?(?=\s|$)|$)",
        normalized_body,
    )

    sentences = [
        sentence.strip()
        for sentence in sentence_matches
        if sentence.strip()
    ]

    if len(sentences) <= 1:
        return [normalized_body]

    paragraphs: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        projected_length = current_length + len(sentence) + 1

        if current_sentences and projected_length > 620:
            paragraphs.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_length = len(sentence)
            continue

        current_sentences.append(sentence)
        current_length = projected_length

        if current_length >= 380 and len(current_sentences) >= 3:
            paragraphs.append(" ".join(current_sentences))
            current_sentences = []
            current_length = 0

    if current_sentences:
        trailing_paragraph = " ".join(current_sentences)

        if paragraphs and len(trailing_paragraph) < 180:
            paragraphs[-1] = f"{paragraphs[-1]} {trailing_paragraph}"
        else:
            paragraphs.append(trailing_paragraph)

    return paragraphs or [normalized_body]


def format_article_body(value: Any) -> str:
    paragraphs = split_article_paragraphs(value)
    rendered_paragraphs: list[str] = []

    for paragraph_index, paragraph in enumerate(paragraphs):
        if paragraph_index != 0:
            rendered_paragraphs.append(
                f"<p>{html.escape(paragraph, quote=True)}</p>"
            )
            continue

        first_letter_index = next(
            (
                index
                for index, character in enumerate(paragraph)
                if character.isalpha()
            ),
            -1,
        )

        if first_letter_index < 0:
            rendered_paragraphs.append(
                f"<p>{html.escape(paragraph, quote=True)}</p>"
            )
            continue

        drop_cap_text = html.escape(
            paragraph[:first_letter_index + 1].lstrip(),
            quote=True,
        )
        remaining_text = html.escape(
            paragraph[first_letter_index + 1:],
            quote=True,
        )

        rendered_paragraphs.append(
            '<p>'
            f'<span class="drop-cap">{drop_cap_text}</span>'
            f'{remaining_text}'
            '</p>'
        )

    return "".join(rendered_paragraphs)


# =========================================================
# 6. Article display components
# =========================================================

def render_featured_article(
    article: dict[str, Any],
) -> None:
    category = safe_text(
        article.get("final_category")
        or article.get("broad_category")
    )

    title = safe_text(
        article.get("title")
        or "Untitled article"
    )

    highlight = safe_text(
        clean_highlight(article.get("final_highlight"))
    )

    metadata = safe_text(
        build_meta_text(article)
    )

    highlight_html = (
        f'<div class="featured-highlight">{highlight}</div>'
        if highlight
        else ""
    )

    metadata_html = (
        f'<div class="article-meta">{metadata}</div>'
        if metadata
        else ""
    )

    render_html(
        f"""
        <a class="story-link" href="{article_href(article)}" target="_self">
            <article class="featured-article">
                <div class="lead-label">
                    <span class="lead-label-dot"></span>
                    Latest news
                </div>
                <div class="article-category">{category}</div>
                <h3 class="featured-title">{title}</h3>
                {highlight_html}
                {metadata_html}
            </article>
        </a>
        """
    )


def render_compact_article(
    article: dict[str, Any],
) -> None:
    category = safe_text(
        article.get("final_category")
        or article.get("broad_category")
    )

    title = safe_text(
        article.get("title")
        or "Untitled article"
    )

    highlight = safe_text(
        clean_highlight(article.get("final_highlight"))
    )

    metadata = safe_text(
        build_meta_text(article)
    )

    category_html = (
        f'<div class="article-category">{category}</div>'
        if category
        else ""
    )

    highlight_html = (
        f'<div class="compact-highlight">{highlight}</div>'
        if highlight
        else ""
    )

    metadata_html = (
        f'<div class="article-meta">{metadata}</div>'
        if metadata
        else ""
    )

    render_html(
        f"""
        <a class="story-link" href="{article_href(article)}" target="_self">
            <article class="story-card">
                {category_html}
                <h3 class="compact-title">{title}</h3>
                {highlight_html}
                {metadata_html}
            </article>
        </a>
        """
    )


def render_vertical_story_rotator(
    articles: list[dict[str, Any]],
    category_name: str,
) -> None:
    """Render an auto-playing vertical carousel with polished manual controls."""
    if not articles:
        return

    category_slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        category_name.lower(),
    ).strip("-") or "section"

    component_id = f"story-rotator-{category_slug}"
    slides: list[str] = []

    for article in articles:
        category = safe_text(
            article.get("final_category")
            or article.get("broad_category")
        )
        title = safe_text(
            article.get("title")
            or "Untitled article"
        )
        highlight = safe_text(
            clean_highlight(article.get("final_highlight"))
            or "Open the story for the full report and context."
        )
        metadata = safe_text(build_meta_text(article))
        destination_url = article_absolute_url(article)

        category_html = (
            f'<div class="card-category">{category}</div>'
            if category
            else ""
        )
        metadata_html = (
            f'<div class="card-meta">{metadata}</div>'
            if metadata
            else ""
        )

        slides.append(
            f"""
            <a
                class="carousel-slide"
                href="{safe_text(destination_url)}"
                target="_top"
                rel="noopener"
                data-source-id="{safe_text(clean_text(article.get('source_id')))}"
                aria-label="Open {title}"
            >
                <article class="carousel-card">
                    <div class="headline-panel">
                        {category_html}
                        <h3 class="card-title">{title}</h3>
                        {metadata_html}
                        <span class="open-mark" aria-hidden="true">
                            <svg viewBox="0 0 24 24" focusable="false">
                                <path d="M8 16 16 8M10 8h6v6" />
                            </svg>
                        </span>
                    </div>
                    <div class="quick-read-panel">
                        <div class="panel-label">Quick read</div>
                        <div class="card-highlight">{highlight}</div>
                    </div>
                </article>
            </a>
            """
        )

    carousel_markup = dedent(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                :root {{
                    color-scheme: light;
                    --accent: {ACCENT_COLOR};
                    --accent-2: #06B6D4;
                    --ink: #0F172A;
                    --muted: #64748B;
                    --card-height: 248px;
                    --viewport-height: 302px;
                    --active-y: 27px;
                    --previous-y: -232px;
                    --next-y: 274px;
                }}

                * {{
                    box-sizing: border-box;
                }}

                html,
                body {{
                    width: 100%;
                    min-height: 100%;
                    margin: 0;
                    overflow: hidden;
                    background: transparent;
                    color: var(--ink);
                    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                        BlinkMacSystemFont, "Segoe UI", sans-serif;
                }}

                .carousel-shell {{
                    width: min(100%, 1120px);
                    margin: 0 auto;
                }}

                .carousel-toolbar {{
                    display: flex;
                    align-items: center;
                    min-height: 34px;
                    padding: 0 0.25rem 0.4rem;
                }}

                .carousel-heading {{
                    display: inline-flex;
                    align-items: center;
                    gap: 0.68rem;
                    color: #64748B;
                    font-size: 0.69rem;
                    font-weight: 790;
                    letter-spacing: 0.115em;
                    text-transform: uppercase;
                    white-space: nowrap;
                }}

                .more-section-icon {{
                    position: relative;
                    width: 20px;
                    height: 16px;
                    flex: 0 0 auto;
                }}

                .more-section-icon::before,
                .more-section-icon::after {{
                    content: "";
                    position: absolute;
                    width: 14px;
                    height: 10px;
                    border-radius: 4px;
                }}

                .more-section-icon::before {{
                    left: 0;
                    top: 0;
                    border: 1.5px solid #A5B4FC;
                    background: #EEF2FF;
                }}

                .more-section-icon::after {{
                    left: 5px;
                    top: 5px;
                    border: 1.5px solid #4F46E5;
                    background: #FFFFFF;
                    box-shadow: 0 3px 8px rgba(79, 70, 229, 0.16);
                }}

                .carousel-viewport {{
                    position: relative;
                    height: var(--viewport-height);
                    overflow: hidden;
                    border-radius: 24px;
                    isolation: isolate;
                }}

                .carousel-viewport::before,
                .carousel-viewport::after {{
                    content: "";
                    position: absolute;
                    left: 0;
                    right: 0;
                    z-index: 20;
                    height: 9px;
                    pointer-events: none;
                }}

                .carousel-viewport::before {{
                    top: 0;
                    background: linear-gradient(
                        180deg,
                        rgba(243, 246, 251, 0.62),
                        rgba(243, 246, 251, 0)
                    );
                }}

                .carousel-viewport::after {{
                    bottom: 0;
                    background: linear-gradient(
                        0deg,
                        rgba(243, 246, 251, 0.62),
                        rgba(243, 246, 251, 0)
                    );
                }}

                .carousel-slide {{
                    position: absolute;
                    inset: 0 0 auto;
                    z-index: 0;
                    display: block;
                    height: var(--card-height);
                    color: inherit;
                    text-decoration: none;
                    opacity: 0;
                    pointer-events: none;
                    transform: translateY(calc(var(--viewport-height) + 20px)) scale(0.965);
                    transition:
                        transform 400ms cubic-bezier(0.22, 1, 0.36, 1),
                        opacity 320ms ease,
                        filter 320ms ease;
                    will-change: transform, opacity;
                }}

                .carousel-slide:visited,
                .carousel-slide:hover,
                .carousel-slide:active {{
                    color: inherit;
                    text-decoration: none;
                }}

                .carousel-slide.is-active {{
                    z-index: 6;
                    opacity: 1;
                    pointer-events: auto;
                    transform: translateY(var(--active-y)) scale(1);
                    filter: none;
                }}

                .carousel-slide.is-previous {{
                    z-index: 2;
                    opacity: 0.6;
                    transform: translateY(var(--previous-y)) scale(0.982);
                    filter: saturate(0.9);
                }}

                .carousel-slide.is-next {{
                    z-index: 3;
                    opacity: 0.68;
                    transform: translateY(var(--next-y)) scale(0.985);
                    filter: saturate(0.94);
                }}

                .carousel-slide.is-hidden-above {{
                    transform: translateY(calc(var(--previous-y) - var(--card-height))) scale(0.96);
                }}

                .carousel-slide.is-hidden-below {{
                    transform: translateY(calc(var(--next-y) + var(--card-height))) scale(0.96);
                }}

                .carousel-card {{
                    position: relative;
                    display: grid;
                    grid-template-columns: minmax(0, 0.94fr) minmax(0, 1.06fr);
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    border: 1px solid rgba(129, 140, 248, 0.36);
                    border-radius: 22px;
                    background: #FFFFFF;
                    box-shadow:
                        0 16px 34px rgba(15, 23, 42, 0.12),
                        0 3px 10px rgba(79, 70, 229, 0.07);
                    cursor: pointer;
                    transition:
                        transform 180ms ease,
                        border-color 180ms ease,
                        box-shadow 180ms ease;
                }}

                .carousel-slide.is-active:hover .carousel-card {{
                    transform: translateY(-2px);
                    border-color: rgba(79, 70, 229, 0.54);
                    box-shadow:
                        0 21px 42px rgba(15, 23, 42, 0.16),
                        0 5px 14px rgba(79, 70, 229, 0.11);
                }}

                .headline-panel,
                .quick-read-panel {{
                    min-width: 0;
                    padding: clamp(1.2rem, 2vw, 1.65rem);
                }}

                .headline-panel {{
                    position: relative;
                    display: flex;
                    flex-direction: column;
                    border-right: 1px solid rgba(148, 163, 184, 0.28);
                    background:
                        radial-gradient(circle at 0% 100%, rgba(99, 102, 241, 0.16), transparent 14rem),
                        linear-gradient(145deg, #FFFFFF 0%, #F1F2FF 100%);
                }}

                .quick-read-panel {{
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    padding-right: clamp(4.25rem, 6vw, 5rem);
                    background:
                        radial-gradient(circle at 100% 0%, rgba(6, 182, 212, 0.18), transparent 15rem),
                        linear-gradient(145deg, #EAF2FF 0%, #E4F7FF 100%);
                }}

                .panel-label,
                .card-category {{
                    color: #4338CA;
                    font-size: 0.66rem;
                    font-weight: 850;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                }}

                .panel-label {{
                    margin-bottom: 0.62rem;
                }}

                .card-category {{
                    margin-bottom: 0.62rem;
                    padding-right: 2rem;
                }}

                .card-title {{
                    margin: 0;
                    padding-right: 1.5rem;
                    color: #0F172A;
                    font-size: clamp(1.2rem, 1.8vw, 1.52rem);
                    font-weight: 790;
                    line-height: 1.15;
                    letter-spacing: -0.035em;
                    display: -webkit-box;
                    -webkit-box-orient: vertical;
                    -webkit-line-clamp: 3;
                    overflow: hidden;
                }}

                .card-highlight {{
                    color: #334155;
                    font-size: clamp(0.9rem, 1.18vw, 1rem);
                    line-height: 1.58;
                    display: -webkit-box;
                    -webkit-box-orient: vertical;
                    -webkit-line-clamp: 4;
                    overflow: hidden;
                }}

                .card-meta {{
                    margin-top: auto;
                    padding-top: 0.82rem;
                    color: #64748B;
                    font-size: 0.71rem;
                    line-height: 1.45;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }}

                .open-mark {{
                    position: absolute;
                    top: 0.95rem;
                    right: 0.95rem;
                    width: 28px;
                    height: 28px;
                    display: grid;
                    place-items: center;
                    border: 1px solid rgba(99, 102, 241, 0.16);
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.66);
                    color: #4F46E5;
                    opacity: 0.82;
                    transition: transform 180ms ease, opacity 180ms ease;
                }}

                .open-mark svg {{
                    width: 15px;
                    height: 15px;
                    fill: none;
                    stroke: currentColor;
                    stroke-width: 1.9;
                    stroke-linecap: round;
                    stroke-linejoin: round;
                }}

                .carousel-slide.is-active:hover .open-mark {{
                    transform: translate(2px, -2px);
                    opacity: 1;
                }}

                .carousel-controls {{
                    position: absolute;
                    z-index: 40;
                    top: 50%;
                    right: 13px;
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                    padding: 4px;
                    transform: translateY(-50%);
                    border: 1px solid rgba(148, 163, 184, 0.25);
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.84);
                    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.12);
                    backdrop-filter: blur(12px);
                }}

                .carousel-control {{
                    width: 34px;
                    height: 34px;
                    display: grid;
                    place-items: center;
                    padding: 0;
                    border: 0;
                    border-radius: 50%;
                    background: transparent;
                    color: #4338CA;
                    cursor: pointer;
                    transition:
                        transform 150ms ease,
                        color 150ms ease,
                        background 150ms ease;
                }}

                .carousel-control svg {{
                    width: 17px;
                    height: 17px;
                    fill: none;
                    stroke: currentColor;
                    stroke-width: 2;
                    stroke-linecap: round;
                    stroke-linejoin: round;
                }}

                .carousel-control:hover {{
                    color: #FFFFFF;
                    background: linear-gradient(135deg, var(--accent), #6366F1);
                    transform: scale(1.04);
                }}

                .carousel-control:active {{
                    transform: scale(0.92);
                }}

                .carousel-control:focus-visible {{
                    outline: 3px solid rgba(79, 70, 229, 0.18);
                    outline-offset: 2px;
                }}

                @media (max-width: 760px) {{
                    :root {{
                        --card-height: 264px;
                        --viewport-height: 318px;
                        --active-y: 27px;
                        --previous-y: -247px;
                        --next-y: 291px;
                    }}

                    .carousel-card {{
                        grid-template-columns: 1fr;
                        grid-template-rows: minmax(0, 1fr) minmax(0, 0.92fr);
                    }}

                    .headline-panel,
                    .quick-read-panel {{
                        padding: 0.92rem 1rem;
                    }}

                    .headline-panel {{
                        border-right: 0;
                        border-bottom: 1px solid rgba(148, 163, 184, 0.28);
                    }}

                    .quick-read-panel {{
                        padding-right: 3.8rem;
                    }}

                    .card-category,
                    .panel-label {{
                        margin-bottom: 0.38rem;
                    }}

                    .card-title {{
                        font-size: 1.03rem;
                        -webkit-line-clamp: 2;
                    }}

                    .card-highlight {{
                        font-size: 0.82rem;
                        line-height: 1.43;
                        -webkit-line-clamp: 3;
                    }}

                    .card-meta {{
                        padding-top: 0.42rem;
                        font-size: 0.65rem;
                    }}

                    .carousel-controls {{
                        right: 9px;
                    }}
                }}

                @media (prefers-reduced-motion: reduce) {{
                    .carousel-slide,
                    .carousel-card,
                    .carousel-control {{
                        transition-duration: 0.01ms !important;
                    }}
                }}
            </style>
        </head>
        <body>
            <section
                class="carousel-shell"
                id="{component_id}"
                aria-label="More from {safe_text(category_name)}"
            >
                <div class="carousel-toolbar">
                    <div class="carousel-heading">
                        <span class="more-section-icon" aria-hidden="true"></span>
                        <span>More from this section</span>
                    </div>
                </div>
                <div class="carousel-viewport">
                    {''.join(slides)}
                    <div class="carousel-controls" aria-label="Carousel controls">
                        <button
                            class="carousel-control previous-control"
                            type="button"
                            aria-label="Show previous story"
                            title="Previous story"
                        >
                            <svg viewBox="0 0 24 24" focusable="false">
                                <path d="m7 14 5-5 5 5" />
                            </svg>
                        </button>
                        <button
                            class="carousel-control next-control"
                            type="button"
                            aria-label="Show next story"
                            title="Next story"
                        >
                            <svg viewBox="0 0 24 24" focusable="false">
                                <path d="m7 10 5 5 5-5" />
                            </svg>
                        </button>
                    </div>
                </div>
            </section>

            <script>
                function sendMessageToStreamlitClient(type, data) {{
                    const payload = Object.assign(
                        {{
                            isStreamlitMessage: true,
                            type: type,
                        }},
                        data || {{}}
                    );
                    window.parent.postMessage(payload, "*");
                }}

                function sendDataToPython(value) {{
                    sendMessageToStreamlitClient(
                        "streamlit:setComponentValue",
                        {{
                            value: value,
                            dataType: "json",
                        }}
                    );
                }}

                sendMessageToStreamlitClient(
                    "streamlit:componentReady",
                    {{apiVersion: 1}}
                );

                (() => {{
                    const root = document.getElementById({json.dumps(component_id)});
                    if (!root) return;

                    const slides = Array.from(
                        root.querySelectorAll('.carousel-slide')
                    );
                    const previousButton = root.querySelector('.previous-control');
                    const nextButton = root.querySelector('.next-control');
                    const intervalMs = {int(ROTATOR_SECONDS_PER_ARTICLE * 1000)};
                    const transitionMs = 420;

                    let currentIndex = 0;
                    let timerId = null;
                    let isDocumentHidden = document.hidden;
                    let interactionLocked = false;

                    const normalizeIndex = (value) => {{
                        const length = slides.length;
                        return ((value % length) + length) % length;
                    }};

                    const renderSlides = () => {{
                        const previousIndex = normalizeIndex(currentIndex - 1);
                        const nextIndex = normalizeIndex(currentIndex + 1);

                        slides.forEach((slide, index) => {{
                            slide.classList.remove(
                                'is-active',
                                'is-previous',
                                'is-next',
                                'is-hidden-above',
                                'is-hidden-below'
                            );

                            if (index === currentIndex) {{
                                slide.classList.add('is-active');
                                slide.setAttribute('aria-hidden', 'false');
                                slide.tabIndex = 0;
                            }} else if (slides.length > 1 && index === previousIndex) {{
                                slide.classList.add('is-previous');
                                slide.setAttribute('aria-hidden', 'true');
                                slide.tabIndex = -1;
                            }} else if (slides.length > 1 && index === nextIndex) {{
                                slide.classList.add('is-next');
                                slide.setAttribute('aria-hidden', 'true');
                                slide.tabIndex = -1;
                            }} else {{
                                const forwardDistance = normalizeIndex(index - currentIndex);
                                const backwardDistance = normalizeIndex(currentIndex - index);
                                slide.classList.add(
                                    backwardDistance < forwardDistance
                                        ? 'is-hidden-above'
                                        : 'is-hidden-below'
                                );
                                slide.setAttribute('aria-hidden', 'true');
                                slide.tabIndex = -1;
                            }}
                        }});
                    }};

                    const clearTimer = () => {{
                        if (timerId !== null) {{
                            window.clearTimeout(timerId);
                            timerId = null;
                        }}
                    }};

                    const scheduleNext = () => {{
                        clearTimer();

                        if (
                            slides.length <= 1 ||
                            isDocumentHidden ||
                            window.matchMedia('(prefers-reduced-motion: reduce)').matches
                        ) {{
                            return;
                        }}

                        timerId = window.setTimeout(() => {{
                            move(1);
                        }}, intervalMs);
                    }};

                    const move = (delta) => {{
                        if (slides.length <= 1 || interactionLocked) return;

                        clearTimer();
                        interactionLocked = true;
                        currentIndex = normalizeIndex(currentIndex + delta);
                        renderSlides();

                        window.setTimeout(() => {{
                            interactionLocked = false;
                            scheduleNext();
                        }}, transitionMs);
                    }};

                    previousButton.addEventListener('click', (event) => {{
                        event.preventDefault();
                        event.stopPropagation();
                        move(-1);
                    }});

                    nextButton.addEventListener('click', (event) => {{
                        event.preventDefault();
                        event.stopPropagation();
                        move(1);
                    }});

                    slides.forEach((slide) => {{
                        slide.addEventListener('click', (event) => {{
                            if (!slide.classList.contains('is-active')) return;

                            const sourceId = slide.dataset.sourceId || '';
                            if (!sourceId) return;

                            event.preventDefault();
                            event.stopPropagation();
                            clearTimer();

                            sendDataToPython({{
                                source_id: sourceId,
                                event_id: `${{Date.now()}}-${{Math.random()}}`,
                            }});
                        }});
                    }});

                    document.addEventListener('visibilitychange', () => {{
                        isDocumentHidden = document.hidden;
                        scheduleNext();
                    }});

                    root.addEventListener('keydown', (event) => {{
                        if (event.key === 'ArrowUp') {{
                            event.preventDefault();
                            move(-1);
                        }} else if (event.key === 'ArrowDown') {{
                            event.preventDefault();
                            move(1);
                        }}
                    }});

                    if (slides.length <= 1) {{
                        previousButton.hidden = true;
                        nextButton.hidden = true;
                    }}

                    renderSlides();
                    scheduleNext();
                    sendMessageToStreamlitClient(
                        "streamlit:setFrameHeight",
                        {{height: 344}}
                    );
                }})();
            </script>
        </body>
        </html>
        """
    ).strip()

    component_root = Path(
        os.getenv(
            "BRIEFLINE_ROTATOR_COMPONENT_DIR",
            str(FRONTEND_ARTIFACT_DIR / "rotator_components"),
        )
    )
    component_path = component_root / category_slug
    component_path.mkdir(parents=True, exist_ok=True)

    index_path = component_path / "index.html"
    existing_markup = ""

    try:
        existing_markup = index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass

    if existing_markup != carousel_markup:
        index_path.write_text(carousel_markup, encoding="utf-8")

    component_name = (
        "briefline_rotator_"
        + category_slug.replace("-", "_")
    )
    rotator_component = components.declare_component(
        component_name,
        path=str(component_path),
    )

    navigation_event = rotator_component(
        default=None,
        key=component_id,
    )

    if isinstance(navigation_event, dict):
        selected_source_id = clean_text(
            navigation_event.get("source_id")
        )
        event_id = clean_text(
            navigation_event.get("event_id")
        )
        handled_event_key = (
            f"handled-rotator-navigation-{category_slug}"
        )

        if (
            selected_source_id
            and event_id
            and st.session_state.get(handled_event_key) != event_id
        ):
            st.session_state[handled_event_key] = event_id
            st.query_params["article"] = selected_source_id
            st.rerun()

def render_article_grid(
    articles: list[dict[str, Any]],
) -> None:
    if not articles:
        return

    if len(articles) == 1:
        render_compact_article(articles[0])
        st.write("")
        return

    for row_start in range(0, len(articles), 2):
        row_articles = articles[row_start:row_start + 2]
        columns = st.columns(2, gap="medium")

        for column_index, article in enumerate(row_articles):
            with columns[column_index]:
                render_compact_article(article)

        st.write("")


def render_search_results_list(
    articles: list[dict[str, Any]],
) -> None:
    """Render search matches as a compact editorial archive list."""
    if not articles:
        return

    rows: list[str] = []

    for article in articles:
        broad_category = safe_text(
            article.get("broad_category")
            or "News"
        )
        fine_category = safe_text(
            article.get("final_category")
            or ""
        )
        title = safe_text(
            article.get("title")
            or "Untitled article"
        )
        highlight = safe_text(
            clean_highlight(article.get("final_highlight"))
        )
        published_date = safe_text(
            format_date(article.get("published_at"))
        )

        keywords = normalize_keywords(
            article.get("final_keywords")
        )[:3]
        keywords_text = safe_text(" · ".join(keywords))

        fine_html = (
            f'<div class="search-result-fine">{fine_category}</div>'
            if fine_category
            else ""
        )
        highlight_html = (
            f'<div class="search-result-highlight">{highlight}</div>'
            if highlight
            else ""
        )
        keywords_html = (
            f'<div class="search-result-keywords">{keywords_text}</div>'
            if keywords_text
            else ""
        )
        date_html = (
            f'<div class="search-result-date">{published_date}</div>'
            if published_date
            else ""
        )

        rows.append(
            f"""
            <a
                class="search-result-link"
                href="{article_href(article)}"
                target="_self"
            >
                <article class="search-result-row">
                    <div class="search-result-taxonomy">
                        <div class="search-result-broad">{broad_category}</div>
                        {fine_html}
                    </div>

                    <div class="search-result-main">
                        <h2 class="search-result-title">{title}</h2>
                        {highlight_html}
                        {keywords_html}
                    </div>

                    <div class="search-result-side">
                        {date_html}
                        <span class="search-result-open" aria-hidden="true">
                            <svg viewBox="0 0 24 24" fill="none">
                                <path
                                    d="M8 16L16 8M10 8H16V14"
                                    stroke="currentColor"
                                    stroke-width="1.8"
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                ></path>
                            </svg>
                        </span>
                    </div>
                </article>
            </a>
            """
        )

    render_html(
        '<section class="search-results-list" aria-label="Search results">'
        + "".join(rows)
        + "</section>"
    )


def render_section_header(
    category_name: str,
) -> None:
    render_html(
        f"""
        <div class="section-header">
            <div class="section-heading-group">
                <div class="section-eyebrow">Section</div>
                <h2 class="section-title">{safe_text(category_name)}</h2>
                <div class="section-subtitle">
                    Latest reporting selected for this topic.
                </div>
            </div>
        </div>
        """
    )


def render_category_hero(
    category_name: str,
    total_articles: int,
    current_page: int,
    total_pages: int,
) -> None:
    render_html(
        f"""
        <section class="category-hero" id="category-results-top">
            <div class="category-hero-label">Browse section</div>
            <h1 class="content-heading">{safe_text(category_name)}</h1>
            <div class="content-description">
                Latest reporting selected for this topic · Page {current_page} of {total_pages}
            </div>
        </section>
        """
    )


def render_search_hero(
    query: str,
    current_page: int,
    total_pages: int,
) -> None:
    page_copy = (
        f"Page {current_page} of {total_pages}"
        if total_pages > 1
        else "Matching stories from the archive"
    )

    render_html(
        f"""
        <section class="search-hero" id="search-results-top">
            <div class="search-hero-label">Search archive</div>
            <h1 class="search-hero-title">
                Results for “{safe_text(query)}”
            </h1>
            <div class="search-hero-copy">{page_copy}</div>
        </section>
        """
    )


def render_recommendation_carousel(
    recommendations: list[dict[str, Any]],
) -> None:
    if not recommendations:
        return

    cards: list[str] = []

    for article in recommendations:
        category = safe_text(
            article.get("final_category")
            or article.get("broad_category")
            or "News"
        )

        title = safe_text(
            article.get("title")
            or "Untitled article"
        )

        highlight = safe_text(
            clean_highlight(article.get("final_highlight"))
        )

        metadata = safe_text(
            build_meta_text(article)
        )

        highlight_html = (
            f'<div class="related-card-highlight">{highlight}</div>'
            if highlight
            else ""
        )

        metadata_html = (
            f'<div class="related-card-meta">{metadata}</div>'
            if metadata
            else ""
        )

        cards.append(
            f"""
            <a
                class="related-card-link"
                href="{article_href(article)}"
                target="_self"
            >
                <article class="related-card">
                    <div class="related-card-category">{category}</div>
                    <h3 class="related-card-title">{title}</h3>
                    {highlight_html}
                    {metadata_html}
                </article>
            </a>
            """
        )

    render_html(
        f"""
        <section class="related-section">
            <h2 class="related-heading">You may also like</h2>
            <div class="related-subtitle">
                Stories connected to what you just read.
            </div>
            <div
                class="related-carousel"
                aria-label="Related articles"
                tabindex="0"
            >
                {''.join(cards)}
            </div>
        </section>
        """
    )


def render_article_detail(
    article: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> None:
    broad_category = clean_text(
        article.get("broad_category")
        or "News"
    )

    final_category = clean_text(
        article.get("final_category")
    )

    title = safe_text(
        article.get("title")
        or "Untitled article"
    )

    published_date = safe_text(
        format_date(article.get("published_at"))
    )

    highlight = safe_text(
        clean_highlight(article.get("final_highlight"))
    )

    keyword_html = build_keyword_pills(
        article.get("final_keywords")
    )

    body_value = article.get("body_text")

    if not clean_text(body_value):
        body_value = article.get("summary")

    body_html = format_article_body(body_value)

    back_column, _ = st.columns(
        [1.45, 4.55],
        gap="small",
    )

    with back_column:
        if st.button(
            f"← Back to {broad_category}",
            key="back-to-article-category",
            use_container_width=True,
        ):
            st.session_state.selected_category = broad_category
            st.session_state.category_page = 1
            st.query_params.clear()
            st.query_params["section"] = broad_category
            st.query_params["page"] = "1"
            st.rerun()

    broad_html = safe_text(broad_category)
    final_html = safe_text(final_category)

    final_category_html = (
        f'<span class="detail-final-category">{final_html}</span>'
        if final_html
        else ""
    )

    render_html(
        f"""
        <section class="detail-hero">
            <div class="detail-category-row">
                <span class="detail-broad-category">{broad_html}</span>
                {final_category_html}
            </div>
            <h1 class="detail-title">{title}</h1>
            <div class="detail-date">Published {published_date}</div>
        </section>
        """
    )

    brief_html = (
        f"""
        <section class="integrated-brief">
            <div class="brief-label">In brief</div>
            <div class="brief-copy">{highlight}</div>
        </section>
        """
        if highlight
        else ""
    )

    mobile_topics_html = (
        f'<div class="mobile-topic-row">{keyword_html}</div>'
        if keyword_html
        else ""
    )

    readable_body_html = (
        f'<div class="article-body" lang="en">{body_html}</div>'
        if body_html
        else (
            '<div class="empty-message">'
            'This article does not currently have readable body text.'
            '</div>'
        )
    )

    aside_keywords = normalize_keywords(
        article.get("final_keywords")
    )

    aside_topics = "<br>".join(
        safe_text(keyword)
        for keyword in aside_keywords
    ) or "No topic labels"

    render_html(
        f"""
        <section class="detail-reading-panel">
            <main class="reading-main">
                {brief_html}
                {mobile_topics_html}
                {readable_body_html}
            </main>

            <div class="story-aside-wrap">
                <aside class="story-aside">
                    <div class="aside-label">About this story</div>
                    <div class="aside-value">{broad_html}</div>
                    <div class="aside-value">{final_html or broad_html}</div>
                    <div class="aside-value">{published_date}</div>

                    <div class="aside-divider"></div>

                    <div class="aside-label">Topics</div>
                    <div class="aside-value">{aside_topics}</div>
                </aside>
            </div>
        </section>
        """
    )

    render_recommendation_carousel(recommendations)


# =========================================================
# 7. Page state
# =========================================================

if "selected_category" not in st.session_state:
    st.session_state.selected_category = "All"

if "category_page" not in st.session_state:
    st.session_state.category_page = 1


# =========================================================
# 8. Site header
# =========================================================

active_search_query = clean_text(
    get_query_value("q")
)

render_html(
    f"""
    <header class="site-header">
        <div class="brand-group">
            <a class="story-link" href="?" target="_self" aria-label="Open homepage">
                <div class="brand-mark">B</div>
            </a>
            <div class="site-name">{safe_text(SITE_NAME)}</div>
        </div>

        <form class="archive-search-form" method="get" action="">
            <input
                class="archive-search-input"
                type="search"
                name="q"
                value="{safe_text(active_search_query)}"
                placeholder="Search the news archive…"
                aria-label="Search the news archive"
                autocomplete="off"
            />
            <button
                class="archive-search-submit"
                type="submit"
                aria-label="Search"
                title="Search"
            >
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
                    <circle cx="10.5" cy="10.5" r="5.75" stroke="currentColor" stroke-width="1.9"></circle>
                    <path d="M15 15L20 20" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"></path>
                </svg>
            </button>
        </form>
    </header>
    """
)


# =========================================================
# 9. Page routing
# =========================================================

article_source_id = get_query_value("article")

if article_source_id:
    try:
        selected_article = fetch_article_detail(
            article_source_id
        )

        recommended_articles = (
            fetch_recommended_articles(article_source_id)
            if selected_article
            else []
        )

    except Exception as error:
        st.error("Unable to load the article details.")
        st.code(str(error))
        st.stop()

    if not selected_article:
        render_html(
            """
            <div class="empty-message">
                The requested article is unavailable.
            </div>
            """
        )

        if st.button(
            "Return to homepage",
            key="return-home-from-missing-article",
        ):
            st.query_params.clear()
            st.rerun()

    else:
        render_article_detail(
            article=selected_article,
            recommendations=recommended_articles,
        )

    st.stop()


search_query = active_search_query

if search_query:
    raw_page = get_query_value("page")

    try:
        current_search_page = max(1, int(raw_page or "1"))
    except ValueError:
        current_search_page = 1

    try:
        search_results, total_search_results = search_articles(
            query=search_query,
            page_number=current_search_page,
            page_size=SEARCH_PAGE_SIZE,
        )
    except Exception as error:
        st.error("Unable to search the news archive.")
        st.code(str(error))
        st.stop()

    total_search_pages = max(
        1,
        math.ceil(total_search_results / SEARCH_PAGE_SIZE),
    )

    if current_search_page > total_search_pages:
        st.query_params["q"] = search_query
        st.query_params["page"] = str(total_search_pages)
        st.rerun()

    render_search_hero(
        query=search_query,
        current_page=current_search_page,
        total_pages=total_search_pages,
    )

    if search_results:
        render_search_results_list(search_results)
    else:
        render_html(
            """
            <div class="empty-message">
                No matching stories were found. Try a different title, topic, or keyword.
            </div>
            """
        )

    if total_search_pages > 1:
        encoded_query = quote(search_query, safe="")

        if current_search_page > 1:
            previous_href = (
                f"?q={encoded_query}&page={current_search_page - 1}"
                "#search-results-top"
            )
            previous_control = f"""
                <a
                    class="search-page-link"
                    href="{previous_href}"
                    target="_self"
                    aria-label="Open previous search-results page"
                >
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path d="M14.5 6L8.5 12L14.5 18" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"></path>
                    </svg>
                    <span>Previous</span>
                </a>
            """
        else:
            previous_control = """
                <span class="search-page-disabled" aria-disabled="true">
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path d="M14.5 6L8.5 12L14.5 18" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"></path>
                    </svg>
                    <span>Previous</span>
                </span>
            """

        if current_search_page < total_search_pages:
            next_href = (
                f"?q={encoded_query}&page={current_search_page + 1}"
                "#search-results-top"
            )
            next_control = f"""
                <a
                    class="search-page-link"
                    href="{next_href}"
                    target="_self"
                    aria-label="Open next search-results page"
                >
                    <span>Next</span>
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path d="M9.5 6L15.5 12L9.5 18" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"></path>
                    </svg>
                </a>
            """
        else:
            next_control = """
                <span class="search-page-disabled" aria-disabled="true">
                    <span>Next</span>
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path d="M9.5 6L15.5 12L9.5 18" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"></path>
                    </svg>
                </span>
            """

        render_html(
            f"""
            <nav class="search-pagination" aria-label="Search-results pagination">
                <div class="search-pagination-side">
                    {previous_control}
                </div>
                <div class="search-page-status">
                    Page {current_search_page} of {total_search_pages}
                </div>
                <div class="search-pagination-side">
                    {next_control}
                </div>
            </nav>
            """
        )

    st.stop()


# =========================================================
# 10. Load category data
# =========================================================

try:
    categories = fetch_categories()

except Exception as error:
    st.error("Unable to connect to PostgreSQL or load category data.")
    st.code(str(error))
    st.stop()


category_names = [
    row["broad_category"]
    for row in categories
]

requested_section = clean_text(
    get_query_value("section")
)

if requested_section in category_names:
    st.session_state.selected_category = requested_section

    raw_category_page = get_query_value("page")

    try:
        requested_category_page = max(
            1,
            int(raw_category_page or "1"),
        )
    except ValueError:
        requested_category_page = 1

    st.session_state.category_page = requested_category_page

elif requested_section:
    st.session_state.selected_category = "All"
    st.session_state.category_page = 1

if (
    st.session_state.selected_category != "All"
    and st.session_state.selected_category not in category_names
):
    st.session_state.selected_category = "All"
    st.session_state.category_page = 1


# =========================================================
# 11. Homepage layout
# =========================================================

navigation_column, content_column = st.columns(
    [1.3, 5.2],
    gap="large",
)


# =========================================================
# 12. Category navigation
# =========================================================

with navigation_column:
    render_html(
        """
        <div class="navigation-heading">Sections</div>
        """
    )

    all_categories = [
        "All",
        *category_names,
    ]

    for category_name in all_categories:
        is_selected = (
            category_name
            == st.session_state.selected_category
        )

        button_clicked = st.button(
            category_name,
            key=f"category-navigation-{category_name}",
            type=(
                "primary"
                if is_selected
                else "secondary"
            ),
            use_container_width=True,
        )

        if button_clicked:
            st.session_state.selected_category = (
                category_name
            )

            st.session_state.category_page = 1
            st.query_params.clear()

            if category_name != "All":
                st.query_params["section"] = category_name
                st.query_params["page"] = "1"

            st.rerun()


# =========================================================
# 13. Article content
# =========================================================

with content_column:
    selected_category = (
        st.session_state.selected_category
    )

    if selected_category == "All":
        try:
            homepage_articles = (
                fetch_all_homepage_articles()
            )

        except Exception as error:
            st.error("Unable to load homepage articles.")
            st.code(str(error))
            st.stop()

        articles_by_category: dict[
            str,
            list[dict[str, Any]],
        ] = defaultdict(list)

        for article in homepage_articles:
            articles_by_category[
                article["broad_category"]
            ].append(article)

        rendered_section_count = 0

        for category_name in category_names:
            category_articles = (
                articles_by_category.get(
                    category_name,
                    [],
                )
            )

            if not category_articles:
                continue

            rendered_section_count += 1

            render_section_header(
                category_name=category_name,
            )

            render_featured_article(
                category_articles[0]
            )

            render_vertical_story_rotator(
                articles=category_articles[1:],
                category_name=category_name,
            )

            render_html(
                '<div class="section-tightener"></div>'
            )

        if rendered_section_count == 0:
            render_html(
                """
                <div class="empty-message">
                    No articles are currently available.
                </div>
                """
            )

    else:
        current_page = (
            st.session_state.category_page
        )

        try:
            category_articles, total_articles = (
                fetch_category_articles(
                    broad_category=selected_category,
                    page_number=current_page,
                    page_size=PAGE_SIZE,
                )
            )

        except Exception as error:
            st.error("Unable to load articles for this category.")
            st.code(str(error))
            st.stop()

        total_pages = max(
            1,
            math.ceil(
                total_articles / PAGE_SIZE
            ),
        )

        if current_page > total_pages:
            st.session_state.category_page = total_pages
            st.query_params.clear()
            st.query_params["section"] = selected_category
            st.query_params["page"] = str(total_pages)
            st.rerun()

        render_category_hero(
            category_name=selected_category,
            total_articles=total_articles,
            current_page=current_page,
            total_pages=total_pages,
        )

        if not category_articles:
            render_html(
                """
                <div class="empty-message">
                    No articles are currently available
                    in this section.
                </div>
                """
            )

        else:
            if current_page == 1:
                render_featured_article(
                    category_articles[0]
                )

                remaining_articles = (
                    category_articles[1:]
                )

            else:
                remaining_articles = (
                    category_articles
                )

            render_article_grid(
                remaining_articles
            )

        if total_pages > 1:
            if current_page > 1:
                previous_href = category_page_href(
                    selected_category,
                    current_page - 1,
                )
                previous_control = f"""
                    <a
                        class="search-page-link"
                        href="{previous_href}"
                        target="_self"
                        aria-label="Open previous section page"
                    >
                        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                            <path d="M14.5 6L8.5 12L14.5 18" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                        <span>Previous</span>
                    </a>
                """
            else:
                previous_control = """
                    <span class="search-page-disabled" aria-disabled="true">
                        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                            <path d="M14.5 6L8.5 12L14.5 18" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                        <span>Previous</span>
                    </span>
                """

            if current_page < total_pages:
                next_href = category_page_href(
                    selected_category,
                    current_page + 1,
                )
                next_control = f"""
                    <a
                        class="search-page-link"
                        href="{next_href}"
                        target="_self"
                        aria-label="Open next section page"
                    >
                        <span>Next</span>
                        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                            <path d="M9.5 6L15.5 12L9.5 18" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                    </a>
                """
            else:
                next_control = """
                    <span class="search-page-disabled" aria-disabled="true">
                        <span>Next</span>
                        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                            <path d="M9.5 6L15.5 12L9.5 18" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                    </span>
                """

            render_html(
                f"""
                <nav class="search-pagination" aria-label="Section pagination">
                    <div class="search-pagination-side">
                        {previous_control}
                    </div>
                    <div class="search-page-status">
                        Page {current_page} of {total_pages}
                    </div>
                    <div class="search-pagination-side">
                        {next_control}
                    </div>
                </nav>
                """
            )
