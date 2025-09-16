from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re
import difflib
from datetime import datetime, timedelta

__all__ = [
    "ensure_json_dataset",
    "load_json_docs",
    "load_pdf_docs",
    "ensure_sample_csv",
    "_find_closest_match",
    "_extract_sql",
    "webpage_to_pdf",
]


def ensure_json_dataset(local_root: Path, repo_id: str) -> Path:
    local_root.mkdir(parents=True, exist_ok=True)
    # Lazy import to avoid requiring huggingface_hub at package import time
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_root),
        local_dir_use_symlinks=False,
    )
    candidates = list(local_root.rglob("training_data.json"))
    if not candidates:
        raise FileNotFoundError("training_data.json not found after snapshot_download.")
    return candidates[0]


def load_json_docs(json_path: Path) -> List[Any]:
    with open(json_path, "r") as f:
        rows: List[Dict[str, Any]] = json.load(f)
    # Lazy import Document to avoid requiring langchain at package import time
    try:
        from langchain_core.documents import Document  # type: ignore
    except Exception:
        Document = dict  # fallback type to avoid import-time failure

    docs: List[Any] = []
    for r in rows:
        text = (r.get("content") or "").strip()
        if not text:
            continue
        meta = {
            "id": r.get("id"),
            "topic": r.get("topic"),
            "title": r.get("title"),
            "source": json_path.name,
        }
        try:
            docs.append(Document(page_content=text, metadata=meta))
        except TypeError:
            docs.append({"page_content": text, "metadata": meta})
    return docs


def load_pdf_docs(pdf_dir: Path) -> List[Any]:
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    out: List[Any] = []
    for p in pdfs:
        try:
            # Lazy import to avoid requiring langchain at package import time
            from langchain_community.document_loaders import PyPDFLoader  # type: ignore
            loader = PyPDFLoader(str(p))
            pages = loader.load()
            for d in pages:
                d.metadata = {
                    **(d.metadata or {}),
                    "title": d.metadata.get("title") or p.stem,
                    "file_path": str(p),
                }
            out.extend(pages)
        except Exception as e:
            print(f"Failed to load PDF {p}: {e}")
    return out


def ensure_sample_csv(
    csv_path: Path,
    wells: Optional[List[str]] = None,
    days: int = 60,
    seed: int = 42,
) -> None:
    if csv_path.exists():
        return
    # Lazy imports to avoid heavy deps at package import time
    import numpy as np
    import pandas as pd

    np.random.seed(seed)
    wells = wells or [
        "Well-15/9-F-1-C",
        "Well-15/9-F-11-A",
        "Well-15/9-F-12-B",
    ]
    start = datetime(2017, 1, 1)
    rows: List[Dict[str, Any]] = []
    for w in wells:
        for d in range(days):
            day = start + timedelta(days=d)
            oil = max(0, 1200 + int(300 * np.cos(d / 9)) + np.random.randint(-50, 50))
            water = max(0, 500 + int(120 * np.sin(d / 7)) + np.random.randint(-30, 30))
            gas = max(0, 8000 + int(1200 * np.sin(d / 11)) + np.random.randint(-500, 500))
            wc = round(100 * water / (oil + water) if (oil + water) > 0 else 0, 2)
            rows.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "well_name": w,
                    "oil_rate_bpd": oil,
                    "water_rate_bpd": water,
                    "gas_rate_scfd": gas,
                    "water_cut_pct": wc,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Created sample CSV: {csv_path}")


def _find_closest_match(text: str, possibilities: List[str]) -> str:
    matches = difflib.get_close_matches(text, possibilities, n=1, cutoff=0.6)
    return matches[0] if matches else text


def _extract_sql(
    text: str,
    table_columns: List[str],
    string_value_domains: Optional[Dict[str, List[str]]] = None,
) -> str:
    m = re.search(r"```sql\s*(.*?)```", text, flags=re.I | re.S)
    sql = m.group(1).strip() if m else text.strip()

    sql = sql.split(";")[0].strip()
    if not re.match(r"^select\b", sql, flags=re.I):
        raise ValueError("Generated SQL is not a single SELECT.")
    forbidden = re.search(
        r"\b(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|ATTACH|COPY|EXPORT)\b", sql, re.I
    )
    if forbidden:
        raise ValueError("Non-SELECT statement detected.")

    # Fix column name casing based on provided columns
    for col in sorted(table_columns, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(col)}\b", flags=re.I)
        sql = pattern.sub(col, sql)

    # Light correction for string literals using provided domains
    if string_value_domains:
        all_values: List[str] = []
        for values in string_value_domains.values():
            all_values.extend(values)
        if all_values:
            def replace_literal(mo: re.Match[str]) -> str:
                literal = mo.group(1)
                closest = _find_closest_match(literal, all_values)
                return f"'{closest}'"

            sql = re.sub(r"'([^']+)'", replace_literal, sql)

    return sql + ";"


async def _webpage_to_pdf_async(url: str, out_pdf: Path, inject_mathjax: bool, wait_ms: int) -> Path:
    from playwright.async_api import async_playwright

    MATHJAX_CDN = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        if inject_mathjax:
            js_check = """() => Boolean(window.MathJax) || Boolean(document.querySelector('[class*="katex"], [id*="katex"]'))"""
            has_math_lib = await page.evaluate(js_check)
            if not has_math_lib:
                await page.add_script_tag(url=MATHJAX_CDN)
        await page.wait_for_timeout(wait_ms)
        await page.emulate_media(media="screen")
        await page.pdf(path=str(out_pdf), format="A4", print_background=True)
        await browser.close()
    return out_pdf


def webpage_to_pdf(url: str, out_pdf: Path, inject_mathjax: bool = True, wait_ms: int = 1500) -> Path:
    """
    Render a webpage (including LaTeX/KaTeX) to a PDF.

    Works in notebooks by using Playwright's async API and nest_asyncio when a loop is running.
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        try:
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
        except Exception:
            pass
        return loop.run_until_complete(_webpage_to_pdf_async(url, out_pdf, inject_mathjax, wait_ms))
    else:
        return asyncio.run(_webpage_to_pdf_async(url, out_pdf, inject_mathjax, wait_ms))
