"""Blue Triangle CLI Reporter - Analyze RUM data and revenue impact."""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from difflib import get_close_matches
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set to DEBUG to see detailed API calls, INFO for normal operation
LOG_LEVEL = os.getenv("BT_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# API Credentials loaded from environment variables
# BT_API_EMAIL, BT_API_KEY, and BT_SITE_PREFIX are loaded from .env file

# Base API URL
BASE_URL = "https://api.bluetriangletech.com"

# Site prefix for API requests
SITE_PREFIX = os.getenv("BT_SITE_PREFIX", "")

# Request timeout in seconds
REQUEST_TIMEOUT = int(os.getenv("BT_REQUEST_TIMEOUT", "30"))

# Headers for API Requests
HEADERS = {
    "X-API-Email": os.getenv("BT_API_EMAIL"),
    "X-API-Key": os.getenv("BT_API_KEY"),
    "Content-Type": "application/json",
}

# API Endpoints
ENDPOINTS: dict[str, str] = {
    "performance": "/performance",
    "performance_hits": "/performance/hits",
    "synthetic_monitor": "/synthetic-monitor",
    "resource": "/resource",
    "error": "/error",
    "revenue_opportunity": "/revenue-opportunity",
    "revenue_opportunity_report": "/revenue-opportunity/report-date",
    "content_security_policy": "/content-security-policies",
    "network_health": "/network-health",
    "network_health_hits": "/network-health/hits",
    "event_markers": "/event-markers",
}

# Time range mapping (days) - single source of truth
DAY_MAP: dict[str, float] = {
    "qd": 0.25,
    "hd": 0.5,
    "24h": 1,
    "xd": 1.5,
    "2d": 2,
    "6d": 6,
    "7d": 7,
    "28d": 28,
    "30d": 30,
    "90d": 90,
    "1y": 365,
    "2y": 730,
    "3y": 1095,
}

# Metric labels mapping
METRIC_LABELS: dict[str, str] = {
    "largestContentfulPaint": "LCP",
    "intToNextPaint": "INP",
    "cumulativeLayoutShift": "CLS",
    "onload": "Onload",
    "firstByte": "First Byte",
    "totalBlockingTime": "TBT",
    "dns": "DNS",
    "tcp": "TCP",
}

# Metric weights for performance scoring
METRIC_WEIGHTS: dict[str, float] = {
    "largestContentfulPaint": 4,
    "intToNextPaint": 3,
    "cumulativeLayoutShift": 3,
    "onload": 0.75,
    "firstByte": 0.5,
    "dns": 0.375,
    "tcp": 0.375,
}

# Fallback list in case the API returns no data
AVAILABLE_PAGES: list[str] = ["homepage"]

# Global time variables (set after parsing time arguments)
now: int | None = None
one_day_ago: int | None = None
two_days_ago: int | None = None

# Selected metrics filter (None means all metrics)
selected_metrics: list[str] | None = None


# ==================
# HELPER FUNCTIONS
# ==================


def normalize_page_name(query: str) -> str | None:
    """Fuzzy-match user input to the canonical page name.

    The input query is normalized by converting to lowercase and removing
    non-alphanumeric characters, while the page names in AVAILABLE_PAGES
    remain unchanged.

    Args:
        query: User input page name to normalize.

    Returns:
        Matched page name or None if no match found.
    """
    normalized_query = re.sub(r"[^a-z0-9]", "", query.strip().lower())

    page_map = {
        page: re.sub(r"[^a-z0-9]", "", page.strip().lower())
        for page in AVAILABLE_PAGES
    }

    if normalized_query in page_map.values():
        return next(
            page for page, normalized in page_map.items()
            if normalized == normalized_query
        )

    matches = get_close_matches(normalized_query, page_map.values(), n=1, cutoff=0.4)
    if matches:
        return next(
            page for page, normalized in page_map.items()
            if normalized == matches[0]
        )
    return None


def validate_api_response(
    data: Any,
    required_key: str | None = None
) -> bool:
    """Validate API response structure.

    Args:
        data: Response data to validate.
        required_key: Optional key that must exist in response.

    Returns:
        True if response is valid, False otherwise.
    """
    if data is None:
        return False
    if required_key and not isinstance(data, dict):
        return False
    if required_key and required_key not in data:
        return False
    return True


def fetch_data(
    endpoint: str,
    payload: dict[str, Any] | None = None,
    method: str = "POST",
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any] | None:
    """Generic function to fetch JSON from the Blue Triangle API.

    Args:
        endpoint: API endpoint path.
        payload: JSON payload for POST requests.
        method: HTTP method (GET or POST).
        params: Query parameters for GET requests.

    Returns:
        Parsed JSON response or None on error.
    """
    url = BASE_URL + endpoint
    try:
        if method == "GET":
            logger.debug("GET %s params=%s", url, params)
            r = requests.get(
                url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT
            )
        else:
            logger.debug("POST %s payload=%s", url, payload)
            r = requests.post(
                url, headers=HEADERS, json=payload, timeout=REQUEST_TIMEOUT
            )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as errh:
        logger.error("HTTP Error: %s", errh)
    except requests.exceptions.ConnectionError as errc:
        logger.error("Connection Error: %s", errc)
    except requests.exceptions.Timeout as errt:
        logger.error("Timeout Error: %s", errt)
    except requests.exceptions.RequestException as err:
        logger.error("Request Error: %s", err)
    return None


def _to_float(val: Any) -> float | None:
    """Convert a field to float, or None if not numeric.

    Args:
        val: Value to convert.

    Returns:
        Float value or None.
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch_top_page_names(
    limit: int = 20,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """Fetch top page names ordered by pageViews.

    Uses the performance endpoint. If start or end are not provided,
    falls back to global time variables.

    Args:
        limit: Maximum number of pages to return.
        start: Start timestamp (epoch seconds).
        end: End timestamp (epoch seconds).

    Returns:
        DataFrame with page names and views.
    """
    global now, one_day_ago
    if start is None:
        start = one_day_ago if one_day_ago is not None else int(time.time()) - 86400
    if end is None:
        end = now if now is not None else int(time.time())

    payload = {
        "site": SITE_PREFIX,
        "start": start,
        "end": end,
        "dataColumns": ["pageViews"],
        "group": ["pageName", "url"],
        "limit": limit,
        "orderBy": [{"field": "pageViews", "direction": "DESC"}],
    }

    logger.debug("Fetching top pages: %s", payload)
    data = fetch_data(ENDPOINTS["performance"], payload)
    logger.debug("Top pages response: %s", data)

    if validate_api_response(data, "data"):
        return pd.DataFrame(data["data"])
    return pd.DataFrame([])


def update_available_pages(limit: int = 20) -> list[str]:
    """Update AVAILABLE_PAGES dynamically from the API.

    Args:
        limit: Maximum number of pages to fetch.

    Returns:
        List of available page names.
    """
    global AVAILABLE_PAGES
    df = fetch_top_page_names(limit=limit)
    if not df.empty and "pageName" in df.columns:
        AVAILABLE_PAGES = df["pageName"].tolist()
    return AVAILABLE_PAGES


def plot_performance_metrics(metrics: dict[str, Any]) -> None:
    """Plot LCP over time.

    Args:
        metrics: Dictionary with 'time' and 'lcp' keys.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metrics["time"], metrics["lcp"], label="LCP")
    plt.title("LCP Over Time")
    plt.legend()
    plt.show()


# ================
# TIME PARSING
# ================


def parse_time_args(
    args: argparse.Namespace,
) -> tuple[int | None, int | None, int | None, int | None, list[str]]:
    """Parse time-related arguments.

    If --multi-range is provided, returns a list of multiple ranges.
    Otherwise, parse single start/end or a named --time-range.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (start, end, prev_start, prev_end, multi_list).
    """
    if args.multi_range:
        ranges = [x.strip() for x in args.multi_range.split(",")]
        return None, None, None, None, ranges

    now_ts = int(time.time())

    if args.start and args.end:
        start = int(args.start)
        end = int(args.end)
        return start, end, None, None, []

    days = DAY_MAP.get(args.time_range, 1)
    end = now_ts
    start = int(now_ts - (days * 86400))
    prev_end = start
    prev_start = int(start - (days * 86400))
    return start, end, prev_start, prev_end, []


def compute_time_window(
    range_str: str,
) -> tuple[int, int, int, int]:
    """Convert a range like '24h' or '28d' to timestamps.

    Args:
        range_str: Time range string (e.g., '24h', '7d').

    Returns:
        Tuple of (start, end, prev_start, prev_end).
    """
    now_ts = int(time.time())
    days = DAY_MAP.get(range_str, 1)
    end = now_ts
    start = int(now_ts - (days * 86400))
    prev_end = start
    prev_start = int(start - (days * 86400))
    return start, end, prev_start, prev_end


# =====================
# PERFORMANCE LOGIC
# =====================


def should_include_metric(metric_key: str) -> bool:
    """Check if a metric should be included based on --metrics filter.

    Args:
        metric_key: The API key for the metric.

    Returns:
        True if metric should be included.
    """
    if selected_metrics is None:
        return True

    label = METRIC_LABELS.get(metric_key, "").upper()
    # Map common variations
    label_map = {
        "LCP": "LCP",
        "INP": "INP",
        "CLS": "CLS",
        "TBT": "TBT",
        "FIRST BYTE": "FB",
        "FB": "FB",
    }
    normalized_label = label_map.get(label, label)
    return normalized_label in selected_metrics


def summarize_performance(current: dict[str, Any], previous: dict[str, Any]) -> str:
    """Generate weighted summary of performance metrics.

    Args:
        current: Current period metrics.
        previous: Previous period metrics.

    Returns:
        Markdown formatted summary string.
    """

    def to_float(value: Any) -> float | None:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    statements: list[str] = []
    weighted_score = 0.0
    total_weight = sum(METRIC_WEIGHTS.values())

    for key, label in METRIC_LABELS.items():
        if not should_include_metric(key):
            continue

        current_value = to_float(current.get(key))
        previous_value = to_float(previous.get(key))

        if current_value is None or previous_value is None:
            continue

        weight = METRIC_WEIGHTS.get(key, 1)

        if current_value < previous_value:
            statements.append(f"{label} improved")
            weighted_score += weight
        elif current_value > previous_value:
            statements.append(f"{label} worsened")
            weighted_score -= weight
        else:
            statements.append(f"{label} stayed the same")

    if weighted_score > 0:
        arrow = "▲"
    elif weighted_score < 0:
        arrow = "▼"
    else:
        arrow = "→"

    if not statements:
        return "### 📝 Summary\nNo performance data to compare.\n"

    summary_list = ", ".join(statements)
    return (
        f"### 📝 Summary ({arrow} [Weighted Score: {weighted_score} / "
        f"Total: {total_weight}])\n- {summary_list} \n"
    )


def get_page_performance(page_name: str) -> str:
    """Fetch performance data for a given page and return a Markdown report.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Markdown formatted performance report.
    """
    payload = {
        "site": SITE_PREFIX,
        "start": one_day_ago,
        "end": now,
        "dataType": "rum",
        "dataColumns": [
            "onload", "dns", "tcp", "firstByte",
            "largestContentfulPaint", "totalBlockingTime",
            "cumulativeLayoutShift", "intToNextPaint",
        ],
        "group": ["time"],
        "limit": 1000,
        "order": "time",
        "sort": "asc",
        "pageName": page_name,
    }

    prev_payload = dict(payload)
    if two_days_ago is not None:
        prev_payload["start"] = two_days_ago
        prev_payload["end"] = one_day_ago

    logger.debug("Fetching current data for %s: %s", page_name, payload)
    logger.debug("Fetching previous data for %s: %s", page_name, prev_payload)

    data_today = fetch_data(ENDPOINTS["performance"], payload)
    data_prev = (
        fetch_data(ENDPOINTS["performance"], prev_payload)
        if two_days_ago
        else None
    )

    if (
        not validate_api_response(data_today, "data")
        or not validate_api_response(data_prev, "data")
    ):
        return f"> ⚠️ No performance data for **{page_name}**.\n"

    df_today = pd.DataFrame(data_today["data"])
    df_prev = pd.DataFrame(data_prev["data"])
    if df_today.empty or df_prev.empty:
        return f"> ⚠️ No usable performance data for **{page_name}**.\n"

    t = df_today.iloc[-1]
    p = df_prev.iloc[-1]

    # If this is a VT page, explicitly set LCP & CLS to None
    if "vt" in page_name.lower():
        t["largestContentfulPaint"] = None
        t["cumulativeLayoutShift"] = None
        p["largestContentfulPaint"] = None
        p["cumulativeLayoutShift"] = None

    def n(val: Any) -> float | str:
        """Return the value as float if numeric, else 'N/A'."""
        if val in (None, ""):
            return "N/A"
        try:
            return float(val)
        except (ValueError, TypeError):
            return "N/A"

    def delta(a: float | str, b: float | str) -> float | str:
        if a == "N/A" or b == "N/A":
            return "N/A"
        return round(float(a) - float(b), 2)

    def percent_change(current: float | str, previous: float | str) -> float | str:
        """Calculate the percentage change from previous to current."""
        if previous is None or previous == 0 or previous == "N/A":
            return "N/A"
        if current == "N/A":
            return "N/A"
        try:
            change = ((float(current) - float(previous)) / float(previous)) * 100
            return round(change, 2)
        except (TypeError, ValueError):
            return "N/A"

    lcp_current = n(t.get("largestContentfulPaint"))
    lcp_previous = n(p.get("largestContentfulPaint"))
    delta_lcp = delta(lcp_current, lcp_previous)

    if lcp_current == "N/A" or lcp_previous == "N/A":
        lcp_insight = "LCP data is not available for comparison."
    elif isinstance(delta_lcp, (int, float)) and delta_lcp > 0:
        lcp_insight = "Actions are crucial for optimizing the LCP metric."
    else:
        lcp_insight = "LCP is stable."

    perf_summary = summarize_performance(t, p)

    # Build report sections based on selected metrics
    current_section = "#### Current Window\n"
    previous_section = "#### Previous Window\n"
    delta_section = "### 📉 Performance Change (Delta)\n"

    metric_items = [
        ("onload", "Onload Time", "ms"),
        ("dns", "DNS Lookup Time", "ms"),
        ("tcp", "TCP Connection Time", "ms"),
        ("firstByte", "First Byte Time", "ms"),
        ("largestContentfulPaint", "Largest Contentful Paint (LCP)", "ms"),
        ("totalBlockingTime", "Total Blocking Time (TBT)", "ms"),
        ("intToNextPaint", "Input Delay (INP)", "ms"),
        ("cumulativeLayoutShift", "Cumulative Layout Shift (CLS)", ""),
    ]

    for key, label, unit in metric_items:
        if not should_include_metric(key):
            continue
        unit_str = f" {unit}" if unit else ""
        current_section += f"- **{label}**: {n(t.get(key))}{unit_str}\n"
        previous_section += f"- **{label}**: {n(p.get(key))}{unit_str}\n"
        d = delta(n(t.get(key)), n(p.get(key)))
        pc = percent_change(n(t.get(key)), n(p.get(key)))
        delta_section += f"- **{label}**: Δ {d}{unit_str} ({pc}%)\n"

    return f"""
{perf_summary}

### 📊 Performance Metrics for {page_name}

{current_section}
{previous_section}
{delta_section}
### 🛠 Optimization Insights
- **Onload Time**: {"Improve server response time or defer offscreen images." if isinstance(delta(n(t.get('onload')), n(p.get('onload'))), (int, float)) and delta(n(t.get('onload')), n(p.get('onload'))) > 0 else "Maintain good performance."}
- **LCP**:
  - **Recommendations**:
    - **Preload** images and video poster images to ensure they load immediately.
    - **Remove lazy-loading** for main content images that contribute to the LCP.
    - {lcp_insight}
- **INP**: Optimize your JavaScript and CSS to reduce blocking time.
- **CLS**: Ensure that dimension attributes for images and video elements are set.
"""


# ============================
# TABLE & METRICS FUNCTIONS
# ============================


def safe_delta(curr: float | None, prev: float | None) -> float | str:
    """Calculate safe delta between two values.

    Args:
        curr: Current value.
        prev: Previous value.

    Returns:
        Difference or 'N/A' if either value is None.
    """
    if curr is None or prev is None:
        return "N/A"
    return round(curr - prev, 2)


def gather_page_metrics(page_name: str) -> dict[str, Any] | None:
    """Fetch current and previous page metrics and return a dictionary.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Dictionary of metrics or None if no data.
    """
    global now, one_day_ago, two_days_ago

    payload = {
        "site": SITE_PREFIX,
        "start": one_day_ago,
        "end": now,
        "dataType": "rum",
        "dataColumns": [
            "onload", "dns", "tcp", "firstByte",
            "largestContentfulPaint", "totalBlockingTime",
            "cumulativeLayoutShift", "intToNextPaint",
        ],
        "group": ["time"],
        "limit": 1000,
        "order": "time",
        "sort": "asc",
        "pageName": page_name,
    }

    prev_payload = dict(payload)
    if two_days_ago:
        prev_payload["start"] = two_days_ago
        prev_payload["end"] = one_day_ago

    data_curr = fetch_data(ENDPOINTS["performance"], payload)
    data_prev = (
        fetch_data(ENDPOINTS["performance"], prev_payload)
        if two_days_ago
        else None
    )

    if (
        not validate_api_response(data_curr, "data")
        or not validate_api_response(data_prev, "data")
    ):
        logger.warning("No data fetched for %s", page_name)
        return None

    df_curr = pd.DataFrame(data_curr["data"])
    df_prev = pd.DataFrame(data_prev["data"])

    if df_curr.empty or df_prev.empty:
        logger.warning("Empty data returned for %s", page_name)
        return None

    latest_curr = df_curr.iloc[-1]
    latest_prev = df_prev.iloc[-1]

    row = {
        "page": page_name,
        "onload_curr": _to_float(latest_curr.get("onload")),
        "onload_prev": _to_float(latest_prev.get("onload")),
        "lcp_curr": _to_float(latest_curr.get("largestContentfulPaint")),
        "lcp_prev": _to_float(latest_prev.get("largestContentfulPaint")),
        "inp_curr": _to_float(latest_curr.get("intToNextPaint")),
        "inp_prev": _to_float(latest_prev.get("intToNextPaint")),
        "cls_curr": _to_float(latest_curr.get("cumulativeLayoutShift")),
        "cls_prev": _to_float(latest_prev.get("cumulativeLayoutShift")),
        "tbt_curr": _to_float(latest_curr.get("totalBlockingTime")),
        "tbt_prev": _to_float(latest_prev.get("totalBlockingTime")),
        "fb_curr": _to_float(latest_curr.get("firstByte")),
        "fb_prev": _to_float(latest_prev.get("firstByte")),
        "dns_curr": _to_float(latest_curr.get("dns")),
        "dns_prev": _to_float(latest_prev.get("dns")),
        "tcp_curr": _to_float(latest_curr.get("tcp")),
        "tcp_prev": _to_float(latest_prev.get("tcp")),
    }

    is_vt = "vt" in page_name.lower()
    onload_delta = safe_delta(row["onload_curr"], row["onload_prev"])
    lcp_delta = safe_delta(row["lcp_curr"], row["lcp_prev"]) if not is_vt else "N/A"
    inp_delta = safe_delta(row["inp_curr"], row["inp_prev"])
    cls_delta = safe_delta(row["cls_curr"], row["cls_prev"]) if not is_vt else "N/A"

    logger.debug(
        "%s Delta: Onload: %s, LCP: %s, INP: %s, CLS: %s",
        page_name, onload_delta, lcp_delta, inp_delta, cls_delta
    )

    return row


def make_summary_table(rows: list[dict[str, Any]]) -> str:
    """Build a Markdown table summarizing key metrics from each page.

    Args:
        rows: List of metric dictionaries for each page.

    Returns:
        Markdown formatted table string.
    """
    if not rows:
        return "(No data for summary table)\n\n"

    md = (
        "| Page | Onload (Curr) | Onload (Prev) | LCP (Curr) | LCP (Prev) | "
        "TBT (Curr) | TBT (Prev) | INP (Curr) | INP (Prev) | CLS (Curr) | CLS (Prev) |\n"
    )
    md += (
        "|------|---------------|---------------|------------|------------|"
        "------------|------------|------------|------------|------------|------------|\n"
    )
    for r in rows:
        page = r["page"]
        oc = r["onload_curr"] or 0
        op = r["onload_prev"] or 0
        lc = r["lcp_curr"] or 0
        lp = r["lcp_prev"] or 0
        tc = r["tbt_curr"] or 0
        tp = r["tbt_prev"] or 0
        ic = r["inp_curr"] or 0
        ip = r["inp_prev"] or 0
        cc = r["cls_curr"] or 0
        cp = r["cls_prev"] or 0
        md += (
            f"| {page} | {round(oc, 2)} | {round(op, 2)} | {round(lc, 2)} | "
            f"{round(lp, 2)} | {round(tc, 2)} | {round(tp, 2)} | {round(ic, 2)} | "
            f"{round(ip, 2)} | {round(cc, 2)} | {round(cp, 2)} |\n"
        )

    md += "\n"
    return md


# ========== OTHER STUFF: Revenue, hits, errors, etc. ==========


def get_event_markers() -> str:
    """Fetch event markers for the site.

    Returns:
        Markdown formatted event markers.
    """
    data = fetch_data(
        ENDPOINTS["event_markers"], method="GET", params={"prefix": SITE_PREFIX}
    )
    if not validate_api_response(data, "data"):
        return "> ⚠️ No event markers found.\n"
    df = pd.DataFrame(data["data"])
    if df.empty:
        return "> ⚠️ No event markers available.\n"
    lines = [
        f"- {row['eventName']}: {row['annotation']} "
        f"({row['eventStart']} - {row.get('eventEnd', 'N/A')})"
        for _, row in df.iterrows()
    ]
    return "### 📅 Event Markers\n" + "\n".join(lines)


def get_js_errors(page_name: str) -> str:
    """Retrieve aggregated JavaScript error data.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Markdown formatted error report.
    """
    global now, one_day_ago
    payload = {
        "site": SITE_PREFIX,
        "start": one_day_ago,
        "end": now,
        "dataType": "rum",
        "dataColumns": ["errorCount"],
        "pageName": page_name,
        "group": ["time", "errorConstructor"],
        "order": "time",
        "sort": "asc",
        "limit": 50000,
    }
    data = fetch_data(ENDPOINTS["error"], payload, method="POST")
    if not data or not isinstance(data, list) or len(data) == 0:
        return f"> ⚠️ No aggregated JS error data available for **{page_name}**.\n"

    lines = [f"### ⚠️ Aggregated JS Errors ({page_name})"]
    for record in data:
        time_epoch = record.get("time", "N/A")
        error_constructor = record.get("errorConstructor", "N/A")
        error_count = record.get("errorCount", "N/A")
        lines.append(f"- Time: {time_epoch} | {error_constructor}: {error_count} errors")

    return "\n".join(lines)


def summarize_resource_usage(
    current_df: pd.DataFrame, prev_df: pd.DataFrame
) -> str:
    """Summarize resource usage changes between periods.

    Args:
        current_df: Current period resource data.
        prev_df: Previous period resource data.

    Returns:
        Markdown formatted summary.
    """
    if current_df.empty or prev_df.empty:
        return "> ⚠️ Cannot summarize resource usage (no data)."

    current_map = dict(zip(current_df["domain"], current_df["duration"]))
    prev_map = dict(zip(prev_df["domain"], prev_df["duration"]))

    changes: list[tuple[str, float]] = []

    for domain in current_map:
        if domain in prev_map:
            c_val = _to_float(current_map[domain])
            p_val = _to_float(prev_map[domain])
            if c_val is not None and p_val is not None:
                diff = c_val - p_val
                changes.append((domain, diff))

    changes.sort(key=lambda x: x[1], reverse=True)

    slowdowns = [d for d in changes if d[1] > 0][:3]
    speedups = sorted([d for d in changes if d[1] < 0], key=lambda x: x[1])[:3]

    def format_changes(lst: list[tuple[str, float]]) -> str:
        return (
            ", ".join(
                [f"**{d}** ({'+' if v > 0 else ''}{round(v, 2)} ms)" for d, v in lst]
            )
            or "(none)"
        )

    summary = "### 📝 Resource Summary\n"
    summary += "#### Top Slowdowns\n"
    summary += f"- {format_changes(slowdowns)}\n"
    summary += "#### Top Speedups\n"
    summary += f"- {format_changes(speedups)}\n"

    return summary


def get_resource_data(page_name: str, compare_previous: bool = True) -> str:
    """Fetch resource data for a page.

    Args:
        page_name: Name of the page to analyze.
        compare_previous: Whether to compare with previous period.

    Returns:
        Markdown formatted resource report.
    """
    global now, one_day_ago, two_days_ago

    payload_curr = {
        "site": SITE_PREFIX,
        "start": one_day_ago,
        "end": now,
        "dataColumns": ["duration", "elementCount"],
        "group": ["domain"],
        "pageName[]": [page_name],
    }

    payload_prev = dict(payload_curr)
    if compare_previous and two_days_ago is not None:
        payload_prev["start"] = two_days_ago
        payload_prev["end"] = one_day_ago

    data_curr = fetch_data(ENDPOINTS["resource"], payload_curr)
    if not validate_api_response(data_curr, "data"):
        return f"> ⚠️ No resource data for **{page_name}**.\n"

    df_curr = pd.DataFrame(data_curr["data"])
    if df_curr.empty:
        return f"> ⚠️ Resource data empty for **{page_name}**.\n"

    resource_text = f"### 📦 Resource Usage ({page_name})\n"
    lines = [
        f"- {row['domain']}: {row['duration']} ms, {row['elementCount']} elements"
        for _, row in df_curr.iterrows()
    ]
    resource_text += "\n".join(lines)

    if compare_previous and two_days_ago is not None:
        data_prev = fetch_data(ENDPOINTS["resource"], payload_prev)
        if validate_api_response(data_prev, "data"):
            df_prev = pd.DataFrame(data_prev["data"])
            if not df_prev.empty:
                summary = summarize_resource_usage(df_curr, df_prev)
                resource_text = summary + "\n\n" + resource_text

    return resource_text


def get_performance_hits(page_name: str) -> str:
    """Fetch performance hits data.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Markdown formatted performance hits report.
    """
    global now, one_day_ago

    payload = {
        "site": SITE_PREFIX,
        "start": one_day_ago,
        "end": now,
        "dataType": "rum",
        "dataColumns": ["measurementTime", "httpCode", "url"],
        "pageName[]": [page_name],
        "limit": 1000,
    }

    data = fetch_data(ENDPOINTS["performance_hits"], payload, method="POST")

    if not validate_api_response(data, "data") or not data["data"]:
        return f"> ⚠️ No performance data for **{page_name}**.\n"

    df = pd.DataFrame(data["data"])
    if df.empty:
        return f"> ⚠️ No performance hits data found for **{page_name}**."

    urls = df["url"].value_counts().head(5)
    return f"### 🔗 Top URLs for {page_name}\n" + "\n".join(
        [f"- {u} ({c} hits)" for u, c in urls.items()]
    )


# ========== REVENUE REPORT ==========


def get_latest_revenue_date() -> int | None:
    """Helper to find the latest revenue date.

    Returns:
        Report date as epoch timestamp or None.
    """
    data = fetch_data(
        ENDPOINTS["revenue_opportunity_report"],
        method="GET",
        params={"prefix": SITE_PREFIX, "salesType": "revenue", "latest": "true"},
    )

    logger.debug("Latest Revenue Date Response: %s", data)

    if data and isinstance(data, list) and len(data) > 0:
        if "reportDate" in data[0]:
            return data[0]["reportDate"]

    return None


def get_latest_revenue_opportunity_date() -> int | None:
    """Fetch the latest report date for revenue opportunity data.

    Returns:
        Report date as epoch timestamp or None.
    """
    params = {
        "prefix": SITE_PREFIX,
        "salesType": "revenue",
        "latest": "true",
    }
    data = fetch_data(ENDPOINTS["revenue_opportunity_report"], method="GET", params=params)
    logger.debug("Latest Revenue Opportunity Date Response: %s", data)
    if data and isinstance(data, list) and len(data) > 0:
        if "reportDate" in data[0]:
            return data[0]["reportDate"]
    return None


def get_revenue_opportunity(
    page_name: str | None = None, device: list[str] | None = None
) -> str:
    """Fetch the Revenue Opportunity report for the given page.

    Args:
        page_name: Name of the page to analyze.
        device: List of device types to filter by.

    Returns:
        Markdown formatted revenue opportunity report.
    """
    report_date = get_latest_revenue_opportunity_date()
    logger.debug("Revenue Opportunity Report Date: %s", report_date)
    if not report_date:
        return "> ⚠️ No Revenue Opportunity report found.\n"

    params: dict[str, Any] = {
        "prefix": SITE_PREFIX,
        "salesType": "revenue",
        "reportDate": report_date,
    }
    if page_name:
        params["pageName[]"] = [page_name]
    if device:
        params["device[]"] = device

    opp_data = fetch_data(ENDPOINTS["revenue_opportunity"], method="GET", params=params)
    logger.debug("Revenue Opportunity Data Response: %s", opp_data)
    if not opp_data:
        return "> ⚠️ No Revenue Opportunity data found. (Empty response)\n"

    devices_to_check = device if device else list(opp_data.keys())

    total_lost_revenue = 0.0
    details: list[str] = []
    for dev in devices_to_check:
        dev_data = opp_data.get(dev)
        if not dev_data:
            continue
        revenue_section = dev_data.get("revenue", {})
        if page_name not in revenue_section:
            continue
        page_data = revenue_section[page_name]
        device_lost_revenue = 0.0

        try:
            if "speedUpToXData" in page_data and "lostRevenue" in page_data["speedUpToXData"]:
                arr = page_data["speedUpToXData"]["lostRevenue"]
                if arr and len(arr) > 0:
                    device_lost_revenue += float(arr[-1])
            if "speedUpByXData" in page_data and "lostRevenue" in page_data["speedUpByXData"]:
                arr = page_data["speedUpByXData"]["lostRevenue"]
                if arr and len(arr) > 0:
                    device_lost_revenue += float(arr[-1])
        except (ValueError, TypeError, KeyError) as exc:
            logger.error("Error converting lost revenue values: %s", exc)
            device_lost_revenue = 0.0

        total_lost_revenue += device_lost_revenue
        details.append(f"{dev}: ${device_lost_revenue}")

    if total_lost_revenue == 0.0 and not details:
        return f"> ⚠️ No Revenue Opportunity details available for **{page_name}**.\n"

    output = "### 💡 Revenue Opportunity Report\n"
    output += f"- Combined Lost Revenue Opportunity for **{page_name}**: ${total_lost_revenue}\n"
    if details:
        output += "\n**Breakdown by Device:**\n"
        for d in details:
            output += f"- {d}\n"
    return output


def get_page_revenue(page_name: str) -> str:
    """Fetch revenue metrics for a given page with previous day comparison.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Markdown formatted revenue report.
    """
    report_date = get_latest_revenue_date()
    logger.debug("Report Date: %s", report_date)

    if not report_date:
        return f"> ⚠️ No revenue data found for **{page_name}**.\n"

    params: dict[str, Any] = {
        "prefix": SITE_PREFIX,
        "salesType": "revenue",
        "reportDate": report_date,
        "pageName[]": [page_name],
    }

    data_today = fetch_data(ENDPOINTS["revenue_opportunity"], method="GET", params=params)

    if not data_today or not (
        data_today.get("Desktop") or data_today.get("Tablet") or data_today.get("Mobile")
    ):
        return f"> ⚠️ No revenue data found for **{page_name}**. (Check API response structure)\n"

    device_data = (
        data_today.get("Desktop")
        or data_today.get("Tablet")
        or data_today.get("Mobile")
    )
    if not device_data or page_name not in device_data.get("conversions", {}):
        return f"> ⚠️ No revenue data available for **{page_name}** on any device type.\n"

    df_today = pd.DataFrame(device_data["conversions"][page_name]["conversionsData"])
    if df_today.empty:
        return f"> ⚠️ Revenue data is empty for **{page_name}**.\n"

    latest_data = df_today.iloc[-1]

    text = f"""### 💰 Revenue Metrics for {page_name}
- Revenue: ${latest_data.get('revenue', 'N/A')}
- Orders: {latest_data.get('orders', 'N/A')}
- Visitors: {latest_data.get('visitors', 'N/A')}
"""

    previous_date = report_date - 86400
    params["reportDate"] = previous_date
    data_prev = fetch_data(ENDPOINTS["revenue_opportunity"], method="GET", params=params)
    if not data_prev or not (
        data_prev.get("Desktop") or data_prev.get("Tablet") or data_prev.get("Mobile")
    ):
        text += "\nNo previous revenue data for comparison."
        return text

    return text


# ========== BUILD PAGE REPORT ==========


def build_page_report(page_name: str) -> str:
    """Combine all reports for a single page.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Markdown formatted complete page report.
    """
    time_range_value = f"Time Range: {one_day_ago} to {now}"
    text = f"## 📄 Page: {page_name}\n{time_range_value}\n"
    text += get_page_performance(page_name) + "\n\n"
    text += get_resource_data(page_name, compare_previous=True) + "\n\n"
    text += get_page_revenue(page_name) + "\n\n"
    text += get_revenue_opportunity(page_name) + "\n\n"
    return text


def generate_full_report(pages: list[str]) -> str:
    """Generate complete report for multiple pages.

    Args:
        pages: List of page names to analyze.

    Returns:
        Markdown formatted complete report.
    """
    table_rows = []
    for pg in pages:
        row = gather_page_metrics(pg)
        if row is not None:
            table_rows.append(row)

    summary_table = make_summary_table(table_rows)
    big_md = "# 🔍 Blue Triangle API Report\n\n"
    big_md += "## Overall Summary Table\n"
    big_md += summary_table

    for pg in pages:
        big_md += build_page_report(pg)
        big_md += "---\n\n"

    big_md += get_event_markers() + "\n"
    return big_md


def generate_multi_range_report(ranges: list[str], pages: list[str]) -> str:
    """Generate report for multiple time ranges.

    Args:
        ranges: List of time range strings.
        pages: List of page names to analyze.

    Returns:
        Markdown formatted multi-range report.
    """
    sections = []
    for rng_str in ranges:
        s, e, ps, pe = compute_time_window(rng_str)
        global now, one_day_ago, two_days_ago
        now = e
        one_day_ago = s
        two_days_ago = ps

        sec_md = f"# Time Range: {rng_str}\n\n"
        sec_md += generate_full_report(pages)
        sections.append(sec_md)
    return "\n\n".join(sections)


def analyze_trends(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance trends over time.

    Args:
        data_frame: DataFrame with LCP, TBT, CLS, INP, FB columns.

    Returns:
        DataFrame with percentage changes.
    """
    trends = {
        "lcp": data_frame["LCP"].pct_change() * 100,
        "tbt": data_frame["TBT"].pct_change() * 100,
        "cls": data_frame["CLS"].pct_change() * 100,
        "inp": data_frame["INP"].pct_change() * 100,
        "fb": data_frame["FB"].pct_change() * 100,
        "date": data_frame["date"],
    }

    trends_df = pd.DataFrame(trends)
    trends_df = trends_df.set_index("date")
    trends_df = trends_df.dropna()
    return trends_df


def visualize_trends(trends: pd.DataFrame) -> None:
    """Visualize performance trends.

    Args:
        trends: DataFrame with trend data.
    """
    plt.figure(figsize=(10, 6))
    for metric in trends.columns:
        plt.plot(trends.index, trends[metric], label=metric)

    plt.title("Performance Metrics Trends")
    plt.ylabel("Percentage Change")
    plt.xlabel("Date")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ========== FILE IO ==========


def save_report(content: str, filename: str) -> None:
    """Save report to file.

    Args:
        content: Report content.
        filename: Output filename.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


# ========== MAIN ==========


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate a Blue Triangle performance & revenue report."
    )
    parser.add_argument(
        "--page",
        nargs="+",
        help="Specify one or more page names (e.g. PDP CDP Checkout).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="full_bluetriangle_report.md",
        help="Output filename",
    )
    parser.add_argument(
        "--time-range",
        choices=list(DAY_MAP.keys()),
        default="7d",
        help="Select one of the predefined time windows.",
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Custom start time (epoch) – overrides --time-range",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="Custom end time (epoch) – overrides --time-range",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["LCP", "TBT", "CLS", "INP", "FB"],
        help="Specify which metrics to include in the report.",
    )
    parser.add_argument(
        "--multi-range",
        type=str,
        help="Comma separated, e.g. '24h,28d,90d' for multiple sections",
    )
    parser.add_argument(
        "--top-pages",
        action="store_true",
        help="Generate report for top pages by page views",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    return parser.parse_args()


def validate_pages(
    input_pages: list[str], available_pages: list[str]
) -> list[str] | None:
    """Validate page names against available pages.

    Args:
        input_pages: List of user-provided page names.
        available_pages: List of valid page names.

    Returns:
        List of valid pages or None if none valid.
    """
    valid_pages = []
    for p in input_pages:
        if p in available_pages:
            valid_pages.append(p)
        else:
            logger.warning("Unknown page '%s'", p)
    return valid_pages if valid_pages else None


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Set up logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Set selected metrics filter
    global selected_metrics
    if args.metrics:
        selected_metrics = args.metrics

    global now, one_day_ago, two_days_ago
    start, end, prev_start, prev_end, multi_list = parse_time_args(args)
    now, one_day_ago, two_days_ago = end, start, prev_start

    if args.top_pages:
        df = fetch_top_page_names(limit=20)
        if df.empty or "pageName" not in df.columns:
            logger.error("Could not fetch top pages")
            return
        pages = df["pageName"].tolist()
    elif args.page:
        pages = validate_pages(args.page, AVAILABLE_PAGES)
        if pages is None:
            return
    else:
        pages = update_available_pages(limit=20)

    try:
        final_md = (
            generate_multi_range_report(multi_list, pages)
            if multi_list
            else generate_full_report(pages)
        )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_md)
        logger.info("Report saved to '%s'", args.output)
    except Exception as e:
        logger.error("Error saving report: %s", e)


if __name__ == "__main__":
    main()
