"""Blue Triangle CLI Reporter - Analyze RUM data and revenue impact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import pickle
import re
import sys
import time
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv

# Try to import yaml for config file support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

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


# ==================
# COLORED OUTPUT
# ==================


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    _enabled = True

    @classmethod
    def disable(cls) -> None:
        """Disable colored output."""
        cls._enabled = False

    @classmethod
    def enable(cls) -> None:
        """Enable colored output."""
        cls._enabled = True

    @classmethod
    def _wrap(cls, text: str, color: str) -> str:
        """Wrap text with color if enabled."""
        if cls._enabled and sys.stdout.isatty():
            return f"{color}{text}{cls.RESET}"
        return text

    @classmethod
    def success(cls, text: str) -> str:
        """Format success message (green)."""
        return cls._wrap(text, cls.GREEN)

    @classmethod
    def error(cls, text: str) -> str:
        """Format error message (red)."""
        return cls._wrap(text, cls.RED)

    @classmethod
    def warning(cls, text: str) -> str:
        """Format warning message (yellow)."""
        return cls._wrap(text, cls.YELLOW)

    @classmethod
    def info(cls, text: str) -> str:
        """Format info message (cyan)."""
        return cls._wrap(text, cls.CYAN)

    @classmethod
    def bold(cls, text: str) -> str:
        """Format bold text."""
        return cls._wrap(text, cls.BOLD)

    @classmethod
    def dim(cls, text: str) -> str:
        """Format dim text."""
        return cls._wrap(text, cls.DIM)


def print_banner() -> None:
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║           Blue Triangle CLI Reporter                     ║
║           Performance & Revenue Analysis                 ║
╚══════════════════════════════════════════════════════════╝
"""
    print(Colors.info(banner))


def print_success(message: str) -> None:
    """Print success message."""
    print(f"{Colors.success('✓')} {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"{Colors.error('✗')} {message}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"{Colors.warning('!')} {message}")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"{Colors.info('→')} {message}")


def print_progress(current: int, total: int, label: str = "") -> None:
    """Print progress indicator.

    Args:
        current: Current item number (1-indexed).
        total: Total number of items.
        label: Optional label to display.
    """
    width = 30
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    percent = int(100 * current / total)
    status = f"[{bar}] {percent}% ({current}/{total})"
    if label:
        status += f" - {label}"
    # Use carriage return to overwrite line
    print(f"\r{Colors.dim(status)}", end="", flush=True)
    if current == total:
        print()  # New line when complete

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

# Alert thresholds (None means no alerting)
alert_thresholds: dict[str, float] | None = None

# Cache settings
CACHE_DIR = Path.home() / ".bt_cache"
CACHE_TTL = int(os.getenv("BT_CACHE_TTL", "300"))  # 5 minutes default
cache_enabled = False


# ==================
# CONFIGURATION FILE
# ==================


def load_config_file(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches default locations.

    Returns:
        Configuration dictionary.
    """
    if not YAML_AVAILABLE:
        return {}

    # Default search paths
    search_paths = [
        Path(config_path) if config_path else None,
        Path("bt_config.yaml"),
        Path("bt_config.yml"),
        Path.home() / ".bt_config.yaml",
        Path.home() / ".config" / "bluetriangle" / "config.yaml",
    ]

    for path in search_paths:
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                    logger.debug("Loaded config from %s", path)
                    return config
            except Exception as e:
                logger.warning("Failed to load config from %s: %s", path, e)

    return {}


def apply_config(config: dict[str, Any]) -> None:
    """Apply configuration values to global settings.

    Args:
        config: Configuration dictionary.
    """
    global SITE_PREFIX, REQUEST_TIMEOUT, alert_thresholds, cache_enabled, CACHE_TTL

    # API settings
    if "api" in config:
        api_config = config["api"]
        if "email" in api_config and not os.getenv("BT_API_EMAIL"):
            os.environ["BT_API_EMAIL"] = api_config["email"]
            HEADERS["X-API-Email"] = api_config["email"]
        if "key" in api_config and not os.getenv("BT_API_KEY"):
            os.environ["BT_API_KEY"] = api_config["key"]
            HEADERS["X-API-Key"] = api_config["key"]
        if "site_prefix" in api_config and not os.getenv("BT_SITE_PREFIX"):
            os.environ["BT_SITE_PREFIX"] = api_config["site_prefix"]
            SITE_PREFIX = api_config["site_prefix"]
        if "timeout" in api_config:
            REQUEST_TIMEOUT = int(api_config["timeout"])

    # Cache settings
    if "cache" in config:
        cache_config = config["cache"]
        cache_enabled = cache_config.get("enabled", False)
        if "ttl" in cache_config:
            CACHE_TTL = int(cache_config["ttl"])

    # Alert thresholds
    if "thresholds" in config:
        alert_thresholds = config["thresholds"]


# ==================
# CACHING
# ==================


def get_cache_key(endpoint: str, payload: dict[str, Any] | None) -> str:
    """Generate cache key from endpoint and payload.

    Args:
        endpoint: API endpoint.
        payload: Request payload.

    Returns:
        Cache key string.
    """
    key_data = f"{endpoint}:{json.dumps(payload, sort_keys=True)}"
    return hashlib.md5(key_data.encode()).hexdigest()


def get_cached_response(cache_key: str) -> dict[str, Any] | list[Any] | None:
    """Retrieve cached response if valid.

    Args:
        cache_key: Cache key.

    Returns:
        Cached data or None if not found/expired.
    """
    if not cache_enabled:
        return None

    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)

        if time.time() - cached["timestamp"] > CACHE_TTL:
            cache_file.unlink()  # Remove expired cache
            return None

        logger.debug("Cache hit for %s", cache_key)
        return cached["data"]
    except Exception:
        return None


def set_cached_response(
    cache_key: str, data: dict[str, Any] | list[Any]
) -> None:
    """Store response in cache.

    Args:
        cache_key: Cache key.
        data: Data to cache.
    """
    if not cache_enabled:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    try:
        with open(cache_file, "wb") as f:
            pickle.dump({"timestamp": time.time(), "data": data}, f)
        logger.debug("Cached response for %s", cache_key)
    except Exception as e:
        logger.warning("Failed to cache response: %s", e)


def clear_cache() -> int:
    """Clear all cached responses.

    Returns:
        Number of files removed.
    """
    if not CACHE_DIR.exists():
        return 0

    count = 0
    for cache_file in CACHE_DIR.glob("*.pkl"):
        try:
            cache_file.unlink()
            count += 1
        except Exception:
            pass

    return count


# ==================
# ALERTING / THRESHOLDS
# ==================


# Default thresholds based on Web Vitals recommendations
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "LCP": {"good": 2500, "poor": 4000},
    "INP": {"good": 200, "poor": 500},
    "CLS": {"good": 0.1, "poor": 0.25},
    "TBT": {"good": 200, "poor": 600},
    "FB": {"good": 800, "poor": 1800},
}


def check_threshold(
    metric: str, value: float | None
) -> tuple[str, str | None]:
    """Check if metric value exceeds threshold.

    Args:
        metric: Metric name (LCP, INP, CLS, etc.).
        value: Metric value.

    Returns:
        Tuple of (status, alert_message).
        Status is 'good', 'needs-improvement', 'poor', or 'unknown'.
    """
    if value is None:
        return "unknown", None

    thresholds = alert_thresholds or DEFAULT_THRESHOLDS
    if metric not in thresholds:
        return "unknown", None

    thresh = thresholds[metric]
    good_thresh = thresh.get("good", float("inf"))
    poor_thresh = thresh.get("poor", float("inf"))

    if value <= good_thresh:
        return "good", None
    elif value <= poor_thresh:
        return "needs-improvement", f"{metric} ({value}) needs improvement (threshold: {good_thresh})"
    else:
        return "poor", f"{metric} ({value}) is POOR (threshold: {poor_thresh})"


def get_threshold_alerts(metrics: dict[str, Any]) -> list[str]:
    """Get all threshold alerts for a set of metrics.

    Args:
        metrics: Dictionary of metric values.

    Returns:
        List of alert messages.
    """
    alerts = []

    metric_mapping = {
        "largestContentfulPaint": "LCP",
        "intToNextPaint": "INP",
        "cumulativeLayoutShift": "CLS",
        "totalBlockingTime": "TBT",
        "firstByte": "FB",
    }

    for api_key, label in metric_mapping.items():
        value = metrics.get(api_key)
        if value is not None:
            try:
                value = float(value)
                status, alert = check_threshold(label, value)
                if alert:
                    alerts.append(alert)
            except (ValueError, TypeError):
                pass

    return alerts


# ==================
# CREDENTIAL VALIDATION
# ==================


def validate_credentials() -> tuple[bool, list[str]]:
    """Validate that required API credentials are configured.

    Returns:
        Tuple of (is_valid, list of missing credential names).
    """
    missing = []

    if not os.getenv("BT_API_EMAIL"):
        missing.append("BT_API_EMAIL")
    if not os.getenv("BT_API_KEY"):
        missing.append("BT_API_KEY")
    if not os.getenv("BT_SITE_PREFIX"):
        missing.append("BT_SITE_PREFIX")

    return len(missing) == 0, missing


def test_api_connection() -> tuple[bool, str]:
    """Test the API connection with current credentials.

    Returns:
        Tuple of (success, message).
    """
    is_valid, missing = validate_credentials()
    if not is_valid:
        return False, f"Missing credentials: {', '.join(missing)}"

    # Try to make a simple API call
    try:
        url = BASE_URL + ENDPOINTS["performance"]
        payload = {
            "site": SITE_PREFIX,
            "start": int(time.time()) - 86400,
            "end": int(time.time()),
            "dataColumns": ["pageViews"],
            "group": ["pageName"],
            "limit": 1,
        }
        response = requests.post(
            url, headers=HEADERS, json=payload, timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:
            return True, "API connection successful"
        elif response.status_code == 401:
            return False, "Authentication failed - check your API key and email"
        elif response.status_code == 403:
            return False, "Access denied - check your site prefix and permissions"
        else:
            return False, f"API returned status {response.status_code}: {response.text[:200]}"

    except requests.exceptions.Timeout:
        return False, f"Connection timed out after {REQUEST_TIMEOUT}s - check your network"
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to API - check your internet connection"
    except Exception as e:
        return False, f"Connection test failed: {str(e)}"


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
    use_cache: bool = True,
) -> dict[str, Any] | list[Any] | None:
    """Generic function to fetch JSON from the Blue Triangle API.

    Args:
        endpoint: API endpoint path.
        payload: JSON payload for POST requests.
        method: HTTP method (GET or POST).
        params: Query parameters for GET requests.
        use_cache: Whether to use caching for this request.

    Returns:
        Parsed JSON response or None on error.
    """
    # Check cache first
    cache_data = payload if method == "POST" else params
    cache_key = get_cache_key(endpoint, cache_data)

    if use_cache:
        cached = get_cached_response(cache_key)
        if cached is not None:
            return cached

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
        data = r.json()

        # Cache successful response
        if use_cache and data is not None:
            set_cached_response(cache_key, data)

        return data
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


def generate_full_report(
    pages: list[str],
    show_progress: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate complete report for multiple pages.

    Args:
        pages: List of page names to analyze.
        show_progress: Whether to show progress indicators.

    Returns:
        Tuple of (Markdown formatted report, list of metric rows for export).
    """
    table_rows: list[dict[str, Any]] = []
    total = len(pages)

    if show_progress:
        print_info(f"Gathering metrics for {total} page(s)...")

    for i, pg in enumerate(pages, 1):
        if show_progress:
            print_progress(i, total, pg)
        row = gather_page_metrics(pg)
        if row is not None:
            table_rows.append(row)

    summary_table = make_summary_table(table_rows)
    big_md = "# 🔍 Blue Triangle API Report\n\n"
    big_md += "## Overall Summary Table\n"
    big_md += summary_table

    if show_progress:
        print_info("Building detailed page reports...")

    for i, pg in enumerate(pages, 1):
        if show_progress:
            print_progress(i, total, pg)
        big_md += build_page_report(pg)
        big_md += "---\n\n"

    big_md += get_event_markers() + "\n"
    return big_md, table_rows


def generate_multi_range_report(
    ranges: list[str],
    pages: list[str],
    show_progress: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate report for multiple time ranges.

    Args:
        ranges: List of time range strings.
        pages: List of page names to analyze.
        show_progress: Whether to show progress indicators.

    Returns:
        Tuple of (Markdown formatted report, all metric rows).
    """
    sections = []
    all_rows: list[dict[str, Any]] = []

    for rng_str in ranges:
        if show_progress:
            print_info(f"Processing time range: {rng_str}")

        s, e, ps, pe = compute_time_window(rng_str)
        global now, one_day_ago, two_days_ago
        now = e
        one_day_ago = s
        two_days_ago = ps

        sec_md = f"# Time Range: {rng_str}\n\n"
        report_md, rows = generate_full_report(pages, show_progress=show_progress)
        sec_md += report_md

        # Add time range to each row for context
        for row in rows:
            row["time_range"] = rng_str
        all_rows.extend(rows)

        sections.append(sec_md)

    return "\n\n".join(sections), all_rows


# ========== EXPORT FUNCTIONS ==========


def export_to_json(data: list[dict[str, Any]], output_file: str) -> None:
    """Export metrics data to JSON format.

    Args:
        data: List of metric dictionaries.
        output_file: Output file path.
    """
    # Ensure .json extension
    if not output_file.endswith(".json"):
        output_file = output_file.rsplit(".", 1)[0] + ".json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_to_csv(data: list[dict[str, Any]], output_file: str) -> None:
    """Export metrics data to CSV format.

    Args:
        data: List of metric dictionaries.
        output_file: Output file path.
    """
    if not data:
        return

    # Ensure .csv extension
    if not output_file.endswith(".csv"):
        output_file = output_file.rsplit(".", 1)[0] + ".csv"

    fieldnames = data[0].keys()
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def export_to_html(
    data: list[dict[str, Any]],
    markdown_content: str,
    output_file: str,
) -> None:
    """Export report as HTML with embedded charts.

    Args:
        data: List of metric dictionaries.
        markdown_content: Markdown report content.
        output_file: Output file path.
    """
    if not output_file.endswith(".html"):
        output_file = output_file.rsplit(".", 1)[0] + ".html"

    # Generate chart data
    chart_data = []
    for row in data:
        chart_data.append({
            "page": row.get("page", "Unknown"),
            "lcp_curr": row.get("lcp_curr", 0) or 0,
            "lcp_prev": row.get("lcp_prev", 0) or 0,
            "inp_curr": row.get("inp_curr", 0) or 0,
            "inp_prev": row.get("inp_prev", 0) or 0,
            "cls_curr": (row.get("cls_curr", 0) or 0) * 1000,  # Scale for visibility
            "cls_prev": (row.get("cls_prev", 0) or 0) * 1000,
            "tbt_curr": row.get("tbt_curr", 0) or 0,
            "tbt_prev": row.get("tbt_prev", 0) or 0,
        })

    # Convert markdown to basic HTML (simple conversion)
    html_body = markdown_content
    # Convert headers
    html_body = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_body, flags=re.MULTILINE)
    html_body = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_body, flags=re.MULTILINE)
    html_body = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_body, flags=re.MULTILINE)
    html_body = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html_body, flags=re.MULTILINE)
    # Convert bold
    html_body = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_body)
    # Convert list items
    html_body = re.sub(r'^- (.+)$', r'<li>\1</li>', html_body, flags=re.MULTILINE)
    # Convert line breaks
    html_body = html_body.replace('\n\n', '</p><p>')
    html_body = f'<p>{html_body}</p>'

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blue Triangle Performance Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --accent: #0f4c75;
            --success: #00d9a5;
            --warning: #ffc107;
            --danger: #e74c3c;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #fff; margin-bottom: 1rem; font-size: 2rem; }}
        h2 {{ color: #3498db; margin: 2rem 0 1rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }}
        h3 {{ color: #9b59b6; margin: 1.5rem 0 0.5rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .chart-container {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1rem;
            height: 300px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{ background: var(--accent); color: #fff; }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        .metric-good {{ color: var(--success); }}
        .metric-warning {{ color: var(--warning); }}
        .metric-poor {{ color: var(--danger); }}
        .report-content {{ background: var(--card-bg); padding: 2rem; border-radius: 12px; }}
        .report-content p {{ margin-bottom: 1rem; }}
        .report-content li {{ margin-left: 1.5rem; margin-bottom: 0.5rem; }}
        .timestamp {{ color: #888; font-size: 0.9rem; margin-bottom: 2rem; }}
        .alerts {{ background: #2c1810; border-left: 4px solid var(--danger); padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0; }}
        .alerts h4 {{ color: var(--danger); margin-bottom: 0.5rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Blue Triangle Performance Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="charts-grid">
            <div class="chart-container">
                <canvas id="lcpChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="inpChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="clsChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="tbtChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Metrics Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Page</th>
                        <th>LCP (ms)</th>
                        <th>INP (ms)</th>
                        <th>CLS</th>
                        <th>TBT (ms)</th>
                    </tr>
                </thead>
                <tbody id="metricsTable"></tbody>
            </table>
        </div>

        <div class="report-content">
            {html_body}
        </div>
    </div>

    <script>
        const chartData = {json.dumps(chart_data)};

        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ labels: {{ color: '#eee' }} }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#eee' }}, grid: {{ color: '#333' }} }},
                y: {{ ticks: {{ color: '#eee' }}, grid: {{ color: '#333' }} }}
            }}
        }};

        function createChart(id, label, currKey, prevKey, unit) {{
            const ctx = document.getElementById(id).getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: chartData.map(d => d.page),
                    datasets: [
                        {{
                            label: 'Current ' + label,
                            data: chartData.map(d => d[currKey]),
                            backgroundColor: 'rgba(52, 152, 219, 0.8)',
                        }},
                        {{
                            label: 'Previous ' + label,
                            data: chartData.map(d => d[prevKey]),
                            backgroundColor: 'rgba(155, 89, 182, 0.6)',
                        }}
                    ]
                }},
                options: {{ ...chartOptions, plugins: {{ ...chartOptions.plugins, title: {{ display: true, text: label + ' (' + unit + ')', color: '#fff' }} }} }}
            }});
        }}

        createChart('lcpChart', 'LCP', 'lcp_curr', 'lcp_prev', 'ms');
        createChart('inpChart', 'INP', 'inp_curr', 'inp_prev', 'ms');
        createChart('clsChart', 'CLS', 'cls_curr', 'cls_prev', 'x1000');
        createChart('tbtChart', 'TBT', 'tbt_curr', 'tbt_prev', 'ms');

        // Populate table
        const tbody = document.getElementById('metricsTable');
        chartData.forEach(row => {{
            const tr = document.createElement('tr');
            const getClass = (val, good, poor) => val <= good ? 'metric-good' : val <= poor ? 'metric-warning' : 'metric-poor';
            tr.innerHTML = `
                <td>${{row.page}}</td>
                <td class="${{getClass(row.lcp_curr, 2500, 4000)}}">${{row.lcp_curr.toFixed(0)}}</td>
                <td class="${{getClass(row.inp_curr, 200, 500)}}">${{row.inp_curr.toFixed(0)}}</td>
                <td class="${{getClass(row.cls_curr/1000, 0.1, 0.25)}}">${{(row.cls_curr/1000).toFixed(3)}}</td>
                <td class="${{getClass(row.tbt_curr, 200, 600)}}">${{row.tbt_curr.toFixed(0)}}</td>
            `;
            tbody.appendChild(tr);
        }});
    </script>
</body>
</html>"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)


# ========== COMPARISON MODE ==========


def compare_time_periods(
    pages: list[str],
    period1_start: int,
    period1_end: int,
    period2_start: int,
    period2_end: int,
    show_progress: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Compare metrics between two arbitrary time periods.

    Args:
        pages: List of page names to analyze.
        period1_start: Start of first period (epoch).
        period1_end: End of first period (epoch).
        period2_start: Start of second period (epoch).
        period2_end: End of second period (epoch).
        show_progress: Whether to show progress indicators.

    Returns:
        Tuple of (Markdown report, comparison data).
    """
    global now, one_day_ago, two_days_ago

    comparison_data = []

    if show_progress:
        print_info("Fetching Period 1 data...")

    # Fetch period 1 data
    now = period1_end
    one_day_ago = period1_start
    two_days_ago = None

    period1_metrics = {}
    for pg in pages:
        row = gather_page_metrics(pg)
        if row:
            period1_metrics[pg] = row

    if show_progress:
        print_info("Fetching Period 2 data...")

    # Fetch period 2 data
    now = period2_end
    one_day_ago = period2_start

    period2_metrics = {}
    for pg in pages:
        row = gather_page_metrics(pg)
        if row:
            period2_metrics[pg] = row

    # Build comparison report
    p1_start_str = datetime.fromtimestamp(period1_start).strftime('%Y-%m-%d %H:%M')
    p1_end_str = datetime.fromtimestamp(period1_end).strftime('%Y-%m-%d %H:%M')
    p2_start_str = datetime.fromtimestamp(period2_start).strftime('%Y-%m-%d %H:%M')
    p2_end_str = datetime.fromtimestamp(period2_end).strftime('%Y-%m-%d %H:%M')

    md = "# Time Period Comparison Report\n\n"
    md += f"**Period 1:** {p1_start_str} to {p1_end_str}\n\n"
    md += f"**Period 2:** {p2_start_str} to {p2_end_str}\n\n"

    md += "## Comparison Summary\n\n"
    md += "| Page | Metric | Period 1 | Period 2 | Change | % Change |\n"
    md += "|------|--------|----------|----------|--------|----------|\n"

    metrics_to_compare = [
        ("lcp_curr", "LCP"),
        ("inp_curr", "INP"),
        ("cls_curr", "CLS"),
        ("tbt_curr", "TBT"),
        ("fb_curr", "First Byte"),
        ("onload_curr", "Onload"),
    ]

    for pg in pages:
        p1 = period1_metrics.get(pg, {})
        p2 = period2_metrics.get(pg, {})

        row_data = {"page": pg}

        for metric_key, metric_label in metrics_to_compare:
            v1 = p1.get(metric_key)
            v2 = p2.get(metric_key)

            row_data[f"p1_{metric_key}"] = v1
            row_data[f"p2_{metric_key}"] = v2

            if v1 is not None and v2 is not None:
                change = v2 - v1
                pct_change = ((v2 - v1) / v1 * 100) if v1 != 0 else 0
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                md += f"| {pg} | {metric_label} | {v1:.2f} | {v2:.2f} | {arrow} {abs(change):.2f} | {pct_change:+.1f}% |\n"
                row_data[f"change_{metric_key}"] = change
                row_data[f"pct_{metric_key}"] = pct_change
            else:
                md += f"| {pg} | {metric_label} | N/A | N/A | - | - |\n"

        comparison_data.append(row_data)

    return md, comparison_data


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
        description="Generate a Blue Triangle performance & revenue report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --page pdp checkout --time-range 7d
  %(prog)s --top-pages --time-range 28d --output monthly_report.md
  %(prog)s --page homepage --metrics LCP INP --format json
  %(prog)s --test-connection

Environment Variables:
  BT_API_EMAIL       Your Blue Triangle account email
  BT_API_KEY         Your Blue Triangle API key
  BT_SITE_PREFIX     Your site prefix
  BT_REQUEST_TIMEOUT Request timeout in seconds (default: 30)
  BT_LOG_LEVEL       Log level: DEBUG, INFO, WARNING, ERROR
""",
    )

    # Page selection
    page_group = parser.add_argument_group("Page Selection")
    page_group.add_argument(
        "--page",
        nargs="+",
        help="Specify one or more page names (e.g. PDP CDP Checkout)",
    )
    page_group.add_argument(
        "--top-pages",
        action="store_true",
        help="Generate report for top 20 pages by page views",
    )

    # Time range
    time_group = parser.add_argument_group("Time Range")
    time_group.add_argument(
        "--time-range",
        choices=list(DAY_MAP.keys()),
        default="7d",
        help="Select a predefined time window (default: 7d)",
    )
    time_group.add_argument(
        "--start",
        type=int,
        help="Custom start time (epoch timestamp) - overrides --time-range",
    )
    time_group.add_argument(
        "--end",
        type=int,
        help="Custom end time (epoch timestamp) - overrides --time-range",
    )
    time_group.add_argument(
        "--multi-range",
        type=str,
        help="Generate multiple reports, e.g. '24h,7d,28d'",
    )

    # Time period comparison
    time_group.add_argument(
        "--compare",
        nargs=4,
        type=int,
        metavar=("P1_START", "P1_END", "P2_START", "P2_END"),
        help="Compare two time periods (4 epoch timestamps)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default="full_bluetriangle_report.md",
        help="Output filename (default: full_bluetriangle_report.md)",
    )
    output_group.add_argument(
        "--format", "-f",
        choices=["markdown", "json", "csv", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    output_group.add_argument(
        "--metrics",
        nargs="+",
        choices=["LCP", "TBT", "CLS", "INP", "FB"],
        help="Filter report to specific metrics",
    )

    # Advanced options
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    advanced_group.add_argument(
        "--cache",
        action="store_true",
        help="Enable API response caching",
    )
    advanced_group.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear API cache and exit",
    )
    advanced_group.add_argument(
        "--alerts",
        action="store_true",
        help="Show threshold alerts for metrics exceeding Web Vitals limits",
    )
    advanced_group.add_argument(
        "--generate-completion",
        choices=["bash", "zsh"],
        help="Generate shell completion script and exit",
    )

    # Utility options
    util_group = parser.add_argument_group("Utility Options")
    util_group.add_argument(
        "--test-connection",
        action="store_true",
        help="Test API connection and exit",
    )
    util_group.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    util_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    util_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    return parser.parse_args()


def generate_shell_completion(shell: str) -> str:
    """Generate shell completion script.

    Args:
        shell: Shell type ('bash' or 'zsh').

    Returns:
        Completion script content.
    """
    if shell == "bash":
        return '''# Bash completion for bt_insights.py
# Add this to your .bashrc or .bash_completion

_bt_insights_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="--page --top-pages --time-range --start --end --multi-range --compare --output --format --metrics --config --cache --clear-cache --alerts --generate-completion --test-connection --no-color --quiet --verbose --help"

    case "${prev}" in
        --time-range)
            COMPREPLY=( $(compgen -W "qd hd 24h xd 2d 6d 7d 28d 30d 90d 1y 2y 3y" -- ${cur}) )
            return 0
            ;;
        --format|-f)
            COMPREPLY=( $(compgen -W "markdown json csv html" -- ${cur}) )
            return 0
            ;;
        --metrics)
            COMPREPLY=( $(compgen -W "LCP TBT CLS INP FB" -- ${cur}) )
            return 0
            ;;
        --generate-completion)
            COMPREPLY=( $(compgen -W "bash zsh" -- ${cur}) )
            return 0
            ;;
        --output|-o|--config)
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
    esac

    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}

complete -F _bt_insights_completions bt_insights.py
complete -F _bt_insights_completions python bt_insights.py
'''

    elif shell == "zsh":
        return '''#compdef bt_insights.py python

# Zsh completion for bt_insights.py
# Add this to your .zshrc or place in your fpath

_bt_insights() {
    local -a opts time_ranges formats metrics shells

    time_ranges=(qd hd 24h xd 2d 6d 7d 28d 30d 90d 1y 2y 3y)
    formats=(markdown json csv html)
    metrics=(LCP TBT CLS INP FB)
    shells=(bash zsh)

    _arguments -C \\
        '--page[Specify page names]:page:' \\
        '--top-pages[Analyze top pages by views]' \\
        '--time-range[Time window]:range:($time_ranges)' \\
        '--start[Custom start time (epoch)]:timestamp:' \\
        '--end[Custom end time (epoch)]:timestamp:' \\
        '--multi-range[Multiple time ranges]:ranges:' \\
        '--compare[Compare two periods]:timestamps:' \\
        {-o,--output}'[Output filename]:file:_files' \\
        {-f,--format}'[Output format]:format:($formats)' \\
        '--metrics[Filter metrics]:metrics:($metrics)' \\
        '--config[Config file path]:file:_files -g "*.yaml *.yml"' \\
        '--cache[Enable caching]' \\
        '--clear-cache[Clear cache]' \\
        '--alerts[Show threshold alerts]' \\
        '--generate-completion[Generate completion script]:shell:($shells)' \\
        '--test-connection[Test API connection]' \\
        '--no-color[Disable colors]' \\
        {-q,--quiet}'[Suppress progress]' \\
        {-v,--verbose}'[Enable debug logging]' \\
        '--help[Show help]'
}

compdef _bt_insights bt_insights.py
compdef _bt_insights "python bt_insights.py"
'''

    return ""


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


def print_summary(
    pages_count: int,
    rows_count: int,
    output_file: str,
    output_format: str,
    elapsed_time: float,
) -> None:
    """Print completion summary.

    Args:
        pages_count: Number of pages analyzed.
        rows_count: Number of data rows collected.
        output_file: Output file path.
        output_format: Output format used.
        elapsed_time: Time taken in seconds.
    """
    print()
    print(Colors.bold("═" * 50))
    print(Colors.bold("  Report Complete"))
    print(Colors.bold("═" * 50))
    print(f"  Pages analyzed:  {Colors.info(str(pages_count))}")
    print(f"  Data rows:       {Colors.info(str(rows_count))}")
    print(f"  Output format:   {Colors.info(output_format)}")
    print(f"  Output file:     {Colors.success(output_file)}")
    print(f"  Time elapsed:    {Colors.dim(f'{elapsed_time:.2f}s')}")
    print(Colors.bold("═" * 50))
    print()


def main() -> None:
    """Main entry point."""
    global cache_enabled

    args = parse_arguments()
    start_time = time.time()

    # Handle color settings
    if args.no_color:
        Colors.disable()

    # Set up logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    show_progress = not args.quiet

    # Handle --generate-completion (no banner needed)
    if args.generate_completion:
        print(generate_shell_completion(args.generate_completion))
        sys.exit(0)

    # Handle --clear-cache
    if args.clear_cache:
        count = clear_cache()
        print_success(f"Cleared {count} cached responses")
        sys.exit(0)

    # Show banner unless quiet mode
    if show_progress:
        print_banner()

    # Load and apply config file
    if args.config or Path("bt_config.yaml").exists() or Path("bt_config.yml").exists():
        config = load_config_file(args.config)
        if config:
            apply_config(config)
            if show_progress:
                print_info("Loaded configuration file")

    # Enable caching if requested
    if args.cache:
        cache_enabled = True
        if show_progress:
            print_info(f"Caching enabled (TTL: {CACHE_TTL}s)")

    # Handle --test-connection
    if args.test_connection:
        print_info("Testing API connection...")
        success, message = test_api_connection()
        if success:
            print_success(message)
            sys.exit(0)
        else:
            print_error(message)
            sys.exit(1)

    # Validate credentials before proceeding
    is_valid, missing = validate_credentials()
    if not is_valid:
        print_error("Missing required credentials:")
        for cred in missing:
            print(f"  - {Colors.warning(cred)}")
        print()
        print_info("Please set these in your .env file or as environment variables.")
        print_info("See .env.example for a template.")
        sys.exit(1)

    if show_progress:
        print_success("Credentials configured")

    # Set selected metrics filter
    global selected_metrics
    if args.metrics:
        selected_metrics = args.metrics
        if show_progress:
            print_info(f"Filtering metrics: {', '.join(args.metrics)}")

    global now, one_day_ago, two_days_ago
    start, end, prev_start, prev_end, multi_list = parse_time_args(args)
    now, one_day_ago, two_days_ago = end, start, prev_start

    if show_progress:
        if args.compare:
            print_info("Mode: Time period comparison")
        elif multi_list:
            print_info(f"Time ranges: {', '.join(multi_list)}")
        else:
            print_info(f"Time range: {args.time_range}")

    # Determine which pages to analyze
    if args.top_pages:
        if show_progress:
            print_info("Fetching top pages by page views...")
        df = fetch_top_page_names(limit=20)
        if df.empty or "pageName" not in df.columns:
            print_error("Could not fetch top pages from API")
            print_info("Check your credentials and network connection")
            sys.exit(1)
        pages = df["pageName"].tolist()
        if show_progress:
            print_success(f"Found {len(pages)} pages")
    elif args.page:
        pages = validate_pages(args.page, AVAILABLE_PAGES)
        if pages is None:
            print_error("No valid pages specified")
            print_info("Check page names and try again")
            sys.exit(1)
    else:
        if show_progress:
            print_info("Fetching available pages...")
        pages = update_available_pages(limit=20)
        if show_progress:
            print_success(f"Found {len(pages)} pages")

    # Generate report
    try:
        # Handle comparison mode
        if args.compare:
            p1_start, p1_end, p2_start, p2_end = args.compare
            final_md, data_rows = compare_time_periods(
                pages, p1_start, p1_end, p2_start, p2_end,
                show_progress=show_progress
            )
        elif multi_list:
            final_md, data_rows = generate_multi_range_report(
                multi_list, pages, show_progress=show_progress
            )
        else:
            final_md, data_rows = generate_full_report(
                pages, show_progress=show_progress
            )

        # Show threshold alerts if requested
        if args.alerts and data_rows:
            all_alerts = []
            for row in data_rows:
                alerts = get_threshold_alerts(row)
                all_alerts.extend(alerts)

            if all_alerts:
                print()
                print_warning("Threshold Alerts:")
                for alert in all_alerts:
                    print(f"  {Colors.warning('⚠')} {alert}")
                print()

        # Handle output format
        output_file = args.output
        output_format = args.format

        if output_format == "json":
            if not output_file.endswith(".json"):
                output_file = output_file.rsplit(".", 1)[0] + ".json"
            export_to_json(data_rows, output_file)
        elif output_format == "csv":
            if not output_file.endswith(".csv"):
                output_file = output_file.rsplit(".", 1)[0] + ".csv"
            export_to_csv(data_rows, output_file)
        elif output_format == "html":
            if not output_file.endswith(".html"):
                output_file = output_file.rsplit(".", 1)[0] + ".html"
            export_to_html(data_rows, final_md, output_file)
        else:  # markdown
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_md)

        elapsed_time = time.time() - start_time

        if show_progress:
            print_summary(
                pages_count=len(pages),
                rows_count=len(data_rows),
                output_file=output_file,
                output_format=output_format,
                elapsed_time=elapsed_time,
            )
        else:
            print_success(f"Report saved to '{output_file}'")

    except KeyboardInterrupt:
        print()
        print_warning("Operation cancelled by user")
        sys.exit(130)
    except PermissionError:
        print_error(f"Permission denied: Cannot write to '{args.output}'")
        print_info("Check file permissions or choose a different output path")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error generating report: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
