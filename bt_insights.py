"""Blue Triangle CLI Reporter - Analyze RUM data and revenue impact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import re
import smtplib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import get_close_matches
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None  # type: ignore

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
selected_time_range_days: float = 1.0  # Number of days in the selected time range

# Selected metrics filter (None means all metrics)
selected_metrics: list[str] | None = None

# Alert thresholds (None means no alerting)
alert_thresholds: dict[str, float] | None = None

# Cache settings
CACHE_DIR = Path.home() / ".bt_cache"
CACHE_TTL = int(os.getenv("BT_CACHE_TTL", "300"))  # 5 minutes default
cache_enabled = False

# Dry run mode - preview actions without making API calls
dry_run_mode = False

# Percentile settings - None means use averages (default)
# Valid percentiles: 50 (median), 75, 90, 95, 99
selected_percentiles: list[int] | None = None

# Data type for performance queries
# Options: "rum" (Real User Monitoring), "synthetic", "native", "basepage"
selected_data_type: str = "rum"

# Resource grouping option
# Options: "domain", "file", "service"
resource_group_by: str = "domain"

# Resource file filter pattern (e.g., "chunk*.js")
resource_file_filter: str | None = None

# Traffic segment filter (e.g., ["eCommerce", "Mobile"])
segment_filter: list[str] | None = None

# Country filter in ISO 3166 format (e.g., ["US", "CA", "GB"])
country_filter: list[str] | None = None

# Device filter (e.g., ["Desktop", "Mobile", "Tablet"])
device_filter: list[str] | None = None

# In-memory cache for performance data to avoid duplicate API calls
# Key: page_name, Value: {"current": DataFrame, "previous": DataFrame}
_performance_data_cache: dict[str, dict[str, Any]] = {}


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
    global selected_percentiles, selected_data_type, resource_group_by

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

    # Analysis settings
    if "analysis" in config:
        analysis_config = config["analysis"]
        if "percentile" in analysis_config:
            p = int(analysis_config["percentile"])
            if p in [50, 75, 90, 95, 99]:
                selected_percentiles = [p]
        if "data_type" in analysis_config:
            dt = analysis_config["data_type"]
            if dt in ["rum", "synthetic", "native", "basepage"]:
                selected_data_type = dt
        if "resource_group" in analysis_config:
            rg = analysis_config["resource_group"]
            if rg in ["domain", "file", "service"]:
                resource_group_by = rg


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


def generate_optimization_insights(metrics: dict[str, Any], delta_metrics: dict[str, float | str]) -> str:
    """Generate dynamic optimization insights based on actual metric values and thresholds.

    Args:
        metrics: Dictionary with current metric values (API field names as keys).
        delta_metrics: Dictionary with delta values (positive = worsened, negative = improved).

    Returns:
        Markdown formatted optimization insights.
    """
    insights = []

    # LCP insights
    lcp_val = metrics.get("largestContentfulPaint")
    if lcp_val is not None and lcp_val != "N/A":
        try:
            lcp_float = float(lcp_val)
            status, _ = check_threshold("LCP", lcp_float)
            lcp_delta = delta_metrics.get("lcp", "N/A")

            if status == "poor":
                insights.append(
                    "- **LCP** (POOR - {:.0f}ms):\n"
                    "  - 🔴 **Critical**: LCP exceeds 4000ms threshold\n"
                    "  - **Preload** the LCP image using `<link rel=\"preload\">`\n"
                    "  - **Remove lazy-loading** from above-the-fold images\n"
                    "  - Consider using a CDN for faster image delivery".format(lcp_float)
                )
            elif status == "needs-improvement":
                insights.append(
                    "- **LCP** (Needs Improvement - {:.0f}ms):\n"
                    "  - 🟡 LCP is between 2500-4000ms\n"
                    "  - **Preload** critical images and fonts\n"
                    "  - Optimize server response time (TTFB)".format(lcp_float)
                )
            else:
                if isinstance(lcp_delta, (int, float)) and lcp_delta < -50:
                    insights.append(f"- **LCP** (Good - {lcp_float:.0f}ms): ✅ Improved by {abs(lcp_delta):.0f}ms")
                else:
                    insights.append(f"- **LCP** (Good - {lcp_float:.0f}ms): ✅ Within target threshold")
        except (ValueError, TypeError):
            pass

    # INP insights
    inp_val = metrics.get("intToNextPaint")
    if inp_val is not None and inp_val != "N/A":
        try:
            inp_float = float(inp_val)
            status, _ = check_threshold("INP", inp_float)
            inp_delta = delta_metrics.get("inp", "N/A")

            if status == "poor":
                insights.append(
                    "- **INP** (POOR - {:.0f}ms):\n"
                    "  - 🔴 **Critical**: INP exceeds 500ms threshold\n"
                    "  - Break up long JavaScript tasks (>50ms)\n"
                    "  - Use `requestIdleCallback` for non-critical work\n"
                    "  - Review and optimize event handlers".format(inp_float)
                )
            elif status == "needs-improvement":
                insights.append(
                    "- **INP** (Needs Improvement - {:.0f}ms):\n"
                    "  - 🟡 INP is between 200-500ms\n"
                    "  - Optimize JavaScript execution time\n"
                    "  - Consider code-splitting to reduce main thread work".format(inp_float)
                )
            else:
                if isinstance(inp_delta, (int, float)) and inp_delta > 20:
                    insights.append(f"- **INP** (Good - {inp_float:.0f}ms): ⚠️ Worsened by {inp_delta:.0f}ms - monitor closely")
                else:
                    insights.append(f"- **INP** (Good - {inp_float:.0f}ms): ✅ Responsive interactions")
        except (ValueError, TypeError):
            pass

    # CLS insights
    cls_val = metrics.get("cumulativeLayoutShift")
    if cls_val is not None and cls_val != "N/A":
        try:
            cls_float = float(cls_val)
            status, _ = check_threshold("CLS", cls_float)

            if status == "poor":
                insights.append(
                    "- **CLS** (POOR - {:.4f}):\n"
                    "  - 🔴 **Critical**: CLS exceeds 0.25 threshold\n"
                    "  - Add `width` and `height` attributes to images/videos\n"
                    "  - Reserve space for dynamic content (ads, embeds)\n"
                    "  - Avoid inserting content above existing content".format(cls_float)
                )
            elif status == "needs-improvement":
                insights.append(
                    "- **CLS** (Needs Improvement - {:.4f}):\n"
                    "  - 🟡 CLS is between 0.1-0.25\n"
                    "  - Set explicit dimensions for media elements\n"
                    "  - Use CSS `aspect-ratio` for responsive images".format(cls_float)
                )
            else:
                insights.append(f"- **CLS** (Good - {cls_float:.4f}): ✅ Minimal layout shift")
        except (ValueError, TypeError):
            pass

    # TBT insights
    tbt_val = metrics.get("totalBlockingTime")
    if tbt_val is not None and tbt_val != "N/A":
        try:
            tbt_float = float(tbt_val)
            status, _ = check_threshold("TBT", tbt_float)

            if status == "poor":
                insights.append(
                    "- **TBT** (POOR - {:.0f}ms):\n"
                    "  - 🔴 High blocking time affecting interactivity\n"
                    "  - Defer non-critical JavaScript\n"
                    "  - Remove unused code and dependencies".format(tbt_float)
                )
            elif status == "needs-improvement":
                insights.append(
                    "- **TBT** (Needs Improvement - {:.0f}ms):\n"
                    "  - 🟡 Consider optimizing JavaScript execution".format(tbt_float)
                )
        except (ValueError, TypeError):
            pass

    # Onload insights (based on delta only, no fixed threshold)
    onload_delta = delta_metrics.get("onload", "N/A")
    if isinstance(onload_delta, (int, float)):
        if onload_delta > 500:
            insights.append(f"- **Onload Time**: 🔴 Worsened by {onload_delta:.0f}ms - investigate recent changes")
        elif onload_delta > 100:
            insights.append(f"- **Onload Time**: 🟡 Increased by {onload_delta:.0f}ms")
        elif onload_delta < -100:
            insights.append(f"- **Onload Time**: ✅ Improved by {abs(onload_delta):.0f}ms")

    if not insights:
        return "### 🛠 Optimization Insights\n✅ All metrics within acceptable ranges.\n"

    return "### 🛠 Optimization Insights\n" + "\n".join(insights) + "\n"


# ==================
# NOTIFICATIONS
# ==================


def send_slack_notification(
    webhook_url: str,
    report_summary: str,
    output_file: str | None = None,
) -> tuple[bool, str]:
    """Send report notification to Slack via webhook.

    Args:
        webhook_url: Slack incoming webhook URL.
        report_summary: Summary text to include in the message.
        output_file: Optional path to the report file.

    Returns:
        Tuple of (success, message).
    """
    if dry_run_mode:
        logger.info("[DRY RUN] Would send Slack notification to webhook")
        return True, "DRY RUN: Slack notification skipped"

    try:
        # Build Slack message payload
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📊 Blue Triangle Performance Report",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": report_summary
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            }
        ]

        if output_file:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📎 Report saved to: `{output_file}`"
                }
            })

        payload = {"blocks": blocks}

        response = requests.post(
            webhook_url,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return True, "Slack notification sent successfully"
        else:
            return False, f"Slack returned status {response.status_code}: {response.text[:200]}"

    except requests.exceptions.Timeout:
        return False, "Slack webhook timed out"
    except requests.exceptions.RequestException as e:
        return False, f"Slack notification failed: {str(e)}"


def send_teams_notification(
    webhook_url: str,
    report_summary: str,
    output_file: str | None = None,
) -> tuple[bool, str]:
    """Send report notification to Microsoft Teams via webhook.

    Args:
        webhook_url: Teams incoming webhook URL.
        report_summary: Summary text to include in the message.
        output_file: Optional path to the report file.

    Returns:
        Tuple of (success, message).
    """
    if dry_run_mode:
        logger.info("[DRY RUN] Would send Teams notification to webhook")
        return True, "DRY RUN: Teams notification skipped"

    try:
        # Build Teams Adaptive Card payload
        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "0076D7",
            "summary": "Blue Triangle Performance Report",
            "sections": [
                {
                    "activityTitle": "📊 Blue Triangle Performance Report",
                    "activitySubtitle": f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "text": report_summary,
                    "markdown": True
                }
            ]
        }

        if output_file:
            card["sections"].append({
                "text": f"📎 Report saved to: `{output_file}`",
                "markdown": True
            })

        response = requests.post(
            webhook_url,
            json=card,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            return True, "Teams notification sent successfully"
        else:
            return False, f"Teams returned status {response.status_code}: {response.text[:200]}"

    except requests.exceptions.Timeout:
        return False, "Teams webhook timed out"
    except requests.exceptions.RequestException as e:
        return False, f"Teams notification failed: {str(e)}"


def send_email_notification(
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    recipient_emails: list[str],
    subject: str,
    report_summary: str,
    attachment_path: str | None = None,
    use_tls: bool = True,
) -> tuple[bool, str]:
    """Send report notification via email.

    Args:
        smtp_server: SMTP server hostname.
        smtp_port: SMTP server port.
        sender_email: Sender email address.
        sender_password: Sender email password or app password.
        recipient_emails: List of recipient email addresses.
        subject: Email subject line.
        report_summary: Report summary for email body.
        attachment_path: Optional path to file to attach.
        use_tls: Whether to use TLS encryption (default: True).

    Returns:
        Tuple of (success, message).
    """
    if dry_run_mode:
        logger.info("[DRY RUN] Would send email to %s", ", ".join(recipient_emails))
        return True, "DRY RUN: Email notification skipped"

    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = ", ".join(recipient_emails)
        msg["Subject"] = subject

        # HTML body
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .header {{ background: #0076D7; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .footer {{ background: #f5f5f5; padding: 10px; font-size: 12px; color: #666; }}
                pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Blue Triangle Performance Report</h1>
            </div>
            <div class="content">
                <pre>{report_summary}</pre>
            </div>
            <div class="footer">
                Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Blue Triangle CLI Reporter
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(html_body, "html"))

        # Attach file if provided
        if attachment_path and Path(attachment_path).exists():
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={Path(attachment_path).name}"
                )
                msg.attach(part)

        # Connect and send
        if use_tls:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)

        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_emails, msg.as_string())
        server.quit()

        return True, f"Email sent successfully to {len(recipient_emails)} recipient(s)"

    except smtplib.SMTPAuthenticationError:
        return False, "Email authentication failed - check credentials"
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        return False, f"Email notification failed: {str(e)}"


def generate_report_summary(
    data_rows: list[dict[str, Any]],
    pages_count: int,
    time_range: str,
) -> str:
    """Generate a text summary of the report for notifications.

    Args:
        data_rows: List of metric data rows.
        pages_count: Number of pages analyzed.
        time_range: Time range string.

    Returns:
        Summary text string.
    """
    if not data_rows:
        return f"Report generated for {pages_count} pages over {time_range}. No data available."

    summary_lines = [
        f"*Report Summary* ({time_range})",
        f"• Pages analyzed: {pages_count}",
        f"• Data rows: {len(data_rows)}",
        "",
        "*Key Metrics:*"
    ]

    # Calculate averages for key metrics
    lcp_values = [r.get("lcp_curr") for r in data_rows if r.get("lcp_curr")]
    inp_values = [r.get("inp_curr") for r in data_rows if r.get("inp_curr")]
    cls_values = [r.get("cls_curr") for r in data_rows if r.get("cls_curr")]

    if lcp_values:
        avg_lcp = sum(lcp_values) / len(lcp_values)
        summary_lines.append(f"• Avg LCP: {avg_lcp:.0f}ms")

    if inp_values:
        avg_inp = sum(inp_values) / len(inp_values)
        summary_lines.append(f"• Avg INP: {avg_inp:.0f}ms")

    if cls_values:
        avg_cls = sum(cls_values) / len(cls_values)
        summary_lines.append(f"• Avg CLS: {avg_cls:.3f}")

    # Add alerts if any metrics are poor
    alerts = []
    for row in data_rows:
        row_alerts = get_threshold_alerts(row)
        alerts.extend(row_alerts)

    if alerts:
        summary_lines.append("")
        summary_lines.append(f"*⚠️ {len(alerts)} threshold alert(s) detected*")

    return "\n".join(summary_lines)


# ==================
# PDF EXPORT
# ==================


def export_to_pdf(
    data: list[dict[str, Any]],
    markdown_content: str,
    output_file: str,
) -> None:
    """Export report as PDF with charts.

    Args:
        data: List of metric dictionaries.
        markdown_content: Markdown report content.
        output_file: Output file path.
    """
    if not output_file.endswith(".pdf"):
        output_file = output_file.rsplit(".", 1)[0] + ".pdf"

    # Use matplotlib for PDF generation
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_file) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(
            0.5, 0.7,
            "Blue Triangle Performance Report",
            ha="center", va="center",
            fontsize=24, fontweight="bold"
        )
        ax.text(
            0.5, 0.5,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ha="center", va="center",
            fontsize=14
        )
        ax.text(
            0.5, 0.4,
            f"Pages analyzed: {len(data)}",
            ha="center", va="center",
            fontsize=12
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not data:
            return

        # Metrics comparison chart
        pages = [row.get("page", "Unknown")[:15] for row in data]
        lcp_curr = [row.get("lcp_curr", 0) or 0 for row in data]
        lcp_prev = [row.get("lcp_prev", 0) or 0 for row in data]

        if any(lcp_curr) or any(lcp_prev):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            x = range(len(pages))
            width = 0.35

            ax.bar([i - width/2 for i in x], lcp_curr, width, label="Current", color="#4CAF50")
            ax.bar([i + width/2 for i in x], lcp_prev, width, label="Previous", color="#2196F3")

            ax.set_xlabel("Page")
            ax.set_ylabel("LCP (ms)")
            ax.set_title("Largest Contentful Paint (LCP) by Page")
            ax.set_xticks(list(x))
            ax.set_xticklabels(pages, rotation=45, ha="right")
            ax.legend()
            ax.axhline(y=2500, color="orange", linestyle="--", label="Good threshold")
            ax.axhline(y=4000, color="red", linestyle="--", label="Poor threshold")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # INP chart
        inp_curr = [row.get("inp_curr", 0) or 0 for row in data]
        inp_prev = [row.get("inp_prev", 0) or 0 for row in data]

        if any(inp_curr) or any(inp_prev):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            x = range(len(pages))

            ax.bar([i - width/2 for i in x], inp_curr, width, label="Current", color="#4CAF50")
            ax.bar([i + width/2 for i in x], inp_prev, width, label="Previous", color="#2196F3")

            ax.set_xlabel("Page")
            ax.set_ylabel("INP (ms)")
            ax.set_title("Interaction to Next Paint (INP) by Page")
            ax.set_xticks(list(x))
            ax.set_xticklabels(pages, rotation=45, ha="right")
            ax.legend()
            ax.axhline(y=200, color="orange", linestyle="--", label="Good threshold")
            ax.axhline(y=500, color="red", linestyle="--", label="Poor threshold")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # CLS chart
        cls_curr = [(row.get("cls_curr", 0) or 0) for row in data]
        cls_prev = [(row.get("cls_prev", 0) or 0) for row in data]

        if any(cls_curr) or any(cls_prev):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            x = range(len(pages))

            ax.bar([i - width/2 for i in x], cls_curr, width, label="Current", color="#4CAF50")
            ax.bar([i + width/2 for i in x], cls_prev, width, label="Previous", color="#2196F3")

            ax.set_xlabel("Page")
            ax.set_ylabel("CLS Score")
            ax.set_title("Cumulative Layout Shift (CLS) by Page")
            ax.set_xticks(list(x))
            ax.set_xticklabels(pages, rotation=45, ha="right")
            ax.legend()
            ax.axhline(y=0.1, color="orange", linestyle="--", label="Good threshold")
            ax.axhline(y=0.25, color="red", linestyle="--", label="Poor threshold")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Summary table page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        # Create table data
        table_data = []
        headers = ["Page", "LCP (ms)", "INP (ms)", "CLS", "TBT (ms)"]

        for row in data[:20]:  # Limit to 20 rows for readability
            table_data.append([
                str(row.get("page", ""))[:20],
                str(int(row.get("lcp_curr", 0) or 0)),
                str(int(row.get("inp_curr", 0) or 0)),
                f"{(row.get('cls_curr', 0) or 0):.3f}",
                str(int(row.get("tbt_curr", 0) or 0)),
            ])

        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=headers,
                loc="center",
                cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)

            # Style header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor("#0076D7")
                table[(0, i)].set_text_props(color="white", fontweight="bold")

            ax.set_title("Performance Metrics Summary", fontsize=14, fontweight="bold", pad=20)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    logger.info("PDF report saved to %s", output_file)


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
            "dataType": "rum",  # Default for connection test
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


def is_virtual_page(page_name: str) -> bool:
    """Check if a page is a virtual page (VT) based on naming convention.

    Virtual pages are client-side transitions that don't have full page metrics
    like LCP and CLS. They typically end with '-VT' or '- VT'.

    Args:
        page_name: Name of the page to check.

    Returns:
        True if page is a virtual page, False otherwise.
    """
    name_lower = page_name.lower()
    # Check for common VT suffix patterns
    return (
        name_lower.endswith("-vt")
        or name_lower.endswith("- vt")
        or "-vt-" in name_lower  # e.g., "cdp-VT-LazyLoad"
    )


def normalize_vt_page_name(page_name: str) -> str:
    """Normalize VT page name to standard casing (uppercase VT suffix).

    Blue Triangle API is case-sensitive. VT pages typically use uppercase
    '-VT' suffix. This function corrects common casing variations.

    Args:
        page_name: Original page name.

    Returns:
        Normalized page name with correct VT casing.
    """
    import re

    # Pattern to match VT suffix variations (case-insensitive)
    # Handles: -vt, -VT, - vt, - VT, -Vt, etc.
    patterns = [
        (r"(-\s*)vt$", r"\1VT"),           # -vt or - vt at end -> -VT or - VT
        (r"(-\s*)vt(-)", r"\1VT\2"),       # -vt- in middle -> -VT-
    ]

    result = page_name
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def apply_global_filters(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply global segment, country, and device filters to an API payload.

    Args:
        payload: The API payload dictionary.

    Returns:
        Updated payload with filters applied.
    """
    if segment_filter:
        payload["trafficSeg"] = segment_filter
    if country_filter:
        payload["country"] = country_filter
    if device_filter:
        payload["device"] = device_filter
    return payload


# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def fetch_data(
    endpoint: str,
    payload: dict[str, Any] | None = None,
    method: str = "POST",
    params: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> dict[str, Any] | list[Any] | None:
    """Generic function to fetch JSON from the Blue Triangle API with retry logic.

    Implements exponential backoff for transient failures (timeouts, 5xx errors).
    Respects Retry-After headers for rate limiting (429 responses).

    Args:
        endpoint: API endpoint path.
        payload: JSON payload for POST requests.
        method: HTTP method (GET or POST).
        params: Query parameters for GET requests.
        use_cache: Whether to use caching for this request.

    Returns:
        Parsed JSON response or None on error.
    """
    # Dry run mode - return mock data
    if dry_run_mode:
        logger.info("[DRY RUN] Would fetch: %s %s", method, endpoint)
        return {"dry_run": True, "endpoint": endpoint}

    # Check cache first
    cache_data = payload if method == "POST" else params
    cache_key = get_cache_key(endpoint, cache_data)

    if use_cache:
        cached = get_cached_response(cache_key)
        if cached is not None:
            return cached

    url = BASE_URL + endpoint

    for attempt in range(MAX_RETRIES + 1):
        try:
            if method == "GET":
                logger.debug("GET %s params=%s (attempt %d)", url, params, attempt + 1)
                r = requests.get(
                    url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT
                )
            else:
                logger.debug("POST %s payload=%s (attempt %d)", url, payload, attempt + 1)
                r = requests.post(
                    url, headers=HEADERS, json=payload, timeout=REQUEST_TIMEOUT
                )

            # Handle rate limiting with Retry-After header
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except ValueError:
                        wait_time = RETRY_BACKOFF_BASE ** attempt
                else:
                    wait_time = RETRY_BACKOFF_BASE ** attempt

                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Rate limited (429), waiting %ds before retry %d/%d",
                        wait_time, attempt + 1, MAX_RETRIES
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Rate limit exceeded after %d retries", MAX_RETRIES)
                    return None

            # Check for other retryable status codes
            if r.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                wait_time = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Retryable status %d, waiting %.1fs before retry %d/%d",
                    r.status_code, wait_time, attempt + 1, MAX_RETRIES
                )
                time.sleep(wait_time)
                continue

            r.raise_for_status()
            data = r.json()

            # Cache successful response
            if use_cache and data is not None:
                set_cached_response(cache_key, data)

            return data

        except requests.exceptions.Timeout as errt:
            if attempt < MAX_RETRIES:
                wait_time = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Timeout, waiting %.1fs before retry %d/%d",
                    wait_time, attempt + 1, MAX_RETRIES
                )
                time.sleep(wait_time)
                continue
            logger.error("Timeout Error after %d retries: %s", MAX_RETRIES, errt)

        except requests.exceptions.ConnectionError as errc:
            if attempt < MAX_RETRIES:
                wait_time = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Connection error, waiting %.1fs before retry %d/%d",
                    wait_time, attempt + 1, MAX_RETRIES
                )
                time.sleep(wait_time)
                continue
            logger.error("Connection Error after %d retries: %s", MAX_RETRIES, errc)

        except requests.exceptions.HTTPError as errh:
            logger.error("HTTP Error: %s", errh)
            break  # Don't retry non-retryable HTTP errors

        except requests.exceptions.RequestException as err:
            logger.error("Request Error: %s", err)
            break

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
    global now, one_day_ago, selected_data_type
    if start is None:
        start = one_day_ago if one_day_ago is not None else int(time.time()) - 86400
    if end is None:
        end = now if now is not None else int(time.time())

    payload = {
        "site": SITE_PREFIX,
        "start": start,
        "end": end,
        "dataType": selected_data_type,
        "dataColumns": ["pageViews"],
        "group": ["pageName"],
        "limit": limit * 10,  # Fetch more to ensure we get enough unique pages
        "orderBy": [{"field": "pageViews", "direction": "DESC"}],
    }

    apply_global_filters(payload)
    logger.debug("Fetching top pages: %s", payload)
    data = fetch_data(ENDPOINTS["performance"], payload)
    logger.debug("Top pages response: %s", data)

    if validate_api_response(data, "data"):
        df = pd.DataFrame(data["data"])
        logger.debug("Performance endpoint returned %d rows", len(df))
        if not df.empty and "pageName" in df.columns:
            original_pages = df["pageName"].tolist()
            logger.debug("Raw page names: %s", original_pages[:10])
            # Filter out aggregate entries like "All Pages"
            df = df[~df["pageName"].str.lower().isin(["all pages", "all-pages", "allpages"])]
            logger.debug("After filtering 'All Pages': %d rows", len(df))
        # If filtering removed all results, try the hits endpoint
        if df.empty or len(df) == 0:
            logger.debug("No individual pages found, trying hits endpoint")
            return _fetch_pages_from_hits(start, end, limit)
        return df
    logger.debug("Performance endpoint validation failed, trying hits endpoint")
    return _fetch_pages_from_hits(start, end, limit)


def _fetch_pages_from_hits(
    start: int,
    end: int,
    limit: int = 20,
) -> pd.DataFrame:
    """Fallback: fetch page names from performance/hits endpoint.

    Uses raw hit data to extract unique page names and count views.

    Args:
        start: Start timestamp (epoch seconds).
        end: End timestamp (epoch seconds).
        limit: Maximum number of pages to return.

    Returns:
        DataFrame with pageName and pageViews columns.
    """
    global selected_data_type

    payload = {
        "site": SITE_PREFIX,
        "start": start,
        "end": end,
        "dataType": selected_data_type,
        "dataColumns": ["pageName"],
        "limit": 10000,  # Fetch many hits to get good page coverage
    }

    logger.debug("Fetching pages from hits: %s", payload)
    data = fetch_data(ENDPOINTS["performance_hits"], payload, method="POST")
    logger.debug("Hits response sample: %s", str(data)[:500] if data else "None")

    if validate_api_response(data, "data"):
        df = pd.DataFrame(data["data"])
        logger.debug("Hits endpoint returned %d rows, columns: %s", len(df), list(df.columns) if not df.empty else [])
        if not df.empty and "pageName" in df.columns:
            unique_before = df["pageName"].nunique()
            logger.debug("Unique page names before filter: %d", unique_before)
            # Filter out aggregate entries
            df = df[~df["pageName"].str.lower().isin(["all pages", "all-pages", "allpages"])]
            # Count page views and get top pages
            page_counts = df["pageName"].value_counts().head(limit).reset_index()
            page_counts.columns = ["pageName", "pageViews"]
            logger.debug("Returning %d pages from hits: %s", len(page_counts), page_counts["pageName"].tolist()[:5])
            return page_counts
        else:
            logger.debug("Hits endpoint: empty df or no pageName column")
    else:
        logger.debug("Hits endpoint validation failed")
    return pd.DataFrame([])


def update_available_pages(limit: int = 20) -> list[str]:
    """Update AVAILABLE_PAGES dynamically from the API.

    Args:
        limit: Maximum number of pages to fetch.

    Returns:
        List of available page names (deduplicated).
    """
    global AVAILABLE_PAGES
    df = fetch_top_page_names(limit=limit)
    if not df.empty and "pageName" in df.columns:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_pages: list[str] = []
        for page in df["pageName"].tolist():
            if page not in seen:
                seen.add(page)
                unique_pages.append(page)
        AVAILABLE_PAGES = unique_pages
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


def validate_epoch_timestamp(ts: int, name: str) -> None:
    """Validate an epoch timestamp is reasonable.

    Args:
        ts: Timestamp to validate.
        name: Name of the parameter for error messages.

    Raises:
        ValueError: If timestamp is invalid.
    """
    now_ts = int(time.time())
    min_ts = 1000000000  # Sep 9, 2001 - reasonable minimum
    max_ts = now_ts + 86400  # 1 day in the future max

    if ts < min_ts:
        raise ValueError(
            f"Invalid {name}: {ts} is too old. "
            f"Expected epoch timestamp (seconds since 1970). "
            f"Example: {now_ts} for current time."
        )
    if ts > max_ts:
        raise ValueError(
            f"Invalid {name}: {ts} is in the future. "
            f"Current time is {now_ts}."
        )


def parse_time_args(
    args: argparse.Namespace,
) -> tuple[int | None, int | None, int | None, int | None, list[str]]:
    """Parse time-related arguments with validation.

    If --multi-range is provided, returns a list of multiple ranges.
    Otherwise, parse single start/end or a named --time-range.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (start, end, prev_start, prev_end, multi_list).

    Raises:
        ValueError: If timestamps are invalid.
    """
    global selected_time_range_days

    if args.multi_range:
        ranges = [x.strip() for x in args.multi_range.split(",")]
        # Validate each range is known
        for r in ranges:
            if r not in DAY_MAP:
                valid_ranges = ", ".join(sorted(DAY_MAP.keys()))
                raise ValueError(
                    f"Unknown time range: '{r}'. Valid options: {valid_ranges}"
                )
        return None, None, None, None, ranges

    now_ts = int(time.time())

    if args.start and args.end:
        start = int(args.start)
        end = int(args.end)

        # Validate timestamps
        validate_epoch_timestamp(start, "--start")
        validate_epoch_timestamp(end, "--end")

        if start >= end:
            raise ValueError(
                f"--start ({start}) must be before --end ({end}). "
                f"These are epoch timestamps (seconds since 1970)."
            )

        # Calculate days from custom timestamps
        selected_time_range_days = (end - start) / 86400
        return start, end, None, None, []

    days = DAY_MAP.get(args.time_range, 1)
    selected_time_range_days = days
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
    global selected_time_range_days
    now_ts = int(time.time())
    days = DAY_MAP.get(range_str, 1)
    selected_time_range_days = days
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

    Uses cached data from gather_page_metrics() if available to avoid
    duplicate API calls.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Markdown formatted performance report.
    """
    global _performance_data_cache

    # Check cache first (populated by gather_page_metrics)
    cached = _performance_data_cache.get(page_name)
    if cached:
        df_today = cached["df_curr"]
        df_prev = cached["df_prev"]
        logger.debug("Using cached performance data for %s", page_name)
    else:
        # Fallback to fetching if not in cache
        payload: dict[str, Any] = {
            "site": SITE_PREFIX,
            "start": one_day_ago,
            "end": now,
            "dataType": selected_data_type,
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

        # Add percentile settings if specified
        if selected_percentiles:
            payload["avgType"] = "percentile"
            payload["percentile"] = selected_percentiles

        prev_payload = dict(payload)
        if two_days_ago is not None:
            prev_payload["start"] = two_days_ago
            prev_payload["end"] = one_day_ago

        apply_global_filters(payload)
        apply_global_filters(prev_payload)
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
    if is_virtual_page(page_name):
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

    # Build delta metrics for dynamic insights
    delta_metrics = {
        "lcp": delta(n(t.get("largestContentfulPaint")), n(p.get("largestContentfulPaint"))),
        "inp": delta(n(t.get("intToNextPaint")), n(p.get("intToNextPaint"))),
        "cls": delta(n(t.get("cumulativeLayoutShift")), n(p.get("cumulativeLayoutShift"))),
        "tbt": delta(n(t.get("totalBlockingTime")), n(p.get("totalBlockingTime"))),
        "onload": delta(n(t.get("onload")), n(p.get("onload"))),
    }

    # Generate dynamic optimization insights based on thresholds
    optimization_insights = generate_optimization_insights(dict(t), delta_metrics)

    return f"""
{perf_summary}

### 📊 Performance Metrics for {page_name}

{current_section}
{previous_section}
{delta_section}
{optimization_insights}
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

    Also caches the raw data in _performance_data_cache for use by
    get_page_performance() to avoid duplicate API calls.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Dictionary of metrics or None if no data.
    """
    global now, one_day_ago, two_days_ago, _performance_data_cache

    payload: dict[str, Any] = {
        "site": SITE_PREFIX,
        "start": one_day_ago,
        "end": now,
        "dataType": selected_data_type,
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

    # Add percentile settings if specified
    if selected_percentiles:
        payload["avgType"] = "percentile"
        payload["percentile"] = selected_percentiles

    prev_payload = dict(payload)
    if two_days_ago:
        prev_payload["start"] = two_days_ago
        prev_payload["end"] = one_day_ago

    apply_global_filters(payload)
    apply_global_filters(prev_payload)
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

    # Cache the dataframes for use by get_page_performance()
    _performance_data_cache[page_name] = {
        "df_curr": df_curr,
        "df_prev": df_prev,
    }

    latest_curr = df_curr.iloc[-1]
    latest_prev = df_prev.iloc[-1]

    # Helper to get metric column name (with percentile suffix if applicable)
    def get_col(base_name: str) -> str:
        if selected_percentiles:
            # Use the first percentile (typically the most useful one like p75 or p90)
            return f"{base_name}-{selected_percentiles[0]}"
        return base_name

    row = {
        "page": page_name,
        "onload_curr": _to_float(latest_curr.get(get_col("onload"))),
        "onload_prev": _to_float(latest_prev.get(get_col("onload"))),
        "lcp_curr": _to_float(latest_curr.get(get_col("largestContentfulPaint"))),
        "lcp_prev": _to_float(latest_prev.get(get_col("largestContentfulPaint"))),
        "inp_curr": _to_float(latest_curr.get(get_col("intToNextPaint"))),
        "inp_prev": _to_float(latest_prev.get(get_col("intToNextPaint"))),
        "cls_curr": _to_float(latest_curr.get(get_col("cumulativeLayoutShift"))),
        "cls_prev": _to_float(latest_prev.get(get_col("cumulativeLayoutShift"))),
        "tbt_curr": _to_float(latest_curr.get(get_col("totalBlockingTime"))),
        "tbt_prev": _to_float(latest_prev.get(get_col("totalBlockingTime"))),
        "fb_curr": _to_float(latest_curr.get(get_col("firstByte"))),
        "fb_prev": _to_float(latest_prev.get(get_col("firstByte"))),
        "dns_curr": _to_float(latest_curr.get(get_col("dns"))),
        "dns_prev": _to_float(latest_prev.get(get_col("dns"))),
        "tcp_curr": _to_float(latest_curr.get(get_col("tcp"))),
        "tcp_prev": _to_float(latest_prev.get(get_col("tcp"))),
    }

    # Add percentile info to row if applicable
    if selected_percentiles:
        row["percentile"] = selected_percentiles[0]

    is_vt = is_virtual_page(page_name)
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
        "dataType": selected_data_type,
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


def get_revenue_curve_for_page(page_name: str) -> dict[str, Any] | None:
    """Fetch revenue curve data for a page to estimate cost per millisecond.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Dictionary with revenue curve data or None if unavailable.
    """
    report_date = get_latest_revenue_opportunity_date()
    if not report_date:
        return None

    params: dict[str, Any] = {
        "prefix": SITE_PREFIX,
        "salesType": "revenue",
        "reportDate": report_date,
        "pageName[]": [page_name],
    }

    opp_data = fetch_data(ENDPOINTS["revenue_opportunity"], method="GET", params=params)
    if not opp_data:
        return None

    # Aggregate revenue curve across all devices
    total_curve: list[float] = []
    timing_steps: list[int] = []

    for dev in ["Desktop", "Mobile", "Tablet"]:
        dev_data = opp_data.get(dev)
        if not dev_data:
            continue
        revenue_section = dev_data.get("revenue", {})
        if page_name not in revenue_section:
            continue
        page_data = revenue_section[page_name]

        # Get the speedUpByXData which has timing steps and lost revenue
        speed_data = page_data.get("speedUpByXData", page_data.get("speedUpToXData", {}))
        if not speed_data:
            continue

        lost_revenue = speed_data.get("lostRevenue", [])
        # Blue Triangle typically uses 100ms steps: [0, 100, 200, 300, ...]
        steps = speed_data.get("timingSteps", [i * 100 for i in range(len(lost_revenue))])

        if not timing_steps and steps:
            timing_steps = steps

        if lost_revenue:
            if not total_curve:
                total_curve = [0.0] * len(lost_revenue)
            for i, val in enumerate(lost_revenue):
                if i < len(total_curve):
                    total_curve[i] += float(val) if val else 0.0

    if not total_curve or not timing_steps:
        return None

    return {
        "timing_steps": timing_steps,
        "lost_revenue": total_curve,
    }


def estimate_resource_cost_impact(
    timing_change_ms: float,
    revenue_curve: dict[str, Any],
) -> float:
    """Estimate revenue impact of a timing change using the revenue curve.

    Args:
        timing_change_ms: Timing change in milliseconds (positive = slowdown, negative = speedup).
        revenue_curve: Revenue curve data from get_revenue_curve_for_page().

    Returns:
        Estimated daily revenue impact (positive = cost/loss, negative = savings/gain).
    """
    global selected_time_range_days

    if not revenue_curve:
        return 0.0

    timing_steps = revenue_curve.get("timing_steps", [])
    lost_revenue = revenue_curve.get("lost_revenue", [])

    if len(timing_steps) < 2 or len(lost_revenue) < 2:
        return 0.0

    # Calculate revenue per millisecond from the curve
    # Use the slope between first two meaningful points
    step_size = timing_steps[1] - timing_steps[0] if len(timing_steps) > 1 else 100
    if step_size == 0:
        step_size = 100

    # Revenue difference per step (e.g., per 100ms)
    revenue_per_step = lost_revenue[-1] / len(lost_revenue) if lost_revenue else 0
    revenue_per_ms = revenue_per_step / step_size

    # Divide by number of days to get daily average
    days = selected_time_range_days if selected_time_range_days > 0 else 1.0

    # Apply timing change (slowdown = positive cost, speedup = negative cost/savings)
    return (timing_change_ms * revenue_per_ms) / days


def summarize_resource_usage_with_cost(
    current_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    page_name: str,
) -> str:
    """Summarize resource usage changes with estimated cost impact.

    Args:
        current_df: Current period resource data.
        prev_df: Previous period resource data.
        page_name: Name of the page for revenue lookup.

    Returns:
        Markdown formatted summary with cost estimates.
    """
    if current_df.empty or prev_df.empty:
        return "> ⚠️ Cannot summarize resource usage (no data)."

    # Get the grouping column (could be domain, file, or service)
    group_col = "domain"
    for col in ["domain", "file", "service"]:
        if col in current_df.columns:
            group_col = col
            break

    current_map = dict(zip(current_df[group_col], current_df["duration"]))
    prev_map = dict(zip(prev_df[group_col], prev_df["duration"]))

    changes: list[tuple[str, float]] = []

    for resource in current_map:
        if resource in prev_map:
            c_val = _to_float(current_map[resource])
            p_val = _to_float(prev_map[resource])
            if c_val is not None and p_val is not None:
                diff = c_val - p_val
                changes.append((resource, diff))

    changes.sort(key=lambda x: x[1], reverse=True)

    slowdowns = [d for d in changes if d[1] > 0][:5]
    speedups = sorted([d for d in changes if d[1] < 0], key=lambda x: x[1])[:5]

    # Get revenue curve for cost estimation
    revenue_curve = get_revenue_curve_for_page(page_name)
    has_cost_data = revenue_curve is not None

    def format_changes_with_cost(lst: list[tuple[str, float]], is_slowdown: bool) -> list[str]:
        lines = []
        for resource, timing_diff in lst:
            cost = estimate_resource_cost_impact(timing_diff, revenue_curve) if has_cost_data else 0
            timing_str = f"{'+' if timing_diff > 0 else ''}{round(timing_diff, 1)}ms"
            if has_cost_data and abs(cost) >= 1:
                if is_slowdown:
                    lines.append(f"  - **{resource}**: {timing_str} → ~${abs(cost):,.0f}/day cost")
                else:
                    lines.append(f"  - **{resource}**: {timing_str} → ~${abs(cost):,.0f}/day savings")
            else:
                lines.append(f"  - **{resource}**: {timing_str}")
        return lines

    summary = "### 📝 Resource Summary\n"

    if slowdowns:
        summary += "#### 🔴 Top Slowdowns (Costing You)\n"
        summary += "\n".join(format_changes_with_cost(slowdowns, is_slowdown=True)) + "\n"
    else:
        summary += "#### 🔴 Top Slowdowns\n  - (none)\n"

    if speedups:
        summary += "#### 🟢 Top Speedups (Saving You)\n"
        summary += "\n".join(format_changes_with_cost(speedups, is_slowdown=False)) + "\n"
    else:
        summary += "#### 🟢 Top Speedups\n  - (none)\n"

    # Add total impact summary
    if has_cost_data:
        total_slowdown_cost = sum(
            estimate_resource_cost_impact(d[1], revenue_curve) for d in slowdowns
        )
        total_speedup_savings = sum(
            abs(estimate_resource_cost_impact(d[1], revenue_curve)) for d in speedups
        )
        net_impact = total_slowdown_cost - total_speedup_savings

        summary += "\n#### 💰 Estimated Daily Impact\n"
        if total_slowdown_cost > 0:
            summary += f"  - Slowdown costs: ~${total_slowdown_cost:,.0f}/day\n"
        if total_speedup_savings > 0:
            summary += f"  - Speedup savings: ~${total_speedup_savings:,.0f}/day\n"
        if net_impact > 0:
            summary += f"  - **Net daily loss: ~${net_impact:,.0f}**\n"
        elif net_impact < 0:
            summary += f"  - **Net daily gain: ~${abs(net_impact):,.0f}**\n"

    return summary


def get_resource_data(page_name: str, compare_previous: bool = True) -> str:
    """Fetch resource data for a page.

    Args:
        page_name: Name of the page to analyze.
        compare_previous: Whether to compare with previous period.

    Returns:
        Markdown formatted resource report.
    """
    global now, one_day_ago, two_days_ago, selected_data_type, resource_group_by

    # Use configured resource grouping (domain, file, or service)
    group_col = resource_group_by

    payload_curr: dict[str, Any] = {
        "site": SITE_PREFIX,
        "start": one_day_ago,
        "end": now,
        "dataType": selected_data_type,
        "dataColumns": ["duration", "elementCount"],
        "group": [group_col],
        "pageName": page_name,
    }

    # Add percentile settings for resource data if specified
    if selected_percentiles:
        payload_curr["avgType"] = "percentile"
        payload_curr["percentile"] = selected_percentiles

    payload_prev = dict(payload_curr)
    if compare_previous and two_days_ago is not None:
        payload_prev["start"] = two_days_ago
        payload_prev["end"] = one_day_ago

    apply_global_filters(payload_curr)
    apply_global_filters(payload_prev)
    data_curr = fetch_data(ENDPOINTS["resource"], payload_curr)
    if not validate_api_response(data_curr, "data"):
        return f"> ⚠️ No resource data for **{page_name}**.\n"

    df_curr = pd.DataFrame(data_curr["data"])
    if df_curr.empty:
        return f"> ⚠️ Resource data empty for **{page_name}**.\n"

    # Get duration column name (with percentile suffix if applicable)
    duration_col = "duration"
    if selected_percentiles:
        duration_col = f"duration-{selected_percentiles[0]}"

    group_label = {"domain": "Domain", "file": "File", "service": "Service"}.get(group_col, group_col)
    resource_text = f"### 📦 Resource Usage by {group_label} ({page_name})\n"

    lines = []
    for _, row in df_curr.iterrows():
        group_val = row.get(group_col, "Unknown")
        duration = row.get(duration_col, row.get("duration", 0))
        element_count = row.get("elementCount", 0)
        lines.append(f"- {group_val}: {duration} ms, {element_count} elements")

    resource_text += "\n".join(lines)

    if compare_previous and two_days_ago is not None:
        data_prev = fetch_data(ENDPOINTS["resource"], payload_prev)
        if validate_api_response(data_prev, "data"):
            df_prev = pd.DataFrame(data_prev["data"])
            if not df_prev.empty:
                # Use the cost-aware summary that includes revenue impact
                summary = summarize_resource_usage_with_cost(df_curr, df_prev, page_name)
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
        "dataType": selected_data_type,
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

    # Filter out "Native" device type as it's not a standard web device category
    excluded_devices = {"Native"}
    if device:
        devices_to_check = device
    elif isinstance(opp_data, dict):
        devices_to_check = [k for k in opp_data.keys() if k not in excluded_devices]
    else:
        devices_to_check = []

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
            # Get lost revenue from speedUpByXData (relative improvement model)
            # Note: speedUpToXData and speedUpByXData are alternative calculations,
            # not cumulative. We use speedUpByXData as it shows potential if page
            # were X% faster, which is the more commonly referenced metric.
            if "speedUpByXData" in page_data and "lostRevenue" in page_data["speedUpByXData"]:
                arr = page_data["speedUpByXData"]["lostRevenue"]
                if arr and len(arr) > 0:
                    device_lost_revenue = float(arr[-1])
            # Fallback to speedUpToXData if speedUpByXData not available
            elif "speedUpToXData" in page_data and "lostRevenue" in page_data["speedUpToXData"]:
                arr = page_data["speedUpToXData"]["lostRevenue"]
                if arr and len(arr) > 0:
                    device_lost_revenue = float(arr[-1])
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


# ========== ADVANCED COST ANALYSIS ==========


def get_metric_cost_breakdown(page_name: str) -> str:
    """Calculate cost impact per Core Web Vital metric.

    Estimates how much each metric's degradation is costing by correlating
    timing changes with the revenue opportunity curve.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Markdown formatted metric cost breakdown.
    """
    global now, one_day_ago, two_days_ago, selected_data_type

    # Fetch current and previous performance data
    payload: dict[str, Any] = {
        "site": SITE_PREFIX,
        "start": one_day_ago,
        "end": now,
        "dataType": selected_data_type,
        "dataColumns": [
            "largestContentfulPaint", "totalBlockingTime",
            "cumulativeLayoutShift", "intToNextPaint", "firstByte",
        ],
        "group": ["pageName"],
        "pageName": page_name,
    }

    if selected_percentiles:
        payload["avgType"] = "percentile"
        payload["percentile"] = selected_percentiles

    prev_payload = dict(payload)
    if two_days_ago is not None:
        prev_payload["start"] = two_days_ago
        prev_payload["end"] = one_day_ago

    apply_global_filters(payload)
    apply_global_filters(prev_payload)
    data_curr = fetch_data(ENDPOINTS["performance"], payload)
    data_prev = fetch_data(ENDPOINTS["performance"], prev_payload) if two_days_ago else None

    if not validate_api_response(data_curr, "data"):
        return "> ⚠️ No metric data available for cost analysis.\n"

    df_curr = pd.DataFrame(data_curr["data"])
    df_prev = pd.DataFrame(data_prev["data"]) if data_prev and validate_api_response(data_prev, "data") else pd.DataFrame()

    if df_curr.empty:
        return "> ⚠️ No metric data available.\n"

    # Get revenue curve for cost estimation
    revenue_curve = get_revenue_curve_for_page(page_name)
    if not revenue_curve:
        return "> ⚠️ No revenue data available for metric cost analysis.\n"

    # Metric mappings
    metrics = {
        "LCP": "largestContentfulPaint",
        "TBT": "totalBlockingTime",
        "INP": "intToNextPaint",
        "TTFB": "firstByte",
    }
    # CLS is a ratio, not milliseconds - handle separately
    cls_col = "cumulativeLayoutShift"

    curr_row = df_curr.iloc[0] if not df_curr.empty else {}
    prev_row = df_prev.iloc[0] if not df_prev.empty else {}

    metric_costs: list[tuple[str, float, float]] = []  # (name, delta_ms, cost)

    for metric_name, col_name in metrics.items():
        curr_val = _to_float(curr_row.get(col_name, 0)) or 0
        prev_val = _to_float(prev_row.get(col_name, 0)) or 0 if not df_prev.empty else curr_val

        delta_ms = curr_val - prev_val
        cost = estimate_resource_cost_impact(delta_ms, revenue_curve)
        metric_costs.append((metric_name, delta_ms, cost))

    # Handle CLS separately (multiply by 1000 to convert to pseudo-ms for cost calc)
    curr_cls = _to_float(curr_row.get(cls_col, 0)) or 0
    prev_cls = _to_float(prev_row.get(cls_col, 0)) or 0 if not df_prev.empty else curr_cls
    cls_delta = (curr_cls - prev_cls) * 1000  # Scale up for cost calculation
    cls_cost = estimate_resource_cost_impact(cls_delta, revenue_curve)
    metric_costs.append(("CLS", cls_delta / 1000, cls_cost))  # Store original delta

    # Sort by absolute cost (highest impact first)
    metric_costs.sort(key=lambda x: abs(x[2]), reverse=True)

    output = "### 📊 Metric Cost Breakdown\n"
    output += "*Which Core Web Vital is costing you the most?*\n\n"

    total_cost = 0.0
    for metric_name, delta, cost in metric_costs:
        total_cost += cost
        if metric_name == "CLS":
            delta_str = f"{'+' if delta > 0 else ''}{delta:.4f}"
        else:
            delta_str = f"{'+' if delta > 0 else ''}{delta:.0f}ms"

        if abs(cost) >= 1:
            if cost > 0:
                output += f"- **{metric_name}**: {delta_str} → ~${cost:,.0f}/day cost\n"
            else:
                output += f"- **{metric_name}**: {delta_str} → ~${abs(cost):,.0f}/day savings\n"
        else:
            output += f"- **{metric_name}**: {delta_str} (minimal impact)\n"

    if abs(total_cost) >= 1:
        output += f"\n**Total metric impact**: ~${abs(total_cost):,.0f}/day "
        output += "cost\n" if total_cost > 0 else "savings\n"

    return output


def get_device_cost_breakdown(page_name: str) -> str:
    """Calculate cost impact by device type (Desktop, Mobile, Tablet).

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Markdown formatted device cost breakdown.
    """
    global selected_time_range_days

    report_date = get_latest_revenue_opportunity_date()
    if not report_date:
        return "> ⚠️ No revenue data available for device breakdown.\n"

    params: dict[str, Any] = {
        "prefix": SITE_PREFIX,
        "salesType": "revenue",
        "reportDate": report_date,
        "pageName[]": [page_name],
    }

    opp_data = fetch_data(ENDPOINTS["revenue_opportunity"], method="GET", params=params)
    if not opp_data:
        return "> ⚠️ No device revenue data available.\n"

    device_costs: list[tuple[str, float]] = []
    days = selected_time_range_days if selected_time_range_days > 0 else 1.0

    for device in ["Mobile", "Desktop", "Tablet"]:
        dev_data = opp_data.get(device)
        if not dev_data:
            continue

        revenue_section = dev_data.get("revenue", {})
        if page_name not in revenue_section:
            continue

        page_data = revenue_section[page_name]
        lost_revenue = 0.0

        try:
            speed_data = page_data.get("speedUpByXData", page_data.get("speedUpToXData", {}))
            if speed_data and "lostRevenue" in speed_data:
                arr = speed_data["lostRevenue"]
                if arr and len(arr) > 0:
                    lost_revenue = float(arr[-1])
        except (ValueError, TypeError, KeyError):
            lost_revenue = 0.0

        if lost_revenue > 0:
            # Divide by number of days to get daily average
            device_costs.append((device, lost_revenue / days))

    if not device_costs:
        return "> ⚠️ No device-specific cost data available.\n"

    # Sort by cost (highest first)
    device_costs.sort(key=lambda x: x[1], reverse=True)
    total_cost = sum(c[1] for c in device_costs)

    output = "### 📱 Device Cost Breakdown\n"
    output += "*Where is performance costing you the most?*\n\n"

    for device, cost in device_costs:
        pct = (cost / total_cost * 100) if total_cost > 0 else 0
        emoji = {"Mobile": "📱", "Desktop": "💻", "Tablet": "📟"}.get(device, "📊")
        output += f"- {emoji} **{device}**: ~${cost:,.0f}/day ({pct:.0f}%)\n"

    output += f"\n**Total**: ~${total_cost:,.0f}/day\n"

    return output


def get_page_cost_ranking(pages: list[str]) -> str:
    """Rank pages by revenue impact (highest cost first).

    Args:
        pages: List of page names to analyze.

    Returns:
        Markdown formatted page cost ranking.
    """
    global selected_time_range_days

    report_date = get_latest_revenue_opportunity_date()
    if not report_date:
        return "> ⚠️ No revenue data available for page ranking.\n"

    params: dict[str, Any] = {
        "prefix": SITE_PREFIX,
        "salesType": "revenue",
        "reportDate": report_date,
    }

    opp_data = fetch_data(ENDPOINTS["revenue_opportunity"], method="GET", params=params)
    if not opp_data:
        return "> ⚠️ No revenue data available.\n"

    page_costs: dict[str, float] = {}
    days = selected_time_range_days if selected_time_range_days > 0 else 1.0

    for page_name in pages:
        total_cost = 0.0
        for device in ["Desktop", "Mobile", "Tablet"]:
            dev_data = opp_data.get(device)
            if not dev_data:
                continue

            revenue_section = dev_data.get("revenue", {})
            if page_name not in revenue_section:
                continue

            page_data = revenue_section[page_name]
            try:
                speed_data = page_data.get("speedUpByXData", page_data.get("speedUpToXData", {}))
                if speed_data and "lostRevenue" in speed_data:
                    arr = speed_data["lostRevenue"]
                    if arr and len(arr) > 0:
                        total_cost += float(arr[-1])
            except (ValueError, TypeError, KeyError):
                pass

        if total_cost > 0:
            # Divide by number of days to get daily average
            page_costs[page_name] = total_cost / days

    if not page_costs:
        return "> ⚠️ No page cost data available.\n"

    # Sort by cost (highest first) and take top 10
    sorted_pages = sorted(page_costs.items(), key=lambda x: x[1], reverse=True)[:10]
    total_all = sum(page_costs.values())

    output = "### 🏆 Top Costly Pages\n"
    output += "*Pages with highest revenue impact from performance*\n\n"

    for i, (page, cost) in enumerate(sorted_pages, 1):
        pct = (cost / total_all * 100) if total_all > 0 else 0
        output += f"{i}. **{page}**: ~${cost:,.0f}/day ({pct:.0f}%)\n"

    output += f"\n**Total across all pages**: ~${total_all:,.0f}/day\n"

    return output


def get_cost_trend(page_name: str) -> str:
    """Calculate cost trend over multiple time periods.

    Compares performance costs across different time ranges to show
    if things are getting better or worse.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Markdown formatted cost trend analysis.
    """
    global now, selected_data_type

    if now is None:
        return "> ⚠️ Time range not configured for trend analysis.\n"

    # Define time periods to compare (in seconds)
    periods = [
        ("Today", 86400),       # Last 24 hours
        ("This Week", 604800),  # Last 7 days
        ("Last Week", 604800),  # 7-14 days ago
    ]

    period_costs: list[tuple[str, float]] = []

    for period_name, duration in periods:
        if period_name == "Last Week":
            end_ts = now - 604800  # Start from 7 days ago
            start_ts = end_ts - duration
        else:
            end_ts = now
            start_ts = now - duration

        # Fetch performance data for this period
        payload: dict[str, Any] = {
            "site": SITE_PREFIX,
            "start": start_ts,
            "end": end_ts,
            "dataType": selected_data_type,
            "dataColumns": ["largestContentfulPaint", "totalBlockingTime"],
            "group": ["pageName"],
            "pageName": page_name,
        }

        apply_global_filters(payload)
        data = fetch_data(ENDPOINTS["performance"], payload)
        if not validate_api_response(data, "data"):
            continue

        df = pd.DataFrame(data["data"])
        if df.empty:
            continue

        # Get average LCP + TBT as a proxy for "total slowness"
        row = df.iloc[0]
        lcp = _to_float(row.get("largestContentfulPaint", 0)) or 0
        tbt = _to_float(row.get("totalBlockingTime", 0)) or 0
        total_time = lcp + tbt

        # Get revenue curve to estimate cost
        revenue_curve = get_revenue_curve_for_page(page_name)
        if revenue_curve:
            # Use the timing as a proxy for "how far from optimal"
            # Assume optimal is 1000ms LCP + 100ms TBT = 1100ms baseline
            baseline = 1100
            delta_from_optimal = total_time - baseline
            if delta_from_optimal > 0:
                cost = estimate_resource_cost_impact(delta_from_optimal, revenue_curve)
                period_costs.append((period_name, cost))
            else:
                period_costs.append((period_name, 0))

    if len(period_costs) < 2:
        return "> ⚠️ Insufficient data for trend analysis.\n"

    output = "### 📈 Cost Trend\n"
    output += "*Is performance costing you more or less over time?*\n\n"

    for period_name, cost in period_costs:
        output += f"- **{period_name}**: ~${cost:,.0f}/day\n"

    # Calculate trend
    if len(period_costs) >= 2:
        current = period_costs[0][1]  # Today/This Week
        previous = period_costs[-1][1]  # Last Week

        if previous > 0:
            change_pct = ((current - previous) / previous) * 100
            diff = current - previous

            output += "\n"
            if diff > 0:
                output += f"⚠️ **Trend**: Getting worse (+${diff:,.0f}/day, +{change_pct:.0f}%)\n"
            elif diff < 0:
                output += f"✅ **Trend**: Improving (-${abs(diff):,.0f}/day, {change_pct:.0f}%)\n"
            else:
                output += "➡️ **Trend**: Stable\n"

    return output


def generate_friction_summary(pages: list[str], show_progress: bool = True) -> str:
    """Generate an executive friction summary.

    Shows total revenue at risk, top friction points, quick wins,
    and pages that need attention (getting worse).

    Args:
        pages: List of page names to analyze.
        show_progress: Whether to show progress messages.

    Returns:
        Formatted friction summary string.
    """
    global selected_time_range_days, now, selected_data_type

    if show_progress:
        print_info("Generating friction summary...")

    # Get revenue opportunity date
    report_date = get_latest_revenue_opportunity_date()
    if not report_date:
        return "⚠️ No revenue data available for friction summary.\n"

    # Fetch revenue opportunity data
    params: dict[str, Any] = {
        "prefix": SITE_PREFIX,
        "salesType": "revenue",
        "reportDate": report_date,
    }

    opp_data = fetch_data(ENDPOINTS["revenue_opportunity"], method="GET", params=params)
    if not opp_data:
        return "⚠️ No revenue data available.\n"

    days = selected_time_range_days if selected_time_range_days > 0 else 1.0

    # Collect page costs and device breakdowns
    page_data: dict[str, dict[str, Any]] = {}

    for page_name in pages:
        total_cost = 0.0
        device_costs: dict[str, float] = {}

        for device in ["Desktop", "Mobile", "Tablet"]:
            dev_data = opp_data.get(device)
            if not dev_data:
                continue

            revenue_section = dev_data.get("revenue", {})
            if page_name not in revenue_section:
                continue

            page_rev_data = revenue_section[page_name]
            try:
                speed_data = page_rev_data.get("speedUpByXData", page_rev_data.get("speedUpToXData", {}))
                if speed_data and "lostRevenue" in speed_data:
                    arr = speed_data["lostRevenue"]
                    if arr and len(arr) > 0:
                        device_cost = float(arr[-1]) / days
                        device_costs[device.lower()] = device_cost
                        total_cost += device_cost
            except (ValueError, TypeError, KeyError):
                pass

        if total_cost > 0:
            page_data[page_name] = {
                "cost": total_cost,
                "devices": device_costs,
                "trend": None,
                "trend_diff": 0,
                "trend_pct": 0,
            }

    if not page_data:
        return "⚠️ No page cost data available for friction summary.\n"

    # Calculate trends for each page
    if show_progress:
        print_info("Analyzing cost trends...")

    for page_name in page_data:
        if now is None:
            continue

        # Compare this week vs last week
        periods = [
            ("this_week", now - 604800, now),
            ("last_week", now - 1209600, now - 604800),
        ]

        period_costs: dict[str, float] = {}

        for period_name, start_ts, end_ts in periods:
            payload: dict[str, Any] = {
                "site": SITE_PREFIX,
                "start": start_ts,
                "end": end_ts,
                "dataType": selected_data_type,
                "dataColumns": ["largestContentfulPaint", "totalBlockingTime"],
                "group": ["pageName"],
                "pageName": page_name,
            }

            apply_global_filters(payload)
            data = fetch_data(ENDPOINTS["performance"], payload)
            if not validate_api_response(data, "data"):
                continue

            df = pd.DataFrame(data["data"])
            if df.empty:
                continue

            row = df.iloc[0]
            lcp = _to_float(row.get("largestContentfulPaint", 0)) or 0
            tbt = _to_float(row.get("totalBlockingTime", 0)) or 0
            total_time = lcp + tbt

            revenue_curve = get_revenue_curve_for_page(page_name)
            if revenue_curve:
                baseline = 1100
                delta_from_optimal = total_time - baseline
                if delta_from_optimal > 0:
                    cost = estimate_resource_cost_impact(delta_from_optimal, revenue_curve)
                    period_costs[period_name] = cost

        if "this_week" in period_costs and "last_week" in period_costs:
            current = period_costs["this_week"]
            previous = period_costs["last_week"]
            diff = current - previous

            if previous > 0:
                pct = ((current - previous) / previous) * 100
                if diff > 0:
                    page_data[page_name]["trend"] = "worse"
                elif diff < 0:
                    page_data[page_name]["trend"] = "better"
                else:
                    page_data[page_name]["trend"] = "stable"
                page_data[page_name]["trend_diff"] = diff
                page_data[page_name]["trend_pct"] = pct

    # Sort pages by cost
    sorted_pages = sorted(page_data.items(), key=lambda x: x[1]["cost"], reverse=True)
    total_cost = sum(p["cost"] for p in page_data.values())

    # Identify quick wins (improving) and watch list (getting worse)
    improving = [(p, d) for p, d in sorted_pages if d["trend"] == "better"]
    worsening = [(p, d) for p, d in sorted_pages if d["trend"] == "worse"]

    # Build summary output
    output = "\n"
    output += "━" * 50 + "\n"
    output += "📊 FRICTION SUMMARY\n"
    output += "━" * 50 + "\n\n"

    # Total at risk
    output += f"💰 **Total Daily Revenue at Risk**: ~${total_cost:,.0f}/day\n\n"

    # Top friction points
    output += "🔴 **Top Friction Points:**\n"
    for i, (page, data) in enumerate(sorted_pages[:5], 1):
        pct = (data["cost"] / total_cost * 100) if total_cost > 0 else 0
        devices = data.get("devices", {})
        top_device = max(devices.items(), key=lambda x: x[1])[0] if devices else ""
        top_device_pct = (devices.get(top_device, 0) / data["cost"] * 100) if data["cost"] > 0 else 0

        line = f"   {i}. **{page}**: ~${data['cost']:,.0f}/day ({pct:.0f}%)"

        if top_device and top_device_pct > 50:
            line += f" [{top_device}: {top_device_pct:.0f}%]"

        if data["trend"] == "worse":
            line += f" ⚠️ +{data['trend_pct']:.0f}%"
        elif data["trend"] == "better":
            line += f" ✅ {data['trend_pct']:.0f}%"

        output += line + "\n"

    output += "\n"

    # Quick wins
    if improving:
        output += "🟢 **Quick Wins (Improving):**\n"
        for page, data in improving[:3]:
            output += f"   - **{page}**: {data['trend_pct']:.0f}% (-${abs(data['trend_diff']):,.0f}/day)\n"
        output += "\n"

    # Watch list
    if worsening:
        output += "⚠️ **Watch List (Getting Worse):**\n"
        for page, data in worsening[:3]:
            output += f"   - **{page}**: +{data['trend_pct']:.0f}% (+${data['trend_diff']:,.0f}/day)\n"
        output += "\n"

    # Device summary
    all_devices: dict[str, float] = {"mobile": 0, "desktop": 0, "tablet": 0}
    for page, data in page_data.items():
        for device, cost in data.get("devices", {}).items():
            if device in all_devices:
                all_devices[device] += cost

    total_device = sum(all_devices.values())
    if total_device > 0:
        output += "📱 **Device Impact:**\n"
        sorted_devices = sorted(all_devices.items(), key=lambda x: x[1], reverse=True)
        icons = {"mobile": "📱", "desktop": "💻", "tablet": "📟"}
        for device, cost in sorted_devices:
            pct = (cost / total_device * 100) if total_device > 0 else 0
            icon = icons.get(device, "")
            output += f"   {icon} {device.capitalize()}: ~${cost:,.0f}/day ({pct:.0f}%)\n"
        output += "\n"

    output += "━" * 50 + "\n"
    output += "Run full report: python bt_insights.py --top-pages --time-range 28d\n"
    output += "━" * 50 + "\n"

    return output


def get_resource_file_analysis(page_name: str) -> str:
    """Analyze specific resource files matching the filter pattern.

    Shows timing changes and cost impact for resources matching the
    --resource-file pattern.

    Args:
        page_name: Name of the page to analyze.

    Returns:
        Markdown formatted resource file analysis.
    """
    global now, one_day_ago, two_days_ago, selected_data_type, resource_file_filter

    if not resource_file_filter:
        return ""

    import fnmatch

    # Fetch resource data grouped by file
    payload_curr: dict[str, Any] = {
        "site": SITE_PREFIX,
        "start": one_day_ago,
        "end": now,
        "dataType": selected_data_type,
        "dataColumns": ["duration", "elementCount"],
        "group": ["file"],
        "pageName": page_name,
    }

    if selected_percentiles:
        payload_curr["avgType"] = "percentile"
        payload_curr["percentile"] = selected_percentiles

    payload_prev = dict(payload_curr)
    if two_days_ago is not None:
        payload_prev["start"] = two_days_ago
        payload_prev["end"] = one_day_ago

    apply_global_filters(payload_curr)
    apply_global_filters(payload_prev)
    data_curr = fetch_data(ENDPOINTS["resource"], payload_curr)
    data_prev = fetch_data(ENDPOINTS["resource"], payload_prev) if two_days_ago else None

    if not validate_api_response(data_curr, "data"):
        return f"> ⚠️ No resource file data for **{page_name}**.\n"

    df_curr = pd.DataFrame(data_curr["data"])
    df_prev = pd.DataFrame(data_prev["data"]) if data_prev and validate_api_response(data_prev, "data") else pd.DataFrame()

    if df_curr.empty or "file" not in df_curr.columns:
        return f"> ⚠️ No file-level resource data for **{page_name}**.\n"

    # Filter files matching the pattern
    pattern = resource_file_filter.lower()
    matching_curr = df_curr[df_curr["file"].str.lower().apply(lambda x: fnmatch.fnmatch(x, pattern))]

    if matching_curr.empty:
        # Try partial match if exact pattern doesn't work
        matching_curr = df_curr[df_curr["file"].str.lower().str.contains(pattern.replace("*", ""), regex=False)]

    if matching_curr.empty:
        return f"> ⚠️ No resources matching '{resource_file_filter}' found for **{page_name}**.\n"

    # Get revenue curve for cost estimation
    revenue_curve = get_revenue_curve_for_page(page_name)

    output = f"### 🔍 Resource File Analysis: `{resource_file_filter}`\n"
    output += f"*Matching {len(matching_curr)} file(s) on {page_name}*\n\n"

    total_duration_curr = 0.0
    total_duration_prev = 0.0
    total_elements = 0
    file_details: list[tuple[str, float, float, float, int]] = []  # (file, curr_dur, prev_dur, delta, elements)

    for _, row in matching_curr.iterrows():
        file_name = row.get("file", "Unknown")
        duration_curr = _to_float(row.get("duration", 0)) or 0
        element_count = int(row.get("elementCount", 0) or 0)

        # Find previous duration for this file
        duration_prev = 0.0
        if not df_prev.empty and "file" in df_prev.columns:
            prev_match = df_prev[df_prev["file"] == file_name]
            if not prev_match.empty:
                duration_prev = _to_float(prev_match.iloc[0].get("duration", 0)) or 0

        delta = duration_curr - duration_prev
        total_duration_curr += duration_curr
        total_duration_prev += duration_prev
        total_elements += element_count
        file_details.append((file_name, duration_curr, duration_prev, delta, element_count))

    # Sort by current duration (slowest first)
    file_details.sort(key=lambda x: x[1], reverse=True)

    # Show individual files (limit to top 10)
    output += "#### Individual Files\n"
    for file_name, dur_curr, dur_prev, delta, elements in file_details[:10]:
        # Truncate long file names
        display_name = file_name if len(file_name) <= 60 else "..." + file_name[-57:]
        delta_str = f"{'+' if delta > 0 else ''}{delta:.0f}ms" if dur_prev > 0 else "new"

        if revenue_curve and abs(delta) > 0:
            cost = estimate_resource_cost_impact(delta, revenue_curve)
            if abs(cost) >= 1:
                cost_str = f" → ~${abs(cost):,.0f}/day {'cost' if cost > 0 else 'savings'}"
            else:
                cost_str = ""
        else:
            cost_str = ""

        output += f"- `{display_name}`: {dur_curr:.0f}ms ({delta_str}){cost_str}\n"

    if len(file_details) > 10:
        output += f"  - *...and {len(file_details) - 10} more files*\n"

    # Summary
    total_delta = total_duration_curr - total_duration_prev
    output += "\n#### Summary\n"
    output += f"- **Total files**: {len(file_details)}\n"
    output += f"- **Total duration**: {total_duration_curr:.0f}ms\n"
    output += f"- **Total elements**: {total_elements:,}\n"

    if total_duration_prev > 0:
        delta_pct = ((total_delta) / total_duration_prev) * 100 if total_duration_prev else 0
        delta_str = f"{'+' if total_delta > 0 else ''}{total_delta:.0f}ms ({delta_pct:+.1f}%)"
        output += f"- **Change from previous period**: {delta_str}\n"

        if revenue_curve:
            total_cost = estimate_resource_cost_impact(total_delta, revenue_curve)
            if abs(total_cost) >= 1:
                if total_cost > 0:
                    output += f"\n💰 **Estimated cost impact**: ~${total_cost:,.0f}/day\n"
                else:
                    output += f"\n💰 **Estimated savings**: ~${abs(total_cost):,.0f}/day\n"

    return output


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

    # Resource file analysis (if --resource-file is specified)
    resource_file_section = get_resource_file_analysis(page_name)
    if resource_file_section:
        text += resource_file_section + "\n\n"

    # JavaScript errors section
    js_errors_section = get_js_errors(page_name)
    if js_errors_section and "No aggregated JS error" not in js_errors_section:
        text += js_errors_section + "\n\n"

    text += get_page_revenue(page_name) + "\n\n"
    text += get_revenue_opportunity(page_name) + "\n\n"

    # Advanced cost analysis sections
    text += get_metric_cost_breakdown(page_name) + "\n\n"
    text += get_device_cost_breakdown(page_name) + "\n\n"
    text += get_cost_trend(page_name) + "\n\n"

    return text


# Concurrency configuration
MAX_WORKERS = 5  # Maximum concurrent API requests


def generate_full_report(
    pages: list[str],
    show_progress: bool = True,
    use_concurrency: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate complete report for multiple pages.

    Uses concurrent processing for faster multi-page reports.
    Shows tqdm progress bars when available.

    Args:
        pages: List of page names to analyze.
        show_progress: Whether to show progress indicators.
        use_concurrency: Whether to use parallel processing.

    Returns:
        Tuple of (Markdown formatted report, list of metric rows for export).
    """
    global _performance_data_cache

    # Clear the performance data cache at the start of each report
    _performance_data_cache.clear()

    table_rows: list[dict[str, Any]] = []
    total = len(pages)

    if show_progress and not TQDM_AVAILABLE:
        print_info(f"Gathering metrics for {total} page(s)...")

    # Use tqdm progress bar if available
    use_tqdm = show_progress and TQDM_AVAILABLE and sys.stdout.isatty()

    # Use concurrent processing for multiple pages
    if use_concurrency and total > 1:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total)) as executor:
            future_to_page = {
                executor.submit(gather_page_metrics, pg): pg for pg in pages
            }

            if use_tqdm:
                futures_iter = tqdm(
                    as_completed(future_to_page),
                    total=total,
                    desc="Gathering metrics",
                    unit="page",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )
            else:
                futures_iter = as_completed(future_to_page)

            completed = 0
            for future in futures_iter:
                pg = future_to_page[future]
                completed += 1
                if show_progress and not use_tqdm:
                    print_progress(completed, total, pg)
                try:
                    row = future.result()
                    if row is not None:
                        table_rows.append(row)
                except Exception as e:
                    logger.error("Error processing page %s: %s", pg, e)

        # Sort rows to maintain page order
        page_order = {pg: i for i, pg in enumerate(pages)}
        table_rows.sort(key=lambda r: page_order.get(r.get("page", ""), 999))
    else:
        # Sequential processing for single page
        if use_tqdm:
            page_iter = tqdm(
                pages,
                desc="Gathering metrics",
                unit="page",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        else:
            page_iter = pages

        for i, pg in enumerate(page_iter if use_tqdm else pages, 1):
            if show_progress and not use_tqdm:
                print_progress(i, total, pg)
            row = gather_page_metrics(pg)
            if row is not None:
                table_rows.append(row)

    summary_table = make_summary_table(table_rows)
    big_md = "# 🔍 Blue Triangle API Report\n\n"

    # Add analysis settings info
    settings_parts = []
    if selected_percentiles:
        settings_parts.append(f"**Percentile:** p{selected_percentiles[0]}")
    else:
        settings_parts.append("**Metric:** Average")
    if selected_data_type != "rum":
        settings_parts.append(f"**Data Type:** {selected_data_type.upper()}")
    if resource_group_by != "domain":
        settings_parts.append(f"**Resource Grouping:** {resource_group_by}")
    if segment_filter:
        settings_parts.append(f"**Segment:** {', '.join(segment_filter)}")
    if country_filter:
        settings_parts.append(f"**Country:** {', '.join(country_filter)}")
    if device_filter:
        settings_parts.append(f"**Device:** {', '.join(device_filter)}")
    if settings_parts:
        big_md += "> " + " | ".join(settings_parts) + "\n\n"

    big_md += "## Overall Summary Table\n"
    big_md += summary_table

    # Add page cost ranking for multi-page reports
    if len(pages) > 1:
        big_md += "\n" + get_page_cost_ranking(pages) + "\n"

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

    Temporarily modifies global time variables, then restores them.

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

    # Save original global state to restore later
    original_now = now
    original_one_day_ago = one_day_ago
    original_two_days_ago = two_days_ago

    comparison_data = []

    try:
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
    finally:
        # Always restore original global state
        now = original_now
        one_day_ago = original_one_day_ago
        two_days_ago = original_two_days_ago

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

    # Data filtering options
    filter_group = parser.add_argument_group("Data Filtering")
    filter_group.add_argument(
        "--segment",
        nargs="+",
        help="Filter by traffic segment(s) - must match exactly as configured in Blue Triangle",
    )
    filter_group.add_argument(
        "--country",
        nargs="+",
        help="Filter by country code(s) in ISO 3166 format (e.g., US, CA, GB)",
    )
    filter_group.add_argument(
        "--device",
        nargs="+",
        choices=["Desktop", "Mobile", "Tablet"],
        help="Filter by device type(s)",
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
        choices=["markdown", "json", "csv", "html", "pdf"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    output_group.add_argument(
        "--metrics",
        nargs="+",
        choices=["LCP", "TBT", "CLS", "INP", "FB"],
        help="Filter report to specific metrics",
    )

    # Notification options
    notify_group = parser.add_argument_group("Notification Options")
    notify_group.add_argument(
        "--slack-webhook",
        type=str,
        help="Slack incoming webhook URL for notifications",
    )
    notify_group.add_argument(
        "--teams-webhook",
        type=str,
        help="Microsoft Teams incoming webhook URL for notifications",
    )
    notify_group.add_argument(
        "--email-to",
        nargs="+",
        help="Email recipient(s) for report notification",
    )
    notify_group.add_argument(
        "--email-subject",
        type=str,
        default="Blue Triangle Performance Report",
        help="Email subject line (default: Blue Triangle Performance Report)",
    )
    notify_group.add_argument(
        "--email-attach",
        action="store_true",
        help="Attach report file to email notification",
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
    advanced_group.add_argument(
        "--percentile",
        type=int,
        choices=[50, 75, 90, 95, 99],
        help="Use percentile instead of average (50=median, 75, 90, 95, 99)",
    )
    advanced_group.add_argument(
        "--data-type",
        choices=["rum", "synthetic", "native", "basepage"],
        default="rum",
        help="Data type for performance queries (default: rum)",
    )
    advanced_group.add_argument(
        "--resource-group",
        choices=["domain", "file", "service"],
        default="domain",
        help="Group resources by domain, file, or service (default: domain)",
    )
    advanced_group.add_argument(
        "--resource-file",
        type=str,
        default=None,
        help="Filter and analyze specific resource files (supports wildcards, e.g., 'chunk.273*.js')",
    )
    advanced_group.add_argument(
        "--summary",
        action="store_true",
        help="Show executive friction summary: total revenue at risk, top issues, quick wins, and watch list",
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
    util_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without making API calls",
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

    opts="--page --top-pages --time-range --start --end --multi-range --compare --segment --country --device --output --format --metrics --slack-webhook --teams-webhook --email-to --email-subject --email-attach --config --cache --clear-cache --alerts --generate-completion --percentile --data-type --resource-group --resource-file --summary --test-connection --no-color --quiet --verbose --dry-run --help"

    case "${prev}" in
        --time-range)
            COMPREPLY=( $(compgen -W "qd hd 24h xd 2d 6d 7d 28d 30d 90d 1y 2y 3y" -- ${cur}) )
            return 0
            ;;
        --format|-f)
            COMPREPLY=( $(compgen -W "markdown json csv html pdf" -- ${cur}) )
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
        --percentile)
            COMPREPLY=( $(compgen -W "50 75 90 95 99" -- ${cur}) )
            return 0
            ;;
        --data-type)
            COMPREPLY=( $(compgen -W "rum synthetic native basepage" -- ${cur}) )
            return 0
            ;;
        --resource-group)
            COMPREPLY=( $(compgen -W "domain file service" -- ${cur}) )
            return 0
            ;;
        --device)
            COMPREPLY=( $(compgen -W "Desktop Mobile Tablet" -- ${cur}) )
            return 0
            ;;
        --output|-o|--config)
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
        --slack-webhook|--teams-webhook|--email-to|--email-subject)
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
    local -a opts time_ranges formats metrics shells percentiles data_types resource_groups devices

    time_ranges=(qd hd 24h xd 2d 6d 7d 28d 30d 90d 1y 2y 3y)
    formats=(markdown json csv html pdf)
    metrics=(LCP TBT CLS INP FB)
    shells=(bash zsh)
    percentiles=(50 75 90 95 99)
    data_types=(rum synthetic native basepage)
    resource_groups=(domain file service)
    devices=(Desktop Mobile Tablet)

    _arguments -C \\
        '--page[Specify page names]:page:' \\
        '--top-pages[Analyze top pages by views]' \\
        '--time-range[Time window]:range:($time_ranges)' \\
        '--start[Custom start time (epoch)]:timestamp:' \\
        '--end[Custom end time (epoch)]:timestamp:' \\
        '--multi-range[Multiple time ranges]:ranges:' \\
        '--compare[Compare two periods]:timestamps:' \\
        '--segment[Filter by traffic segment]:segment:' \\
        '--country[Filter by country code (ISO 3166)]:country:' \\
        '--device[Filter by device type]:device:($devices)' \\
        {-o,--output}'[Output filename]:file:_files' \\
        {-f,--format}'[Output format]:format:($formats)' \\
        '--metrics[Filter metrics]:metrics:($metrics)' \\
        '--slack-webhook[Slack webhook URL]:url:' \\
        '--teams-webhook[Teams webhook URL]:url:' \\
        '--email-to[Email recipients]:emails:' \\
        '--email-subject[Email subject]:subject:' \\
        '--email-attach[Attach report to email]' \\
        '--config[Config file path]:file:_files -g "*.yaml *.yml"' \\
        '--cache[Enable caching]' \\
        '--clear-cache[Clear cache]' \\
        '--alerts[Show threshold alerts]' \\
        '--generate-completion[Generate completion script]:shell:($shells)' \\
        '--percentile[Use percentile instead of average]:percentile:($percentiles)' \\
        '--data-type[Data type for queries]:type:($data_types)' \\
        '--resource-group[Group resources by]:grouping:($resource_groups)' \\
        '--resource-file[Filter resource files]:pattern:' \\
        '--summary[Show executive friction summary]' \\
        '--test-connection[Test API connection]' \\
        '--no-color[Disable colors]' \\
        {-q,--quiet}'[Suppress progress]' \\
        {-v,--verbose}'[Enable debug logging]' \\
        '--dry-run[Preview without API calls]' \\
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

    # Enable dry-run mode if requested
    global dry_run_mode
    if args.dry_run:
        dry_run_mode = True
        print_warning("DRY RUN MODE - No API calls will be made")

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

    # Set percentile, data type, and resource grouping options
    global selected_percentiles, selected_data_type, resource_group_by, resource_file_filter
    if args.percentile:
        selected_percentiles = [args.percentile]
        if show_progress:
            print_info(f"Using percentile: p{args.percentile}")
    if args.data_type:
        selected_data_type = args.data_type
        if show_progress and args.data_type != "rum":
            print_info(f"Data type: {args.data_type}")
    if args.resource_group:
        resource_group_by = args.resource_group
        if show_progress and args.resource_group != "domain":
            print_info(f"Resource grouping: {args.resource_group}")
    if args.resource_file:
        resource_file_filter = args.resource_file
        # When filtering by file, force resource grouping to "file"
        resource_group_by = "file"
        if show_progress:
            print_info(f"Resource file filter: {args.resource_file}")

    # Set segment, country, and device filters
    global segment_filter, country_filter, device_filter
    if args.segment:
        segment_filter = args.segment
        if show_progress:
            print_info(f"Traffic segment filter: {', '.join(args.segment)}")
    if args.country:
        country_filter = [c.upper() for c in args.country]  # Normalize to uppercase
        if show_progress:
            print_info(f"Country filter: {', '.join(country_filter)}")
    if args.device:
        device_filter = args.device
        if show_progress:
            print_info(f"Device filter: {', '.join(args.device)}")

    global now, one_day_ago, two_days_ago
    try:
        start, end, prev_start, prev_end, multi_list = parse_time_args(args)
    except ValueError as e:
        print_error(f"Invalid time argument: {e}")
        sys.exit(1)

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
        # Deduplicate while preserving order
        seen: set[str] = set()
        pages = []
        for page in df["pageName"].tolist():
            if page not in seen:
                seen.add(page)
                pages.append(page)
        if show_progress:
            print_success(f"Found {len(pages)} pages")
    elif args.page:
        # Normalize VT page names to correct casing (API is case-sensitive)
        pages = [normalize_vt_page_name(p) for p in args.page]
        if show_progress:
            # Show if any pages were normalized
            for orig, norm in zip(args.page, pages):
                if orig != norm:
                    print_info(f"Normalized page name: '{orig}' → '{norm}'")
            print_info(f"Analyzing {len(pages)} page(s): {', '.join(pages)}")
    else:
        if show_progress:
            print_info("Fetching available pages...")
        pages = update_available_pages(limit=20)
        if show_progress:
            print_success(f"Found {len(pages)} pages")

    # Handle --summary mode
    if args.summary:
        try:
            summary = generate_friction_summary(pages, show_progress=show_progress)
            print(summary)
            elapsed_time = time.time() - start_time
            if show_progress:
                print_info(f"Summary generated in {elapsed_time:.1f}s")
            sys.exit(0)
        except KeyboardInterrupt:
            print()
            print_warning("Operation cancelled by user")
            sys.exit(130)
        except Exception as e:
            print_error(f"Error generating summary: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

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
        elif output_format == "pdf":
            if not output_file.endswith(".pdf"):
                output_file = output_file.rsplit(".", 1)[0] + ".pdf"
            export_to_pdf(data_rows, final_md, output_file)
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

        # Send notifications if configured
        time_range_str = args.time_range if not args.compare else "comparison"
        report_summary = generate_report_summary(data_rows, len(pages), time_range_str)

        # Slack notification
        if args.slack_webhook:
            if show_progress:
                print_info("Sending Slack notification...")
            success, message = send_slack_notification(
                args.slack_webhook, report_summary, output_file
            )
            if success:
                print_success(message)
            else:
                print_error(message)

        # Teams notification
        if args.teams_webhook:
            if show_progress:
                print_info("Sending Teams notification...")
            success, message = send_teams_notification(
                args.teams_webhook, report_summary, output_file
            )
            if success:
                print_success(message)
            else:
                print_error(message)

        # Email notification
        if args.email_to:
            smtp_server = os.getenv("BT_SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("BT_SMTP_PORT", "587"))
            sender_email = os.getenv("BT_SMTP_EMAIL", "")
            sender_password = os.getenv("BT_SMTP_PASSWORD", "")

            if not sender_email or not sender_password:
                print_error("Email notification requires BT_SMTP_EMAIL and BT_SMTP_PASSWORD environment variables")
            else:
                if show_progress:
                    print_info(f"Sending email to {len(args.email_to)} recipient(s)...")
                attachment = output_file if args.email_attach else None
                success, message = send_email_notification(
                    smtp_server=smtp_server,
                    smtp_port=smtp_port,
                    sender_email=sender_email,
                    sender_password=sender_password,
                    recipient_emails=args.email_to,
                    subject=args.email_subject,
                    report_summary=report_summary,
                    attachment_path=attachment,
                )
                if success:
                    print_success(message)
                else:
                    print_error(message)

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
