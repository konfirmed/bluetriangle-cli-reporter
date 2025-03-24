import time
import re
import requests
import pandas as pd
import argparse
from difflib import get_close_matches
import matplotlib.pyplot as plt

# =====================
# 1) CONFIG & CONSTANTS
# =====================

EMAIL = "xxx"
API_KEY = "xxx"

BASE_URL = "https://api.bluetriangletech.com"
HEADERS = {
    "X-API-Email": EMAIL,
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

ENDPOINTS = {
    "performance": "/performance",
    # "performance_hits": "/performance/hits",
    "resource": "/resource",
    "error": "/error",
    "event_markers": "/event-markers",
    "revenue_report": "/revenue-opportunity/report-date",
    "revenue": "/revenue-opportunity"
}

# You can still define a fallback list in case the API returns no data:
AVAILABLE_PAGES = ["homepage", "story-women", "story-mens", "story-holiday-gift-guide",
                   "cdp", "cdp-VT", "cdp-VT-LazyLoad", "search",
                   "pdp", "pdp-VT", "add-to-bag-cta", 
                   "My Bag", "My Bag-VT", "mybag-refresh-bag",
                   "checkout", "checkout-VT",
                   "account-login", "account-dashboard", "Membership",
                   ]

# We'll define these as None, so we can set them after parsing time arguments:
now = None
one_day_ago = None
two_days_ago = None

# ==================
# 2) HELPER FUNCTIONS
# ==================


def normalize_page_name(query: str):
    """Fuzzy-match user input to the canonical page name.

    The input query is normalized by converting to lowercase and removing non-alphanumeric characters,
    while the page names in AVAILABLE_PAGES remain unchanged.
    """
    # Normalize the query (remove non-alphanumerics and convert to lowercase)
    normalized_query = re.sub(r'[^a-z0-9]', '', query.strip().lower())

    # Build a mapping from normalized query to original page name in AVAILABLE_PAGES
    page_map = {
        page: re.sub(r'[^a-z0-9]', '', page.strip().lower())  # map raw page to its normalized form
        for page in AVAILABLE_PAGES
    }

    # If there's an exact match (query vs normalized page), return the raw page name
    if normalized_query in page_map.values():
        return next(page for page, normalized in page_map.items() if normalized == normalized_query)

    # Otherwise, use fuzzy matching to find the closest match
    matches = get_close_matches(normalized_query, page_map.values(), n=1, cutoff=0.4)
    if matches:
        return next(page for page, normalized in page_map.items() if normalized == matches[0])
    else:
        return None


def fetch_data(endpoint: str, payload=None, method="POST", params=None):
    """Generic function to fetch JSON from the Blue Triangle API."""
    url = BASE_URL + endpoint
    try:
        if method == "GET":
            r = requests.get(url, headers=HEADERS, params=params)
        else:
            r = requests.post(url, headers=HEADERS, json=payload)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")  # Log HTTP errors
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")  # Log connection errors
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")  # Log timeout errors
    except requests.exceptions.RequestException as err:
        print(f"Oops: {err}")  # Log other errors
    return None


def _to_float(val):
    """Convert a field to float, or None if not numeric."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch_top_page_names(limit=20, start=None, end=None):
    """
    Fetch top page names ordered by pageViews.
    Uses the performance endpoint.
    If start or end are not provided, falls back to global time variables.
    """
    global now, one_day_ago
    if start is None:
        start = one_day_ago if one_day_ago is not None else int(time.time()) - 86400
    if end is None:
        end = now if now is not None else int(time.time())
    payload = {
        "site": ADD_PREFIX,
        "start": start,
        "end": end,
        "dataColumns": ["pageViews"],
        "group": ["pageName", "url"],
        "limit": limit,
        "orderBy": [{"field": "pageViews", "direction": "DESC"}]
    }
    print("DEBUG: Top pages payload:", payload)
    data = fetch_data(ENDPOINTS["performance"], payload)
    print("DEBUG: Top pages API response:", data)
    if data and "data" in data:
        return pd.DataFrame(data["data"])
    else:
        return pd.DataFrame([])


def update_available_pages(limit=20):
    """Update AVAILABLE_PAGES dynamically from the API, keeping the raw page names."""
    global AVAILABLE_PAGES
    df = fetch_top_page_names(limit=limit)
    if not df.empty and 'pageName' in df.columns:
        # Store the raw page names as they are (no normalization applied)
        AVAILABLE_PAGES = df['pageName'].tolist()
    return AVAILABLE_PAGES


def plot_performance_metrics(metrics: dict):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['time'], metrics['lcp'], label='LCP')
    plt.title('LCP Over Time')
    plt.legend()
    plt.show()

# ================
# 3) TIME PARSING
# ================


def parse_time_args(args):
    """
    If --multi-range is provided, we return a list of multiple ranges (e.g., "24h,28d").
    Otherwise, parse single start/end or a named --time-range.

    Returns (start, end, prev_start, prev_end, multi_list).
    If multi_list is non-empty, skip single-range logic.
    """
    if args.multi_range:
        ranges = [x.strip() for x in args.multi_range.split(",")]
        return None, None, None, None, ranges

    # single range
    now_ts = int(time.time())

    if args.start and args.end:
        # custom epoch
        start = int(args.start)
        end = int(args.end)
        return start, end, None, None, []

    # else parse --time-range
    day_map = {
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
        "3y": 1095
    }
    days = day_map.get(args.time_range, 1)
    end = now_ts
    start = now_ts - (days * 86400)
    prev_end = start
    prev_start = start - (days * 86400)
    return start, end, prev_start, prev_end, []


def compute_time_window(range_str: str):
    """
    Convert a range like '24h' or '28d' to (start,end,prev_start,prev_end).
    """
    now_ts = int(time.time())
    day_map = {
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
        "3y": 1095
    }
    days = day_map.get(range_str, 1)
    end = now_ts
    start = now_ts - (days*86400)
    prev_end = start
    prev_start = start - (days*86400)
    return start, end, prev_start, prev_end

# =====================
# 4) PERFORMANCE LOGIC
# =====================


def summarize_performance(current: dict, previous: dict) -> str:
    """Weighted summary of performance metrics => arrow + bullet list."""
    
    def to_float(value):  # Changed to lowercase
        try:
            return float(value)
        except (ValueError, TypeError):  # Specify the exceptions being caught
            return None

    # Weight assignments for each metric
    weights = {
        "largestContentfulPaint": 4,
        "intToNextPaint": 3,           
        "cumulativeLayoutShift": 3,
        "onload": 0.75,                 
        "firstByte": 0.5,               
        "dns": 0.375,                     
        "tcp": 0.375                      
    }

    labels = {
        "largestContentfulPaint": "LCP",
        "intToNextPaint": "INP",
        "cumulativeLayoutShift": "CLS",
        "onload": "Onload",
        "firstByte": "First Byte",
        "dns": "DNS",
        "tcp": "TCP"
    }

    statements = []
    weighted_score = 0
    total_weight = sum(weights.values())  # Calculate total weight

    for key, label in labels.items():
        current_value = to_float(current.get(key))
        previous_value = to_float(previous.get(key))

        if current_value is None or previous_value is None:
            continue

        weight = weights.get(key, 1)  # Default weight to 1 if key is not in weights

        if current_value < previous_value:  # Indicates improvement
            statements.append(f"{label} improved")
            weighted_score += weight
        elif current_value > previous_value:  # Indicates worsening
            statements.append(f"{label} worsened")
            weighted_score -= weight
        else:  # No change
            statements.append(f"{label} stayed the same")

    # Determine arrow direction
    if weighted_score > 0:
        arrow = "▲"
    elif weighted_score < 0:
        arrow = "▼"
    else:
        arrow = "→"
    
    if not statements:
        return "### 📝 Summary\nNo performance data to compare.\n"

    summary_list = ", ".join(statements)

    # Return the summary including the weighted score and total, and time range
    # Add time range value to the summary
    return f"### 📝 Summary ({arrow} [Weighted Score: {weighted_score} / Total: {total_weight}])\n- {summary_list} \n"

def get_page_performance(page_name: str) -> str:
    """
    Fetch performance data for a given page and return a Markdown report.
    """
    # Set up payload for current window
    payload = {
        "site": "ADD_PREFIX",
        "start": one_day_ago,
        "end": now,
        "dataType": "rum",
        "dataColumns": [
            "onload", "dns", "tcp", "firstByte",
            "largestContentfulPaint", "totalBlockingTime",
            "cumulativeLayoutShift", "intToNextPaint"
        ],
        "group": ["time"],
        "limit": 1000,
        "order": "time",
        "sort": "asc",
        "pageName": page_name
    }

    # Prepare payload for previous window
    prev_payload = dict(payload)
    if two_days_ago is not None:
        prev_payload["start"] = two_days_ago
        prev_payload["end"] = one_day_ago

    # Add debug print statements here:
    print(f"[DEBUG] Fetching current data for {page_name}: {payload}")
    print(f"[DEBUG] Fetching previous data for {page_name}: {prev_payload}")

    data_today = fetch_data(ENDPOINTS["performance"], payload)
    data_prev = fetch_data(ENDPOINTS["performance"], prev_payload) if two_days_ago else None

    if not data_today or not data_prev or "data" not in data_today or "data" not in data_prev:
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

    # Helper functions for formatting
    def n(val):
        """Return the value as float if numeric, else 'N/A'."""
        if val in (None, ""):
            return "N/A"

        try:
            return float(val)
        except (ValueError, TypeError):
            return "N/A"

    def delta(a, b):
        if a == "N/A" or b == "N/A":
            return "N/A"
        return round(float(a) - float(b), 2)

    def percent_change(current, previous):
        """Calculate the percentage change from previous to current."""
        if previous is None or previous == 0:
            return "N/A"  # return 'N/A' if previous is None or zero to avoid division by zero

        try:
            change = ((current - previous) / previous) * 100
            return round(change, 2)  # round to two decimal places
        except TypeError:
            return "N/A"  # return 'N/A' if the inputs are of incorrect type

    lcp_current = n(t.get('largestContentfulPaint'))
    lcp_previous = n(p.get('largestContentfulPaint'))

    delta_lcp = delta(lcp_current, lcp_previous)  # Use your existing delta function for diffs.

    # Add logic to check if LCP is "N/A"
    if lcp_current == "N/A" or lcp_previous == "N/A":
        lcp_insight = "LCP data is not available for comparison."
    else:
        if delta_lcp > 0:
            lcp_insight = "Actions are crucial for optimizing the LCP metric."
        else:
            lcp_insight = "LCP is stable."

    perf_summary = summarize_performance(t, p)

    return f"""
    {perf_summary}

    ### 📊 Performance Metrics for {page_name}

    #### Current Window
    - **Onload Time**: {n(t.get('onload'))} ms
    - **DNS Lookup Time**: {n(t.get('dns'))} ms
    - **TCP Connection Time**: {n(t.get('tcp'))} ms
    - **First Byte Time**: {n(t.get('firstByte'))} ms
    - **Largest Contentful Paint (LCP)**: {n(t.get('largestContentfulPaint'))} ms
    - **Total Blocking Time (TBT)**: {n(t.get('totalBlockingTime'))} ms
    - **Input Delay (INP)**: {n(t.get('intToNextPaint'))} ms
    - **Cumulative Layout Shift (CLS)**: {n(t.get('cumulativeLayoutShift'))}

    #### Previous Window
    - **Onload Time**: {n(p.get('onload'))} ms
    - **DNS Lookup Time**: {n(p.get('dns'))} ms
    - **TCP Connection Time**: {n(p.get('tcp'))} ms
    - **First Byte Time**: {n(p.get('firstByte'))} ms
    - **Largest Contentful Paint (LCP)**: {n(p.get('largestContentfulPaint'))} ms
    - **Total Blocking Time (TBT)**: {n(p.get('totalBlockingTime'))} ms
    - **Input Delay (INP)**: {n(p.get('intToNextPaint'))} ms
    - **Cumulative Layout Shift (CLS)**: {n(p.get('cumulativeLayoutShift'))}

    ### 📉 Performance Change (Delta)
    - **Onload Time**: Δ {delta(n(t.get('onload')), n(p.get('onload')))} ms ({percent_change(n(t.get('onload')), n(p.get('onload')))}%)
    - **DNS Lookup Time**: Δ {delta(n(t.get('dns')), n(p.get('dns')))} ms ({percent_change(n(t.get('dns')), n(p.get('dns')))}%)
    - **TCP Connection Time**: Δ {delta(n(t.get('tcp')), n(p.get('tcp')))} ms ({percent_change(n(t.get('tcp')), n(p.get('tcp')))}%)
    - **First Byte Time**: Δ {delta(n(t.get('firstByte')), n(p.get('firstByte')))} ms ({percent_change(n(t.get('firstByte')), n(p.get('firstByte')))}%)
    - **Largest Contentful Paint (LCP)**: Δ {delta(n(t.get('largestContentfulPaint')), n(p.get('largestContentfulPaint')))} ms ({percent_change(n(t.get('largestContentfulPaint')), n(p.get('largestContentfulPaint')))}%)
    - **Total Blocking Time (TBT)**: Δ {delta(n(t.get('totalBlockingTime')), n(p.get('totalBlockingTime')))} ms ({percent_change(n(t.get('totalBlockingTime')), n(p.get('totalBlockingTime')))}%)
    - **Input Delay (INP)**: Δ {delta(n(t.get('intToNextPaint')), n(p.get('intToNextPaint')))} ms ({percent_change(n(t.get('intToNextPaint')), n(p.get('intToNextPaint')))}%)
    - **Cumulative Layout Shift (CLS)**: Δ {delta(n(t.get('cumulativeLayoutShift')), n(p.get('cumulativeLayoutShift')))} ({percent_change(n(t.get('cumulativeLayoutShift')), n(p.get('cumulativeLayoutShift')))}%)

    ### 🛠 Optimization Insights
    - **Onload Time**: {"Improve server response time or defer offscreen images." if delta(n(t.get('onload')), n(p.get('onload'))) > 0 else "Maintain good performance."}
    - **LCP**: 
      - **Recommendations**: 
        - **Preload** images and video poster images to ensure they load immediately.
        - **Remove lazy-loading** for main content images that contribute to the LCP.
        - {lcp_insight}
    - **INP**: "Optimize your JavaScript and CSS to reduce blocking time.
    - **CLS**: "Ensure that dimension attributes for images and video elements are set.
    """

# ============================
# TABLE & METRICS FUNCTIONS
# ============================


def safe_delta(curr, prev):
    if curr is None or prev is None:
        return "N/A"
    return round(curr - prev, 2)


def gather_page_metrics(page_name: str):
    """
    Fetch current and previous page metrics and return a dictionary.
    """
    global now, one_day_ago, two_days_ago

    payload = {
        "site": "ADD_PREFIX",
        "start": one_day_ago,
        "end": now,
        "dataType": "rum",
        "dataColumns": [
            "onload", "dns", "tcp", "firstByte",
            "largestContentfulPaint", "totalBlockingTime",
            "cumulativeLayoutShift", "intToNextPaint"
        ],
        "group": ["time"],
        "limit": 1000,
        "order": "time",
        "sort": "asc",
        "pageName": page_name
    }

    prev_payload = dict(payload)
    if two_days_ago:
        prev_payload["start"] = two_days_ago
        prev_payload["end"] = one_day_ago

    data_curr = fetch_data(ENDPOINTS["performance"], payload)
    data_prev = fetch_data(ENDPOINTS["performance"], prev_payload) if two_days_ago else None

    if not data_curr or not data_prev or "data" not in data_curr or "data" not in data_prev:
        print(f"⚠️ No data fetched for {page_name}")
        return None

    df_curr = pd.DataFrame(data_curr["data"])
    df_prev = pd.DataFrame(data_prev["data"])

    if df_curr.empty or df_prev.empty:
        print(f"⚠️ Empty data returned for {page_name}")
        return None

    latest_curr = df_curr.iloc[-1]
    latest_prev = df_prev.iloc[-1]

    row = {
        "page": page_name,
        "onload_curr": _to_float(latest_curr.get("onload")),
        "onload_prev": _to_float(latest_prev.get("onload")),
        "lcp_curr":    _to_float(latest_curr.get("largestContentfulPaint")),
        "lcp_prev":    _to_float(latest_prev.get("largestContentfulPaint")),
        "inp_curr":    _to_float(latest_curr.get("intToNextPaint")),
        "inp_prev":    _to_float(latest_prev.get("intToNextPaint")),
        "cls_curr":    _to_float(latest_curr.get("cumulativeLayoutShift")),
        "cls_prev":    _to_float(latest_prev.get("cumulativeLayoutShift")),
        "tbt_curr":    _to_float(latest_curr.get("totalBlockingTime")),
        "tbt_prev":    _to_float(latest_prev.get("totalBlockingTime")),
        "fb_curr":     _to_float(latest_curr.get("firstByte")),
        "fb_prev":     _to_float(latest_prev.get("firstByte")),
        "dns_curr":    _to_float(latest_curr.get("dns")),
        "dns_prev":    _to_float(latest_prev.get("dns")),
        "tcp_curr":    _to_float(latest_curr.get("tcp")),
        "tcp_prev":    _to_float(latest_prev.get("tcp")),
    }
    # Check if this is a VT page
    is_vt = "vt" in page_name.lower()
    # Debug output for troubleshooting
    # Debug info: use safe_delta as before but conditionally
    onload_delta = safe_delta(row['onload_curr'], row['onload_prev'])
    lcp_delta = safe_delta(row['lcp_curr'], row['lcp_prev']) if not is_vt else "N/A"
    inp_delta = safe_delta(row['inp_curr'], row['inp_prev'])
    cls_delta = safe_delta(row['cls_curr'], row['cls_prev']) if not is_vt else "N/A"

    print(f"[DEBUG] {page_name} Delta: Onload: {onload_delta}, "
          f"LCP: {lcp_delta}, "
          f"INP: {inp_delta}, "
          f"CLS: {cls_delta}")

    return row


def make_summary_table(rows: list) -> str:
    """
    Build a Markdown table summarizing key metrics from each page.
    """
    if not rows:
        return "(No data for summary table)\n\n"

    md = ("| Page | Onload (Curr) | Onload (Prev) | LCP (Curr) | LCP (Prev) | TBT (Curr) | TBT (Prev) | INP (Curr) "
          "| INP (Prev) | CLS (Curr) | CLS (Prev) |\n")
    md += ("|------|---------------|---------------|------------|------------|------------|------------|------------"
           "|------------|------------|------------|\n")
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
        md += (f"| {page} | {round(oc,2)} | {round(op,2)} | {round(lc,2)} | {round(lp,2)} | {round(tc,2)} "
               f"| {round(tp,2)} | {round(ic,2)} | {round(ip,2)} | {round(cc,2)} | {round(cp,2)} |\n")

    md += "\n"
    return md

# ========== OTHER STUFF: Revenue, hits, errors, etc. ==========


def get_event_markers() -> str:
    data = fetch_data(ENDPOINTS["event_markers"], method="GET", params={"prefix": "ADD_PREFIX"})
    if not data or "data" not in data:
        return "> ⚠️ No event markers found.\n"
    df = pd.DataFrame(data["data"])
    if df.empty:
        return "> ⚠️ No event markers available.\n"
    lines = [f"- {row['eventName']}: {row['annotation']} "
             f"({row['eventStart']} - {row.get('eventEnd','N/A')})"
             for _, row in df.iterrows()]
    return "### 📅 Event Markers\n" + "\n".join(lines)


def get_js_errors(page_name: str) -> str:
    """
    Retrieve aggregated JavaScript error data (grouped by time and errorConstructor)
    from the /error endpoint.

    Expected response records:
    {
        "time": "1691002800000",
        "errorConstructor": "CSP Violation",
        "errorCount": "8"
    }
    """
    global now, one_day_ago
    payload = {
        "site": "ADD_PREFIX",
        "start": one_day_ago,
        "end": now,
        "dataType": "rum",  # Using RUM errors for JavaScript issues.
        "dataColumns": ["errorCount"],
        "pageName": page_name,
        "group": ["time", "errorConstructor"],
        "order": "time",
        "sort": "asc",
        "limit": 50000
    }
    # All requests to the /error endpoint must be POST.
    data = fetch_data(ENDPOINTS["error"], payload, method="POST")
    if not data or not isinstance(data, list) or len(data) == 0:
        return f"> ⚠️ No aggregated JS error data available for **{page_name}**.\n"

    lines = [f"### ⚠️ Aggregated JS Errors ({page_name})"]
    for record in data:
        # The API returns the time in epoch milliseconds.
        time_epoch = record.get("time", "N/A")
        error_constructor = record.get("errorConstructor", "N/A")
        error_count = record.get("errorCount", "N/A")
        lines.append(f"- Time: {time_epoch} | {error_constructor}: {error_count} errors")

    return "\n".join(lines)


def summarize_resource_usage(current_df: pd.DataFrame, prev_df: pd.DataFrame) -> str:
    # Check if either DataFrame is empty
    if current_df.empty or prev_df.empty:
        return "> ⚠️ Cannot summarize resource usage (no data)."

    # Use a dictionary to map domain to duration for both current and previous data
    current_map = dict(zip(current_df["domain"], current_df["duration"]))
    prev_map = dict(zip(prev_df["domain"], prev_df["duration"]))

    # List to hold changes in duration per domain
    changes = []

    # Calculate differences only for domains present in both maps
    for domain in current_map:
        if domain in prev_map:
            c_val = _to_float(current_map[domain])
            p_val = _to_float(prev_map[domain])
            if c_val is not None and p_val is not None:
                diff = c_val - p_val
                changes.append((domain, diff))

    # Sort changes to identify slowdowns and speedups
    changes.sort(key=lambda x: x[1], reverse=True)

    # Get top slowdowns and speedups
    slowdowns = [d for d in changes if d[1] > 0][:3]
    speedups = sorted([d for d in changes if d[1] < 0], key=lambda x: x[1])[:3]

    def format_changes(lst):
        """Format the changes for output."""
        return ", ".join([f"**{d}** ({'+' if v > 0 else ''}{round(v, 2)} ms)" for d, v in lst]) or "(none)"

    # Prepare the summary output
    summary = "### 📝 Resource Summary\n"
    summary += "#### Top Slowdowns\n"
    summary += f"- {format_changes(slowdowns)}\n"
    summary += "#### Top Speedups\n"
    summary += f"- {format_changes(speedups)}\n"

    return summary


def get_resource_data(page_name: str, compare_previous=True) -> str:
    global now, one_day_ago, two_days_ago

    payload_curr = {
        "site": "ADD_PREFIX",
        "start": one_day_ago,
        "end": now,
        "dataColumns": ["duration", "elementCount"],
        "group": ["domain"],
        "pageName[]": [page_name]
    }

    payload_prev = dict(payload_curr)
    if compare_previous and two_days_ago is not None:
        payload_prev["start"] = two_days_ago
        payload_prev["end"] = one_day_ago

    # Fetch current data
    data_curr = fetch_data(ENDPOINTS["resource"], payload_curr)
    if not data_curr or "data" not in data_curr:
        return f"> ⚠️ No resource data for **{page_name}**.\n"

    df_curr = pd.DataFrame(data_curr["data"])
    if df_curr.empty:
        return f"> ⚠️ Resource data empty for **{page_name}**.\n"

    # Start building the resource markdown
    resource_text = f"### 📦 Resource Usage ({page_name})\n"
    lines = [
        f"- {row['domain']}: {row['duration']} ms, {row['elementCount']} elements"
        for _, row in df_curr.iterrows()
    ]
    resource_text += "\n".join(lines)

    # Compare with previous if available
    if compare_previous and two_days_ago is not None:
        data_prev = fetch_data(ENDPOINTS["resource"], payload_prev)
        if data_prev and "data" in data_prev:
            df_prev = pd.DataFrame(data_prev["data"])
            if not df_prev.empty:
                summary = summarize_resource_usage(df_curr, df_prev)
                resource_text = summary + "\n\n" + resource_text

    return resource_text


def get_performance_hits(page_name: str) -> str:
    global now, one_day_ago

    payload = {
        "site": "ADD_PREFIX",
        "start": one_day_ago,
        "end": now,
        "dataType": "rum",
        "dataColumns": ["measurementTime", "httpCode", "url"],
        "pageName[]": [page_name],
        "limit": 1000
    }

    data = fetch_data(ENDPOINTS["performance_hits"], payload, method="POST")

    if not data or "data" not in data or not data["data"]:
        return f"> ⚠️ No performance data for **{page_name}** (Current window).\n"

    if not data or "data" not in data or not data["data"]:
        return f"> ⚠️ No performance data for **{page_name}** (Previous window).\n"

    df = pd.DataFrame(data["data"])
    if df.empty:
        return f"> ⚠️ No performance hits data found for **{page_name}**."

    urls = df["url"].value_counts().head(5)
    return f"### 🔗 Top URLs for {page_name}\n" + "\n".join([f"- {u} ({c} hits)" for u, c in urls.items()])

# ========== REVENUE REPORT ==========


def get_latest_revenue_date():
    """Helper to find the latest revenue date if needed."""
    data = fetch_data(ENDPOINTS["revenue_report"], method="GET", params={
        "prefix": "ADD_PREFIX",
        "salesType": "revenue",
        "latest": "true"
    })
    
    print("Latest Revenue Date Response:", data)  # Debugging Line
    
    if data and isinstance(data, list) and len(data) > 0:
        if "reportDate" in data[0]:
            return data[0]["reportDate"]
    
    return None


def get_latest_revenue_opportunity_date():
    """
    Fetch the latest report date for revenue opportunity data.
    Endpoint: /revenue-opportunity/report-date
    """
    params = {
        "prefix": "ADD_PREFIX",
        "salesType": "revenue",  # or "brand" if using the Brand Opportunity endpoint
        "latest": "true"
    }
    data = fetch_data(ENDPOINTS["revenue_report"], method="GET", params=params)
    print("Latest Revenue Opportunity Date Response:", data)  # Debugging
    if data and isinstance(data, list) and len(data) > 0:
        if "reportDate" in data[0]:
            return data[0]["reportDate"]
    return None


def get_revenue_opportunity(page_name: str = None, device: list = None) -> str:
    """
    Fetch the Revenue Opportunity report for the given page.
    The function iterates over device-specific groups (e.g. Desktop, Tablet)
    and combines the lost revenue numbers by taking only the most recent reported values.
    """
    report_date = get_latest_revenue_opportunity_date()
    print("Revenue Opportunity Report Date:", report_date)
    if not report_date:
        return "> ⚠️ No Revenue Opportunity report found.\n"

    params = {
        "prefix": "ADD_PREFIX",
        "salesType": "revenue",
        "reportDate": report_date,
    }
    # If filtering by page or device, add those parameters.
    if page_name:
        params["pageName[]"] = [page_name]
    if device:
        params["device[]"] = device

    opp_data = fetch_data(ENDPOINTS["revenue"], method="GET", params=params)
    print("Revenue Opportunity Data Response:", opp_data)
    if not opp_data:
        return "> ⚠️ No Revenue Opportunity data found. (Empty response)\n"

    # Determine which device groups to iterate through.
    # If the 'device' parameter was provided, use that list; otherwise inspect all top-level keys.
    devices_to_check = device if device else list(opp_data.keys())
    
    total_lost_revenue = 0.0
    details = []
    for dev in devices_to_check:
        dev_data = opp_data.get(dev)
        if not dev_data:
            continue
        revenue_section = dev_data.get("revenue", {})
        if page_name not in revenue_section:
            continue
        page_data = revenue_section[page_name]
        device_lost_revenue = 0.0

        # Instead of summing all values, take only the last value from each list (if available)
        try:
            if "speedUpToXData" in page_data and "lostRevenue" in page_data["speedUpToXData"]:
                arr = page_data["speedUpToXData"]["lostRevenue"]
                if arr and len(arr) > 0:
                    device_lost_revenue += float(arr[-1])
            if "speedUpByXData" in page_data and "lostRevenue" in page_data["speedUpByXData"]:
                arr = page_data["speedUpByXData"]["lostRevenue"]
                if arr and len(arr) > 0:
                    device_lost_revenue += float(arr[-1])
        except Exception as exc:
            print(f"Error converting lost revenue values: {exc}")
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
    """Fetch revenue metrics for a given page with previous day comparison."""

    report_date = get_latest_revenue_date()
    print("Report Date:", report_date)  # Debugging Line

    if not report_date:
        return f"> ⚠️ No revenue data found for **{page_name}**.\n"

    params = {
        "prefix": "ADD_PREFIX",
        "salesType": "revenue",
        "reportDate": report_date,
        "pageName[]": [page_name]
    }

    # Fetch today's revenue data
    data_today = fetch_data(ENDPOINTS["revenue"], method="GET", params=params)

    # Instead of checking for "data", check if data_today is not empty and has device keys.
    if not data_today or not (data_today.get("Desktop") or data_today.get("Tablet") or data_today.get("Mobile")):
        return f"> ⚠️ No revenue data found for **{page_name}**. (Check API response structure)\n"

    # Check if we have specific device data (we expect "conversions" to have our page).
    device_data = data_today.get("Desktop") or data_today.get("Tablet") or data_today.get("Mobile")
    if not device_data or page_name not in device_data.get("conversions", {}):
        return f"> ⚠️ No revenue data available for **{page_name}** on any device type.\n"

    # Convert today's revenue conversion data into a DataFrame
    df_today = pd.DataFrame(device_data["conversions"][page_name]["conversionsData"])
    if df_today.empty:
        return f"> ⚠️ Revenue data is empty for **{page_name}**.\n"

    # Extract the latest metrics from the DataFrame
    latest_data = df_today.iloc[-1]

    # Prepare output for today's data (note: adjust if needed; sometimes there is no 'orders' key)
    text = f"""### 💰 Revenue Metrics for {page_name}
- Revenue: ${latest_data.get('revenue', 'N/A')}
- Orders: {latest_data.get('orders', 'N/A')}
- Visitors: {latest_data.get('visitors', 'N/A')}
"""

    # Then follow similar logic for previous day's data...
    previous_date = report_date - 86400
    params["reportDate"] = previous_date
    data_prev = fetch_data(ENDPOINTS["revenue"], method="GET", params=params)
    if not data_prev or not (data_prev.get("Desktop") or data_prev.get("Tablet") or data_prev.get("Mobile")):
        text += "\nNo previous revenue data for comparison."
        return text

    # (Continue similar extraction for previous day's revenue data here.)

    return text
# ========== BUILD PAGE REPORT ==========


def build_page_report(page_name: str) -> str:
    """
    Combine performance, resource usage, revenue, hits, JS errors
    for a single page into one markdown block.
    """
    time_range_value = f"Time Range: {one_day_ago} to {now}"
    text = f"## 📄 Page: {page_name}\n{time_range_value}\n"
    text += get_page_performance(page_name) + "\n\n"
    text += get_resource_data(page_name, compare_previous=True) + "\n\n"
    text += get_page_revenue(page_name) + "\n\n"
    text += get_revenue_opportunity(page_name) + "\n\n"
    # text += get_performance_hits(page_name) + "\n\n"
    # text += get_js_errors(page_name) + "\n\n"
    return text


def generate_full_report(pages):
    """
    1) Create a summary table for all pages
    2) Then produce the big page breakdown for each
    3) Then add event markers at the very end
    """
    # gather table rows
    table_rows = []
    for pg in pages:
        row = gather_page_metrics(pg)
        if row is not None:
            table_rows.append(row)

    summary_table = make_summary_table(table_rows)
    big_md = "# 🔍 Blue Triangle API Report\n\n"
    big_md += "## Overall Summary Table\n"
    big_md += summary_table

    # detailed breakdown
    for pg in pages:
        big_md += build_page_report(pg)
        big_md += "---\n\n"

    # optional: event markers
    big_md += get_event_markers() + "\n"
    return big_md


def generate_multi_range_report(ranges, pages):
    """
    For each range (like '24h','7d','28d'), we build a big block
    with a summary table + page detail, then combine them.
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


def analyze_trends(data_frame: pd.DataFrame):
    trends = {'lcp': data_frame['LCP'].pct_change() * 100, 'tbt': data_frame['TBT'].pct_change() * 100,
              'cls': data_frame['CLS'].pct_change() * 100, 'inp': data_frame['INP'].pct_change() * 100,
              'fb': data_frame['FB'].pct_change() * 100, 'date': data_frame['date']}

    trends = pd.DataFrame(trends)
    trends = trends.set_index('date')
    trends = trends.dropna()
    return trends


def visualize_trends(trends):
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


def save_report(content: str, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

# ========== MAIN ==========


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a Blue Triangle performance & revenue report.")
    parser.add_argument("--page", nargs='+', help="Specify one or more page names (e.g. PDP CDP Checkout).")
    parser.add_argument("--output", type=str, default="full_bluetriangle_report.md", help="Output filename")
    parser.add_argument("--time-range", choices=["qd","hd","24h", "xd", "2d", "6d", "7d", "28d", "30d",
                                                 "90d", "1y", "2y", "3y"], default="7d",
                        help="Select one of the predefined time windows.")
    parser.add_argument("--start", type=int, help="Custom start time (epoch) – overrides --time-range")
    parser.add_argument("--end", type=int, help="Custom end time (epoch) – overrides --time-range")
    parser.add_argument("--metrics", nargs='+', choices=["LCP", "TBT", "CLS", "INP", "FB"],
                        help="Specify which metrics to include in the report.")
    parser.add_argument("--multi-range", type=str,
                        help="Comma separated, e.g. '24h,28d,90d' for multiple sections in a single report")
    parser.add_argument("--top-pages", action="store_true", help="Generate report for top pages by page views")
    return parser.parse_args()


def validate_pages(input_pages, available_pages):
    valid_pages = []
    for p in input_pages:
        if p in available_pages:
            valid_pages.append(p)
        else:
            print(f"❌ Unknown page '{p}'")
    return valid_pages if valid_pages else None



def main():
    args = parse_arguments()

    global now, one_day_ago, two_days_ago
    start, end, prev_start, prev_end, multi_list = parse_time_args(args)
    now, one_day_ago, two_days_ago = end, start, prev_start

    if args.top_pages:
        pages = fetch_top_page_names(limit=20)['pageName'].tolist()
        if pages is None:
            return
    elif args.page:
        pages = validate_pages(args.page, AVAILABLE_PAGES)
        if pages is None:
            return
    else:
        pages = update_available_pages(limit=20)

    try:
        final_md = generate_multi_range_report(multi_list, pages) if multi_list else generate_full_report(pages)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_md)
        print(f"✅ Report saved to '{args.output}'")
    except Exception as e:
        print(f"❌ Error saving report: {e}")

if __name__ == "__main__":
    main()
