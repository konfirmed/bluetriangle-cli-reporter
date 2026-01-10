# Blue Triangle CLI Reporter

![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/github/license/konfirmed/bluetriangle-cli-reporter)
![Issues](https://img.shields.io/github/issues/konfirmed/bluetriangle-cli-reporter)
![Last Commit](https://img.shields.io/github/last-commit/konfirmed/bluetriangle-cli-reporter)

A powerful Python CLI that connects to the [Blue Triangle API](https://help.bluetriangle.com/hc/en-us/articles/360034915953-The-Blue-Triangle-API-Overview) to analyze real user monitoring (RUM) data and revenue impact, helping teams **track performance regressions**, **identify slow resources**, and **uncover revenue opportunities**.

---

## Features

- Fetch real-time and historical performance data (LCP, TBT, CLS, INP, etc.)
- Compare current vs. previous time ranges (delta and % change)
- Get revenue and lost revenue opportunity estimates
- Identify slow/fast resources per domain
- Generate reports across multiple time ranges
- Export to **Markdown**, **JSON**, or **CSV** formats
- Colored terminal output with progress indicators
- Supports top N pages by page views

### Sample Report Screenshot
<img width="747" alt="Sample Report Screenshot" src="https://github.com/user-attachments/assets/b792bd2c-150b-455c-abfc-0c19f9cfd24c" />

---

## Requirements

- Python 3.9+
- Blue Triangle site prefix, API Key, and email

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/konfirmed/bluetriangle-cli-reporter.git
cd bluetriangle-cli-reporter
pip install -r requirements.txt
```

### 2. Configure Credentials

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your Blue Triangle credentials:

```bash
BT_API_EMAIL=your@email.com
BT_API_KEY=your_api_key_here
BT_SITE_PREFIX=your_site_prefix
```

### 3. Test Your Connection

```bash
python bt_insights.py --test-connection
```

### 4. Generate a Report

```bash
python bt_insights.py --page pdp checkout --time-range 7d --output report.md
```

---

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `BT_API_EMAIL` | Your Blue Triangle account email |
| `BT_API_KEY` | Your Blue Triangle API key |
| `BT_SITE_PREFIX` | Your site prefix (from dashboard URL) |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BT_REQUEST_TIMEOUT` | `30` | API request timeout in seconds |
| `BT_LOG_LEVEL` | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR |

---

## Usage Examples

### Basic Usage

```bash
# Analyze specific pages over 7 days
python bt_insights.py --page pdp checkout --time-range 7d

# Analyze top 20 pages by traffic
python bt_insights.py --top-pages --time-range 28d

# Focus on specific metrics only
python bt_insights.py --page homepage --metrics LCP INP CLS
```

### Output Formats

```bash
# Default: Markdown report
python bt_insights.py --page pdp -o report.md

# Export as JSON (for programmatic use)
python bt_insights.py --page pdp --format json -o metrics.json

# Export as CSV (for spreadsheets)
python bt_insights.py --page pdp --format csv -o metrics.csv

# Export as PDF with charts
python bt_insights.py --page pdp --format pdf -o report.pdf
```

### Multi-Range Reports

```bash
# Compare across multiple time ranges
python bt_insights.py --page pdp --multi-range 24h,7d,28d -o comparison.md
```

### Quiet Mode (for scripts)

```bash
# Suppress progress output
python bt_insights.py --page pdp --quiet -o report.md

# Disable colors (for piping/logging)
python bt_insights.py --page pdp --no-color
```

### HTML Reports with Charts

```bash
# Generate HTML report with embedded Chart.js visualizations
python bt_insights.py --page pdp checkout --format html -o report.html
```

### API Caching

```bash
# Enable caching to speed up repeated requests (1 hour TTL)
python bt_insights.py --page pdp --cache

# Clear the cache
python bt_insights.py --clear-cache
```

### Configuration File

```bash
# Use a custom config file
python bt_insights.py --page pdp --config my_config.yaml

# Default locations searched: bt_config.yaml, ~/.bt_config.yaml
```

### Threshold Alerts

```bash
# Show alerts for metrics exceeding Web Vitals thresholds
python bt_insights.py --page pdp --alerts
```

### Period Comparison

```bash
# Compare two time periods (epoch timestamps)
python bt_insights.py --page pdp --compare 1704067200 1704672000 1704672000 1705276800
```

### Shell Completion

```bash
# Generate bash completion script
python bt_insights.py --generate-completion bash > /etc/bash_completion.d/bt_insights

# Generate zsh completion script
python bt_insights.py --generate-completion zsh > ~/.zsh/completions/_bt_insights
```

### Debugging

```bash
# Enable verbose logging
python bt_insights.py --page pdp --verbose

# Test API connection
python bt_insights.py --test-connection
```

### Notifications (Slack/Teams/Email)

```bash
# Send report notification to Slack
python bt_insights.py --page pdp --slack-webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Send report notification to Microsoft Teams
python bt_insights.py --page pdp --teams-webhook https://outlook.office.com/webhook/YOUR/WEBHOOK/URL

# Send report via email (requires SMTP environment variables)
python bt_insights.py --page pdp --email-to recipient@example.com --email-subject "Weekly Report"

# Attach report file to email
python bt_insights.py --page pdp --email-to recipient@example.com --email-attach

# Combine multiple notifications
python bt_insights.py --page pdp --slack-webhook URL --email-to user@example.com
```

**Email Configuration:** Set these environment variables for email notifications:
- `BT_SMTP_SERVER` - SMTP server hostname (default: smtp.gmail.com)
- `BT_SMTP_PORT` - SMTP server port (default: 587)
- `BT_SMTP_EMAIL` - Sender email address
- `BT_SMTP_PASSWORD` - Sender email password or app password

---

## Command-Line Reference

```
python bt_insights.py --help
```

### Page Selection

| Flag | Description | Example |
|------|-------------|---------|
| `--page` | Specify page names to analyze | `--page pdp checkout` |
| `--top-pages` | Analyze top 20 pages by views | `--top-pages` |

### Time Range

| Flag | Description | Example |
|------|-------------|---------|
| `--time-range` | Predefined time window (default: 7d) | `--time-range 28d` |
| `--start` | Custom start time (epoch) | `--start 1710796800` |
| `--end` | Custom end time (epoch) | `--end 1710883200` |
| `--multi-range` | Multiple ranges, comma-separated | `--multi-range 24h,7d,28d` |

### Output Options

| Flag | Description | Example |
|------|-------------|---------|
| `--output`, `-o` | Output filename | `-o report.md` |
| `--format`, `-f` | Output format: markdown, json, csv, html, pdf | `--format pdf` |
| `--metrics` | Filter to specific metrics | `--metrics LCP TBT CLS` |
| `--alerts` | Show threshold alerts for metrics | `--alerts` |

### Notification Options

| Flag | Description | Example |
|------|-------------|---------|
| `--slack-webhook` | Slack incoming webhook URL | `--slack-webhook URL` |
| `--teams-webhook` | Microsoft Teams incoming webhook URL | `--teams-webhook URL` |
| `--email-to` | Email recipient(s) | `--email-to user@example.com` |
| `--email-subject` | Email subject line | `--email-subject "Report"` |
| `--email-attach` | Attach report file to email | `--email-attach` |

### Caching & Configuration

| Flag | Description | Example |
|------|-------------|---------|
| `--cache` | Enable API response caching | `--cache` |
| `--clear-cache` | Clear cache and exit | `--clear-cache` |
| `--config` | Path to YAML config file | `--config my_config.yaml` |

### Comparison Mode

| Flag | Description | Example |
|------|-------------|---------|
| `--compare` | Compare two time periods (4 epoch timestamps) | `--compare START1 END1 START2 END2` |

### Utility Options

| Flag | Description |
|------|-------------|
| `--test-connection` | Test API connection and exit |
| `--generate-completion` | Generate shell completion (bash/zsh) |
| `--dry-run` | Preview actions without making API calls |
| `--no-color` | Disable colored output |
| `--quiet`, `-q` | Suppress progress output |
| `--verbose`, `-v` | Enable debug logging |

---

## Available Time Ranges

| Code | Description |
|------|-------------|
| `qd` | Quarter day (~6 hours) |
| `hd` | Half day (12 hours) |
| `24h` | Last 24 hours |
| `xd` | 1.5 days |
| `2d` | Last 2 days |
| `6d` | Last 6 days |
| `7d` | Last 7 days (Default) |
| `28d` | Last 28 days |
| `30d` | Last 30 days |
| `90d` | Last 90 days |
| `1y` | Last 1 year |
| `2y` | Last 2 years |
| `3y` | Last 3 years |

---

## Available Metrics

| Code | Full Name |
|------|-----------|
| `LCP` | Largest Contentful Paint |
| `TBT` | Total Blocking Time |
| `CLS` | Cumulative Layout Shift |
| `INP` | Interaction to Next Paint |
| `FB` | First Byte (TTFB) |

---

## Sample Output

### Summary Table (Markdown)

```
| Page     | Onload (Curr) | Onload (Prev) | LCP (Curr) | LCP (Prev) | ...
|----------|---------------|---------------|------------|------------|-----
| pdp      | 2800          | 2900          | 1800       | 2100       | ...
```

### JSON Output

```json
[
  {
    "page": "pdp",
    "onload_curr": 2800,
    "onload_prev": 2900,
    "lcp_curr": 1800,
    "lcp_prev": 2100
  }
]
```

### Insights Included

- Performance deltas (e.g., "LCP improved", "INP worsened")
- Weighted performance scores
- Resource usage breakdowns (domain-level)
- Revenue & lost revenue opportunity
- Event markers from the Blue Triangle dashboard

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## TODOs & Future Improvements

- [x] Export reports to CSV and JSON
- [x] Implement `--metrics` filtering in reports
- [x] Add colored terminal output
- [x] Add progress indicators
- [x] Add connection testing
- [x] Add HTML reports with Chart.js visualizations
- [x] Add API response caching
- [x] Add YAML config file support
- [x] Add threshold alerting (Web Vitals)
- [x] Add time period comparison mode
- [x] Add shell completion scripts (bash/zsh)
- [x] Add CI/CD pipeline (GitHub Actions)
- [x] Add retry logic with exponential backoff
- [x] Add concurrent API requests
- [x] Add progress bars (tqdm)
- [x] Add dry-run mode
- [x] Add input validation
- [x] Add Slack/Teams/email integration for auto-sharing
- [x] Add PDF export option

---

## Author

**Kanmi Obasa**
[github.com/konfirmed](https://github.com/konfirmed)

---

## Want Dashboards, Alerts, or Hosted UI?

Building a premium version with GitHub/Slack integration, automatic trend detection, and visual dashboards.

## Questions or Requests

Have questions, ideas, or feature requests?
[Open an issue](https://github.com/konfirmed/bluetriangle-cli-reporter/issues)
