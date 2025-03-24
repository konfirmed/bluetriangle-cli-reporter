# 📊 Blue Triangle Performance CLI

A powerful Python CLI that connects to the [Blue Triangle API](https://help.bluetriangle.com/hc/en-us/articles/360034915953-The-Blue-Triangle-API-Overview) to analyze real user monitoring (RUM) data and revenue impact — helping teams **track performance regressions**, **identify slow resources**, and **uncover revenue opportunities**.

---

## 🚀 Features

- Fetch real-time and historical performance data (LCP, TBT, CLS, INP, etc.)
- Compare current vs. previous time ranges (delta and % change)
- Get revenue and lost revenue opportunity estimates
- Identify slow/fast resources per domain
- Generate reports across multiple time ranges
- Supports top N pages by page views
- Outputs detailed reports in clean Markdown format

### Sample Report Screenshot 
<img width="747" alt="Sample Report Screenshot" src="https://github.com/user-attachments/assets/b792bd2c-150b-455c-abfc-0c19f9cfd24c" />

---

## 🧰 Requirements

- Python 3.7+
- Blue Triangle site prefix, API Key, and email

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
```
pandas
requests
matplotlib
```

---

## 🔐 Configuration

Before running the CLI, open the script and set your credentials in the config section at the top:

```python
EMAIL = "your@email.com"
API_KEY = "your_api_key"
SITE_PREFIX = "your_site_prefix"
```

---

## 🛠 Usage

Run the CLI with your desired flags:

```bash
python script.py [OPTIONS]
```

### 🧪 Example

```bash
python script.py --page pdp checkout --time-range 7d
```

Generates a performance report for the PDP and Checkout pages over the last 7 days.

---

## 🆘 Command-Line Help

```bash
python script.py --help
```

```
usage: script.py [options]

Options:
  --page            One or more page names to analyze
  --output          Output filename (default: full_bluetriangle_report.md)
  --time-range      Time window (e.g. 24h, 7d, 28d)
  --start --end     Custom start/end epoch timestamps
  --metrics         Select metrics (LCP, TBT, CLS, etc.)
  --multi-range     Run report over multiple ranges (comma-separated)
  --top-pages       Use top 20 pages by views
```
---

## ⚙️ Supported Flags

| Flag            | Description                                    | Example                                |
|-----------------|------------------------------------------------|----------------------------------------|
| `--page`        | One or more specific page names to analyze     | `--page pdp checkout`                  |
| `--output`      | Output Markdown file name                      | `--output pdp_report.md`               |
| `--time-range`  | Predefined time range for analysis             | `--time-range 28d`                     |
| `--start`       | Custom start time (epoch timestamp)            | `--start 1710796800`                   |
| `--end`         | Custom end time (epoch timestamp)              | `--end 1710883200`                     |
| `--metrics`     | Filter report to show only selected metrics    | `--metrics LCP TBT CLS`                |
| `--multi-range` | Generate multiple reports for given ranges     | `--multi-range 24h,7d,28d`             |
| `--top-pages`   | Analyze top 20 pages by page views             | `--top-pages`                          |

---

## 📆 Available Time Ranges

| Code | Description              |
|------|--------------------------|
| qd   | Quarter day (~6 hours)   |
| hd   | Half day (12 hours)      |
| 24h  | Last 24 hours            |
| xd   | 1.5 days                 |
| 2d   | Last 2 days              |
| 6d   | Last 6 days              |
| 7d   | Last 7 days (Default)    |
| 28d  | Last 28 days             |
| 30d  | Last 30 days             |
| 90d  | Last 90 days             |
| 1y   | Last 1 year              |
| 2y   | Last 2 years             |
| 3y   | Last 3 years             |

---

## 📁 Sample Output

### 📋 Summary Table

```
| Page     | Onload (Curr) | Onload (Prev) | LCP (Curr) | LCP (Prev) | ...
|----------|---------------|---------------|------------|------------|-----
| pdp      | 2800          | 2900          | 1800       | 2100       | ...
```

### 🔍 Insights

- Performance deltas (e.g., "LCP improved", "INP worsened")
- Resource usage breakdowns (domain-level)
- Revenue & lost revenue opportunity
- Event markers from the Blue Triangle dashboard

---

## 📤 Output

Reports are saved as `.md` files and can be viewed in:

- Any Markdown viewer
- GitHub or GitLab
- VS Code preview

🔧 You can extend this tool to export CSV or HTML formats in future versions.

---

## ✅ TODOs & Improvements

- [ ] Export reports to CSV or PDF
- [ ] Add Slack/email integration for auto-sharing
- [ ] Implement `--metrics` filtering in the report body
- [ ] Auto-visualize reports in Jupyter or inline via CLI

---

## 👨🏽‍💻 Author

**Kanmi Obasa**  
[github.com/konfirmed](https://github.com/konfirmed) • [knfrmd.dev](https://knfrmd.dev)

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 🚀 Want Dashboards, Alerts, or Hosted UI?

Building a premium version with GitHub/Slack integration, automatic trend detection, and visual dashboards.

