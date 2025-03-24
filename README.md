# 📊 Blue Triangle Performance CLI

A Python-based CLI tool that connects to the Blue Triangle API and generates detailed Markdown reports on web performance, JavaScript errors, resource usage, and revenue opportunity metrics for specific pages on your site.

---

## 🚀 Features

- Fetch real-time and historical performance data (LCP, TBT, CLS, INP, etc.)
- Compare current vs. previous time ranges (delta and % change)
- Get revenue metrics and lost revenue opportunities
- Identify slow/fast resources per domain
- Generate reports across multiple time ranges
- Supports top N pages by page views
- Outputs report in clean Markdown format

---

## 🧰 Requirements

- Python 3.7+
- Blue Triangle API Key and Email

### Install dependencies:
```
pip install -r requirements.txt
```

#### Contents of requirements.txt:
```
pandas
requests
matplotlib
```
### 🔐 Configuration
Before running the CLI, open the script and set your credentials in the config section at the top:

```
EMAIL = "your@email.com"
API_KEY = "your_api_key"
ADD_PREFIX should be changed to your prefix
```
### 🛠 Usage
```
python script.py [OPTIONS]
```
#### Basic Example
```
python script.py --page pdp checkout --time-range 7d
```
Generates a performance report for pdp and checkout pages over the last 7 days.

### ⚙️ Command Line Options
```
Flag	Description	Example
--page	One or more specific page names to analyze	--page pdp checkout
--output	Output Markdown file name	--output pdp_report.md
--time-range	Predefined time range for analysis	--time-range 28d
--start	Custom start time (epoch timestamp)	--start 1710796800
--end	Custom end time (epoch timestamp)	--end 1710883200
--metrics	Filter report to show only selected metrics	--metrics LCP TBT CLS
--multi-range	Generate multiple reports for given ranges	--multi-range 24h,7d,28d
--top-pages	Analyze top 20 pages by page views	--top-pages
--help	Show CLI help	--help
```

### 🗂 Available Time Ranges
```
Code	Description
qd	Quarter day (~6 hours)
hd	Half day (12 hours)
24h	Last 24 hours
xd	1.5 days
2d	Last 2 days
6d	Last 6 days
7d	Last 7 days (Default)
28d	Last 28 days
30d	Last 30 days
90d	Last 90 days
1y	Last 1 year
2y	Last 2 years
3y	Last 3 years
```
### 📁 Sample Outputs
Summary Table
```
| Page     | Onload (Curr) | Onload (Prev) | LCP (Curr) | LCP (Prev) | ...
|----------|---------------|---------------|------------|------------|-----
| pdp      | 2800          | 2900          | 1800       | 2100       | ...
```

### Insights
- Performance deltas and insights (e.g., "LCP improved", "INP worsened")

- Resource usage comparison

- Revenue and lost revenue estimates

- Event markers from the Blue Triangle dashboard

### 🧪 Sample Full Command
```
python script.py \
  --page pdp checkout \
  --output report.md \
  --time-range 28d \
  --metrics LCP TBT \
  --multi-range 24h,7d,28d
```
### 📤 Output
Report is saved as a .md file.

Can be viewed in any Markdown viewer or GitHub/GitLab.

Includes charts (if enabled via matplotlib).

Optionally extend to export CSV or HTML in future versions.

### ✅ TODOs & Improvements
 Add support for exporting CSV or PDF

 Add Slack/email integration for auto-sharing

 Implement --metrics filtering in output

 Visualize charts automatically in Jupyter/CLI

##### 👨🏽‍💻 Author
Kanmi Obasa


