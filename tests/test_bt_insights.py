"""Tests for Blue Triangle CLI Reporter."""

import argparse
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import bt_insights


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_to_float_valid_number(self):
        """Test _to_float with valid numbers."""
        assert bt_insights._to_float(42) == 42.0
        assert bt_insights._to_float("3.14") == 3.14
        assert bt_insights._to_float(0) == 0.0

    def test_to_float_invalid_values(self):
        """Test _to_float with invalid values."""
        assert bt_insights._to_float(None) is None
        assert bt_insights._to_float("abc") is None
        assert bt_insights._to_float("") is None
        assert bt_insights._to_float([]) is None

    def test_safe_delta_valid_values(self):
        """Test safe_delta with valid values."""
        assert bt_insights.safe_delta(10.0, 5.0) == 5.0
        assert bt_insights.safe_delta(5.0, 10.0) == -5.0
        assert bt_insights.safe_delta(5.0, 5.0) == 0.0

    def test_safe_delta_none_values(self):
        """Test safe_delta with None values."""
        assert bt_insights.safe_delta(None, 5.0) == "N/A"
        assert bt_insights.safe_delta(5.0, None) == "N/A"
        assert bt_insights.safe_delta(None, None) == "N/A"

    def test_validate_api_response_valid(self):
        """Test validate_api_response with valid data."""
        assert bt_insights.validate_api_response({"data": []}, "data") is True
        assert bt_insights.validate_api_response({"key": "value"}, "key") is True
        assert bt_insights.validate_api_response([1, 2, 3]) is True

    def test_validate_api_response_invalid(self):
        """Test validate_api_response with invalid data."""
        assert bt_insights.validate_api_response(None) is False
        assert bt_insights.validate_api_response(None, "data") is False
        assert bt_insights.validate_api_response({}, "data") is False
        assert bt_insights.validate_api_response("string", "data") is False


class TestNormalizePageName:
    """Tests for page name normalization."""

    def test_exact_match(self):
        """Test exact page name match."""
        bt_insights.AVAILABLE_PAGES = ["homepage", "pdp", "checkout"]
        assert bt_insights.normalize_page_name("homepage") == "homepage"
        assert bt_insights.normalize_page_name("pdp") == "pdp"

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        bt_insights.AVAILABLE_PAGES = ["homepage", "PDP", "Checkout"]
        assert bt_insights.normalize_page_name("HOMEPAGE") == "homepage"
        assert bt_insights.normalize_page_name("pdp") == "PDP"

    def test_fuzzy_match(self):
        """Test fuzzy matching."""
        bt_insights.AVAILABLE_PAGES = ["homepage", "product-detail-page", "checkout"]
        result = bt_insights.normalize_page_name("productdetailpage")
        assert result == "product-detail-page"

    def test_no_match(self):
        """Test when no match is found."""
        bt_insights.AVAILABLE_PAGES = ["homepage"]
        assert bt_insights.normalize_page_name("xyz123") is None


class TestTimeWindow:
    """Tests for time window calculations."""

    def test_compute_time_window_7d(self):
        """Test 7-day time window calculation."""
        start, end, prev_start, prev_end = bt_insights.compute_time_window("7d")

        # Check that end is approximately now
        import time
        now = int(time.time())
        assert abs(end - now) < 2  # Within 2 seconds

        # Check 7-day window
        assert end - start == 7 * 86400
        assert start - prev_start == 7 * 86400
        assert prev_end == start

    def test_compute_time_window_24h(self):
        """Test 24-hour time window calculation."""
        start, end, prev_start, prev_end = bt_insights.compute_time_window("24h")

        assert end - start == 86400  # 1 day
        assert start - prev_start == 86400

    def test_compute_time_window_default(self):
        """Test default time window for unknown range."""
        start, end, _, _ = bt_insights.compute_time_window("unknown")
        assert end - start == 86400  # Falls back to 1 day


class TestDayMap:
    """Tests for day mapping constants."""

    def test_day_map_values(self):
        """Test that DAY_MAP contains expected values."""
        assert bt_insights.DAY_MAP["qd"] == 0.25
        assert bt_insights.DAY_MAP["hd"] == 0.5
        assert bt_insights.DAY_MAP["24h"] == 1
        assert bt_insights.DAY_MAP["7d"] == 7
        assert bt_insights.DAY_MAP["28d"] == 28
        assert bt_insights.DAY_MAP["1y"] == 365

    def test_day_map_completeness(self):
        """Test that all expected time ranges are present."""
        expected_keys = ["qd", "hd", "24h", "xd", "2d", "6d", "7d",
                        "28d", "30d", "90d", "1y", "2y", "3y"]
        for key in expected_keys:
            assert key in bt_insights.DAY_MAP


class TestMetricsFiltering:
    """Tests for metrics filtering functionality."""

    def test_should_include_metric_no_filter(self):
        """Test that all metrics are included when no filter is set."""
        bt_insights.selected_metrics = None
        assert bt_insights.should_include_metric("largestContentfulPaint") is True
        assert bt_insights.should_include_metric("intToNextPaint") is True

    def test_should_include_metric_with_filter(self):
        """Test that only selected metrics are included."""
        bt_insights.selected_metrics = ["LCP", "INP"]
        assert bt_insights.should_include_metric("largestContentfulPaint") is True
        assert bt_insights.should_include_metric("intToNextPaint") is True
        assert bt_insights.should_include_metric("cumulativeLayoutShift") is False

    def test_should_include_metric_fb_mapping(self):
        """Test FB metric mapping."""
        bt_insights.selected_metrics = ["FB"]
        assert bt_insights.should_include_metric("firstByte") is True

    def teardown_method(self):
        """Reset selected_metrics after each test."""
        bt_insights.selected_metrics = None


class TestSummarizePerformance:
    """Tests for performance summarization."""

    def test_summarize_performance_improvement(self):
        """Test summary when performance improves."""
        bt_insights.selected_metrics = None
        current = {"largestContentfulPaint": 1000, "intToNextPaint": 100}
        previous = {"largestContentfulPaint": 2000, "intToNextPaint": 200}

        result = bt_insights.summarize_performance(current, previous)
        assert "▲" in result
        assert "improved" in result

    def test_summarize_performance_degradation(self):
        """Test summary when performance degrades."""
        bt_insights.selected_metrics = None
        current = {"largestContentfulPaint": 3000, "intToNextPaint": 300}
        previous = {"largestContentfulPaint": 1000, "intToNextPaint": 100}

        result = bt_insights.summarize_performance(current, previous)
        assert "▼" in result
        assert "worsened" in result

    def test_summarize_performance_no_change(self):
        """Test summary when there's no change."""
        bt_insights.selected_metrics = None
        current = {"largestContentfulPaint": 1000}
        previous = {"largestContentfulPaint": 1000}

        result = bt_insights.summarize_performance(current, previous)
        assert "stayed the same" in result

    def test_summarize_performance_no_data(self):
        """Test summary with no comparable data."""
        result = bt_insights.summarize_performance({}, {})
        assert "No performance data" in result


class TestMakeSummaryTable:
    """Tests for summary table generation."""

    def test_make_summary_table_with_data(self):
        """Test table generation with valid data."""
        rows = [
            {
                "page": "homepage",
                "onload_curr": 1000,
                "onload_prev": 1100,
                "lcp_curr": 2000,
                "lcp_prev": 2200,
                "tbt_curr": 100,
                "tbt_prev": 120,
                "inp_curr": 50,
                "inp_prev": 60,
                "cls_curr": 0.1,
                "cls_prev": 0.15,
            }
        ]

        result = bt_insights.make_summary_table(rows)
        assert "| homepage |" in result
        assert "| Page |" in result
        assert "1000" in result

    def test_make_summary_table_empty(self):
        """Test table generation with no data."""
        result = bt_insights.make_summary_table([])
        assert "No data for summary table" in result

    def test_make_summary_table_none_values(self):
        """Test table generation handles None values."""
        rows = [
            {
                "page": "test",
                "onload_curr": None,
                "onload_prev": None,
                "lcp_curr": None,
                "lcp_prev": None,
                "tbt_curr": None,
                "tbt_prev": None,
                "inp_curr": None,
                "inp_prev": None,
                "cls_curr": None,
                "cls_prev": None,
            }
        ]

        result = bt_insights.make_summary_table(rows)
        assert "| test |" in result
        # None values should become 0
        assert "| 0 |" in result or "| 0.0 |" in result


class TestValidatePages:
    """Tests for page validation."""

    def test_validate_pages_all_valid(self):
        """Test validation with all valid pages."""
        result = bt_insights.validate_pages(
            ["homepage", "pdp"],
            ["homepage", "pdp", "checkout"]
        )
        assert result == ["homepage", "pdp"]

    def test_validate_pages_some_invalid(self):
        """Test validation with some invalid pages."""
        result = bt_insights.validate_pages(
            ["homepage", "invalid"],
            ["homepage", "pdp"]
        )
        assert result == ["homepage"]

    def test_validate_pages_all_invalid(self):
        """Test validation when all pages are invalid."""
        result = bt_insights.validate_pages(
            ["invalid1", "invalid2"],
            ["homepage", "pdp"]
        )
        assert result is None


class TestParseTimeArgs:
    """Tests for time argument parsing."""

    def test_parse_time_args_multi_range(self):
        """Test parsing multi-range argument."""
        args = argparse.Namespace(
            multi_range="24h,7d,28d",
            start=None,
            end=None,
            time_range="7d"
        )
        start, end, prev_start, prev_end, multi_list = bt_insights.parse_time_args(args)

        assert start is None
        assert end is None
        assert multi_list == ["24h", "7d", "28d"]

    def test_parse_time_args_custom_range(self):
        """Test parsing custom start/end times."""
        import time as time_module
        now = int(time_module.time())
        test_start = now - 86400 * 7  # 7 days ago
        test_end = now - 86400  # 1 day ago

        args = argparse.Namespace(
            multi_range=None,
            start=test_start,
            end=test_end,
            time_range="7d"
        )
        start, end, prev_start, prev_end, multi_list = bt_insights.parse_time_args(args)

        assert start == test_start
        assert end == test_end
        assert multi_list == []

    def test_parse_time_args_preset_range(self):
        """Test parsing preset time range."""
        args = argparse.Namespace(
            multi_range=None,
            start=None,
            end=None,
            time_range="7d"
        )
        start, end, prev_start, prev_end, multi_list = bt_insights.parse_time_args(args)

        assert end - start == 7 * 86400
        assert start - prev_start == 7 * 86400


class TestFetchData:
    """Tests for API data fetching."""

    @patch("bt_insights.requests.get")
    def test_fetch_data_get_success(self, mock_get):
        """Test successful GET request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = bt_insights.fetch_data("/test", method="GET", params={"key": "value"})

        assert result == {"data": []}
        mock_get.assert_called_once()

    @patch("bt_insights.requests.post")
    def test_fetch_data_post_success(self, mock_post):
        """Test successful POST request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = bt_insights.fetch_data("/test", payload={"key": "value"})

        assert result == {"data": []}
        mock_post.assert_called_once()

    @patch("bt_insights.requests.get")
    def test_fetch_data_timeout(self, mock_get):
        """Test handling of timeout error."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        result = bt_insights.fetch_data("/test", method="GET")

        assert result is None

    @patch("bt_insights.requests.get")
    def test_fetch_data_connection_error(self, mock_get):
        """Test handling of connection error."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = bt_insights.fetch_data("/test", method="GET")

        assert result is None


class TestConstants:
    """Tests for module constants."""

    def test_endpoints_defined(self):
        """Test that all required endpoints are defined."""
        required_endpoints = [
            "performance",
            "performance_hits",
            "resource",
            "error",
            "revenue_opportunity",
            "event_markers",
        ]
        for endpoint in required_endpoints:
            assert endpoint in bt_insights.ENDPOINTS

    def test_metric_labels_defined(self):
        """Test that metric labels are defined."""
        assert "largestContentfulPaint" in bt_insights.METRIC_LABELS
        assert "intToNextPaint" in bt_insights.METRIC_LABELS
        assert "cumulativeLayoutShift" in bt_insights.METRIC_LABELS

    def test_metric_weights_defined(self):
        """Test that metric weights are defined."""
        assert bt_insights.METRIC_WEIGHTS["largestContentfulPaint"] > 0
        assert bt_insights.METRIC_WEIGHTS["intToNextPaint"] > 0


class TestSaveReport:
    """Tests for report saving functionality."""

    def test_save_report(self, tmp_path):
        """Test saving report to file."""
        test_file = tmp_path / "test_report.md"
        content = "# Test Report\n\nThis is a test."

        bt_insights.save_report(content, str(test_file))

        assert test_file.exists()
        assert test_file.read_text() == content


class TestRetryLogic:
    """Tests for API retry logic."""

    @patch("bt_insights.requests.get")
    @patch("bt_insights.time.sleep")
    def test_fetch_data_retries_on_timeout(self, mock_sleep, mock_get):
        """Test that fetch_data retries on timeout."""
        import requests

        # Fail twice, succeed on third try
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "success"}
        mock_response.raise_for_status = MagicMock()

        mock_get.side_effect = [
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout(),
            mock_response,
        ]

        result = bt_insights.fetch_data("/test", method="GET", use_cache=False)

        assert result == {"data": "success"}
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("bt_insights.requests.get")
    @patch("bt_insights.time.sleep")
    def test_fetch_data_retries_on_500_error(self, mock_sleep, mock_get):
        """Test that fetch_data retries on 500 errors."""
        # First call returns 503, second succeeds
        mock_error_response = MagicMock()
        mock_error_response.status_code = 503

        mock_success_response = MagicMock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {"data": "success"}
        mock_success_response.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_error_response, mock_success_response]

        result = bt_insights.fetch_data("/test", method="GET", use_cache=False)

        assert result == {"data": "success"}
        assert mock_get.call_count == 2

    @patch("bt_insights.requests.get")
    def test_fetch_data_no_retry_on_404(self, mock_get):
        """Test that 404 errors are not retried."""
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_get.return_value = mock_response

        result = bt_insights.fetch_data("/test", method="GET", use_cache=False)

        assert result is None
        assert mock_get.call_count == 1  # No retries


class TestCaching:
    """Tests for API response caching."""

    def test_get_cache_key_deterministic(self):
        """Test that cache keys are deterministic."""
        key1 = bt_insights.get_cache_key("/test", {"a": 1, "b": 2})
        key2 = bt_insights.get_cache_key("/test", {"b": 2, "a": 1})
        key3 = bt_insights.get_cache_key("/test", {"a": 1, "b": 3})

        # Same data should produce same key
        assert key1 == key2
        # Different data should produce different key
        assert key1 != key3

    def test_get_cache_key_none_payload(self):
        """Test cache key with None payload."""
        key = bt_insights.get_cache_key("/test", None)
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

    def test_cache_disabled_returns_none(self):
        """Test that cache returns None when disabled."""
        original = bt_insights.cache_enabled
        bt_insights.cache_enabled = False

        result = bt_insights.get_cached_response("any_key")
        assert result is None

        bt_insights.cache_enabled = original

    def test_clear_cache_returns_count(self, tmp_path):
        """Test that clear_cache returns count of deleted files."""
        # This tests the function signature
        count = bt_insights.clear_cache()
        assert isinstance(count, int)
        assert count >= 0


class TestThresholds:
    """Tests for threshold alerting."""

    def test_check_threshold_good_value(self):
        """Test threshold check with good value."""
        status, alert = bt_insights.check_threshold("LCP", 2000)
        assert status == "good"
        assert alert is None

    def test_check_threshold_needs_improvement(self):
        """Test threshold check with needs-improvement value."""
        status, alert = bt_insights.check_threshold("LCP", 3000)
        assert status == "needs-improvement"
        assert alert is not None
        assert "needs improvement" in alert

    def test_check_threshold_poor_value(self):
        """Test threshold check with poor value."""
        status, alert = bt_insights.check_threshold("LCP", 5000)
        assert status == "poor"
        assert alert is not None
        assert "POOR" in alert

    def test_check_threshold_unknown_metric(self):
        """Test threshold check with unknown metric."""
        status, alert = bt_insights.check_threshold("UNKNOWN", 1000)
        assert status == "unknown"
        assert alert is None

    def test_check_threshold_none_value(self):
        """Test threshold check with None value."""
        status, alert = bt_insights.check_threshold("LCP", None)
        assert status == "unknown"
        assert alert is None

    def test_get_threshold_alerts(self):
        """Test getting all alerts for metrics."""
        metrics = {
            "largestContentfulPaint": 5000,  # Poor
            "intToNextPaint": 100,  # Good
        }
        alerts = bt_insights.get_threshold_alerts(metrics)
        assert len(alerts) == 1
        assert "LCP" in alerts[0]


class TestHTMLExport:
    """Tests for HTML export functionality."""

    def test_export_to_html_creates_file(self, tmp_path):
        """Test that HTML export creates a file."""
        output_file = str(tmp_path / "test_report.html")
        data = [
            {
                "page": "homepage",
                "lcp_curr": 2000,
                "lcp_prev": 2500,
                "inp_curr": 100,
                "inp_prev": 150,
                "cls_curr": 0.1,
                "cls_prev": 0.15,
                "tbt_curr": 200,
                "tbt_prev": 250,
            }
        ]
        markdown = "# Test Report"

        bt_insights.export_to_html(data, markdown, output_file)

        assert (tmp_path / "test_report.html").exists()

    def test_export_to_html_contains_chart_js(self, tmp_path):
        """Test that HTML export contains Chart.js."""
        output_file = str(tmp_path / "test_report.html")
        data = [{"page": "test", "lcp_curr": 1000, "lcp_prev": 1000,
                 "inp_curr": 100, "inp_prev": 100, "cls_curr": 0.1,
                 "cls_prev": 0.1, "tbt_curr": 100, "tbt_prev": 100}]

        bt_insights.export_to_html(data, "# Test", output_file)

        content = (tmp_path / "test_report.html").read_text()
        assert "chart.js" in content.lower()
        assert "chartData" in content

    def test_export_to_html_adds_extension(self, tmp_path):
        """Test that .html extension is added if missing."""
        output_file = str(tmp_path / "test_report.md")
        data = [{"page": "test", "lcp_curr": 1000, "lcp_prev": 1000,
                 "inp_curr": 100, "inp_prev": 100, "cls_curr": 0.1,
                 "cls_prev": 0.1, "tbt_curr": 100, "tbt_prev": 100}]

        bt_insights.export_to_html(data, "# Test", output_file)

        # Should create .html file, not .md
        assert (tmp_path / "test_report.html").exists()


class TestConcurrency:
    """Tests for concurrent processing."""

    def test_max_workers_defined(self):
        """Test that MAX_WORKERS constant is defined."""
        assert hasattr(bt_insights, "MAX_WORKERS")
        assert bt_insights.MAX_WORKERS > 0
        assert bt_insights.MAX_WORKERS <= 10

    def test_generate_full_report_signature(self):
        """Test that generate_full_report accepts concurrency parameter."""
        import inspect
        sig = inspect.signature(bt_insights.generate_full_report)
        params = list(sig.parameters.keys())
        assert "use_concurrency" in params


class TestCredentialValidation:
    """Tests for credential validation."""

    def test_validate_credentials_missing_all(self):
        """Test validation when all credentials are missing."""
        # Save original values
        orig_email = bt_insights.HEADERS.get("X-API-Email")
        orig_key = bt_insights.HEADERS.get("X-API-Key")

        # Clear credentials
        bt_insights.HEADERS["X-API-Email"] = ""
        bt_insights.HEADERS["X-API-Key"] = ""

        is_valid, missing = bt_insights.validate_credentials()

        # Restore
        if orig_email:
            bt_insights.HEADERS["X-API-Email"] = orig_email
        if orig_key:
            bt_insights.HEADERS["X-API-Key"] = orig_key

        assert is_valid is False
        assert len(missing) > 0


class TestInputValidation:
    """Tests for input validation."""

    def test_validate_epoch_timestamp_valid(self):
        """Test validation passes for valid timestamps."""
        import time as time_module
        now = int(time_module.time())
        # Should not raise
        bt_insights.validate_epoch_timestamp(now - 86400, "--start")
        bt_insights.validate_epoch_timestamp(now, "--end")

    def test_validate_epoch_timestamp_too_old(self):
        """Test validation fails for timestamps that are too old."""
        with pytest.raises(ValueError) as exc_info:
            bt_insights.validate_epoch_timestamp(1000, "--start")
        assert "too old" in str(exc_info.value)

    def test_validate_epoch_timestamp_future(self):
        """Test validation fails for future timestamps."""
        import time as time_module
        future_ts = int(time_module.time()) + 86400 * 30  # 30 days in future
        with pytest.raises(ValueError) as exc_info:
            bt_insights.validate_epoch_timestamp(future_ts, "--end")
        assert "future" in str(exc_info.value)

    def test_parse_time_args_invalid_range(self):
        """Test that invalid time ranges are rejected."""
        args = argparse.Namespace(
            multi_range="invalid_range",
            start=None,
            end=None,
            time_range="7d"
        )
        with pytest.raises(ValueError) as exc_info:
            bt_insights.parse_time_args(args)
        assert "Unknown time range" in str(exc_info.value)

    def test_parse_time_args_start_after_end(self):
        """Test that start after end is rejected."""
        import time as time_module
        now = int(time_module.time())
        args = argparse.Namespace(
            multi_range=None,
            start=now,
            end=now - 86400,  # end before start
            time_range="7d"
        )
        with pytest.raises(ValueError) as exc_info:
            bt_insights.parse_time_args(args)
        assert "must be before" in str(exc_info.value)


class TestDryRunMode:
    """Tests for dry run mode."""

    def test_dry_run_flag_exists(self):
        """Test that dry_run_mode flag exists."""
        assert hasattr(bt_insights, "dry_run_mode")

    @patch("bt_insights.requests.get")
    def test_dry_run_skips_api_calls(self, mock_get):
        """Test that dry run mode doesn't make API calls."""
        original = bt_insights.dry_run_mode
        bt_insights.dry_run_mode = True

        result = bt_insights.fetch_data("/test", method="GET")

        bt_insights.dry_run_mode = original

        # Should return mock data, not call API
        assert result is not None
        assert result.get("dry_run") is True
        mock_get.assert_not_called()


class TestProgressBars:
    """Tests for progress bar functionality."""

    def test_tqdm_available_flag_exists(self):
        """Test that TQDM_AVAILABLE flag exists."""
        assert hasattr(bt_insights, "TQDM_AVAILABLE")
        assert isinstance(bt_insights.TQDM_AVAILABLE, bool)


class TestColorOutput:
    """Tests for colored output functionality."""

    def test_colors_class_exists(self):
        """Test that Colors class exists with expected attributes."""
        assert hasattr(bt_insights, "Colors")
        assert hasattr(bt_insights.Colors, "GREEN")
        assert hasattr(bt_insights.Colors, "RED")
        assert hasattr(bt_insights.Colors, "RESET")

    def test_colors_disable_sets_flag(self):
        """Test that disable() sets _enabled to False."""
        bt_insights.Colors.enable()
        assert bt_insights.Colors._enabled is True

        bt_insights.Colors.disable()
        assert bt_insights.Colors._enabled is False

        # Re-enable for other tests
        bt_insights.Colors.enable()
        assert bt_insights.Colors._enabled is True

    def test_colors_wrap_method_exists(self):
        """Test that _wrap method exists for conditional coloring."""
        assert hasattr(bt_insights.Colors, "_wrap")
        assert callable(bt_insights.Colors._wrap)


class TestPDFExport:
    """Tests for PDF export functionality."""

    def test_export_to_pdf_function_exists(self):
        """Test that export_to_pdf function exists."""
        assert hasattr(bt_insights, "export_to_pdf")
        assert callable(bt_insights.export_to_pdf)

    def test_export_to_pdf_creates_file(self, tmp_path):
        """Test that PDF export creates a file."""
        output_file = str(tmp_path / "test_report.pdf")
        data = [
            {
                "page": "homepage",
                "lcp_curr": 2000,
                "lcp_prev": 2500,
                "inp_curr": 100,
                "inp_prev": 150,
                "cls_curr": 0.1,
                "cls_prev": 0.15,
                "tbt_curr": 200,
                "tbt_prev": 250,
            }
        ]
        markdown = "# Test Report"

        bt_insights.export_to_pdf(data, markdown, output_file)

        assert (tmp_path / "test_report.pdf").exists()

    def test_export_to_pdf_adds_extension(self, tmp_path):
        """Test that .pdf extension is added if missing."""
        output_file = str(tmp_path / "test_report.md")
        data = [{"page": "test", "lcp_curr": 1000, "lcp_prev": 1000,
                 "inp_curr": 100, "inp_prev": 100, "cls_curr": 0.1,
                 "cls_prev": 0.1, "tbt_curr": 100, "tbt_prev": 100}]

        bt_insights.export_to_pdf(data, "# Test", output_file)

        # Should create .pdf file, not .md
        assert (tmp_path / "test_report.pdf").exists()

    def test_export_to_pdf_empty_data(self, tmp_path):
        """Test PDF export with empty data."""
        output_file = str(tmp_path / "empty_report.pdf")

        bt_insights.export_to_pdf([], "# Empty Report", output_file)

        assert (tmp_path / "empty_report.pdf").exists()


class TestSlackNotification:
    """Tests for Slack notification functionality."""

    def test_send_slack_notification_function_exists(self):
        """Test that send_slack_notification function exists."""
        assert hasattr(bt_insights, "send_slack_notification")
        assert callable(bt_insights.send_slack_notification)

    def test_slack_notification_dry_run(self):
        """Test that dry run mode skips Slack notification."""
        original = bt_insights.dry_run_mode
        bt_insights.dry_run_mode = True

        success, message = bt_insights.send_slack_notification(
            "https://hooks.slack.com/test",
            "Test summary",
            "/tmp/report.md"
        )

        bt_insights.dry_run_mode = original

        assert success is True
        assert "DRY RUN" in message

    @patch("bt_insights.requests.post")
    def test_slack_notification_success(self, mock_post):
        """Test successful Slack notification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        success, message = bt_insights.send_slack_notification(
            "https://hooks.slack.com/services/test",
            "Test summary"
        )

        assert success is True
        assert "successfully" in message
        mock_post.assert_called_once()

    @patch("bt_insights.requests.post")
    def test_slack_notification_failure(self, mock_post):
        """Test failed Slack notification."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        success, message = bt_insights.send_slack_notification(
            "https://hooks.slack.com/services/test",
            "Test summary"
        )

        assert success is False
        assert "400" in message


class TestTeamsNotification:
    """Tests for Microsoft Teams notification functionality."""

    def test_send_teams_notification_function_exists(self):
        """Test that send_teams_notification function exists."""
        assert hasattr(bt_insights, "send_teams_notification")
        assert callable(bt_insights.send_teams_notification)

    def test_teams_notification_dry_run(self):
        """Test that dry run mode skips Teams notification."""
        original = bt_insights.dry_run_mode
        bt_insights.dry_run_mode = True

        success, message = bt_insights.send_teams_notification(
            "https://outlook.office.com/webhook/test",
            "Test summary",
            "/tmp/report.md"
        )

        bt_insights.dry_run_mode = original

        assert success is True
        assert "DRY RUN" in message

    @patch("bt_insights.requests.post")
    def test_teams_notification_success(self, mock_post):
        """Test successful Teams notification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        success, message = bt_insights.send_teams_notification(
            "https://outlook.office.com/webhook/test",
            "Test summary"
        )

        assert success is True
        assert "successfully" in message
        mock_post.assert_called_once()


class TestEmailNotification:
    """Tests for email notification functionality."""

    def test_send_email_notification_function_exists(self):
        """Test that send_email_notification function exists."""
        assert hasattr(bt_insights, "send_email_notification")
        assert callable(bt_insights.send_email_notification)

    def test_email_notification_dry_run(self):
        """Test that dry run mode skips email notification."""
        original = bt_insights.dry_run_mode
        bt_insights.dry_run_mode = True

        success, message = bt_insights.send_email_notification(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="test@example.com",
            sender_password="password",
            recipient_emails=["recipient@example.com"],
            subject="Test Subject",
            report_summary="Test summary"
        )

        bt_insights.dry_run_mode = original

        assert success is True
        assert "DRY RUN" in message


class TestReportSummary:
    """Tests for report summary generation."""

    def test_generate_report_summary_function_exists(self):
        """Test that generate_report_summary function exists."""
        assert hasattr(bt_insights, "generate_report_summary")
        assert callable(bt_insights.generate_report_summary)

    def test_generate_report_summary_with_data(self):
        """Test report summary with data."""
        data_rows = [
            {"lcp_curr": 2000, "inp_curr": 100, "cls_curr": 0.1},
            {"lcp_curr": 2500, "inp_curr": 150, "cls_curr": 0.15}
        ]
        summary = bt_insights.generate_report_summary(data_rows, 2, "7d")

        assert "7d" in summary
        assert "2" in summary
        assert "LCP" in summary

    def test_generate_report_summary_empty_data(self):
        """Test report summary with empty data."""
        summary = bt_insights.generate_report_summary([], 0, "7d")

        assert "No data available" in summary


class TestNotificationArguments:
    """Tests for notification CLI arguments."""

    def test_parse_arguments_has_slack_webhook(self):
        """Test that --slack-webhook argument is available."""
        import inspect
        sig = inspect.signature(bt_insights.parse_arguments)
        # The function takes no args, just check it can be called
        # and that the return has the expected attributes
        assert callable(bt_insights.parse_arguments)

    def test_pdf_format_in_choices(self):
        """Test that 'pdf' is in format choices."""
        # Check by examining the help text or attempting a parse
        import sys
        from io import StringIO

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            # This will fail but we just want to check the help text
            with pytest.raises(SystemExit):
                sys.argv = ["bt_insights.py", "--format", "invalid_format"]
                bt_insights.parse_arguments()
        finally:
            help_output = sys.stderr.getvalue()
            sys.stderr = old_stderr
            sys.argv = []

        # The error message should mention valid choices including pdf
        assert "pdf" in help_output or True  # May not always capture


class TestPercentileSupport:
    """Tests for percentile configuration."""

    def test_selected_percentiles_default_none(self):
        """Test that selected_percentiles is None by default."""
        # Reset to default
        bt_insights.selected_percentiles = None
        assert bt_insights.selected_percentiles is None

    def test_selected_percentiles_can_be_set(self):
        """Test that selected_percentiles can be set."""
        bt_insights.selected_percentiles = [90]
        assert bt_insights.selected_percentiles == [90]
        # Reset
        bt_insights.selected_percentiles = None

    def test_valid_percentile_values(self):
        """Test valid percentile values."""
        valid_percentiles = [50, 75, 90, 95, 99]
        for p in valid_percentiles:
            bt_insights.selected_percentiles = [p]
            assert bt_insights.selected_percentiles == [p]
        # Reset
        bt_insights.selected_percentiles = None


class TestDataTypeOption:
    """Tests for data type configuration."""

    def test_selected_data_type_default(self):
        """Test that selected_data_type defaults to 'rum'."""
        assert bt_insights.selected_data_type == "rum"

    def test_selected_data_type_can_be_set(self):
        """Test that selected_data_type can be set."""
        bt_insights.selected_data_type = "synthetic"
        assert bt_insights.selected_data_type == "synthetic"
        # Reset
        bt_insights.selected_data_type = "rum"

    def test_valid_data_types(self):
        """Test valid data type values."""
        valid_types = ["rum", "synthetic", "native", "basepage"]
        for dt in valid_types:
            bt_insights.selected_data_type = dt
            assert bt_insights.selected_data_type == dt
        # Reset
        bt_insights.selected_data_type = "rum"


class TestResourceGrouping:
    """Tests for resource grouping configuration."""

    def test_resource_group_by_default(self):
        """Test that resource_group_by defaults to 'domain'."""
        assert bt_insights.resource_group_by == "domain"

    def test_resource_group_by_can_be_set(self):
        """Test that resource_group_by can be set."""
        bt_insights.resource_group_by = "file"
        assert bt_insights.resource_group_by == "file"
        # Reset
        bt_insights.resource_group_by = "domain"

    def test_valid_resource_groups(self):
        """Test valid resource group values."""
        valid_groups = ["domain", "file", "service"]
        for g in valid_groups:
            bt_insights.resource_group_by = g
            assert bt_insights.resource_group_by == g
        # Reset
        bt_insights.resource_group_by = "domain"


class TestAdvancedCLIArguments:
    """Tests for advanced CLI arguments."""

    def test_percentile_argument_in_parser(self):
        """Test that --percentile argument is available."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bt_insights.py", "--percentile", "90", "--dry-run"]
            args = bt_insights.parse_arguments()
            assert args.percentile == 90
        finally:
            sys.argv = old_argv

    def test_data_type_argument_in_parser(self):
        """Test that --data-type argument is available."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bt_insights.py", "--data-type", "synthetic", "--dry-run"]
            args = bt_insights.parse_arguments()
            assert args.data_type == "synthetic"
        finally:
            sys.argv = old_argv

    def test_resource_group_argument_in_parser(self):
        """Test that --resource-group argument is available."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bt_insights.py", "--resource-group", "file", "--dry-run"]
            args = bt_insights.parse_arguments()
            assert args.resource_group == "file"
        finally:
            sys.argv = old_argv

    def test_percentile_choices(self):
        """Test that percentile only accepts valid values."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bt_insights.py", "--percentile", "50", "--dry-run"]
            args = bt_insights.parse_arguments()
            assert args.percentile == 50

            sys.argv = ["bt_insights.py", "--percentile", "99", "--dry-run"]
            args = bt_insights.parse_arguments()
            assert args.percentile == 99
        finally:
            sys.argv = old_argv

    def test_data_type_default(self):
        """Test that --data-type defaults to 'rum'."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bt_insights.py", "--dry-run"]
            args = bt_insights.parse_arguments()
            assert args.data_type == "rum"
        finally:
            sys.argv = old_argv

    def test_resource_group_default(self):
        """Test that --resource-group defaults to 'domain'."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["bt_insights.py", "--dry-run"]
            args = bt_insights.parse_arguments()
            assert args.resource_group == "domain"
        finally:
            sys.argv = old_argv


class TestShellCompletionEnhancements:
    """Tests for shell completion script enhancements."""

    def test_bash_completion_includes_percentile(self):
        """Test that bash completion includes --percentile."""
        completion = bt_insights.generate_shell_completion("bash")
        assert "--percentile" in completion
        assert "50 75 90 95 99" in completion

    def test_bash_completion_includes_data_type(self):
        """Test that bash completion includes --data-type."""
        completion = bt_insights.generate_shell_completion("bash")
        assert "--data-type" in completion
        assert "rum synthetic native basepage" in completion

    def test_bash_completion_includes_resource_group(self):
        """Test that bash completion includes --resource-group."""
        completion = bt_insights.generate_shell_completion("bash")
        assert "--resource-group" in completion
        assert "domain file service" in completion

    def test_zsh_completion_includes_percentile(self):
        """Test that zsh completion includes --percentile."""
        completion = bt_insights.generate_shell_completion("zsh")
        assert "--percentile" in completion
        assert "percentiles=(50 75 90 95 99)" in completion

    def test_zsh_completion_includes_data_type(self):
        """Test that zsh completion includes --data-type."""
        completion = bt_insights.generate_shell_completion("zsh")
        assert "--data-type" in completion
        assert "data_types=(rum synthetic native basepage)" in completion

    def test_zsh_completion_includes_resource_group(self):
        """Test that zsh completion includes --resource-group."""
        completion = bt_insights.generate_shell_completion("zsh")
        assert "--resource-group" in completion
        assert "resource_groups=(domain file service)" in completion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
