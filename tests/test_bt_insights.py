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
        args = argparse.Namespace(
            multi_range=None,
            start=1000000,
            end=2000000,
            time_range="7d"
        )
        start, end, prev_start, prev_end, multi_list = bt_insights.parse_time_args(args)

        assert start == 1000000
        assert end == 2000000
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
