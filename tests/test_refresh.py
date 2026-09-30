"""Regressions for the dispatch outage that stopped the refresh on 30 Sep."""
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from refresh_radar import github_request, next_slot, recover


class RefreshTests(unittest.TestCase):
    def response(self):
        response = MagicMock()
        response.__enter__.return_value.status = 204
        return response

    @patch("refresh_radar.time.sleep")
    @patch("refresh_radar.urlopen")
    def test_dispatch_retries_http_500(self, open_url, sleep):
        open_url.side_effect = [HTTPError("https://api.github.com/", 500, "error", {}, BytesIO()), self.response()]
        github_request("Maymonkey/sbm-2-radar-analysis", "test-token", "dispatches", {"ref": "main"})
        self.assertEqual(open_url.call_count, 2)
        sleep.assert_called_once_with(5)

    @patch("refresh_radar.time.sleep")
    @patch("refresh_radar.urlopen")
    def test_exhausted_network_retries(self, open_url, sleep):
        open_url.side_effect = URLError("offline")
        with self.assertRaisesRegex(RuntimeError, "after 5 attempts"):
            github_request("owner/repo", "test-token", "runs")
        self.assertEqual(open_url.call_count, 5)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [5, 10, 20, 40])

    @patch("refresh_radar.time.sleep")
    @patch("refresh_radar.urlopen")
    def test_auth_error_is_not_retried(self, open_url, sleep):
        open_url.side_effect = HTTPError("https://api.github.com/", 403, "forbidden", {}, BytesIO())
        with self.assertRaises(HTTPError):
            github_request("owner/repo", "test-token", "runs")
        sleep.assert_not_called()

    @patch("refresh_radar.time.sleep")
    @patch("refresh_radar.urlopen")
    def test_rate_limit_retry_after(self, open_url, sleep):
        open_url.side_effect = [HTTPError("https://api.github.com/", 429, "limited", {"Retry-After": "20"}, BytesIO()), self.response()]
        github_request("owner/repo", "test-token", "dispatches", {})
        sleep.assert_called_once_with(20)

    def test_next_slot_crosses_ict_midnight(self):
        now = datetime(2026, 9, 30, 16, 59, 59, tzinfo=timezone.utc)
        self.assertEqual(next_slot(now), datetime(2026, 9, 30, 17, 2, tzinfo=timezone.utc))
        self.assertEqual(next_slot(next_slot(now)).minute, 8)

    @patch("refresh_radar.dispatch")
    @patch("refresh_radar.github_request")
    def test_recovery_skips_active_and_pending_runs(self, request, dispatch):
        for status in ("in_progress", "queued", "pending", "waiting"):
            request.return_value = {"workflow_runs": [{"head_branch": "main", "status": status}]}
            self.assertFalse(recover("owner/repo", "test-token", datetime.now(timezone.utc)))
        dispatch.assert_not_called()

    @patch("refresh_radar.dispatch")
    @patch("refresh_radar.github_request")
    def test_recovery_restarts_failed_chain(self, request, dispatch):
        request.return_value = {"workflow_runs": [{"head_branch": "main", "status": "completed", "updated_at": "2026-09-30T04:08:07Z"}]}
        self.assertTrue(recover("owner/repo", "test-token", datetime(2026, 9, 30, 8, tzinfo=timezone.utc)))
        dispatch.assert_called_once_with("owner/repo", "test-token")

    @patch("refresh_radar.dispatch")
    @patch("refresh_radar.github_request")
    def test_recovery_allows_dispatch_visibility_delay(self, request, dispatch):
        request.return_value = {"workflow_runs": [{"head_branch": "main", "status": "completed", "updated_at": "2026-09-30T08:00:00Z"}]}
        self.assertFalse(recover("owner/repo", "test-token", datetime(2026, 9, 30, 8, 0, 30, tzinfo=timezone.utc)))
        dispatch.assert_not_called()
