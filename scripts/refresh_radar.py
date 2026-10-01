"""Continue radar refreshes and recover stopped chains after GitHub outages."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def github_request(repository, token, resource, payload=None, expected_status=204):
    """Retry transient GitHub errors; never log credentials or response bodies."""
    request = Request(
        f"https://api.github.com/repos/{repository}/{resource}",
        data=None if payload is None else json.dumps(payload).encode(),
        method="GET" if payload is None else "POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sbm-2-radar-analysis",
        },
    )
    for attempt in range(5):
        try:
            with urlopen(request, timeout=30) as response:
                if payload is not None:
                    if response.status != expected_status:
                        raise RuntimeError(f"Unexpected GitHub status: HTTP {response.status}")
                    return None
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in (408, 429) and not 500 <= exc.code <= 599:
                raise
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            reason = f"HTTP {exc.code}"
            exc.close()
        except (URLError, TimeoutError, OSError):
            retry_after, reason = "", "network error"
        if attempt == 4:
            raise RuntimeError(f"GitHub request failed after 5 attempts ({reason})")
        delay = max(5 * 2**attempt, min(60, int(retry_after))) if retry_after.isdigit() else 5 * 2**attempt
        print(f"GitHub {reason}; retry {attempt + 2}/5 in {delay}s", flush=True)
        time.sleep(delay)


def next_slot(now):
    slot = now.replace(second=0, microsecond=0)
    while slot <= now or slot.minute % 6 != 2:
        slot += timedelta(minutes=1)
    return slot


def dispatch(repository, token):
    github_request(repository, token, "actions/workflows/check-source.yml/dispatches",
                   {"ref": "main", "inputs": {"continue_refresh": "true"}})
    print("Next refresh dispatched successfully (HTTP 204).", flush=True)


def recover(repository, token, now):
    runs = github_request(repository, token,
                          "actions/workflows/check-source.yml/runs?per_page=100")["workflow_runs"]
    runs = [run for run in runs if run["head_branch"] == "main"]
    # Waiting for an environment does not consume a job's timeout. A waiting
    # deployment previously blocked the chain indefinitely and fooled recovery.
    # Bound liveness by run age, even when GitHub still calls the run active.
    active = [run for run in runs if run["status"] != "completed"]
    stalled = []
    for run in active:
        started = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
        if (now - started).total_seconds() >= 30 * 60:
            stalled.append(run)
            print(f"Cancelling stalled radar run {run['id']} ({run['status']}).", flush=True)
            try:
                github_request(repository, token, f"actions/runs/{run['id']}/force-cancel",
                               {}, expected_status=202)
            except HTTPError as exc:
                if exc.code != 409:  # The run may have completed while we checked.
                    raise
                exc.close()
    if any(run not in stalled for run in active):
        print("Radar refresh is active or queued; recovery skipped.", flush=True)
        return False
    if runs and not stalled:
        latest = max(datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00")) for run in runs)
        if (now - latest).total_seconds() < 120:
            print("A refresh just finished; allow the next dispatch to appear.", flush=True)
            return False
    dispatch(repository, token)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("continue", "recover"))
    args = parser.parse_args()
    repository, token = os.environ["GH_REPOSITORY"], os.environ["GH_TOKEN"]
    now = datetime.now(timezone.utc)
    if args.mode == "recover":
        recover(repository, token, now)
        return
    slot = next_slot(now)
    print(f"Next refresh ICT: {slot.astimezone(timezone(timedelta(hours=7))).isoformat()}", flush=True)
    while (remaining := (slot - datetime.now(timezone.utc)).total_seconds()) > 0:
        time.sleep(min(30, remaining))
    dispatch(repository, token)


if __name__ == "__main__":
    main()
