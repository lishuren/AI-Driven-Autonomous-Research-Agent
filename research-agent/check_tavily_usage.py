#!/usr/bin/env python3
"""Check Tavily API key usage and credit balance."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_HISTORY_FILE = Path(__file__).parent / "data" / "tavily_usage_history.jsonl"

# ---------------------------------------------------------------------------
# Load .env from the same directory as this script
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    # Read file as UTF-8 and fall back safely on undecodable bytes.
    # Using errors='replace' avoids UnicodeDecodeError on Windows locales
    # when the .env contains characters outside the system encoding.
    with env_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TAVILY_USAGE_URL = "https://api.tavily.com/usage"
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        print("ERROR: TAVILY_API_KEY not set. Export it or add it to .env", file=sys.stderr)
        sys.exit(1)
    return key


def _fetch_usage(api_key: str) -> dict | None:
    """Call /usage endpoint; return normalised dict or None."""
    req = urllib.request.Request(
        _TAVILY_USAGE_URL,
        headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        # Tavily /usage response shape (as of 2026):
        # { "key": { "usage", "limit", "search_usage", "crawl_usage", ... },
        #   "account": { "current_plan", "plan_usage", "plan_limit", "search_usage", ... } }
        acct = data.get("account", {})
        key_data = data.get("key", {})
        if not acct and not key_data:
            # Flat legacy format — try generic field name mapping
            result: dict = {}
            for src, dst in (
                ("used", "credits_used"), ("credits_used", "credits_used"),
                ("limit", "credits_limit"), ("credits_limit", "credits_limit"),
                ("remaining", "credits_remaining"), ("credits_remaining", "credits_remaining"),
            ):
                if src in data and dst not in result:
                    result[dst] = data[src]
            return result if result else None

        plan_used = acct.get("plan_usage", key_data.get("usage"))
        plan_limit = acct.get("plan_limit", key_data.get("limit"))
        plan_remaining = (plan_limit - plan_used) if (plan_used is not None and plan_limit is not None) else None
        return {
            # Key-specific usage (matches the "Usage" column on app.tavily.com/api-keys)
            "key_usage": key_data.get("usage"),
            "key_limit": key_data.get("limit"),
            # Plan-level aggregated usage (across all keys on this account)
            "credits_used": plan_used,
            "credits_limit": plan_limit,
            "credits_remaining": plan_remaining,
            "current_plan": acct.get("current_plan"),
            "search_usage": acct.get("search_usage", key_data.get("search_usage")),
            "extract_usage": acct.get("extract_usage", key_data.get("extract_usage")),
            "crawl_usage": acct.get("crawl_usage", key_data.get("crawl_usage")),
            "paygo_usage": acct.get("paygo_usage"),
            "paygo_limit": acct.get("paygo_limit"),
        }
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 405, 501):
            return None  # endpoint not available on this plan
        print(f"ERROR: /usage HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"ERROR: /usage request failed: {exc}", file=sys.stderr)
        return None


def _probe_search(api_key: str) -> dict | None:
    """Make a minimal 1-result search to verify the key works and read usage from response."""
    payload = json.dumps({
        "api_key": api_key,
        "query": "test",
        "max_results": 1,
        "include_usage": True,
    }).encode()
    req = urllib.request.Request(
        _TAVILY_SEARCH_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data.get("usage")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode(errors="replace")
        except Exception:
            pass
        if exc.code == 401:
            print("ERROR: API key is invalid or unauthorised (HTTP 401)", file=sys.stderr)
        elif exc.code == 429:
            print("ERROR: Rate-limited (HTTP 429) — you may be out of credits", file=sys.stderr)
        else:
            print(f"ERROR: Search probe HTTP {exc.code}: {body[:200]}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"ERROR: Search probe failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Local history tracking
# ---------------------------------------------------------------------------

def _save_history(api_key_tail: str, used: int | None, limit: int | None, remaining: int | None, source: str, key_usage: int | None = None) -> None:
    """Append a usage snapshot to the local JSONL history file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "key_tail": api_key_tail,
        "source": source,
        "key_usage": key_usage,       # per-key usage (matches website)
        "credits_used": used,         # plan-level aggregate
        "credits_limit": limit,
        "credits_remaining": remaining,
    }
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _HISTORY_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"  (Warning: could not save history: {exc})", file=sys.stderr)


def _load_history(key_tail: str, max_records: int = 20) -> list[dict]:
    """Load the last `max_records` history entries for this API key tail."""
    if not _HISTORY_FILE.exists():
        return []
    records = []
    try:
        with _HISTORY_FILE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("key_tail") == key_tail:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return records[-max_records:]


def _print_history(records: list[dict]) -> None:
    """Print a compact usage history table."""
    if not records:
        print("  No local history yet — this is the first recorded check.")
        return

    _W = 76
    print(f"\n{'─' * _W}")
    print(f"  Usage History  ({len(records)} record(s) in {_HISTORY_FILE.relative_to(Path.cwd()) if _HISTORY_FILE.is_relative_to(Path.cwd()) else _HISTORY_FILE})")
    print(f"{'─' * _W}")
    print(f"  {'Timestamp (UTC)':<19}  {'Key':>7}  {'Plan':>7}  {'Limit':>7}  {'Remaining':>10}  Source")
    print(f"  {'─' * 19}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 10}  {'─' * 12}")

    prev_key_usage: int | None = None
    for rec in records:
        ts = rec.get("timestamp", "")[:19].replace("T", " ")
        key_usage = rec.get("key_usage")
        used = rec.get("credits_used")
        limit = rec.get("credits_limit")
        remaining = rec.get("credits_remaining")
        source = rec.get("source", "")[:12]

        key_str = f"{key_usage:,}" if key_usage is not None else "—"
        used_str = f"{used:,}" if used is not None else "—"
        limit_str = f"{limit:,}" if limit is not None else "—"
        rem_str = f"{remaining:,}" if remaining is not None else "—"

        # Δ delta annotation on key-specific usage
        delta = ""
        display_used = key_usage if key_usage is not None else used
        prev_used = prev_key_usage
        if display_used is not None and prev_used is not None:
            diff = display_used - prev_used
            if diff > 0:
                delta = f" \033[91m(+{diff:,})\033[0m"
            elif diff < 0:
                delta = f" \033[92m({diff:,})\033[0m"
        prev_key_usage = display_used

        print(f"  {ts:<19}  {key_str:>7}  {used_str:>7}  {limit_str:>7}  {rem_str:>10}  {source}{delta}")

    print(f"{'─' * _W}\n")


def _bar(used: int, limit: int, width: int = 30) -> str:
    if limit <= 0:
        return "[" + "?" * width + "]"
    filled = min(width, round(used / limit * width))
    pct = used / limit * 100
    colour = "\033[92m" if pct < 60 else "\033[93m" if pct < 85 else "\033[91m"
    reset = "\033[0m"
    return f"[{colour}{'█' * filled}{'░' * (width - filled)}{reset}] {pct:.1f}%"


def _print_usage(used: int | None, limit: int | None, remaining: int | None, source: str, extra: dict | None = None) -> None:
    print(f"\n{'─' * 50}")
    print(f"  Tavily Credit Usage  (source: {source})")
    print(f"{'─' * 50}")

    if used is None and remaining is None and (not extra or extra.get("key_usage") is None):
        print("  No usage data available from this endpoint.")
        print(f"{'─' * 50}\n")
        return

    if extra and extra.get("current_plan"):
        print(f"  Plan      : {extra['current_plan']}")

    # Key-specific usage (what the website shows under "Usage" per API key)
    if extra and extra.get("key_usage") is not None:
        key_used = extra["key_usage"]
        key_limit = extra.get("key_limit")
        key_str = f"{key_used:,}"
        if key_limit is not None:
            key_str += f" / {key_limit:,}"
        print(f"  This Key  : {key_str:>10}  ← matches app.tavily.com API Keys tab")

    # Plan-level aggregated usage
    if used is not None:
        print(f"  Plan Used : {used:>10,}  (all keys combined)")
    if limit is not None:
        print(f"  Plan Limit: {limit:>10,}")
    if remaining is not None:
        print(f"  Remaining : {remaining:>10,}")

    # Progress bar against plan limit
    bar_used = extra.get("key_usage") if extra and extra.get("key_usage") is not None else used
    if bar_used is not None and limit is not None and limit > 0:
        print(f"  Progress  : {_bar(bar_used, limit)}")
    elif used is not None and remaining is not None:
        total_est = used + remaining
        print(f"  Progress  : {_bar(used, total_est)}  (estimated limit: {total_est:,})")

    if extra:
        breakdown = [(k.replace("_usage", "").capitalize(), extra[k])
                     for k in ("search_usage", "extract_usage", "crawl_usage")
                     if extra.get(k) is not None]
        if breakdown:
            print(f"  Breakdown :", end="")
            print("  " + "  │  ".join(f"{label}: {val:,}" for label, val in breakdown))
        if extra.get("paygo_usage") is not None:
            paygo_limit = extra.get("paygo_limit")
            paygo_str = f"{extra['paygo_usage']:,}" + (f" / {paygo_limit:,}" if paygo_limit else "")
            print(f"  Pay-as-go : {paygo_str:>10}")

    print(f"{'─' * 50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = _get_api_key()
    key_tail = api_key[-4:]
    masked = api_key[:8] + "..." + key_tail
    print(f"API key  : {masked}")
    print("Checking usage...")

    used: int | None = None
    limit: int | None = None
    remaining: int | None = None
    source: str = ""

    # 1. Try the dedicated /usage endpoint first
    usage = _fetch_usage(api_key)

    if usage:
        used = usage.get("credits_used")
        limit = usage.get("credits_limit")
        remaining = usage.get("credits_remaining")
        source = "/usage endpoint"
        _print_usage(used, limit, remaining, source, extra=usage)

    else:
        # 2. Fall back to a minimal search probe (reads usage from search response)
        print("  /usage endpoint unavailable — probing via search API...")
        probe = _probe_search(api_key)

        if probe:
            used = probe.get("credits")
            source = "search probe"
            _print_usage(used, None, None, source)
        else:
            print("\nCould not retrieve credit data from Tavily.")
            print("Visit https://app.tavily.com to view your usage dashboard.")

    # Show history (before saving so the table reflects prior runs)
    history = _load_history(key_tail)
    _print_history(history)

    # Persist this reading if we got any data
    key_usage_val = usage.get("key_usage") if usage else None
    if used is not None or remaining is not None or key_usage_val is not None:
        _save_history(key_tail, used, limit, remaining, source, key_usage=key_usage_val)
        print(f"  Snapshot saved → {_HISTORY_FILE}")


if __name__ == "__main__":
    main()
