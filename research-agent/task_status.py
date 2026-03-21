"""Quick utility to inspect the current task.json state."""
import json
from collections import defaultdict
from pathlib import Path

TASK_JSON = Path(r"D:\Dev\A-Share-Scout\Stock Selection Strategies\output\task.json")

# Display order and labels for each status bucket
_STATUS_ORDER = [
    ("consolidated", "Consolidated", "✔"),
    ("completed",    "Completed",    "✔"),
    ("pending",      "Pending",      "○"),
    ("analyzing",    "Analyzing",    "…"),
    ("researching",  "Researching",  "…"),
    ("failed",       "Failed",       "✘"),
    ("unknown",      "Unknown",      "?"),
]

_DIVIDER = "─" * 60


def _header(label: str, count: int, icon: str) -> str:
    title = f"  {icon}  {label}  ({count})"
    return f"\n{title}\n{'─' * max(len(title), 40)}"


def main() -> None:
    data = json.loads(TASK_JSON.read_text(encoding="utf-8"))

    status = data.get("status", "N/A").upper()
    topic  = data.get("topic", "N/A")
    nodes: dict = data.get("graph", {}).get("nodes", {})

    # Deduplicate by node id (the raw dict already uses id as key, but guard anyway)
    unique_nodes = {nid: n for nid, n in nodes.items()}

    by_status: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for node in unique_nodes.values():
        st   = node.get("status", "unknown")
        name = node.get("name", node.get("id", "?"))
        depth = node.get("depth", 0)
        by_status[st].append((depth, name))

    total = len(unique_nodes)
    done  = len(by_status.get("consolidated", [])) + len(by_status.get("completed", []))
    pend  = len(by_status.get("pending", []))
    fail  = len(by_status.get("failed", []))

    # ── Summary header ────────────────────────────────────────────────────────
    print(_DIVIDER)
    print(f"  Task   : {topic}")
    print(f"  Status : {status}")
    print(f"  Nodes  : {total} total  |  {done} done  |  {pend} pending  |  {fail} failed")
    print(_DIVIDER)

    # ── Per-status sections ───────────────────────────────────────────────────
    printed: set[str] = set()
    for key, label, icon in _STATUS_ORDER:
        entries = by_status.get(key, [])
        if not entries:
            continue
        printed.add(key)
        print(_header(label, len(entries), icon))
        for depth, name in sorted(entries):
            prefix = "    " + "  " * depth
            print(f"{prefix}d{depth}  {name}")

    # Catch any status values not in the predefined order
    for key, entries in sorted(by_status.items()):
        if key in printed:
            continue
        print(_header(key.capitalize(), len(entries), "·"))
        for depth, name in sorted(entries):
            prefix = "    " + "  " * depth
            print(f"{prefix}d{depth}  {name}")

    print(f"\n{_DIVIDER}")


if __name__ == "__main__":
    main()
