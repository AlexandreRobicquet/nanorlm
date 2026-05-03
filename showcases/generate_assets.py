from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


COLORS = {
    "direct_full_context": "#6c6f7d",
    "keep_recent": "#b35c44",
    "summary_only": "#c49b2e",
    "single_critic_topk": "#2e6f95",
    "pairwise_tournament": "#1d8a5b",
}


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def summary_table(payload: dict) -> str:
    lines = ["| policy | answer | prov | compact | avg toks |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in payload["summaries"]:
        lines.append(
            f"| {row['policy']} | {row['answer_accuracy']:.3f} | {row['provenance_score']:.3f} | {row['compactness']:.3f} | {row['avg_retained_tokens']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def render_architecture_svg() -> str:
    width = 1160
    height = 720
    font = "Menlo, ui-monospace, SFMono-Regular, monospace"
    stroke = "#2a2f38"
    muted = "#5a6170"
    panel_stroke = "#d9d1c3"

    def rect(
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        fill: str,
        stroke_color: str = panel_stroke,
        stroke_width: int = 2,
        rx: int = 18,
        dash: str = "",
    ) -> str:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke_color}" stroke-width="{stroke_width}"{dash_attr}/>'
        )

    def text(
        x: float,
        y: float,
        label: str,
        *,
        size: int,
        fill: str = stroke,
        anchor: str = "start",
        weight: int = 400,
    ) -> str:
        return (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{font}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(label)}</text>'
        )

    def multiline(
        x: float,
        y: float,
        lines: list[str],
        *,
        size: int,
        fill: str = stroke,
        line_height: int = 18,
        anchor: str = "start",
        weight: int = 400,
    ) -> list[str]:
        return [text(x, y + index * line_height, line, size=size, fill=fill, anchor=anchor, weight=weight) for index, line in enumerate(lines)]

    def badge(x: int, y: int, w: int, label: str, fill: str) -> list[str]:
        return [
            rect(x, y, w, 34, fill=fill, stroke_color="none", stroke_width=0, rx=17),
            text(x + 14, y + 23, label, size=14, weight=700),
        ]

    def card(
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        lines: list[str],
        *,
        fill: str,
        stroke_color: str,
        dash: str = "",
    ) -> list[str]:
        parts = [rect(x, y, w, h, fill=fill, stroke_color=stroke_color, stroke_width=2, rx=16, dash=dash)]
        parts.append(text(x + 14, y + 24, title, size=14, weight=700))
        parts.extend(multiline(x + 14, y + 46, lines, size=13, fill=muted, line_height=16))
        return parts

    def arrow(x1: int, y1: int, x2: int, y2: int) -> list[str]:
        if y1 != y2:
            raise ValueError("architecture arrows are expected to be horizontal")
        return [
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="3"/>',
            f'<polygon points="{x2 - 12},{y2 - 8} {x2},{y2} {x2 - 12},{y2 + 8}" fill="{stroke}"/>',
        ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f4ec"/>',
        text(40, 48, "What nanoRLM Builds", size=30, weight=700),
        text(
            40,
            78,
            "Recursive inference with explicit memory retention: recurse, inspect, retain under budget, then answer from kept evidence.",
            size=16,
            fill=muted,
        ),
    ]

    panels = [
        (40, 108, 336, "1. Recurse over context", "#efe4c8"),
        (396, 108, 336, "2. Inspect leaves into memory", "#dce9f1"),
        (752, 108, 368, "3. Retain under budget, answer from memory", "#dff2e7"),
    ]
    for x, y, w, label, accent in panels:
        parts.append(rect(x, y, w, 564, fill="#fffdf7"))
        parts.extend(badge(x + 18, y + 16, min(w - 36, 292), label, accent))

    parts.extend(arrow(376, 390, 396, 390))
    parts.extend(arrow(732, 390, 752, 390))

    recurse_x = 40
    recurse_y = 108
    parts.append(text(recurse_x + 18, recurse_y + 72, "input: root query + ContextBlock[]", size=14, fill=muted))
    parts.extend(
        card(
            recurse_x + 18,
            recurse_y + 92,
            300,
            74,
            "root query",
            ["For pair-000, what is the full code?"],
            fill="#fff8ea",
            stroke_color="#c79f52",
        )
    )
    source_cards = [
        ("left.txt", ["PAIR_ID: pair-000", "FACT_VALUE: amber"], "#eef8f1", "#1d8a5b"),
        ("ops.log", ["deployment noise", "no answer signal"], "#f4efe4", panel_stroke),
        ("right.txt", ["PAIR_ID: pair-000", "FACT_VALUE: orbit"], "#eef8f1", "#1d8a5b"),
    ]
    for index, (title, lines, fill, stroke_color) in enumerate(source_cards):
        y = recurse_y + 194 + index * 88
        parts.extend(card(recurse_x + 18, y, 132, 62, title, lines, fill=fill, stroke_color=stroke_color))

    tree_left = recurse_x + 188
    root_x = tree_left + 34
    root_y = recurse_y + 218
    parts.append(rect(root_x, root_y, 96, 42, fill="#fff8ea", stroke_color="#c79f52", rx=14))
    parts.append(text(root_x + 48, root_y + 26, "depth 0", size=14, anchor="middle", weight=700))
    branch_boxes = [
        (tree_left + 8, recurse_y + 314, 78, 40, "chunk A"),
        (tree_left + 122, recurse_y + 314, 78, 40, "chunk B"),
    ]
    leaf_boxes = [
        (tree_left, recurse_y + 430, 52, 36, "A1"),
        (tree_left + 60, recurse_y + 430, 52, 36, "A2"),
        (tree_left + 122, recurse_y + 430, 52, 36, "B1"),
        (tree_left + 182, recurse_y + 430, 52, 36, "B2"),
    ]
    parts.append(f'<line x1="{root_x + 48}" y1="{root_y + 42}" x2="{root_x + 48}" y2="{recurse_y + 294}" stroke="{stroke}" stroke-width="3"/>')
    parts.append(f'<line x1="{tree_left + 47}" y1="{recurse_y + 294}" x2="{tree_left + 161}" y2="{recurse_y + 294}" stroke="{stroke}" stroke-width="3"/>')
    for x, y, w, h, label in branch_boxes:
        parts.append(f'<line x1="{x + w / 2}" y1="{recurse_y + 294}" x2="{x + w / 2}" y2="{y}" stroke="{stroke}" stroke-width="3"/>')
        parts.append(rect(x, y, w, h, fill="#fffdf7", stroke_color="#d5b184", rx=14))
        parts.append(text(x + w / 2, y + 24, label, size=13, anchor="middle", weight=700))
    for x, y, w, h, label in leaf_boxes:
        parts.append(rect(x, y, w, h, fill="#fffdf7", stroke_color="#2e6f95", rx=12))
        parts.append(text(x + w / 2, y + 23, label, size=14, anchor="middle", weight=700))
    for child_center, leaves in [
        (branch_boxes[0][0] + branch_boxes[0][2] / 2, leaf_boxes[:2]),
        (branch_boxes[1][0] + branch_boxes[1][2] / 2, leaf_boxes[2:]),
    ]:
        join_y = recurse_y + 398
        parts.append(f'<line x1="{child_center}" y1="{recurse_y + 354}" x2="{child_center}" y2="{join_y}" stroke="{stroke}" stroke-width="3"/>')
        parts.append(f'<line x1="{leaves[0][0] + 26}" y1="{join_y}" x2="{leaves[1][0] + 26}" y2="{join_y}" stroke="{stroke}" stroke-width="3"/>')
        for leaf_x, leaf_y, leaf_w, _, _ in leaves:
            parts.append(f'<line x1="{leaf_x + leaf_w / 2}" y1="{join_y}" x2="{leaf_x + leaf_w / 2}" y2="{leaf_y}" stroke="{stroke}" stroke-width="3"/>')
    parts.extend(
        multiline(
            recurse_x + 18,
            recurse_y + 540,
            ["Recursion shrinks one noisy prompt into", "small, inspectable leaves."],
            size=14,
            fill=muted,
            line_height=18,
        )
    )

    inspect_x = 396
    inspect_y = 108
    parts.append(text(inspect_x + 18, inspect_y + 72, "each leaf -> MemoryItem", size=14, fill=muted))
    inspection_cards = [
        ("A1 inspection", ["summary: pair-000 left = amber", "provenance: left.txt", "answer_candidate: amber"], "#eef8f1", "#1d8a5b"),
        ("A2 inspection", ["summary: deploy checklist", "weak overlap; low value"], "#f5efe3", panel_stroke),
        ("B1 inspection", ["summary: pair-000 right = orbit", "provenance: right.txt", "answer_candidate: orbit"], "#eef8f1", "#1d8a5b"),
        ("B2 inspection", ["summary: on-call rota", "weak overlap; low value"], "#f5efe3", panel_stroke),
    ]
    for index, (title, lines, fill, stroke_color) in enumerate(inspection_cards):
        height = 84 if len(lines) == 3 else 68
        y = inspect_y + 96 + index * 92
        parts.extend(card(inspect_x + 18, y, 300, height, title, lines, fill=fill, stroke_color=stroke_color))
    parts.append(rect(inspect_x + 18, inspect_y + 508, 300, 60, fill="#edf4f8", stroke_color="#9ebfd4", rx=14))
    parts.extend(
        multiline(
            inspect_x + 32,
            inspect_y + 531,
            ["Leaves are cheap to inspect because", "each shard is already small."],
            size=14,
            fill=muted,
            line_height=18,
        )
    )

    retain_x = 752
    retain_y = 108
    parts.append(text(retain_x + 18, retain_y + 72, "policy: pairwise_tournament", size=14, fill=muted))
    parts.append(text(retain_x + 18, retain_y + 92, "root query decides what survives", size=13, fill=muted))
    parts.append(text(retain_x + 18, retain_y + 114, "memory_budget_tokens = 60", size=13, fill=muted))
    parts.append(rect(retain_x + 18, retain_y + 126, 220, 16, fill="#ece7db", stroke_color="none", stroke_width=0, rx=8))
    parts.append(rect(retain_x + 18, retain_y + 126, 176, 16, fill="#1d8a5b", stroke_color="none", stroke_width=0, rx=8))
    parts.append(text(retain_x + 246, retain_y + 139, "52 used", size=12, fill=muted))
    retention_cards = [
        (retain_x + 18, retain_y + 162, "keep", ["left = amber", "complements right"], "#eef8f1", "#1d8a5b", ""),
        (retain_x + 194, retain_y + 162, "keep", ["right = orbit", "complements left"], "#eef8f1", "#1d8a5b", ""),
        (retain_x + 18, retain_y + 246, "drop", ["deploy checklist", "weak / redundant"], "#fbf4f1", "#c34747", "7 6"),
        (retain_x + 194, retain_y + 246, "drop", ["on-call rota", "weak / redundant"], "#fbf4f1", "#c34747", "7 6"),
    ]
    for x, y, title, lines, fill, stroke_color, dash in retention_cards:
        parts.extend(card(x, y, 156, 62, title, lines, fill=fill, stroke_color=stroke_color, dash=dash))
    parts.append(rect(retain_x + 18, retain_y + 346, 332, 148, fill="#eef8f1", stroke_color="#1d8a5b", stroke_width=3, rx=18))
    parts.append(text(retain_x + 34, retain_y + 378, "answer from kept_items", size=14, fill=muted))
    parts.append(text(retain_x + 184, retain_y + 418, "amber orbit", size=28, anchor="middle", weight=700))
    evidence_lines = [
        "left.txt -> amber",
        "right.txt -> orbit",
    ]
    for index, label in enumerate(evidence_lines):
        y = retain_y + 444 + index * 28
        parts.append(text(retain_x + 34, y, label, size=13))
    parts.append(rect(retain_x + 18, retain_y + 508, 332, 60, fill="#f3f8f5", stroke_color="#9bcdb1", rx=14))
    parts.extend(
        multiline(
            retain_x + 30,
            retain_y + 531,
            ["If retention drops a needed fact,", "the answer loses it too."],
            size=14,
            fill=muted,
            line_height=18,
        )
    )

    parts.append(text(40, 704, "The whole repo is this loop in a small, inspectable form.", size=15, fill=muted))
    parts.append("</svg>")
    return "\n".join(parts)


def render_curve_svg(curves_payload: dict, metric: str = "answer_accuracy") -> str:
    aggregates = curves_payload["aggregates"]
    depth = max(row["depth"] for row in aggregates)
    rows = [row for row in aggregates if row["depth"] == depth]
    budgets = sorted({row["budget"] for row in rows})
    policies = sorted({row["policy"] for row in rows}, key=lambda value: (value != "pairwise_tournament", value))
    width = 920
    height = 420
    margin_left = 70
    margin_top = 40
    chart_width = 760
    chart_height = 280
    points_by_policy = {policy: [row for row in rows if row["policy"] == policy] for policy in policies}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f4ec"/>',
        f'<text x="{margin_left}" y="28" font-family="Menlo, monospace" font-size="22" fill="#2a2f38">Budget Curve ({metric}, depth={depth})</text>',
    ]
    for tick in range(6):
        y = margin_top + chart_height - (chart_height * tick / 5)
        value = tick / 5
        parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + chart_width}" y2="{y}" stroke="#ddd7c9" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 5}" text-anchor="end" font-family="Menlo, monospace" font-size="12" fill="#5a6170">{value:.1f}</text>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#2a2f38" stroke-width="2"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#2a2f38" stroke-width="2"/>')
    for index, budget in enumerate(budgets):
        x = margin_left + (chart_width * index / max(1, len(budgets) - 1))
        parts.append(f'<text x="{x}" y="{margin_top + chart_height + 24}" text-anchor="middle" font-family="Menlo, monospace" font-size="12" fill="#5a6170">{budget}</text>')
    for policy in policies:
        series = sorted(points_by_policy[policy], key=lambda row: row["budget"])
        if not series:
            continue
        coords = []
        for index, row in enumerate(series):
            x = margin_left + (chart_width * index / max(1, len(budgets) - 1))
            y = margin_top + chart_height - chart_height * float(row[metric])
            coords.append((x, y))
        path = " ".join(("M" if index == 0 else "L") + f"{x:.1f},{y:.1f}" for index, (x, y) in enumerate(coords))
        color = COLORS.get(policy, "#2a2f38")
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4"/>')
        for x, y in coords:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
    legend_y = 350
    for index, policy in enumerate(policies):
        x = margin_left + index * 170
        color = COLORS.get(policy, "#2a2f38")
        parts.append(f'<rect x="{x}" y="{legend_y}" width="20" height="8" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{x + 28}" y="{legend_y + 9}" font-family="Menlo, monospace" font-size="12" fill="#2a2f38">{escape(policy)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def render_trace_svg(tree_text: str) -> str:
    lines = tree_text.splitlines()[:20]
    height = 48 + len(lines) * 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
        '<rect width="100%" height="100%" fill="#1f2430"/>',
        '<text x="24" y="30" font-family="Menlo, monospace" font-size="20" fill="#f6f3eb">Retained Trace</text>',
    ]
    for index, line in enumerate(lines):
        y = 56 + index * 20
        parts.append(f'<text x="24" y="{y}" font-family="Menlo, monospace" font-size="14" fill="#d8dee9">{escape(line)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render launch-ready showcase assets from saved benchmark outputs.")
    parser.add_argument("--report-dir", type=str, required=True)
    parser.add_argument("--assets-dir", type=str, default="")
    parser.add_argument("--metric", type=str, default="answer_accuracy")
    parser.add_argument("--trace-policy", type=str, default="pairwise_tournament")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    assets_dir = Path(args.assets_dir) if args.assets_dir else report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = load_payload(report_dir / "summary.json")
    curves_payload = load_payload(report_dir / "curves.json")
    (assets_dir / "benchmark_snapshot.md").write_text(summary_table(summary_payload))
    (assets_dir / "architecture.svg").write_text(render_architecture_svg())
    (assets_dir / "policy_curve.svg").write_text(render_curve_svg(curves_payload, metric=args.metric))

    trace_dir = report_dir / "trace_examples" / args.trace_policy
    tree_files = sorted(trace_dir.glob("*.tree.txt"))
    if tree_files:
        (assets_dir / "trace_card.svg").write_text(render_trace_svg(tree_files[0].read_text()))

    manifest = {
        "report_dir": str(report_dir),
        "assets_dir": str(assets_dir),
        "files": sorted(path.name for path in assets_dir.iterdir() if path.is_file()),
    }
    (assets_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
