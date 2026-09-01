"""Make a Stage 2 learning-curve report from existing telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _episode_blocks(path: Path) -> list[dict[str, float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    blocks: list[dict[str, float]] = []
    old_count = 0
    old_time_sum = 0.0
    for row in rows:
        count = int(row["completed_episodes"])
        time_sum = float(row["mean_completed_time_in_band_fraction"]) * count
        if count > old_count:
            blocks.append(
                {
                    "iteration": float(row["iteration"]),
                    "time_in_band_fraction": (time_sum - old_time_sum) / (count - old_count),
                    "episodes": float(count - old_count),
                }
            )
            old_count = count
            old_time_sum = time_sum
    for index, block in enumerate(blocks):
        start = max(0, index - 4)
        window = blocks[start : index + 1]
        block["rolling_five_blocks"] = sum(
            item["time_in_band_fraction"] for item in window
        ) / len(window)
    return blocks


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def analyze(inputs: list[Path], labels: list[str], output: Path) -> dict[str, object]:
    if len(inputs) != len(labels):
        raise ValueError("each input must have one label")
    series = []
    for label, path in zip(labels, inputs):
        blocks = _episode_blocks(path)
        series.append({"label": label, "path": path.as_posix(), "blocks": blocks})

    width, height = 960, 540
    left, top, right, bottom = 70, 30, 25, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum_iteration = max(item["blocks"][-1]["iteration"] for item in series)
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
    ]
    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + (1.0 - value) * plot_height
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>')
    for index, item in enumerate(series):
        points = []
        for block in item["blocks"]:
            x = left + block["iteration"] / maximum_iteration * plot_width
            y = top + (1.0 - block["rolling_five_blocks"]) * plot_height
            points.append((x, y))
        color = colors[index % len(colors)]
        lines.append(f'<polyline points="{_polyline(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{left + index * 150}" y="{height-15}" font-family="sans-serif" font-size="13" fill="{color}">{item["label"]}</text>')
    lines.append(f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="13">iteration</text>')
    lines.append(f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="sans-serif" font-size="13">episode time in band</text>')
    lines.append("</svg>")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".svg").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report: dict[str, object] = {
        "metric": "completed_episode_block_time_in_band_fraction",
        "smoothing": "five completed-episode blocks",
        "warning": "This is on-policy training telemetry. It is not a fixed deterministic evaluation.",
        "series": series,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input, args.label, args.output)
    print(json.dumps({"series": len(report["series"]), "output": args.output.as_posix()}))


if __name__ == "__main__":
    main()
