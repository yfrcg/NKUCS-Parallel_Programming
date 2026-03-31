import csv
from pathlib import Path


ROOT = Path(r"D:\WorkStation\并行程序设计\CPU架构编程")
RESULTS = ROOT / "results"
PROFILE = Path(r"C:\AMD_Profiling")


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_amduprof_metrics(path: Path):
    metrics = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            key, value = line.split(",", 1)
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pick(rows, algorithm):
    return [row for row in rows if row["algorithm"] == algorithm]


def make_line_chart(path: Path, title, x_labels, series, y_label):
    width = 920
    height = 520
    margin_left = 90
    margin_right = 30
    margin_top = 60
    margin_bottom = 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_values = [value for item in series for value in item["values"]]
    y_max = max(all_values) * 1.15 if all_values else 1.0
    y_max = max(y_max, 1.0)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    grid_lines = 5

    def sx(i):
        if len(x_labels) == 1:
            return margin_left + plot_w / 2
        return margin_left + i * (plot_w / (len(x_labels) - 1))

    def sy(v):
        return margin_top + plot_h - (v / y_max) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="24" font-family="Segoe UI">{title}</text>',
        f'<text x="22" y="{margin_top + plot_h / 2}" transform="rotate(-90 22 {margin_top + plot_h / 2})" '
        f'font-size="16" font-family="Segoe UI">{y_label}</text>',
    ]

    for i in range(grid_lines + 1):
        value = y_max * i / grid_lines
        y = sy(value)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="12" font-family="Consolas">{value:.2f}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="2"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="2"/>')

    for idx, label in enumerate(x_labels):
        x = sx(idx)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top + plot_h}" x2="{x:.2f}" y2="{margin_top + plot_h + 6}" stroke="#222" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 40}" text-anchor="middle" font-size="13" font-family="Consolas">{label}</text>')

    legend_x = margin_left + 10
    legend_y = height - 20
    for idx, item in enumerate(series):
        color = colors[idx % len(colors)]
        points = " ".join(f"{sx(i):.2f},{sy(v):.2f}" for i, v in enumerate(item["values"]))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        for i, value in enumerate(item["values"]):
            parts.append(f'<circle cx="{sx(i):.2f}" cy="{sy(value):.2f}" r="4" fill="{color}"/>')
            parts.append(f'<text x="{sx(i):.2f}" y="{sy(value) - 10:.2f}" text-anchor="middle" font-size="11" font-family="Consolas" fill="{color}">{value:.2f}</text>')
        lx = legend_x + idx * 180
        parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 24}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{lx + 30}" y="{legend_y + 5}" font-size="13" font-family="Segoe UI">{item["name"]}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_grouped_bar_chart(path: Path, title, categories, series, y_label):
    width = 920
    height = 520
    margin_left = 90
    margin_right = 30
    margin_top = 60
    margin_bottom = 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_values = [value for item in series for value in item["values"]]
    y_max = max(all_values) * 1.15 if all_values else 1.0
    y_max = max(y_max, 1.0)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    def sy(v):
        return margin_top + plot_h - (v / y_max) * plot_h

    group_w = plot_w / len(categories)
    bar_w = group_w / (len(series) + 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="24" font-family="Segoe UI">{title}</text>',
        f'<text x="22" y="{margin_top + plot_h / 2}" transform="rotate(-90 22 {margin_top + plot_h / 2})" '
        f'font-size="16" font-family="Segoe UI">{y_label}</text>',
    ]

    for i in range(6):
        value = y_max * i / 5
        y = sy(value)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="12" font-family="Consolas">{value:.2f}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="2"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="2"/>')

    for c_idx, category in enumerate(categories):
        center_x = margin_left + c_idx * group_w + group_w / 2
        parts.append(f'<text x="{center_x:.2f}" y="{height - 45}" text-anchor="middle" font-size="13" font-family="Consolas">{category}</text>')
        for s_idx, item in enumerate(series):
            x = margin_left + c_idx * group_w + (s_idx + 0.5) * bar_w
            value = item["values"][c_idx]
            y = sy(value)
            h = margin_top + plot_h - y
            color = colors[s_idx % len(colors)]
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.8:.2f}" height="{h:.2f}" fill="{color}"/>')
            parts.append(f'<text x="{x + bar_w * 0.4:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="11" font-family="Consolas" fill="{color}">{value:.2f}</text>')

    legend_x = margin_left + 10
    legend_y = height - 18
    for idx, item in enumerate(series):
        color = colors[idx % len(colors)]
        lx = legend_x + idx * 180
        parts.append(f'<rect x="{lx}" y="{legend_y - 12}" width="18" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx + 25}" y="{legend_y - 2}" font-size="13" font-family="Segoe UI">{item["name"]}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    matrix_rows = read_csv_rows(RESULTS / "matrix_timing.csv")
    sum_rows = read_csv_rows(RESULTS / "sum_timing.csv")

    naive_l1l2 = parse_amduprof_metrics(PROFILE / "matrix_naive_l1l2.csv")
    cache_l1l2 = parse_amduprof_metrics(PROFILE / "matrix_cache_l1l2.csv")
    naive_l3 = parse_amduprof_metrics(PROFILE / "matrix_naive_l3.csv")
    cache_l3 = parse_amduprof_metrics(PROFILE / "matrix_cache_l3.csv")

    profile_rows = []
    for name, l1l2, l3 in [
        ("naive", naive_l1l2, naive_l3),
        ("cache_opt", cache_l1l2, cache_l3),
    ]:
        dc_access = l1l2["DC Access (pti)"]
        l2_access_dc = l1l2["L2 Access from DC Miss (pti)"]
        l2_access = l1l2["L2 Access (pti)"]
        l2_hit = l1l2["L2 Hit (pti)"]
        profile_rows.append({
            "algorithm": name,
            "l1d_hit_rate_pct": round((1.0 - l2_access_dc / dc_access) * 100.0, 2),
            "l2_hit_rate_pct": round((l2_hit / l2_access) * 100.0, 2),
            "l3_hit_rate_pct": round(l3["L3 Hit %"], 2),
            "ipc": round(l3["IPC (Sys + User)"], 2),
            "dc_access_pti": round(dc_access, 2),
            "l2_access_pti": round(l2_access, 2),
            "l3_access_pti": round(l3["L3 Access (pti)"], 2),
        })

    write_csv(
        RESULTS / "matrix_profile_summary.csv",
        ["algorithm", "l1d_hit_rate_pct", "l2_hit_rate_pct", "l3_hit_rate_pct", "ipc", "dc_access_pti", "l2_access_pti", "l3_access_pti"],
        profile_rows,
    )

    sum_summary_rows = []
    for row in sum_rows:
        if row["algorithm"] == "naive":
            continue
        sum_summary_rows.append({
            "n": int(row["n"]),
            "algorithm": row["algorithm"],
            "ns_per_elem": round(float(row["ns_per_elem"]), 6),
            "gadds": round(float(row["gadds"]), 6),
            "speedup": round(float(row["speedup"]), 6),
            "abs_err": round(float(row["abs_err"]), 12),
        })

    write_csv(
        RESULTS / "sum_key_metrics.csv",
        ["n", "algorithm", "ns_per_elem", "gadds", "speedup", "abs_err"],
        sum_summary_rows,
    )

    x_matrix = [row["n"] for row in pick(matrix_rows, "naive")]
    make_line_chart(
        RESULTS / "matrix_runtime.svg",
        "Experiment 1 Runtime Comparison",
        x_matrix,
        [
            {"name": "naive", "values": [float(row["ms_per_call"]) for row in pick(matrix_rows, "naive")]},
            {"name": "cache_opt", "values": [float(row["ms_per_call"]) for row in pick(matrix_rows, "cache_opt")]},
        ],
        "ms per call",
    )

    make_grouped_bar_chart(
        RESULTS / "matrix_cache_hit_rates.svg",
        "Experiment 1 Cache Hit Rates (AMDuProf)",
        ["L1D", "L2", "L3"],
        [
            {"name": "naive", "values": [profile_rows[0]["l1d_hit_rate_pct"], profile_rows[0]["l2_hit_rate_pct"], profile_rows[0]["l3_hit_rate_pct"]]},
            {"name": "cache_opt", "values": [profile_rows[1]["l1d_hit_rate_pct"], profile_rows[1]["l2_hit_rate_pct"], profile_rows[1]["l3_hit_rate_pct"]]},
        ],
        "hit rate (%)",
    )

    x_sum = [row["n"] for row in pick(sum_rows, "naive")]
    make_line_chart(
        RESULTS / "sum_ns_per_elem.svg",
        "Experiment 2 ns/element Comparison",
        x_sum,
        [
            {"name": "naive", "values": [float(row["ns_per_elem"]) for row in pick(sum_rows, "naive")]},
            {"name": "superscalar_2way", "values": [float(row["ns_per_elem"]) for row in pick(sum_rows, "superscalar_2way")]},
            {"name": "recursive", "values": [float(row["ns_per_elem"]) for row in pick(sum_rows, "recursive")]},
        ],
        "ns per element",
    )

    make_line_chart(
        RESULTS / "sum_speedup.svg",
        "Experiment 2 Speedup Comparison",
        x_sum,
        [
            {"name": "superscalar_2way", "values": [float(row["speedup"]) for row in pick(sum_rows, "superscalar_2way")]},
            {"name": "recursive", "values": [float(row["speedup"]) for row in pick(sum_rows, "recursive")]},
        ],
        "speedup vs naive",
    )


if __name__ == "__main__":
    main()
