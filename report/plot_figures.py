"""Draw the two research figures from the bench tables.

Reads bench/ROOFLINE.md and bench/FUSED.md and writes SVG files under
report/figures. Roofline cells marked launch are omitted: their time is
below 0.05 ms, so the percent of peak is not a throughput result.
"""

import html
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOFLINE = ROOT / "bench" / "ROOFLINE.md"
FUSED = ROOT / "bench" / "FUSED.md"
OUT = ROOT / "report" / "figures"

FP32_PEAK = 8.1e3
TENSOR_PEAK = 65.0e3
BANDWIDTH = 320.0
FP32_RIDGE = FP32_PEAK / BANDWIDTH
TENSOR_RIDGE = TENSOR_PEAK / BANDWIDTH


def table_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or cells[0] == "---" or set(cells[0]) == {"-"}:
            continue
        if cells[0] in {"Kernel", "Head width"}:
            continue
        rows.append(cells)
    return rows


def esc(text):
    return html.escape(str(text), quote=True)


class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.parts = []

    def add(self, text):
        self.parts.append(text)

    def text(self, x, y, body, size=13, anchor="start", weight="400"):
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="#1a1a1a" font-family="sans-serif">{esc(body)}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke, width=1.5, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{extra}/>'
        )

    def circle(self, x, y, r, fill, stroke="#1a1a1a"):
        self.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        )

    def rect(self, x, y, w, h, fill):
        self.add(
            f'<rect x="{x - w / 2:.1f}" y="{y - h / 2:.1f}" width="{w}" height="{h}" '
            f'fill="{fill}" stroke="#1a1a1a" stroke-width="1"/>'
        )

    def diamond(self, x, y, r, fill):
        points = f"{x:.1f},{y - r:.1f} {x + r:.1f},{y:.1f} {x:.1f},{y + r:.1f} {x - r:.1f},{y:.1f}"
        self.add(f'<polygon points="{points}" fill="{fill}" stroke="#1a1a1a" stroke-width="1"/>')

    def triangle(self, x, y, r, fill):
        points = f"{x:.1f},{y - r:.1f} {x + r:.1f},{y + r * 0.8:.1f} {x - r:.1f},{y + r * 0.8:.1f}"
        self.add(f'<polygon points="{points}" fill="{fill}" stroke="#1a1a1a" stroke-width="1"/>')

    def save(self, path, title, desc):
        body = "\n".join(self.parts)
        path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img">\n'
            f"<title>{esc(title)}</title>\n<desc>{esc(desc)}</desc>\n"
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            f"{body}\n</svg>\n",
            encoding="utf-8",
        )


def log_x(value, xmin, xmax, left, width):
    span = math.log10(xmax) - math.log10(xmin)
    return left + (math.log10(value) - math.log10(xmin)) / span * width


def log_y(value, ymin, ymax, top, height):
    span = math.log10(ymax) - math.log10(ymin)
    return top + (1.0 - (math.log10(value) - math.log10(ymin)) / span) * height


def draw_roofline(rows):
    width, height = 860, 560
    left, right, top, bottom = 78, 210, 48, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    xmin, xmax = 0.4, 600.0
    ymin, ymax = 4.0, 100000.0
    canvas = Canvas(width, height)
    canvas.text(left, 28, "T4 roofline", 18, weight="500")

    def px(intensity):
        return log_x(intensity, xmin, xmax, left, plot_w)

    def py(gflops):
        return log_y(gflops, ymin, ymax, top, plot_h)

    for tick in (1, 10, 100):
        x = px(tick)
        canvas.line(x, top, x, top + plot_h, "#e4e4e4", 1)
        canvas.text(x, top + plot_h + 18, str(tick), 11, "middle")
    for tick in (10, 100, 1000, 10000):
        y = py(tick)
        canvas.line(left, y, left + plot_w, y, "#e4e4e4", 1)
        canvas.text(left - 8, y + 4, str(tick), 11, "end")

    roof = [(0.4, BANDWIDTH * 0.4), (TENSOR_RIDGE, TENSOR_PEAK), (xmax, TENSOR_PEAK)]
    canvas.line(px(roof[0][0]), py(roof[0][1]), px(roof[1][0]), py(roof[1][1]), "#1a1a1a", 1.75)
    canvas.line(px(roof[1][0]), py(roof[1][1]), px(roof[2][0]), py(roof[2][1]), "#1a1a1a", 1.75)
    canvas.line(px(FP32_RIDGE), py(FP32_PEAK), px(xmax), py(FP32_PEAK), "#1a1a1a", 1.25, "5 4")
    canvas.text(px(80), py(FP32_PEAK) - 8, "8.1 TFLOP/s", 11, "start")
    canvas.text(px(220), py(TENSOR_PEAK) + 16, "65 TFLOP/s", 11, "start")
    canvas.text(px(1.2), py(BANDWIDTH * 1.2) - 8, "320 GB/s", 11, "start")

    series = {
        "cublas": ("#1f4e79", "circle"),
        "tiled": ("#c45911", "square"),
        "wmma": ("#2d6a4f", "diamond"),
        "softmax": ("#6c3483", "triangle"),
        "layernorm": ("#7b4b2a", "square"),
        "scored attention": ("#1a5276", "circle"),
        "fused attention": ("#922b21", "triangle"),
    }
    kept = [row for row in rows if row[-1] != "launch"]
    for kernel, intensity, gflops in (
        (row[0], float(row[5]), float(row[3])) for row in kept
    ):
        color, shape = series[kernel]
        x, y = px(intensity), py(gflops)
        if shape == "circle":
            canvas.circle(x, y, 5, color)
        elif shape == "square":
            canvas.rect(x, y, 9, 9, color)
        elif shape == "diamond":
            canvas.diamond(x, y, 6, color)
        else:
            canvas.triangle(x, y, 6, color)

    labels = [
        ("cuBLAS 1024³", 170.67, 4172.98, 8, -10),
        ("tiled 1024³", 170.67, 2499.67, 8, 14),
        ("WMMA 1024³", 256.0, 2159.69, 8, 14),
    ]
    for label, intensity, gflops, dx, dy in labels:
        canvas.text(px(intensity) + dx, py(gflops) + dy, label, 11)

    canvas.line(left, top, left, top + plot_h, "#1a1a1a", 1.25)
    canvas.line(left, top + plot_h, left + plot_w, top + plot_h, "#1a1a1a", 1.25)
    canvas.text(left + plot_w / 2, height - 16, "Arithmetic intensity (FLOP/byte)", 13, "middle")
    canvas.add(
        f'<text x="18" y="{top + plot_h / 2:.1f}" font-size="13" fill="#1a1a1a" '
        f'font-family="sans-serif" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_h / 2:.1f})">Achieved GFLOP/s</text>'
    )

    legend = [
        ("cuBLAS", "cublas"),
        ("tiled", "tiled"),
        ("WMMA", "wmma"),
        ("softmax", "softmax"),
        ("LayerNorm", "layernorm"),
        ("scored attention", "scored attention"),
        ("fused attention", "fused attention"),
    ]
    legend_x = left + plot_w + 24
    canvas.text(legend_x, top + 4, "Kernel", 13, weight="500")
    for index, (label, key) in enumerate(legend):
        color, shape = series[key]
        y = top + 28 + index * 26
        if shape == "circle":
            canvas.circle(legend_x + 6, y - 4, 5, color)
        elif shape == "square":
            canvas.rect(legend_x + 6, y - 4, 9, 9, color)
        elif shape == "diamond":
            canvas.diamond(legend_x + 6, y - 4, 6, color)
        else:
            canvas.triangle(legend_x + 6, y - 4, 6, color)
        canvas.text(legend_x + 20, y, label, 12)
    canvas.text(legend_x, top + 230, "Launch cells under", 11)
    canvas.text(legend_x, top + 246, "0.05 ms are omitted.", 11)
    canvas.save(
        OUT / "roofline.svg",
        "T4 roofline",
        "Achieved GFLOP/s against arithmetic intensity. "
        "Roofs are 8.1 TFLOP/s, 65 TFLOP/s, and 320 GB/s. Launch cells are omitted.",
    )


def draw_latency(rows):
    width, height = 760, 480
    left, right, top, bottom = 68, 180, 48, 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    lengths = [64, 128, 256, 512, 1024]
    ymax = 3.6
    canvas = Canvas(width, height)
    canvas.text(left, 28, "Attention latency", 18, weight="500")

    by_width = {}
    for cells in rows:
        width_head = int(cells[0])
        length = int(cells[1])
        by_width.setdefault(width_head, {})[length] = (float(cells[2]), float(cells[4]))

    def px(length):
        return left + (math.log2(length) - math.log2(64)) / (math.log2(1024) - math.log2(64)) * plot_w

    def py(ms):
        return top + (1.0 - ms / ymax) * plot_h

    for ms in (0, 1, 2, 3):
        y = py(ms)
        canvas.line(left, y, left + plot_w, y, "#e4e4e4", 1)
        canvas.text(left - 8, y + 4, str(ms), 11, "end")
    for length in lengths:
        x = px(length)
        canvas.line(x, top, x, top + plot_h, "#e4e4e4", 1)
        canvas.text(x, top + plot_h + 18, str(length), 11, "middle")

    styles = [
        (8, 0, "#1f4e79", None, "cuBLAS, head 8"),
        (8, 1, "#922b21", None, "fused, head 8"),
        (32, 0, "#1f4e79", "6 4", "cuBLAS, head 32"),
        (32, 1, "#922b21", "6 4", "fused, head 32"),
    ]
    for head, column, color, dash, _label in styles:
        points = []
        for length in lengths:
            ms = by_width[head][length][column]
            points.append((px(length), py(ms)))
        for start, stop in zip(points, points[1:]):
            canvas.line(start[0], start[1], stop[0], stop[1], color, 2, dash)
        for x, y in points:
            if dash:
                canvas.rect(x, y, 8, 8, color)
            else:
                canvas.circle(x, y, 4.5, color)

    canvas.line(left, top, left, top + plot_h, "#1a1a1a", 1.25)
    canvas.line(left, top + plot_h, left + plot_w, top + plot_h, "#1a1a1a", 1.25)
    canvas.text(left + plot_w / 2, height - 14, "Sequence length", 13, "middle")
    canvas.add(
        f'<text x="18" y="{top + plot_h / 2:.1f}" font-size="13" fill="#1a1a1a" '
        f'font-family="sans-serif" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_h / 2:.1f})">Time (ms)</text>'
    )

    legend_x = left + plot_w + 20
    canvas.text(legend_x, top + 4, "Path", 13, weight="500")
    for index, (head, column, color, dash, label) in enumerate(styles):
        y = top + 32 + index * 28
        canvas.line(legend_x, y - 4, legend_x + 28, y - 4, color, 2, dash)
        if dash:
            canvas.rect(legend_x + 14, y - 4, 8, 8, color)
        else:
            canvas.circle(legend_x + 14, y - 4, 4.5, color)
        canvas.text(legend_x + 36, y, label, 12)
    canvas.text(legend_x, top + 188, "Four heads. Custom", 11)
    canvas.text(legend_x, top + 204, "stays beside cuBLAS", 11)
    canvas.text(legend_x, top + 220, "and is omitted.", 11)
    canvas.save(
        OUT / "attention_latency.svg",
        "Attention latency versus sequence length",
        "cuBLAS scored attention and fused attention for head widths 8 and 32.",
    )


def write_pdf(stem):
    try:
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg
    except ImportError:
        return
    drawing = svg2rlg(str(OUT / f"{stem}.svg"))
    if drawing is None:
        return
    renderPDF.drawToFile(drawing, str(OUT / f"{stem}.pdf"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw_roofline(table_rows(ROOFLINE))
    draw_latency(table_rows(FUSED))
    for stem in ("roofline", "attention_latency"):
        write_pdf(stem)
        print(OUT / f"{stem}.svg")


if __name__ == "__main__":
    main()
