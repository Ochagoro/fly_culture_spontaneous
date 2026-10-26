"""Compose the research video "自発的な『点火』はどこから始まるか — 培養神経とハエの脳".

1280x720, 30 fps, Japanese captions drawn with PIL, H.264 via imageio-ffmpeg.
Everything is rebuilt from the project data (results/...); nothing is cached.

    python compose.py            # render video/fly_culture_ignition_ja.mp4
    python compose.py --check    # write one check frame per segment to video/build/

Colour code: culture = blue, fly = orange, ignition = yellow/red.
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
sys.path.insert(0, str(HERE))
from brain_render import BrainCanvas  # noqa: E402

W, H, FPS = 1280, 720, 30
BG = (14, 19, 30)
PANEL = (24, 31, 46)
BAND = (5, 7, 12)
INK = (244, 245, 248)
INK2 = (172, 180, 194)
GREY = (96, 106, 124)
BLUE = (72, 152, 245)          # culture
ORANGE = (240, 120, 50)        # fly
YELLOW = (255, 214, 64)        # ignition
RED = (238, 72, 64)            # ignition / runaway
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
_fonts: dict = {}

CULTURE_EX = ROOT / "results/culture/example_dandi_sub-23312_ses-23312-240504.npz"
CULTURE_MAPS = ROOT / "results/culture/ignition_maps_dandi.npz"
FLY_SPIKES = ROOT / "results/fly/video/spikes_fixAL_g2.50.npz"
FLY_GAIN_RUNS = {g: ROOT / f"results/fly/bursts/fixAL_g{g}_s0.npz" for g in ("1.00", "2.00", "2.50")}
SILENCE_TRACES = ROOT / "results/fly/video/silence_traces.npz"
SILENCE_CONFIRM = ROOT / "results/fly/silencing_confirm.csv"
TURNER_REGIONS = ROOT / "results/fly/turner_region_summary.csv"
TURNER_AL = ROOT / "results/fly/turner_AL_vs_others.csv"


# ------------------------------------------------------------------ drawing
def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT_R, size, index=0)
    return _fonts[k]


def text(d, xy, s, size=24, bold=False, fill=INK, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def text_w(s, size, bold=False):
    return font(size, bold).getlength(s)


def blank():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def caption(d, s, size=32, note=None, note_size=20):
    """White caption(s) on a dark band at the bottom of the frame; optional smaller note lines below."""
    lines = s.split("\n") if s else []
    notes = note.split("\n") if note else []
    if not lines and not notes:
        return
    lh, nh = int(size * 1.45), int(note_size * 1.4)
    bh = 22 + lh * len(lines) + nh * len(notes) + (6 if lines and notes else 0)
    wmax = max([text_w(x, size, True) for x in lines] + [text_w(x, note_size) for x in notes])
    x0, x1 = W // 2 - wmax / 2 - 30, W // 2 + wmax / 2 + 30
    y1 = H - 14
    d.rounded_rectangle([x0, y1 - bh, x1, y1], 12, fill=BAND)
    for k, ln in enumerate(lines):
        text(d, (W // 2, y1 - bh + 11 + lh * k + lh / 2), ln, size, True, INK, anchor="mm")
    ny = y1 - bh + 11 + lh * len(lines) + (6 if lines else 0)
    for k, ln in enumerate(notes):
        text(d, (W // 2, ny + nh * k + nh / 2), ln, note_size, False, (205, 211, 221), anchor="mm")


def cap_at(t, sched):
    s = ""
    for t0, txt in sched:
        if t >= t0:
            s = txt
    return s


def header(d, badge, title, col, sub=None):
    bw = text_w(badge, 22, True) + 28
    d.rounded_rectangle([36, 24, 36 + bw, 62], 8, fill=col)
    text(d, (36 + bw / 2, 43), badge, 22, True, BG, anchor="mm")
    text(d, (56 + bw, 43), title, 32, True, INK, anchor="lm")
    if sub:
        text(d, (38, 84), sub, 21, False, INK2, anchor="lm")


def ring(d, cx, cy, r, col, width=3):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=width)


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def polyline(d, xs, ys, col, width=2):
    pts = list(zip(np.asarray(xs, float).tolist(), np.asarray(ys, float).tolist()))
    if len(pts) > 1:
        d.line(pts, fill=col, width=width, joint="curve")


# --------------------------------------------------------------- segments
class Seg:
    """A segment = number of frames + a function rendering frame i (PIL image).
    check: time (s) or fraction (<1) of the frame written by --check; extra: more check times (s)."""

    def __init__(self, name, sec, fn, fade=8, check=0.6, extra=()):
        self.name, self.n, self.fn, self.fade, self.check = name, int(round(sec * FPS)), fn, fade, check
        self.extra = extra

    def frame(self, i):
        arr = np.asarray(self.fn(i, i / FPS)).astype(np.float32)
        a = min(1.0, (i + 1) / self.fade, (self.n - i) / self.fade) if self.fade else 1.0
        if a < 1:
            arr = arr * a + np.array(BG, np.float32) * (1 - a)
        return arr.astype(np.uint8)


# ---------------------------------------------------------------- culture data
class Culture:
    def __init__(self):
        d = np.load(CULTURE_EX)
        self.fr = d["spike_frame_20kHz"]
        self.ch = d["spike_channel_index"].astype(np.int64)
        self.x, self.y = d["channel_x_um"], d["channel_y_um"]
        self.active = d["channel_active"]
        self.bs, self.be = d["burst_start_s"], d["burst_end_s"]
        self.ix, self.iy = d["ign_x_um"], d["ign_y_um"]
        self.pop10 = d["pop_count_10ms"]
        pm = d["ign_pmap"]
        i, j = np.unravel_index(np.nanargmax(pm), pm.shape)
        self.hot = ((i + 0.5) * 100.0, (j + 0.5) * 100.0)      # hot spot centre (um)
        self.nch = len(self.x)

    def intensity(self, t, tau=0.04, span=0.2):
        lo, hi = np.searchsorted(self.fr, [(t - span) * 20000, t * 20000])
        ts = self.fr[lo:hi] / 20000.0
        w = np.exp(-(t - ts) / tau)
        return np.bincount(self.ch[lo:hi], weights=w, minlength=self.nch)


class ChipCanvas:
    """Electrode plane (3.9 x 2.1 mm), each active electrode a dot that glows after each spike."""
    PW, PH = 3900.0, 2100.0

    def __init__(self, cul: Culture, width):
        self.s = width / self.PW
        self.w, self.h = int(width), int(round(self.PH * self.s))
        self.px = np.clip((cul.x * self.s).round().astype(int), 0, self.w - 1)
        self.py = np.clip(((self.PH - cul.y) * self.s).round().astype(int), 0, self.h - 1)
        self.act = cul.active
        base = np.zeros((self.h, self.w), np.float32)
        np.add.at(base, (self.py[self.act], self.px[self.act]), 1.0)
        base = np.clip(gaussian_filter(base, 1.1) * 5, 0, 1)
        bg = np.array([20, 27, 42], np.float32)
        dot = np.array([62, 84, 118], np.float32)
        self.base = bg[None, None] + base[..., None] * (dot - bg)[None, None]

    def xy(self, x_um, y_um):
        return x_um * self.s, (self.PH - y_um) * self.s

    def frame(self, inten, gain=2.2):
        img = self.base.copy()
        buf = np.zeros((self.h, self.w), np.float32)
        m = self.act & (inten > 1e-3)
        np.add.at(buf, (self.py[m], self.px[m]), 1 - np.exp(-1.5 * inten[m]))
        glow = gaussian_filter(buf, 1.3) * 6.0 + gaussian_filter(buf, 5.0) * 12.0
        a = np.clip(glow * gain, 0, 1)[..., None]
        c0 = np.array(BLUE, np.float32)[None, None]
        c1 = np.array([235, 246, 255], np.float32)[None, None]
        col = c0 * (1 - a) + c1 * a
        img = img * (1 - a) + col * a
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


_cul = None


def culture():
    global _cul
    if _cul is None:
        _cul = Culture()
    return _cul


# -------------------------------------------------------------------- fly data
class Fly:
    def __init__(self):
        d = np.load(FLY_SPIKES)
        self.t = d["t"]          # 0.1 ms steps, sorted
        self.n = d["n"].astype(np.int64)
        self.pop = d["pop"]
        self.on, self.off = d["on"], d["off"]
        self.N = 138639
        ann = pd.read_csv(ROOT / "data/neuron_annotations.tsv", sep="\t", usecols=["root_id", "cell_type", "side"])
        comp = pd.read_csv(ROOT / "data/shiu_model/Completeness_783.csv", index_col=0)
        ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(comp.index)
        self.lln = np.flatnonzero((ann.cell_type == "lLN1_bc").to_numpy())
        self.lln_side = ann.side.to_numpy()[self.lln]
        self.is_lln = np.zeros(self.N, bool)
        self.is_lln[self.lln] = True
        self.xy = np.load(HERE / "brain_xy.npy")

    def intensity(self, T, tau=8.0, span=40.0):
        lo, hi = np.searchsorted(self.t, [(T - span) * 10, T * 10])
        w = np.exp(-(T - self.t[lo:hi] / 10.0) / tau).astype(np.float32)
        x = np.zeros(self.N, np.float32)
        np.add.at(x, self.n[lo:hi], w)
        return x

    def lln_rate(self, t0, t1, bin_ms=2.0):
        """lLN1_bc spikes per bin over [t0, t1) ms."""
        lo, hi = np.searchsorted(self.t, [t0 * 10, t1 * 10])
        m = self.is_lln[self.n[lo:hi]]
        tt = self.t[lo:hi][m] / 10.0
        edges = np.arange(t0, t1 + bin_ms, bin_ms)
        return np.histogram(tt, edges)[0]


_fly = None


def fly():
    global _fly
    if _fly is None:
        _fly = Fly()
    return _fly


_canv = {}


def brain_canvas(w, h):
    if (w, h) not in _canv:
        _canv[(w, h)] = BrainCanvas(fly().xy, w, h)
    return _canv[(w, h)]


def lln_rings(cv):
    """(cx, cy, r) on canvas for the lLN1_bc cluster of each side."""
    f = fly()
    out = []
    for s in ("left", "right"):
        m = f.lln[f.lln_side == s]
        px, py = cv.px[m].astype(float), cv.py[m].astype(float)
        cx, cy = px.mean(), py.mean()
        r = np.percentile(np.hypot(px - cx, py - cy), 90) + 16
        out.append((cx, cy, r))
    return out


# ====================================================================== 1 title
def seg_title():
    def fn(i, t):
        im, d = blank()
        text(d, (W // 2, 170), "自発的な「点火」はどこから始まるか", 54, True, anchor="mm")
        text(d, (W // 2, 245), "— 培養神経とハエの脳 —", 32, False, INK2, anchor="mm")
        p = smoothstep(t / 0.8)
        lx = int(-420 + (190 + 420) * p)
        rx = int(W + 40 - (W + 40 - 720) * p)
        for x0, col, a, b in [(lx, BLUE, "培養神経", "ラットの神経・電極チップ"),
                              (rx, ORANGE, "ハエの脳", "全脳配線図モデル＋本物のハエ")]:
            d.rounded_rectangle([x0, 330, x0 + 370, 470], 16, fill=PANEL, outline=col, width=4)
            text(d, (x0 + 185, 382), a, 44, True, col, anchor="mm")
            text(d, (x0 + 185, 438), b, 22, False, INK2, anchor="mm")
        if t > 0.8:
            sz = int(64 + 10 * max(0.0, 1 - (t - 0.8) / 0.3))
            text(d, (W // 2, 400), "VS", sz, True, YELLOW, anchor="mm")
        if t > 1.6:
            text(d, (W // 2, 560), "2 つの神経ネットワークに、同じ問題を出してみました", 30, True, INK, anchor="mm")
        return im
    return Seg("title", 5.0, fn, check=3.5)


# ====================================================================== 2 intro
def seg_intro():
    cul = culture()
    p = cul.pop10[: 2400].astype(float)          # first 24 s, 10-ms bins
    p = p.reshape(-1, 4).sum(1)                   # 40-ms bins
    xs = np.linspace(110, 1170, len(p))
    y0, y1 = 470, 250
    ys = y0 - (y0 - y1) * p / p.max()
    caps = [(0.0, "神経細胞は、外から何も入力がなくても\n勝手に発火します（自発活動）"),
            (3.4, "お皿で育てた神経では、ときどき全体が\nいっせいに発火します（バースト）"),
            (6.8, "その「火」は、どこで点くのでしょう？")]

    def fn(i, t):
        im, d = blank()
        text(d, (W // 2, 60), "入力ゼロでも、神経は勝手に燃え上がる", 34, True, anchor="mm")
        d.rounded_rectangle([80, 110, 1200, 500], 14, fill=PANEL)
        text(d, (110, 140), "培養神経の発火数（チップ全体・40 ms ごと）", 22, True, BLUE, anchor="lm")
        prog = min(1.0, t / 6.0)
        k = max(2, int(len(p) * prog))
        polyline(d, xs[:k], ys[:k], BLUE, 2)
        d.line([110, y0 + 2, 1170, y0 + 2], fill=GREY, width=1)
        for s in range(0, 25, 5):
            x = 110 + 1060 * s / 24
            text(d, (x, y0 + 20), f"{s} 秒", 18, False, INK2, anchor="mm")
        if t > 3.4:
            # mark the biggest bursts
            pk = np.argsort(p[:k])[::-1]
            shown = []
            for j in pk:
                if all(abs(j - s) > 25 for s in shown) and p[j] > 0.5 * p.max():
                    shown.append(j)
                if len(shown) >= 4:
                    break
            for j in shown:
                text(d, (xs[j], ys[j] - 18), "バースト", 18, True, YELLOW, anchor="mm")
        if t > 6.8:
            qs = ["Q1 どこから点火する？", "Q2 何がそれを決める？", "Q3 暴走まであとどれくらい？"]
            for k2, q in enumerate(qs):
                a = smoothstep((t - 6.8 - 0.4 * k2) / 0.4)
                if a > 0:
                    x0 = 120 + k2 * 360
                    d.rounded_rectangle([x0, 520, x0 + 330, 580], 10, fill=(40, 34, 20), outline=YELLOW, width=2)
                    text(d, (x0 + 165, 550), q, 22, True, YELLOW, anchor="mm")
        caption(d, cap_at(t, caps))
        return im
    return Seg("intro", 10.0, fn, check=8.5)


# ================================================================ round cards
def seg_round(num, who, col, question, sec=3.0):
    def fn(i, t):
        im, d = blank()
        p = smoothstep(t / 0.5)
        text(d, (W // 2, 230), f"ROUND {num}", int(70 + 20 * (1 - p)), True, col, anchor="mm")
        text(d, (W // 2, 330), who, 46, True, INK, anchor="mm")
        if t > 0.6:
            text(d, (W // 2, 430), question, 32, True, INK2, anchor="mm")
        return im
    return Seg(f"round{num}", sec, fn, check=1.5)


# ============================================================ 3a chip animation
def seg_chip():
    cul = culture()
    cv = ChipCanvas(cul, 900)
    X0, Y0 = (W - cv.w) // 2, 104
    # piecewise timeline: (video seconds, recording start, recording end, label)
    # 1 video s = 0.25 s of recording when playing slowly; the quiet gap is fast-forwarded.
    plan = [(1.1, 32.300, 32.575), (0.45, 32.575, 32.592), (2.4, 32.592, 32.592),
            (2.7, 32.592, 32.700), (0.8, 32.700, 32.900), (1.0, 32.900, 34.050),
            (0.7, 34.050, 34.118), (1.8, 34.118, 34.118), (2.1, 34.118, 34.202)]
    starts = np.cumsum([0] + [p[0] for p in plan])
    sec = starts[-1] + 0.4
    bursts = [k for k in range(len(cul.bs)) if 32.3 < cul.bs[k] < 34.3]
    hx, hy = cv.xy(*cul.hot)
    caps = [(0.0, "電極チップの上の培養神経。光る点 = その場所で発火"),
            (starts[1], "まず、この 1 か所が強く光ります（点火！）"),
            (starts[3], "そこから、まわりへ燃え広がります"),
            (starts[5], "しばらく静かになって…"),
            (starts[6], "次のバーストも、同じ場所から点火します")]

    def rec_time(t):
        k = int(np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(plan) - 1))
        dur, r0, r1 = plan[k]
        u = np.clip((t - starts[k]) / dur, 0, 1)
        if r1 == r0:
            lab = "一時停止"
        elif (r1 - r0) / dur > 0.5:
            lab = "早送り"
        else:
            lab = f"{dur / (r1 - r0):.0f} 倍スロー"
        return r0 + (r1 - r0) * u, lab

    def fn(i, t):
        im, d = blank()
        tr, lab = rec_time(t)
        header(d, "ROUND 1", "培養神経：どこから点火する？", BLUE,
               "ラットの大脳皮質の神経を、3.9 × 2.1 mm の電極チップで記録（公開データ）")
        im.paste(cv.frame(cul.intensity(tr)), (X0, Y0))
        d.rectangle([X0 - 1, Y0 - 1, X0 + cv.w, Y0 + cv.h], outline=(70, 82, 104), width=2)
        sb = 1000 * cv.s
        xr, yb = X0 + cv.w - 24, Y0 + cv.h - 22
        d.line([xr - sb, yb, xr, yb], fill=INK, width=3)
        text(d, (xr - sb / 2, yb - 16), "1 mm", 18, True, INK2, anchor="mm")
        for k in bursts:
            dt = tr - cul.bs[k]
            if 0 <= dt < 0.3:
                cx, cy = cv.xy(cul.ix[k], cul.iy[k])
                cx, cy = X0 + cx, Y0 + cy
                r = 44 + 80 * dt
                ring(d, cx, cy, r, YELLOW, 4)
                ring(d, cx, cy, 7, RED, 4)
                text(d, (cx + r + 12, cy - 4), "点火！", 34, True, YELLOW, anchor="lm")
        if bursts and cul.be[bursts[0]] < tr < cul.bs[bursts[-1]]:
            ring(d, X0 + hx, Y0 + hy, 30, (200, 170, 60), 2)
            text(d, (X0 + hx + 40, Y0 + hy + 2), "さっきの点火場所", 22, True, (225, 195, 90), anchor="lm")
        text(d, (W - 40, 30), f"t = {tr:5.2f} 秒", 24, True, anchor="ra")
        text(d, (W - 40, 62), lab, 20, True, YELLOW if not lab.endswith("スロー") else INK2, anchor="ra")
        text(d, (X0, Y0 + cv.h + 20), "1 点 = 1 つの電極（活動している 797 個）", 20, False, INK2, anchor="lm")
        caption(d, cap_at(t, caps))
        return im
    return Seg("chip", sec, fn, check=2.6, extra=(5.2, 6.2, 8.9, 10.6))


# ============================================================== 3b day maps
def seg_maps():
    m = np.load(CULTURE_MAPS)
    sel = np.flatnonzero(m["culture"] == "23312")
    sess = m["session"][sel]
    want = ["240428", "240429", "240501", "240503", "240504", "240505", "240509", "240512"]
    idx = [sel[np.flatnonzero([s.endswith(w) for s in sess])[0]] for w in want]
    labels = [f"{int(w[2:4])}月{int(w[4:6])}日" for w in want]
    cell = 7
    mw, mh = 39 * cell, 21 * cell
    tiles = []
    for j in idx:
        pm = np.nan_to_num(m["ign_pmap"][j]).T[::-1]          # rows = y (top = 2.1 mm)
        sup = m["support"][j].T[::-1]
        v = pm / max(pm.max(), 1e-9)
        v = np.clip(v / 0.8, 0, 1)
        stops = np.array([[22, 28, 42], [150, 30, 30], [238, 72, 64], [255, 214, 64], [255, 250, 220]], float)
        pos = np.array([0, 0.15, 0.4, 0.75, 1.0])
        rgb = np.stack([np.interp(v, pos, stops[:, c]) for c in range(3)], -1)
        rgb[~sup] = (44, 50, 64)
        img = Image.fromarray(rgb.astype(np.uint8)).resize((mw, mh), Image.NEAREST)
        tiles.append(img)
    hot_cell = np.unravel_index(np.nanargmax(np.nan_to_num(m["ign_pmap"][idx[4]])), (39, 21))
    hx, hy = (hot_cell[0] + 0.5) * cell, (21 - hot_cell[1] - 0.5) * cell
    caps = [(0.0, "同じ培養を何日も記録して、点火した場所を地図に"),
            (3.6, "同じホットスポットが、日をまたいで何度も現れます"),
            (7.2, "ちがう日に別の場所になることもありますが、\nまた戻ってきます")]

    def fn(i, t):
        im, d = blank()
        header(d, "ROUND 1", "点火しやすい場所は、毎日同じ？", BLUE,
               "同じ培養（No. 23312）の点火地図。明るいほど、そこから点火したバーストが多い")
        for k, tile in enumerate(tiles):
            a = smoothstep((t - 0.35 * k) / 0.3)
            if a <= 0:
                continue
            c, r = k % 4, k // 4
            x0, y0 = 62 + c * 296, 132 + r * 214
            if a < 1:
                tile_a = Image.blend(Image.new("RGB", tile.size, BG), tile, a)
            else:
                tile_a = tile
            im.paste(tile_a, (x0, y0 + 30))
            d.rectangle([x0 - 1, y0 + 29, x0 + mw, y0 + 30 + mh], outline=(70, 82, 104), width=1)
            text(d, (x0, y0 + 14), labels[k], 22, True, INK, anchor="lm")
            if t > 3.6:
                ring(d, x0 + hx, y0 + 30 + hy, 20, YELLOW, 3)
        if t > 3.6:
            text(d, (W // 2, 575), "地図の似かた（次の記録日との相関）：  同じ培養 r = 0.43 ／ ちがう培養どうし r = −0.06",
                 22, True, INK, anchor="mm")
        caption(d, cap_at(t, caps))
        return im
    return Seg("maps", 10.5, fn, check=6.0)


# ======================================================= 3c predictor + m gauge
def seg_predict():
    caps = [(0.0, "点火場所を当てたのは「よく発火する場所」ではなく…"),
            (3.0, "バーストの合間に「いっしょに発火する」\nまとまりの最強パターン（先頭モード）でした"),
            (6.6, "そしてネットワークは、暴走のほんの一歩手前にいます")]

    def fn(i, t):
        im, d = blank()
        header(d, "ROUND 1", "点火場所を決めるのは何？", BLUE,
               "7 つの培養・114 回の記録・30,349 回のバーストを解析")
        # --- left: predictor bars
        d.rounded_rectangle([40, 110, 640, 560], 14, fill=PANEL)
        text(d, (70, 145), "ホットスポットをどれだけ当てられるか", 24, True, INK, anchor="lm")
        text(d, (70, 178), "（点火の多さとの順位相関 ρ、64 記録の中央値）", 18, False, INK2, anchor="lm")
        base_y, top_y = 470, 262
        scale = (base_y - top_y) / 0.3
        for k, (lab1, lab2, v, col, t0) in enumerate([("よく発火する", "場所", 0.0, GREY, 0.3),
                                                      ("いっしょに発火する", "最強パターン", 0.25, BLUE, 3.0)]):
            x0 = 130 + k * 260
            g = smoothstep((t - t0) / 0.8)
            hh = max(3, v * scale * g)
            d.rounded_rectangle([x0, base_y - hh, x0 + 150, base_y], 6, fill=col)
            if g > 0:
                text(d, (x0 + 75, base_y - hh - 22), f"ρ = {v:.2f}" if k else "ρ ≈ 0", 26, True,
                     col if k else INK2, anchor="mm")
            text(d, (x0 + 75, base_y + 26), lab1, 20, True, INK, anchor="mm")
            text(d, (x0 + 75, base_y + 52), lab2, 20, True, INK, anchor="mm")
        d.line([100, base_y, 600, base_y], fill=GREY, width=2)
        if t > 3.8:
            text(d, (465, 212), "勝ち！", 28, True, YELLOW, anchor="mm")
        # --- right: branching ratio gauge
        if t > 6.3:
            a = smoothstep((t - 6.3) / 0.5)
            d.rounded_rectangle([670, 110, 1240, 560], 14, fill=PANEL)
            text(d, (700, 145), "暴走メーター（分岐比 m）", 24, True, INK, anchor="lm")
            text(d, (700, 182), "1 個の発火が、次に平均何個の発火を呼ぶか", 18, False, INK2, anchor="lm")
            gx0, gx1, gy = 720, 1190, 330
            lo, hi = 0.90, 1.10
            for k in range(60):
                v = lo + (hi - lo) * k / 60
                c = BLUE if v < 1.0 else RED
                x = gx0 + (gx1 - gx0) * k / 60
                d.rectangle([x, gy - 18, x + (gx1 - gx0) / 60 + 1, gy + 18], fill=c)
            for v in (0.90, 0.95, 1.00, 1.05, 1.10):
                x = gx0 + (gx1 - gx0) * (v - lo) / (hi - lo)
                text(d, (x, gy + 42), f"{v:.2f}", 18, False, INK2, anchor="mm")
            text(d, (gx0 + 8, gy - 44), "静まる", 20, True, BLUE, anchor="lm")
            text(d, (gx1 - 8, gy - 44), "暴走", 20, True, RED, anchor="rm")
            mv = lo + (0.99 - lo) * a
            x = gx0 + (gx1 - gx0) * (mv - lo) / (hi - lo)
            d.polygon([(x, gy - 26), (x - 14, gy - 50), (x + 14, gy - 50)], fill=YELLOW)
            d.line([x, gy - 26, x, gy + 26], fill=YELLOW, width=4)
            text(d, (955, 440), "m ≈ 0.99", 48, True, YELLOW, anchor="mm")
            text(d, (955, 500), "1 を超えると、発火がねずみ算式に増える", 20, False, INK2, anchor="mm")
        caption(d, cap_at(t, caps))
        return im
    return Seg("predict", 10.0, fn, check=8.5)


# ========================================================== 4a fly model setup
def seg_fly_setup():
    f = fly()
    cv = brain_canvas(1000, 470)
    X0, Y0 = (W - 1000) // 2, 108
    T0, SPEED = 4400.0, 100.0          # quiet stretch between bursts; 1 video s = 100 ms
    caps = [(0.0, "ハエの全脳の配線図（ニューロン 138,639 個）で\n脳をまるごとシミュレーション"),
            (3.2, "培養と同じ条件：弱いノイズ＋使うと疲れるシナプス\n入力はそれだけ")]

    def fn(i, t):
        im, d = blank()
        header(d, "ROUND 2", "ハエの脳：同じ条件を与えると？", ORANGE,
               "FlyWire コネクトームのスパイキングモデル。1 点 = 1 個のニューロン（正面から見た図）")
        T = T0 + t * SPEED
        im.paste(Image.fromarray(cv.frame(f.intensity(T, tau=10.0))), (X0, Y0))
        caption(d, cap_at(t, caps))
        return im
    return Seg("fly_setup", 6.4, fn, check=4.5)


# ====================================================== 4b gain comparison
def seg_gain():
    rows = []
    for g, lab in (("1.00", "ふつう（1 倍）"), ("2.00", "2 倍"), ("2.50", "2.5 倍")):
        d0 = np.load(FLY_GAIN_RUNS[g], allow_pickle=True)
        p = d0["pop"].astype(float)
        p = p[: len(p) // 5 * 5].reshape(-1, 5).mean(1)       # 5-ms means
        rows.append((lab, p, np.asarray(d0["on"]), len(d0["on"]), len(d0["pop"]) / 1000.0))
    ymax = max(r[1][60:].max() for r in rows)            # ignore the start-up transient
    caps = [(0.0, "ふつうの強さ（1 倍）では、ハエの脳は\nバーストを起こしません"),
            (4.4, "シナプスを 2 倍に強めると、バーストが起こり始めます\n（強めるほど、たくさん起こります）")]

    def fn(i, t):
        im, d = blank()
        header(d, "ROUND 2", "シナプスの強さを変えてみる", ORANGE,
               "脳全体の発火数（5 ms ごと）。シミュレーション 15 秒間・▼ = バースト")
        prog = min(1.0, t / 5.5)
        for k, (lab, p, on, nb, dur) in enumerate(rows):
            y0 = 130 + k * 152
            d.rounded_rectangle([40, y0, 1240, y0 + 136], 12, fill=PANEL)
            col = [(170, 130, 100), ORANGE, (255, 150, 70)][k]
            text(d, (64, y0 + 40), lab, 26, True, col, anchor="lm")
            xs = np.linspace(250, 1060, len(p))
            ys = y0 + 122 - 100 * np.minimum(p / ymax, 1.04)
            kk = max(2, int(len(p) * prog))
            polyline(d, xs[:kk], ys[:kk], col, 2)
            tcur = prog * dur
            shown = [o for o in on if o / 1000.0 <= tcur]
            for o in shown:
                x = 250 + 810 * (o / 1000.0) / dur
                d.polygon([(x, y0 + 16), (x - 6, y0 + 6), (x + 6, y0 + 6)], fill=YELLOW)
            text(d, (1220, y0 + 50), f"{len(shown)} 回", 34, True, YELLOW if shown else INK2, anchor="rm")
            text(d, (1220, y0 + 92), "バースト", 18, False, INK2, anchor="rm")
        caption(d, cap_at(t, caps))
        return im
    return Seg("gain", 9.0, fn, check=8.0)


# ================================================= 4c slow motion of a burst
def seg_slowmo(burst_ms, rel0, rel1, freeze_rel, caps, name, freeze_sec=3.5, ms_per_s=30.0, badge_note="",
               note=None, tail=0.8):
    f = fly()
    cv = brain_canvas(860, 400)
    X0, Y0 = 30, 104
    rings = lln_rings(cv)
    play1 = (freeze_rel - rel0) / ms_per_s
    play2 = (rel1 - freeze_rel) / ms_per_s
    sec = play1 + freeze_sec + play2 + tail
    # mini traces
    pop = f.pop[int(burst_ms + rel0): int(burst_ms + rel1)].astype(float)
    pop = np.convolve(pop, np.ones(3) / 3, mode="same")
    lr = f.lln_rate(burst_ms + rel0, burst_ms + rel1, 1.0).astype(float)
    lr = np.convolve(lr, np.ones(5) / 5, mode="same")
    TX0, TX1 = 925, 1245

    def rel_at(t):
        if t < play1:
            return rel0 + t * ms_per_s, False
        if t < play1 + freeze_sec:
            return freeze_rel, True
        return min(rel1, freeze_rel + (t - play1 - freeze_sec) * ms_per_s), False

    def fn(i, t):
        im, d = blank()
        rel, frozen = rel_at(t)
        header(d, "ROUND 2", "スローモーション：バーストはどこから？", ORANGE,
               f"結合を 2.5 倍にしたモデル。約 {1000 / ms_per_s:.0f} 倍スロー{badge_note}")
        T = burst_ms + rel
        im.paste(Image.fromarray(cv.frame(f.intensity(T, tau=7.0))), (X0, Y0))
        show_label = frozen or rel > freeze_rel
        for (cx, cy, r) in rings:
            ring(d, X0 + cx, Y0 + cy, r, YELLOW if show_label else (150, 130, 60), 3 if show_label else 1)
        if show_label:
            lx = X0 + 430
            ly = Y0 + 418
            for (cx, cy, r) in rings:
                d.line([X0 + cx, Y0 + cy + r, lx, ly - 18], fill=YELLOW, width=2)
            text(d, (lx, ly), "触角葉（においの中枢）のアセチルコリン性局所ニューロン", 22, True, YELLOW, anchor="mm")
            text(d, (lx, ly + 31), "（lLN1_bc：片側 約 15 個、左右で 30 個）", 21, True, YELLOW, anchor="mm")
        if frozen:
            text(d, (X0 + 20, Y0 + 14), "一時停止", 24, True, INK, anchor="la")
        # right column: time + traces
        lab = f"{rel:+.0f} ms".replace("-", "−")
        text(d, (TX0, 130), "バースト開始まで" if rel < 0 else "バースト開始から", 20, False, INK2, anchor="lm")
        text(d, (TX0, 172), lab, 46, True, RED if rel >= 0 else INK, anchor="lm")
        for k, (arr, col, title_, y0) in enumerate([(lr, YELLOW, "lLN1_bc（30 個）の発火", 250),
                                                    (pop, INK, "脳全体の発火", 410)]):
            text(d, (TX0, y0 - 18), title_, 19, True, col, anchor="lm")
            d.rectangle([TX0, y0, TX1, y0 + 110], fill=PANEL)
            xs = np.linspace(TX0 + 4, TX1 - 4, len(arr))
            ys = y0 + 104 - 96 * arr / max(arr.max(), 1e-9)
            kk = int(np.clip((rel - rel0) / (rel1 - rel0), 0, 1) * (len(arr) - 1)) + 1
            polyline(d, xs[:kk], ys[:kk], col, 2)
            x_on = TX0 + 4 + (TX1 - TX0 - 8) * (0 - rel0) / (rel1 - rel0)
            for yy in range(y0, y0 + 110, 8):
                d.line([x_on, yy, x_on, yy + 4], fill=RED, width=1)
        text(d, (TX0 + 4 + (TX1 - TX0 - 8) * (0 - rel0) / (rel1 - rel0), 536), "開始", 16, True, RED, anchor="mm")
        caption(d, cap_at(t, caps(play1, freeze_sec)), note=note(play1, t) if note else None)
        return im
    return Seg(name, sec, fn, check=None), (play1 + 1.0) / sec


def seg_slow1():
    caps = lambda p1, fz: [(0.0, "バースト直前の 0.1 秒間をゆっくり再生します"),
                           (p1 - 1.2, "最初に光るのは、触角葉（においの中枢）"),
                           (p1 + 0.3, "左右の触角葉にいる、ごく少数の細胞が\n先に発火していました"),
                           (p1 + fz + 0.2, "そこから脳全体へ燃え広がります")]
    seg, chk = seg_slowmo(5587, -95, 40, -18, caps, "slowmo1")
    seg.check = chk
    return seg


def seg_slow2():
    caps = lambda p1, fz: [(0.0, "別のバーストでも…"),
                           (p1 - 0.6, "やはり触角葉の同じ細胞たちから点火します"),
                           (p1 + fz + 0.2, "")]
    note_txt = ("※伝達物質の仮定の 1 つ（最小補正）での結果。文献と矛盾しない他の 3 つの仮定では、\n"
                "この細胞は常に発火し続けて脳を駆動し、バーストは視覚系から始まります")
    note = lambda p1, t: note_txt if t >= p1 - 0.6 else None
    seg, chk = seg_slowmo(7012, -80, 30, -40, caps, "slowmo2", freeze_sec=2.0, note=note, tail=3.0)
    seg.check = chk
    return seg


# =================================================== 4d lead + leading mode
def seg_lead():
    f = fly()
    cv = brain_canvas(520, 260)
    inten = np.zeros(f.N, np.float32)
    inten[f.lln] = 3.0
    thumb = Image.fromarray(cv.frame(inten))
    rings = lln_rings(cv)
    caps = [(0.0, "29 回のバーストで、この細胞たちは\n脳全体より 30〜45 ms 早く発火していました"),
            (4.2, "しかも、配線図だけから計算した「最強ループ」\n（先頭モード）も、同じこの細胞たちでした")]

    def fn(i, t):
        im, d = blank()
        header(d, "ROUND 2", "点火役の正体", ORANGE, "lLN1_bc：触角葉のアセチルコリン性局所ニューロン（左 14 個・右 16 個）")
        # left: lead bars
        d.rounded_rectangle([40, 110, 640, 560], 14, fill=PANEL)
        text(d, (70, 145), "① 動きから：一番乗り", 26, True, YELLOW, anchor="lm")
        text(d, (70, 180), "バースト開始より何 ms 早く発火が高まるか（中央値）", 18, False, INK2, anchor="lm")
        zx = 560
        for k, (lab, v) in enumerate([("lLN1_bc（左）", 44.0), ("lLN1_bc（右）", 29.5)]):
            y = 260 + k * 110
            g = smoothstep((t - 0.4 - 0.3 * k) / 0.8)
            L = 7.0 * v * g
            d.rounded_rectangle([zx - L, y, zx, y + 50], 6, fill=YELLOW)
            text(d, (zx - L - 10, y + 25), f"{v * g:.0f} ms 早い", 24, True, YELLOW, anchor="rm") if g > 0.05 else None
            text(d, (zx - 6, y + 72), lab, 20, True, INK, anchor="rm")
        d.line([zx, 230, zx, 500], fill=RED, width=3)
        text(d, (zx, 520), "脳全体のバースト開始", 18, True, RED, anchor="mm")
        # right: wiring
        if t > 4.0:
            d.rounded_rectangle([670, 110, 1240, 560], 14, fill=PANEL)
            text(d, (700, 145), "② 配線から：最強ループ", 26, True, YELLOW, anchor="lm")
            text(d, (700, 180), "興奮性の配線図の先頭モードが集中する場所", 18, False, INK2, anchor="lm")
            im.paste(thumb, (695, 215))
            for (cx, cy, r) in rings:
                ring(d, 695 + cx, 215 + cy, r, YELLOW, 3)
            text(d, (955, 510), "脳の 0.02%（30 / 138,639 個）", 24, True, INK, anchor="mm")
        caption(d, cap_at(t, caps))
        return im
    return Seg("lead", 8.5, fn, check=7.5)


# ============================================================ 5 silencing
def seg_silence():
    sc = pd.read_csv(SILENCE_CONFIRM)
    sc["grp"] = sc.silenced.str.replace(r"_#\d+", "", regex=True)
    mean = sc.groupby(["gain", "grp"]).bursts_per_min.mean()
    groups = [("そのまま", "intact", ORANGE), ("lLN1_bc 30 個を\n黙らせる", "lLN1_bc", YELLOW),
              ("ランダムな 30 個を\n黙らせる", "random_30", GREY)]
    traces = None
    if SILENCE_TRACES.exists():
        z = np.load(SILENCE_TRACES)
        keys = ["fixAL_intact", "fixAL_lLN1_bc_silenced", "fixAL_random30_silenced"]
        if all(k in z.files for k in keys):
            traces = [z[k].astype(float)[300:] for k in keys]        # drop start-up, as in the simulation log
    if traces is not None:
        # burst rule of src/fly/bursts.py: 20-ms smoothed rate > 1.5 x baseline + 30 spikes/ms for >= 10 ms
        thr = 1.5 * 138639 * 0.3e-3 + 30
        tr_on, tr_disp = [], []
        for p in traces:
            x = np.convolve(p, np.ones(20) / 20, mode="same")
            dd = np.diff(np.r_[0, (x > thr).astype(np.int8), 0])
            on, off = np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)
            tr_on.append(on[(off - on) >= 10])
            tr_disp.append(p[: len(p) // 5 * 5].reshape(-1, 5).mean(1))
        tr_ymax = np.percentile(np.concatenate(tr_disp), 99.9)
    t_tr = 8.0 if traces is not None else 0.0
    t_conf = t_tr + 10.0
    sec = t_conf + 11.0
    conf_lines = ["確認実験（4 通りの仮定 × 2 つの強さ × 3 回）：",
                  "触角葉のアセチルコリン性局所ニューロン 111 個を黙らせると、",
                  "バーストは 0.25〜0.47 倍に。シナプス数をそろえた",
                  "別の細胞（平均 0.60 倍）より強く効く。ただし、同じループの",
                  "投射ニューロンなど『最も強く駆動する』細胞を黙らせると、",
                  "さらに減る（0.03〜0.31 倍）。",
                  "→ 鍵は触角葉の興奮性ループ全体"]
    caps = []
    if traces is not None:
        caps += [(0.0, "点火役の 30 個だけを黙らせて（発火できなくして）みます"),
                 (4.0, f"点火役を黙らせると、バーストは {len(tr_on[0])} 回 → {len(tr_on[1])} 回に\n（ランダムな 30 個では {len(tr_on[2])} 回。この 1 回の例では少し減少）")]
    caps += [(t_tr, "くり返し確かめると：30 個（脳の 0.02%）を黙らせると\nバーストは約 3 分の 1 に（結合 2 倍のとき）"),
             (t_tr + 5.0, "ランダムな 30 個を黙らせても、ほとんど減りません"),
             (t_conf, "")]

    def draw_traces(d, t):
        lab = ["そのまま", "lLN1_bc 30 個を\n黙らせる", "ランダムな 30 個\nを黙らせる"]
        cols = [ORANGE, YELLOW, (150, 160, 178)]
        prog = min(1.0, t / 5.0)
        for k, (p, on) in enumerate(zip(tr_disp, tr_on)):
            y0 = 130 + k * 152
            d.rounded_rectangle([40, y0, 1240, y0 + 136], 12, fill=PANEL)
            for j, ln in enumerate(lab[k].split("\n")):
                text(d, (64, y0 + 50 + 34 * j - 17 * lab[k].count("\n")), ln, 24, True, cols[k], anchor="lm")
            xs = np.linspace(300, 1060, len(p))
            ys = y0 + 122 - 100 * np.minimum(p / tr_ymax, 1.04)
            kk = max(2, int(len(p) * prog))
            polyline(d, xs[:kk], ys[:kk], cols[k], 2)
            shown = [o for o in on if o / 5 <= kk]
            for o in shown:
                x = 300 + 760 * (o / 5) / len(p)
                d.polygon([(x, y0 + 16), (x - 6, y0 + 6), (x + 6, y0 + 6)], fill=YELLOW)
            text(d, (1220, y0 + 50), f"{len(shown)} 回", 34, True, cols[k], anchor="rm")
            text(d, (1220, y0 + 92), "バースト", 18, False, INK2, anchor="rm")

    def fn(i, t):
        im, d = blank()
        if t >= t_conf:
            header(d, "ROUND 3", "点火役を黙らせてみる", YELLOW,
                   "伝達物質の 4 通りの仮定 × 結合の強さ 2 通り × 3 回")
            d.rounded_rectangle([40, 110, 1240, 590], 14, fill=PANEL)
            a = smoothstep((t - t_conf) / 0.5)
            for j, ln in enumerate(conf_lines):
                col = YELLOW if j in (0, len(conf_lines) - 1) else INK
                c = tuple(int(BG[q] + (col[q] - BG[q]) * a) for q in range(3))
                text(d, (W // 2, 160 + 63 * j), ln, 31, True, c, anchor="mm")
        elif traces is not None and t < t_tr:
            header(d, "ROUND 3", "点火役を黙らせてみる", YELLOW, "結合 2.5 倍のモデル。脳全体の発火数（5 ms ごと、約 15 秒間・1 回の例）▼ = バースト")
            draw_traces(d, t)
        else:
            tt = t - t_tr
            header(d, "ROUND 3", "点火役を黙らせてみる", YELLOW,
                   "自発バーストの回数（そのまま = 100%）。3 回の平均、ランダムは 5 組 × 3 回")
            d.rounded_rectangle([40, 110, 1240, 590], 14, fill=PANEL)
            base_y, top_y = 470, 190
            for gi, (gain, glab) in enumerate([(2.0, "結合 2 倍"), (2.5, "結合 2.5 倍")]):
                gx = 120 + gi * 580
                text(d, (gx + 230, 150), glab, 26, True, INK, anchor="mm")
                ref = mean[(gain, "intact")]
                for k, (lab, key, col) in enumerate(groups):
                    v = mean[(gain, key)]
                    frac = v / ref
                    g = smoothstep((tt - 0.3 - 0.5 * k) / 0.7)
                    hh = max(2, (base_y - top_y) * frac * g)
                    x0 = gx + k * 160
                    d.rounded_rectangle([x0, base_y - hh, x0 + 120, base_y], 6, fill=col)
                    if g > 0.05:
                        text(d, (x0 + 60, base_y - hh - 20), f"{100 * frac * g:.0f}%", 26, True,
                             col if k != 2 else INK2, anchor="mm")
                    for j, ln in enumerate(lab.split("\n")):
                        text(d, (x0 + 60, base_y + 24 + 24 * j), ln, 17, True, INK, anchor="mm")
                    text(d, (x0 + 60, base_y + 24 + 24 * (lab.count("\n") + 1)), f"{v:.0f} 回/分", 16, False,
                         INK2, anchor="mm")
                d.line([gx - 10, base_y, gx + 470, base_y], fill=GREY, width=2)
        caption(d, cap_at(t, caps))
        return im
    return Seg("silence", sec, fn, check=t_tr + 8.5, extra=((6.5,) if traces is not None else ()) + (t_conf + 6.0,))


# ============================================================ 6 real flies
def seg_real():
    reg = pd.read_csv(TURNER_REGIONS).sort_values("ac1", ascending=False).reset_index(drop=True)
    al = pd.read_csv(TURNER_AL)
    n_fl = len(al)
    n_hi = int((al.ac1_AL > al.ac1_oth).sum())
    caps = [(0.0, "本物のハエ 20 匹の、何もしていないときの脳活動\n（カルシウムイメージング）"),
            (3.8, "74 の脳領域のうち、触角葉のゆらぎがいちばんゆっくり"),
            (8.0, f"{n_fl} 匹中 {n_hi} 匹で、触角葉はほかの領域より\nゆっくりゆらいでいました"),
            (11.6, "強い自己増幅ループがあると、活動はゆっくりゆらぎます。\nモデルの予測と合う手がかりです（まだ証明ではありません）")]

    def fn(i, t):
        im, d = blank()
        header(d, "ROUND 4", "本物のハエで答え合わせ", ORANGE,
               "公開データ（Turner ら 2021）。ゆらぎのゆっくりさ = となりの時点との相関（自己相関）")
        # left: 74 regions sorted
        d.rounded_rectangle([40, 110, 800, 570], 14, fill=PANEL)
        text(d, (66, 140), "74 の脳領域（ゆっくりな順）", 22, True, INK, anchor="lm")
        x0, x1, base_y, top_y = 80, 770, 520, 200
        bw = (x1 - x0) / len(reg)
        g = smoothstep((t - 0.6) / 1.6)
        for k, row in reg.iterrows():
            isal = row.region.startswith("AL_")
            col = ORANGE if isal and t > 3.8 else (GREY if not isal else (150, 120, 100))
            hh = (base_y - top_y) * max(row.ac1, 0) * g
            d.rectangle([x0 + k * bw + 1, base_y - hh, x0 + (k + 1) * bw - 1, base_y], fill=col)
            if isal and t > 3.8:
                side = "右" if row.region.endswith("_R") else "左"
                ty = 178 if side == "右" else 212
                text(d, (x0 + k * bw + 2, ty), f"触角葉（{side}）{k + 1} 位", 20, True, ORANGE, anchor="lm")
                d.line([x0 + (k + 0.5) * bw, ty + 12, x0 + (k + 0.5) * bw, base_y - hh - 4], fill=ORANGE, width=2)
        d.line([x0, base_y, x1, base_y], fill=GREY, width=2)
        text(d, (x0, base_y + 22), "ゆっくり", 18, False, INK2, anchor="lm")
        text(d, (x1, base_y + 22), "せわしない", 18, False, INK2, anchor="rm")
        # right: per-fly pairs
        if t > 7.6:
            a = smoothstep((t - 7.6) / 1.0)
            d.rounded_rectangle([830, 110, 1240, 570], 14, fill=PANEL)
            text(d, (856, 140), "1 匹ずつ比べると", 22, True, INK, anchor="lm")
            xa, xb = 930, 1140
            yb, yt = 500, 200
            lo, hi = 0.2, 1.0
            fy = lambda v: yb - (yb - yt) * (np.clip(v, lo, hi) - lo) / (hi - lo)
            for _, r in al.iterrows():
                up = r.ac1_AL > r.ac1_oth
                col = ORANGE if up else INK2
                ya = fy(r.ac1_AL)
                yo = fy(r.ac1_oth)
                yo_a = ya + (yo - ya) * a
                d.line([xa, ya, xa + (xb - xa) * a, yo_a], fill=col, width=2)
                d.ellipse([xa - 5, ya - 5, xa + 5, ya + 5], fill=col)
                d.ellipse([xa + (xb - xa) * a - 5, yo_a - 5, xa + (xb - xa) * a + 5, yo_a + 5], fill=col)
            text(d, (xa, yb + 26), "触角葉", 20, True, ORANGE, anchor="mm")
            text(d, (xb, yb + 26), "ほかの領域", 20, True, INK2, anchor="mm")
            text(d, (1035, 175), f"{n_hi} / {n_fl} 匹", 30, True, ORANGE, anchor="mm")
        caption(d, cap_at(t, caps))
        return im
    return Seg("real", 15.6, fn, check=10.5, extra=(14.5,))


# ================================================================ 7 summary
def seg_verdict():
    def fn(i, t):
        im, d = blank()
        p = smoothstep(t / 0.5)
        text(d, (W // 2, 230), "判定", 40, True, INK2, anchor="mm")
        text(d, (W // 2, 320), "引き分け！", int(80 + 20 * (1 - p)), True, YELLOW, anchor="mm")
        if t > 0.8:
            text(d, (W // 2, 440), "培養神経も、ハエの脳も、同じルールで点火していました", 32, True, INK, anchor="mm")
        return im
    return Seg("verdict", 4.0, fn, check=2.5)


def seg_summary():
    cul = culture()
    cv = ChipCanvas(cul, 470)
    k = int(np.flatnonzero((cul.bs > 32.5) & (cul.bs < 32.7))[0])
    chip = cv.frame(cul.intensity(32.592), gain=1.2)          # the frozen ignition moment of round 1
    cx, cy = cv.xy(cul.ix[k], cul.iy[k])
    f = fly()
    bc = brain_canvas(470, 250)
    brain = Image.fromarray(bc.frame(f.intensity(5587 - 18, tau=7.0)))
    rings = lln_rings(bc)
    caps = [(0.0, "まとめ"),
            (0.8, "バーストは「いちばんよく発火する場所」ではなく、\nネットワークの最強ループから点火します"),
            (6.0, "ハエでは、その正体は触角葉の興奮性ループ\n（アセチルコリン性局所ニューロン＋投射ニューロン）。\n本物のハエで確かめられる『予測』です")]

    def fn(i, t):
        im, d = blank()
        text(d, (W // 2, 50), "共通のルール", 34, True, YELLOW, anchor="mm")
        for x0, img, col, lab in [(90, chip, BLUE, "培養神経：決まったホットスポット"),
                                  (720, brain, ORANGE, "ハエの脳：触角葉の興奮性ループ（モデル）")]:
            d.rounded_rectangle([x0 - 16, 90, x0 + 486, 420], 14, fill=PANEL, outline=col, width=3)
            im.paste(img, (x0, 150))
            ls = lab.split("\n")
            for j, ln in enumerate(ls):
                text(d, (x0 + 235, 122 + 26 * (j - (len(ls) - 1) / 2)), ln, 22, True, col, anchor="mm")
        ring(d, 90 + cx, 150 + cy, 30, YELLOW, 3)
        for (rx, ry, r) in rings:
            ring(d, 720 + rx, 150 + ry, r, YELLOW, 3)
        text(d, (W // 2, 255), "＝", 50, True, INK, anchor="mm")
        if t > 6.0:
            d.rounded_rectangle([90, 445, 1190, 525], 12, fill=(44, 36, 16), outline=YELLOW, width=2)
            text(d, (W // 2, 470), "注意：ハエの結果はシミュレーションによる予測です。", 22, True, YELLOW, anchor="mm")
            text(d, (W // 2, 502), "本物のハエでこの細胞たちを黙らせ、自発バーストが減るかを確かめるのが次の一歩", 20, False, INK,
                 anchor="mm")
        else:
            text(d, (W // 2, 480), "データ：DANDI 公開 MaxOne 記録（培養 7 個）・FlyWire 配線図・Turner ら 2021（ハエ 20 匹）",
                 18, False, INK2, anchor="mm")
        caption(d, cap_at(t, caps))
        return im
    return Seg("summary", 12.0, fn, fade=12, check=9.0)


# ===================================================================== main
def segments():
    return [seg_title(), seg_intro(),
            seg_round(1, "培養神経", BLUE, "Q. バーストは、チップのどこから点火する？"),
            seg_chip(), seg_maps(), seg_predict(),
            seg_round(2, "ハエの脳", ORANGE, "Q. 同じ条件のハエの脳では、どこから点火する？"),
            seg_fly_setup(), seg_gain(), seg_slow1(), seg_slow2(), seg_lead(),
            seg_round(3, "黙らせてみる", YELLOW, "Q. 点火役を消したら、バーストは減る？", 2.5),
            seg_silence(),
            seg_round(4, "本物のハエ", ORANGE, "Q. 本物のハエの触角葉にも、その気配はある？", 2.5),
            seg_real(), seg_verdict(), seg_summary()]


def main():
    BUILD.mkdir(exist_ok=True)
    segs = segments()
    if "--check" in sys.argv:
        only = [a for a in sys.argv[1:] if not a.startswith("--")]
        for k, s in enumerate(segs):
            if only and s.name not in only:
                continue
            fr = s.frame(min(s.n - 1, int(s.n * s.check)) if s.check < 1 else int(s.check * FPS))
            Image.fromarray(fr).save(BUILD / f"check_{k:02d}_{s.name}.png")
            for j, te in enumerate(s.extra):
                Image.fromarray(s.frame(min(s.n - 1, int(te * FPS)))).save(BUILD / f"check_{k:02d}_{s.name}_{j + 1}.png")
            print(k, s.name, f"{s.n / FPS:.1f} s")
        print("total", sum(s.n for s in segs) / FPS, "s")
        return
    out = HERE / "fly_culture_ignition_ja.mp4"
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                           pixelformat="yuv420p", ffmpeg_params=["-preset", "medium"])
    n = 0
    for s in segs:
        for i in range(s.n):
            w.append_data(s.frame(i))
            n += 1
        print(s.name, f"{s.n / FPS:.1f} s", flush=True)
    w.close()
    print("wrote", out, n / FPS, "s")


if __name__ == "__main__":
    main()
