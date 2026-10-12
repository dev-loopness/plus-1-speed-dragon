#!/usr/bin/env python3
"""Draws the game's icon set (plan section 2.6): flat two-tone pictures with a soft gloss and a thick
ink outline, 256 x 256 PNGs with transparency. Drawn at 1024 and scaled down for clean edges.
Usage: tools/icons.py <out_dir>"""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

S = 1024
OUT = 256
W = 46  # outline width at 1024

INK = (20, 26, 46)
GOLD, GOLD_D, GOLD_L = (255, 201, 74), (232, 146, 20), (255, 232, 150)
GREEN, GREEN_D = (75, 211, 123), (36, 150, 80)
BLUE, BLUE_D = (80, 160, 255), (40, 100, 210)
PURPLE, PURPLE_D = (175, 95, 255), (120, 60, 200)
RED, RED_D = (255, 90, 95), (200, 50, 60)
WHITE = (244, 241, 232)
CREAM = (250, 238, 200)
GREY, GREY_D = (170, 178, 204), (110, 118, 150)
TEAL, TEAL_D, TEAL_L = (56, 218, 200), (20, 160, 150), (170, 250, 240)
BROWN, BROWN_D = (176, 116, 64), (120, 76, 40)
ORANGE, ORANGE_D = (255, 159, 28), (210, 110, 10)
YELLOW, YELLOW_D = (255, 221, 60), (240, 160, 20)
PINK = (255, 150, 170)


def blank():
    return Image.new("L", (S, S), 0)


def arr(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def ellipse(x0, y0, x1, y1):
    im = blank()
    ImageDraw.Draw(im).ellipse((x0, y0, x1, y1), fill=255)
    return arr(im)


def rrect(x0, y0, x1, y1, r):
    im = blank()
    ImageDraw.Draw(im).rounded_rectangle((x0, y0, x1, y1), radius=r, fill=255)
    return arr(im)


def poly(points):
    im = blank()
    ImageDraw.Draw(im).polygon([tuple(p) for p in points], fill=255)
    return arr(im)


def line(points, width):
    im = blank()
    d = ImageDraw.Draw(im)
    d.line([tuple(p) for p in points], fill=255, width=width, joint="curve")
    r = width / 2
    for x, y in (points[0], points[-1]):
        d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return arr(im)


def arc(box, start, end, width):
    im = blank()
    d = ImageDraw.Draw(im)
    d.arc(box, start, end, fill=255, width=width)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    rx = (box[2] - box[0]) / 2 - width / 2
    ry = (box[3] - box[1]) / 2 - width / 2
    r = width / 2
    for a in (start, end):
        x, y = cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a))
        d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return arr(im)


def star(cx, cy, r_out, r_in, points=5, rot=-90):
    pts = []
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        a = math.radians(rot + i * 180 / points)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return poly(pts)


def union(*masks):
    out = np.zeros((S, S), np.float32)
    for m in masks:
        out = np.maximum(out, m)
    return out


def minus(a, b):
    return np.clip(a - b, 0, 1)


def inter(a, b):
    return np.minimum(a, b)


def half(mask, frac=0.5, top=True):
    """The part of a mask above (or below) a horizontal line at frac of its own height."""
    rows = np.where(mask.max(axis=1) > 0.5)[0]
    if len(rows) == 0:
        return mask
    y = rows[0] + (rows[-1] - rows[0]) * frac
    cut = np.zeros_like(mask)
    if top:
        cut[: int(y), :] = 1
    else:
        cut[int(y) :, :] = 1
    return inter(mask, cut)


class Icon:
    def __init__(self):
        self.layers = []  # (mask, color, alpha)
        self.solid = []  # masks that get the outline

    def add(self, mask, color, outline=True, alpha=1.0):
        self.layers.append((mask, color, alpha))
        if outline:
            self.solid.append(mask)
        return mask

    def shade(self, mask, color, frac=0.62):
        """A darker band over the lower part of a shape (two-tone shading, no outline of its own)."""
        self.layers.append((half(mask, frac, top=False), color, 1.0))

    def gloss(self, mask, cx, cy, rx, ry, alpha=0.32):
        self.layers.append((inter(mask, ellipse(cx - rx, cy - ry, cx + rx, cy + ry)), (255, 255, 255), alpha))

    def render(self):
        shape = union(*self.solid) > 0.5
        dist = distance_transform_edt(~shape)
        outline = np.clip((W + 1 - dist), 0, 1).astype(np.float32)
        rgb = np.zeros((S, S, 3), np.float32)
        alpha = np.zeros((S, S), np.float32)

        def over(mask, color, a=1.0):
            nonlocal rgb, alpha
            m = mask * a
            rgb = rgb * (1 - m[..., None]) + np.array(color, np.float32)[None, None, :] * m[..., None]
            alpha = alpha + m * (1 - alpha)

        over(outline, INK)
        for mask, color, a in self.layers:
            over(mask, color, a)
        img = Image.fromarray(np.dstack([rgb, alpha * 255]).clip(0, 255).astype(np.uint8), "RGBA")
        return img.resize((OUT, OUT), Image.LANCZOS)


def coin():
    i = Icon()
    body = i.add(ellipse(120, 120, 904, 904), GOLD)
    i.shade(body, GOLD_D, 0.6)
    inner = ellipse(230, 230, 794, 794)
    i.add(inner, GOLD_D, outline=False)
    i.add(ellipse(250, 250, 774, 774), GOLD, outline=False)
    i.add(rrect(440, 330, 584, 694, 60), GOLD_L, outline=False)
    i.gloss(body, 380, 330, 260, 170)
    return i


def training():
    i = Icon()
    bolt = i.add(poly([(600, 70), (230, 560), (470, 560), (380, 954), (800, 400), (550, 400), (680, 70)]), YELLOW)
    i.shade(bolt, YELLOW_D, 0.55)
    i.gloss(bolt, 520, 260, 200, 180, 0.35)
    return i


def slow():
    i = Icon()
    body = i.add(rrect(120, 640, 900, 820, 90), GREEN)
    i.add(line([(780, 650), (850, 470)], 44), GREEN)
    i.add(line([(860, 650), (930, 500)], 44), GREEN)
    i.add(ellipse(820, 430, 890, 500), WHITE)
    i.add(ellipse(900, 460, 960, 520), WHITE)
    shell = i.add(ellipse(180, 250, 690, 760), ORANGE)
    i.shade(shell, ORANGE_D, 0.6)
    i.add(arc((300, 370, 570, 640), 0, 300, 50), ORANGE_D, outline=False)
    i.add(arc((390, 460, 480, 550), 0, 300, 40), ORANGE_D, outline=False)
    i.gloss(shell, 350, 360, 160, 100)
    i.shade(body, GREEN_D, 0.6)
    return i


def rewards():
    i = Icon()
    box = i.add(rrect(170, 470, 854, 930, 50), RED)
    i.shade(box, RED_D, 0.55)
    lid = i.add(rrect(120, 330, 904, 500, 50), RED)
    i.add(rrect(447, 330, 577, 930, 10), GOLD, outline=False)
    i.add(rrect(120, 395, 904, 445, 10), GOLD, outline=False)
    bow_l = i.add(ellipse(270, 140, 520, 350), GOLD)
    bow_r = i.add(ellipse(504, 140, 754, 350), GOLD)
    i.add(ellipse(330, 200, 470, 300), GOLD_D, outline=False)
    i.add(ellipse(554, 200, 694, 300), GOLD_D, outline=False)
    i.add(ellipse(440, 260, 584, 380), GOLD)
    i.gloss(lid, 360, 360, 220, 60)
    return i


def index():
    i = Icon()
    pages = i.add(rrect(230, 160, 880, 900, 50), WHITE)
    cover = i.add(rrect(150, 110, 800, 860, 60), BLUE)
    i.shade(cover, BLUE_D, 0.6)
    i.add(rrect(150, 110, 250, 860, 40), BLUE_D, outline=False)
    i.add(star(500, 450, 190, 85), GOLD)
    i.gloss(cover, 400, 250, 230, 120)
    return i


def settings():
    i = Icon()
    teeth = []
    for k in range(8):
        a = math.radians(k * 45)
        cx, cy = 512 + 330 * math.cos(a), 512 + 330 * math.sin(a)
        dx, dy = math.cos(a), math.sin(a)
        px, py = -dy, dx
        w, h = 85, 110
        pts = [(cx + px * w - dx * h, cy + py * w - dy * h), (cx - px * w - dx * h, cy - py * w - dy * h), (cx - px * w * 0.8 + dx * h, cy - py * w * 0.8 + dy * h), (cx + px * w * 0.8 + dx * h, cy + py * w * 0.8 + dy * h)]
        teeth.append(poly(pts))
    gear = minus(union(ellipse(200, 200, 824, 824), *teeth), ellipse(392, 392, 632, 632))
    i.add(gear, GREY)
    i.shade(gear, GREY_D, 0.58)
    i.gloss(gear, 400, 330, 240, 150)
    return i


def quests():
    i = Icon()
    paper = i.add(rrect(170, 100, 760, 900, 60), WHITE)
    for k, y in enumerate((280, 450, 620)):
        i.add(rrect(270, y - 28, 660 - k * 60, y + 28, 28), GREY_D, outline=False)
    badge = i.add(ellipse(560, 600, 910, 950), GREEN)
    i.shade(badge, GREEN_D, 0.6)
    i.add(line([(645, 780), (715, 850), (835, 705)], 60), WHITE, outline=False)
    return i


def shop():
    i = Icon()
    i.add(arc((330, 110, 694, 520), 180, 360, 64), ORANGE_D)
    bag = i.add(poly([(190, 330), (834, 330), (900, 900), (124, 900)]), ORANGE)
    i.shade(bag, ORANGE_D, 0.62)
    i.add(ellipse(330, 410, 400, 480), INK, outline=False)
    i.add(ellipse(624, 410, 694, 480), INK, outline=False)
    i.add(ellipse(390, 560, 634, 804), GOLD)
    i.add(ellipse(440, 610, 584, 754), GOLD_D, outline=False)
    i.gloss(bag, 380, 420, 200, 90)
    return i


def nest():
    i = Icon()
    egg = i.add(ellipse(330, 110, 694, 600), CREAM)
    i.add(ellipse(410, 250, 490, 330), GREEN, outline=False)
    i.add(ellipse(550, 350, 610, 410), GREEN, outline=False)
    i.gloss(egg, 440, 240, 90, 120, 0.5)
    bowl = i.add(poly([(110, 520), (914, 520), (790, 880), (234, 880)]), BROWN)
    i.add(rrect(110, 480, 914, 600, 60), BROWN)
    i.shade(bowl, BROWN_D, 0.45)
    for y, x0, x1 in ((560, 160, 700), (680, 220, 820), (790, 280, 740)):
        i.add(line([(x0, y), (x1, y - 30)], 26), BROWN_D, outline=False)
    return i


def rebirth():
    i = Icon()
    a = i.add(arc((150, 150, 874, 874), 200, 340, 130), PURPLE)
    b = i.add(arc((150, 150, 874, 874), 20, 160, 130), PURPLE)
    i.add(poly([(780, 300), (950, 420), (740, 520)]), PURPLE)
    i.add(poly([(244, 724), (74, 604), (284, 504)]), PURPLE)
    i.shade(union(a, b), PURPLE_D, 0.5)
    i.add(star(512, 512, 150, 65), GOLD)
    return i


def scales():
    i = Icon()
    gem = i.add(poly([(260, 220), (764, 220), (920, 420), (512, 900), (104, 420)]), TEAL)
    i.add(poly([(260, 220), (764, 220), (920, 420), (104, 420)]), TEAL_L, outline=False)
    i.add(poly([(104, 420), (920, 420), (512, 900)]), TEAL, outline=False)
    i.add(poly([(512, 420), (920, 420), (512, 900)]), TEAL_D, outline=False)
    i.add(poly([(380, 220), (512, 420), (644, 220)]), (220, 255, 250), outline=False)
    return i


def trophy():
    i = Icon()
    i.add(arc((80, 200, 400, 560), 90, 270, 70), GOLD)
    i.add(arc((624, 200, 944, 560), 270, 450, 70), GOLD)
    cup = i.add(poly([(220, 150), (804, 150), (720, 520), (304, 520)]), GOLD)
    i.add(ellipse(304, 380, 720, 640), GOLD)
    i.add(rrect(440, 600, 584, 780, 20), GOLD_D)
    base = i.add(rrect(270, 760, 754, 900, 40), BROWN)
    i.shade(cup, GOLD_D, 0.7)
    i.add(star(512, 360, 120, 52), GOLD_L, outline=False)
    i.gloss(cup, 380, 260, 140, 100)
    return i


def dragons():
    i = Icon()
    i.add(poly([(330, 120), (420, 330), (300, 330)]), CREAM)
    i.add(poly([(520, 90), (600, 320), (480, 320)]), CREAM)
    head = i.add(rrect(170, 260, 720, 760, 160), GREEN)
    snout = i.add(rrect(520, 440, 920, 760, 120), GREEN)
    jaw = i.add(rrect(560, 700, 880, 860, 70), CREAM)
    i.shade(union(head, snout), GREEN_D, 0.72)
    i.add(ellipse(780, 520, 830, 570), INK, outline=False)
    eye = i.add(ellipse(330, 340, 520, 560), WHITE)
    i.add(ellipse(400, 400, 500, 530), INK, outline=False)
    i.add(ellipse(420, 410, 460, 450), (255, 255, 255), outline=False)
    i.gloss(head, 330, 320, 160, 70, 0.25)
    return i


def egg():
    i = Icon()
    e = i.add(ellipse(230, 90, 794, 930), CREAM)
    i.shade(e, (226, 210, 160), 0.65)
    i.add(ellipse(330, 330, 450, 450), GREEN, outline=False)
    i.add(ellipse(560, 520, 700, 660), GREEN, outline=False)
    i.add(ellipse(380, 660, 460, 740), GREEN, outline=False)
    i.gloss(e, 400, 290, 110, 150, 0.5)
    return i


def lock():
    i = Icon()
    i.add(arc((280, 90, 744, 580), 180, 360, 90), GREY)
    i.add(rrect(280, 330, 370, 520, 20), GREY)
    i.add(rrect(654, 330, 744, 520, 20), GREY)
    body = i.add(rrect(170, 440, 854, 920, 80), GOLD)
    i.shade(body, GOLD_D, 0.6)
    i.add(ellipse(452, 560, 572, 680), INK, outline=False)
    i.add(poly([(480, 640), (544, 640), (570, 800), (454, 800)]), INK, outline=False)
    i.gloss(body, 360, 520, 200, 70)
    return i


def close():
    i = Icon()
    i.add(line([(220, 220), (804, 804)], 170), WHITE)
    i.add(line([(804, 220), (220, 804)], 170), WHITE)
    return i


def check():
    i = Icon()
    c = i.add(ellipse(100, 100, 924, 924), GREEN)
    i.shade(c, GREEN_D, 0.6)
    i.add(line([(300, 530), (450, 680), (730, 360)], 120), WHITE, outline=False)
    i.gloss(c, 380, 300, 230, 130)
    return i


def plus():
    i = Icon()
    i.add(rrect(422, 150, 602, 874, 80), WHITE)
    i.add(rrect(150, 422, 874, 602, 80), WHITE)
    return i


def star_icon():
    i = Icon()
    s = i.add(star(512, 540, 430, 190), GOLD)
    i.shade(s, GOLD_D, 0.62)
    i.gloss(s, 450, 380, 170, 120)
    return i


def fuse():
    i = Icon()
    for (cx, cy), col, dark in (((260, 300), RED, RED_D), ((764, 300), BLUE, BLUE_D), ((512, 790), GREEN, GREEN_D)):
        o = i.add(ellipse(cx - 150, cy - 150, cx + 150, cy + 150), col)
        i.shade(o, dark, 0.6)
        i.gloss(o, cx - 40, cy - 60, 70, 45, 0.45)
    i.add(star(512, 470, 200, 80, 4, -90), GOLD)
    return i


def trails():
    i = Icon()
    for (y0, y1, col, w) in ((330, 560, PURPLE, 90), (520, 650, BLUE, 80), (700, 740, TEAL, 70)):
        i.add(line([(110, y0), (560, y1 - 60)], w), col)
    s = i.add(star(690, 480, 260, 115, 5, -80), GOLD)
    i.shade(s, GOLD_D, 0.6)
    return i


def clock():
    i = Icon()
    rim = i.add(ellipse(110, 110, 914, 914), BLUE)
    i.add(ellipse(200, 200, 824, 824), WHITE, outline=False)
    for k in range(12):
        a = math.radians(k * 30)
        r0, r1 = 250, 290 if k % 3 else 230
        i.add(line([(512 + r0 * math.cos(a), 512 + r0 * math.sin(a)), (512 + 290 * math.cos(a), 512 + 290 * math.sin(a))], 24 if k % 3 else 40), GREY_D, outline=False)
    i.add(line([(512, 512), (512, 300)], 56), INK, outline=False)
    i.add(line([(512, 512), (660, 600)], 56), INK, outline=False)
    i.add(ellipse(472, 472, 552, 552), RED, outline=False)
    i.gloss(rim, 380, 300, 230, 120, 0.25)
    return i


ICONS = {
    "Coin": coin,
    "Training": training,
    "Slow": slow,
    "Rewards": rewards,
    "Index": index,
    "Settings": settings,
    "Quests": quests,
    "Shop": shop,
    "Nest": nest,
    "Rebirth": rebirth,
    "Scales": scales,
    "Trophy": trophy,
    "Dragons": dragons,
    "Egg": egg,
    "Lock": lock,
    "Close": close,
    "Check": check,
    "Plus": plus,
    "Star": star_icon,
    "Fuse": fuse,
    "Trails": trails,
    "Clock": clock,
}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "icons"
    os.makedirs(out, exist_ok=True)
    names = sys.argv[2:] or list(ICONS)
    for name in names:
        ICONS[name]().render().save(os.path.join(out, f"{name}.png"))
        print("drew", name)
