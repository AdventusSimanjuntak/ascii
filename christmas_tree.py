# Christmas tree in ASCII
# after @adventus_simanjuntak's christmas tree, 2026
#
#   pip install numpy pygame-ce
#   pip install opencv-python            (needed to write a video)
#   python christmas_tree.py             play
#   python christmas_tree.py --video     render one loop to tree_<time>.mp4
#
# A tree is planted: the trunk rises, tier on tier of needles spreads out to the top, the tinsel
# winds up it, the baubles pop on one by one, parcels settle at its foot, the lights come up in a
# wave and the star at the top catches fire, throwing sparks. Snow drifts down all the while,
# stars twinkle round it and "M E R R Y   C H R I S T M A S" is typed out underneath.
#
# Space pause, R restart, S screenshot, V video, Esc quit

import sys
import time
from pathlib import Path

import numpy as np
import pygame

CREDIT = "after @adventus_simanjuntak's christmas tree, 2026"
MESSAGE = "M E R R Y   C H R I S T M A S"

TAU = np.pi * 2
LOOP = 24.0                                     # length of one run (s)

# timeline (s)
T_TRUNK = 0.35                                  # the trunk rises
T_TREE = (0.5, 3.1)                             # tier on tier of needles spreads out, bottom to top
T_GARLAND = (2.7, 4.5)                          # the tinsel winds up
T_BALLS = (3.3, 5.2)                            # the baubles pop on, one by one
T_GIFTS = (3.0, 4.0)                            # parcels settle at the foot
T_STAR = 5.1                                    # the star catches fire
T_LIGHTS = (5.4, 7.4)                           # the bulbs light up, bottom to top
T_MSG = 6.6                                     # the message is typed out, 10 letters a second
GROWN = 5.4                                     # everything but the lights is there
FLASH = 0.6                                     # how long something new glows white (s)

POOLS = ["1il!;:", "rtfjcvxz=+", "aeoun0{}[]<>", "08B$#&%Q"]   # fill letters, dark -> bright
C = lambda ch: ord(ch) - 32                     # character -> glyph index
FILL = np.array([[C(p[k % len(p)]) for k in range(12)] for p in POOLS])

# what each point is, and its colour
(TRUNK, NEEDLE, NEEDLE_LIT, GARLAND, BALL, CAP, STAR, GIFT, GIFT_RIBBON, GLINT, GROUND,
 SNOW, SNOW_BLUE) = range(13)
COLORS = np.array([[92, 62, 44], [24, 92, 46], [92, 205, 112], [255, 196, 92], [214, 32, 48],
                   [255, 208, 120], [255, 226, 140], [188, 36, 52], [255, 198, 96], [255, 255, 255],
                   [206, 226, 248], [255, 255, 255], [186, 214, 255]], np.float32) / 255
WHITE = np.array([1.0, 1.0, 1.0], np.float32)

# the baubles: deep red, gold, blue, silver, pink, emerald
BALL_SHADES = np.array([[214, 32, 48], [255, 190, 70], [52, 110, 214], [226, 232, 242],
                        [206, 64, 150], [52, 168, 102]], np.float32) / 255

# the tree: a cone of tiered skirts, needles on the surface
BASE_Y, TOP_Y, R_BASE = -1.22, 1.98, 1.00
N_TIERS = 9
CONE_H = TOP_Y - BASE_Y

# camera
FOCAL = 2.9                                     # how much of the frame the tree fills
MIDDLE = 0.485                                  # where the camera's aim sits, down the frame
EYE, TARGET = np.array([0.0, 1.95, 7.6]), np.array([0.0, 0.40, 0.0])

FOLD = 0.06                                     # a layer this far in front of its neighbour gets a stroke


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def normals_of(p, axis_u, axis_v, outward):
    # surface normals of a grid of points, turned to face `outward`
    n = np.cross(np.gradient(p, axis=axis_u), np.gradient(p, axis=axis_v))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return n * np.sign((n * outward).sum(-1, keepdims=True) + 1e-9)


def spin_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def tube(curve, radius, around=6):
    # a round tube along a polyline of points
    d = np.gradient(curve, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    side = np.cross(d, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
    up = np.cross(side, d)
    a = np.linspace(0, TAU, around, endpoint=False)
    n = np.cos(a)[None, :, None] * side[:, None] + np.sin(a)[None, :, None] * up[:, None]
    return curve[:, None] + radius * n, n


def cone_radius(y):
    u = np.clip((TOP_Y - y) / CONE_H, 0.0, 1.0)
    return R_BASE * u ** 0.95


def grow_time(y):
    # when the tree reaches height y
    return T_TREE[0] + (T_TREE[1] - T_TREE[0]) * np.clip((y - BASE_Y) / CONE_H, 0, 1)


# ---------------------------------------------------------------------------------------------
# the tree itself

class Builder:
    # gathers every point that never changes on its own: where, normal, kind, colour, part,
    # when it shows up, whether it pops on whole, and the centre it pops on around

    def __init__(self):
        self.P, self.N, self.T, self.K = [], [], [], []
        self.tone, self.appear, self.pop, self.centre = [], [], [], []

    def add(self, p, n, tag, rgb, part, appear, pop=False, centre=None):
        p = np.asarray(p, np.float32).reshape(-1, 3)
        n = np.asarray(n, np.float32).reshape(-1, 3)
        k = len(p)
        self.P.append(p)
        self.N.append(n)
        self.T.append(np.broadcast_to(np.asarray(tag, np.int64), (k,)))
        self.tone.append(np.broadcast_to(np.asarray(rgb, np.float32), (k, 3)))
        self.K.append(np.broadcast_to(np.asarray(part, np.int64), (k,)))
        self.appear.append(np.broadcast_to(np.asarray(appear, np.float32), (k,)))
        self.pop.append(np.broadcast_to(np.asarray(pop, bool), (k,)))
        cen = np.zeros(3, np.float32) if centre is None else np.asarray(centre, np.float32)
        self.centre.append(np.broadcast_to(cen, (k, 3)).copy())

    def done(self):
        v = {"P": np.concatenate(self.P), "N": np.concatenate(self.N),
             "T": np.concatenate(self.T), "K": np.concatenate(self.K),
             "tone": np.concatenate(self.tone).astype(np.float32),
             "appear": np.concatenate(self.appear).astype(np.float32),
             "pop": np.concatenate(self.pop), "centre": np.concatenate(self.centre).astype(np.float32)}
        v["gain"] = np.where(v["T"] == NEEDLE, 0.72, np.where(v["T"] == NEEDLE_LIT, 1.25, 1.0)).astype(np.float32)
        v["gloss"] = np.select([v["T"] == BALL, v["T"] == CAP, v["T"] == STAR, v["T"] == GARLAND,
                                v["T"] == GIFT_RIBBON, v["T"] == GROUND],
                               [1.0, 1.1, 1.25, 1.1, 0.9, 0.35], 0.30).astype(np.float32)
        return v


def trunk():
    y = np.linspace(BASE_Y - 0.06, BASE_Y + 0.30, 30)
    curve = np.stack([np.zeros_like(y), y, np.zeros_like(y)], 1)
    p, n = tube(curve, 0.085, 22)
    return p, n


def foliage():
    # ten skirts, each wider at its rim than at its top, with a scalloped, drooping edge; the
    # needles are a little rough, and some points are a lighter green for texture
    rng = np.random.default_rng(9)
    P, N, TAG, TONE, APP = [], [], [], [], []
    for i in range(N_TIERS):
        y0 = BASE_Y + CONE_H * i / N_TIERS
        y1 = BASE_Y + CONE_H * min((i + 1.35) / N_TIERS, 1.0)
        r0 = cone_radius(y0) * 1.12
        r1 = cone_radius(y1) * 0.85
        a, s = np.meshgrid(np.linspace(0, TAU, 210, endpoint=False), np.linspace(0, 1, 10))
        droop = (1 - s) ** 3
        ripple = 0.5 + 0.5 * np.cos(13 * a + 1.1 * i)                 # the branch tips
        rr = (r0 + (r1 - r0) * s) * (1 + 0.075 * ripple * droop)
        rr = rr + 0.005 * np.sin(19 * a + 7 * s + 2 * i)              # needles
        yy = y0 + (y1 - y0) * s - 0.105 * droop - 0.014 * ripple * droop
        p = np.stack([rr * np.sin(a), yy, rr * np.cos(a)], -1)
        out = np.stack([np.sin(a), np.full_like(a, 0.5), np.cos(a)], -1)
        n = normals_of(p, 1, 0, out)
        light = rng.random(yy.shape) < 0.04
        P.append(p.reshape(-1, 3))
        N.append(n.reshape(-1, 3))
        rim = (light | (s < 0.16))
        TAG.append(np.where(rim.ravel(), NEEDLE_LIT, NEEDLE))
        fac = (1.0 - 0.34 * s ** 1.2).ravel()
        TONE.append(np.where(rim.ravel()[:, None], COLORS[NEEDLE_LIT], COLORS[NEEDLE]) * fac[:, None])
        APP.append(grow_time(yy.ravel()) + 0.18 * droop.ravel() * rng.random(len(droop.ravel())))
    return (np.concatenate(P), np.concatenate(N), np.concatenate(TAG),
            np.concatenate(TONE).astype(np.float32), np.concatenate(APP))


def ground():
    # a soft bank of snow under the tree, the parcels resting on it
    x, z = np.meshgrid(np.linspace(-1.55, 1.55, 130), np.linspace(-0.55, 0.62, 34))
    keep = (x / 1.55) ** 2 + (z / 0.62) ** 2 <= 1
    y = BASE_Y - 0.035 + 0.045 * np.exp(-((x / 1.15) ** 2 + (z / 0.42) ** 2))
    p = np.stack([x, y, z], -1)
    n = np.broadcast_to(np.array([0.0, 1.0, 0.0], np.float32), p.shape).copy()
    m = keep
    return p[m], n[m], T_TRUNK + 0.1


def garland():
    # one gold strand winding up the tree, with a scallop between its turns
    u = np.linspace(0, 1, 430)
    y = BASE_Y + 0.16 + (TOP_Y - BASE_Y - 0.34) * u
    a = u * 7.0 * TAU
    r = cone_radius(y) * 1.13 + 0.020
    r = r + 0.035 * np.sin(u * 90 * np.pi) ** 2
    curve = np.stack([r * np.sin(a), y + 0.022 * np.sin(u * 90 * np.pi), r * np.cos(a)], 1)
    p, n = tube(curve, 0.019, 6)
    return p, n, np.repeat(T_GARLAND[0] + (T_GARLAND[1] - T_GARLAND[0]) * u, 6)


N_BALLS = 22
_ball_rng = np.random.default_rng(7)


def balls():
    # the baubles: spheres hung just under the tier rims, each with a little gold cap
    u, v = np.meshgrid(np.linspace(0, TAU, 26, endpoint=False), np.linspace(-1, 1, 17))
    disk = np.sqrt(np.maximum(1 - v ** 2, 0))
    sp = np.stack([disk * np.cos(u), v, disk * np.sin(u)], -1)
    sn = sp / (np.linalg.norm(sp, axis=-1, keepdims=True) + 1e-9)
    n_pts = sp.shape[0] * sp.shape[1]
    cu, cq = np.meshgrid(np.linspace(0, TAU, 16, endpoint=False), np.linspace(0, 1, 4))
    cap_p = np.stack([0.30 * cq * np.cos(cu), 0.94 + 0.20 * cq, 0.30 * cq * np.sin(cu)], -1)
    cap_n = np.stack([np.cos(cu), np.full_like(cu, 0.6), np.sin(cu)], -1)
    cap_n /= np.linalg.norm(cap_n, axis=-1, keepdims=True)
    ys, aa, rr, rads = np.empty(N_BALLS), np.empty(N_BALLS), np.empty(N_BALLS), np.empty(N_BALLS)
    for k in range(N_BALLS):
        tier = k % N_TIERS
        ys[k] = BASE_Y + 0.24 + (CONE_H - 0.24) * tier / N_TIERS
        aa[k] = (0.35 + k * 2.399963) % TAU
        rads[k] = (0.125 + 0.045 * _ball_rng.random()) * (0.70 + 0.30 * cone_radius(ys[k]) / R_BASE)
        rr[k] = cone_radius(ys[k]) * 1.20
    order = np.argsort(ys)
    starts = np.empty(N_BALLS)
    starts[order] = T_BALLS[0] + (T_BALLS[1] - T_BALLS[0]) * np.arange(N_BALLS) / max(N_BALLS - 1, 1)
    out = []
    for k in range(N_BALLS):
        rad = rads[k]
        centre = np.array([rr[k] * np.sin(aa[k]), ys[k] - rad * 1.1, rr[k] * np.cos(aa[k])], np.float32)
        col = BALL_SHADES[k % len(BALL_SHADES)]
        tint = np.empty((n_pts, 3), np.float32)
        tint[:] = col
        tag = np.full(n_pts, BALL)
        glint = _ball_rng.choice(np.flatnonzero((v.ravel() > 0.25) & (v.ravel() < 0.8)), 6, replace=False)
        tag[glint] = GLINT
        out.append((centre + sp.reshape(-1, 3) * rad, sn.reshape(-1, 3), tag, tint,
                    1000 + 100 * k, starts[k], rad, centre + cap_p.reshape(-1, 3) * rad,
                    cap_n.reshape(-1, 3), 1000 + 100 * k))
    return out, starts


def star_shape(Ro=0.38, Ri=0.148):
    # a five-pointed star, filled, a little thick, facing the camera
    verts = []
    for j in range(10):
        a = np.pi / 2 + j * np.pi / 5
        r = Ro if j % 2 == 0 else Ri
        verts.append((r * np.cos(a), r * np.sin(a)))
    poly = np.array(verts)
    g = np.linspace(-Ro, Ro, 96)
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    inside = np.zeros(len(pts), bool)
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % 10]
        cond = ((y1 > pts[:, 1]) != (y2 > pts[:, 1])) & \
               (pts[:, 0] < (x2 - x1) * (pts[:, 1] - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= cond
    thickness = 0.030
    P, N, TAG = [], [], []
    for z, nz in ((thickness, 1.0), (-thickness, -1.0)):
        p = np.stack([pts[inside, 0], pts[inside, 1], np.full(inside.sum(), z)], 1)
        P.append(p)
        N.append(np.broadcast_to(np.array([0.0, 0.0, nz]), p.shape))
        TAG.append(np.zeros(len(p), np.int64))
    for i in range(10):                                             # the rim
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % 10]
        q = np.linspace(0, 1, 40)[:, None]
        edge = (1 - q) * np.array([x1, y1]) + q * np.array([x2, y2])
        for z in np.linspace(-thickness, thickness, 3):
            P.append(np.stack([edge[:, 0], edge[:, 1], np.full(len(edge), z)], 1))
            N.append(np.broadcast_to(np.array([0.0, 0.0, 1.0]), (len(edge), 3)))
            TAG.append(np.zeros(len(edge), np.int64))
    p = np.concatenate(P)
    n = np.concatenate(N)
    centre = np.array([0.0, TOP_Y + 0.36, 0.0], np.float32)
    return p + centre, n, np.concatenate(TAG), centre


def gift(cx, cz, w, h, d, rot, body_col, ribbon_col, appear):
    # a parcel: a box with a ribbon crossing every face and a bow on the lid
    R = spin_y(rot)
    faces = [
        (np.array([0, -h, 0]), np.array([w, 0, 0]), np.array([0, 0, d]), np.array([0, -1, 0])),
        (np.array([0, h, 0]), np.array([w, 0, 0]), np.array([0, 0, d]), np.array([0, 1, 0])),
        (np.array([0, 0, d]), np.array([w, 0, 0]), np.array([0, h, 0]), np.array([0, 0, 1])),
        (np.array([0, 0, -d]), np.array([w, 0, 0]), np.array([0, h, 0]), np.array([0, 0, -1])),
        (np.array([w, 0, 0]), np.array([0, 0, d]), np.array([0, h, 0]), np.array([1, 0, 0])),
        (np.array([-w, 0, 0]), np.array([0, 0, d]), np.array([0, h, 0]), np.array([-1, 0, 0])),
    ]
    P, N, TAG, TONE = [], [], [], []
    u, v = np.meshgrid(np.linspace(-1, 1, 24), np.linspace(-1, 1, 24))
    for o, e1, e2, nrm in faces:
        p = o + u[..., None] * e1 + v[..., None] * e2
        ribbon = (np.abs(u) < 0.13) | (np.abs(v) < 0.13)
        P.append(p.reshape(-1, 3))
        N.append(np.broadcast_to(nrm, p.shape).reshape(-1, 3).copy())
        TAG.append(np.where(ribbon.ravel(), GIFT_RIBBON, GIFT))
        tone = np.empty((p.size // 3, 3), np.float32)
        tone[:] = body_col
        tone[ribbon.ravel()] = ribbon_col
        TONE.append(tone)
    p = np.concatenate(P) @ R.T
    n = np.concatenate(N) @ R.T
    tag = np.concatenate(TAG)
    tone_all = np.concatenate(TONE)
    for side in (-1, 1):                                            # the bow's two loops
        q = np.linspace(0, TAU, 30)
        path = np.stack([side * (0.02 + 0.075 * np.cos(q)), h + 0.02 + 0.060 * np.sin(q),
                         0.012 * np.sin(q * 0.5)], -1) @ R.T
        bp, bn = tube(path, 0.013, 5)
        bp, bn = bp.reshape(-1, 3), bn.reshape(-1, 3)
        p = np.concatenate([p, bp])
        n = np.concatenate([n, bn])
        tag = np.concatenate([tag, np.full(len(bp), GIFT_RIBBON)])
        tone_all = np.concatenate([tone_all, np.broadcast_to(ribbon_col, (len(bp), 3))])
    centre = np.array([cx, BASE_Y + h, cz], np.float32)
    return p + centre, n, tag, tone_all, centre


def build():
    b = Builder()
    tp, tn = trunk()
    b.add(tp, tn, TRUNK, COLORS[TRUNK], 2, T_TRUNK)
    fp, fn, ft, ftone, fa = foliage()
    b.add(fp, fn, ft, ftone, 10 + np.repeat(np.arange(N_TIERS), len(fp) // N_TIERS), fa)
    gp, gn, ga = ground()
    b.add(gp, gn, GROUND, COLORS[GROUND], 3, ga)
    rp, rn, ra = garland()
    b.add(rp, rn, GARLAND, COLORS[GARLAND], 4, ra)
    out, starts = balls()
    for k, (bp, bn, bt, bcol, part, when, rad, cp, cn, cpart) in enumerate(out):
        b.add(bp, bn, bt, bcol, part, when, pop=True, centre=bp.mean(0))
        b.add(cp, cn, CAP, COLORS[CAP], cpart, when, centre=bp.mean(0))
    sp, sn, st, sc = star_shape()
    b.add(sp, sn, np.where(st == 0, STAR, STAR), COLORS[STAR], 5, T_STAR, pop=True, centre=sc)
    core = sc + np.array([[0.0, 0.0, 0.06], [0.014, 0.010, 0.05], [-0.014, -0.010, 0.05]], np.float32)
    b.add(core, np.tile(np.array([0.0, 0.0, 1.0], np.float32), (3, 1)), GLINT, WHITE, 6, T_STAR,
          pop=True, centre=sc)
    gifts = [gift(-0.66, 0.56, 0.30, 0.20, 0.24, -0.25, np.array([0.72, 0.10, 0.16], np.float32),
                  np.array([1.0, 0.78, 0.36], np.float32), T_GIFTS[0]),
             gift(0.60, 0.66, 0.24, 0.15, 0.20, 0.30, np.array([0.08, 0.42, 0.22], np.float32),
                  np.array([0.92, 0.16, 0.22], np.float32), T_GIFTS[0] + 0.35),
             gift(-0.06, 0.86, 0.18, 0.11, 0.16, 0.05, np.array([0.10, 0.24, 0.56], np.float32),
                  np.array([0.86, 0.90, 0.96], np.float32), T_GIFTS[0] + 0.7)]
    for j, (p, n, tg, tn, cen) in enumerate(gifts):
        b.add(p, n, tg, tn, 200 + 20 * j, T_GIFTS[0] + 0.35 * j, pop=True, centre=cen)
    return b.done(), starts


STATIC, BALL_START = build()


# ---------------------------------------------------------------------------------------------
# lights, snow, sparks

N_BULB = 66
_b = np.random.default_rng(19)
_up = (np.arange(N_BULB) + 0.5) / N_BULB
BULB_Y = BASE_Y + 0.18 + (TOP_Y - BASE_Y - 0.42) * _up
BULB_A = _up * 7.0 * TAU + 0.22
BULB_R = cone_radius(BULB_Y) * 1.14 + 0.035 + 0.030 * np.sin(_up * 90 * np.pi) ** 2
BULB_P = np.stack([BULB_R * np.sin(BULB_A), BULB_Y + 0.022 * np.sin(_up * 90 * np.pi),
                   BULB_R * np.cos(BULB_A)], 1).astype(np.float32)
BULB_COL = np.array([[1.0, 0.82, 0.42], [1.0, 0.94, 0.78], [1.0, 0.30, 0.28],
                     [0.45, 0.65, 1.0], [0.45, 0.95, 0.62]], np.float32)[
    np.random.default_rng(29).integers(0, 5, N_BULB)]
BULB_PHASE = _b.uniform(0, TAU, N_BULB)
BULB_SPEED = _b.uniform(0.9, 2.4, N_BULB)
BULB_APPEAR = np.linspace(T_LIGHTS[0], T_LIGHTS[1], N_BULB) + _b.uniform(0, 0.25, N_BULB)

N_SNOW = 190
_s = np.random.default_rng(33)
SNOWFLAKES = _s.random((N_SNOW, 7))
SNOW_TOP, SNOW_BOTTOM = TOP_Y + 0.9, BASE_Y - 0.05

N_SPARK = 90
_sp = np.random.default_rng(37)
SPARK = _sp.random((N_SPARK, 5))


def bulbs(t):
    # the bulbs on the tree, each on its own beat, and a wave of light running up it
    lit = np.clip((t - BULB_APPEAR) / FLASH, 0, 1)
    if not lit.any():
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    age = t - BULB_APPEAR
    twinkle = 0.70 + 0.30 * np.sin(t * BULB_SPEED + BULB_PHASE) ** 6
    wave_y = BASE_Y + ((t / 5.0) % 1.0) * (TOP_Y - BASE_Y)
    wave = np.exp(-((BULB_Y - wave_y) / 0.18) ** 2) * 0.9
    glow = np.where(age >= 0, (0.78 + 0.32 * twinkle + wave) * lit + 1.5 * np.clip(1 - age / FLASH, 0, 1) ** 2, 0)
    m = glow > 0.04
    return BULB_P[m], glow[m], BULB_COL[m]


def snow(t):
    # snow falling past the tree, drifting a little, slowing to rest near the ground
    if t < 1.5:
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    g = SNOWFLAKES
    x = 2.7 * g[:, 0] - 1.35 + 0.16 * np.sin(t * (0.4 + g[:, 4]) + g[:, 5] * TAU)
    span = SNOW_TOP - SNOW_BOTTOM
    y = SNOW_TOP - ((g[:, 2] * span + t * (0.22 + 0.42 * g[:, 3])) % span)
    z = -1.0 + 2.9 * g[:, 1]
    size = 0.30 + 0.80 * g[:, 3]
    return np.stack([x, y, z], 1), size, np.where((g[:, 6] < 0.25)[:, None],
                                                  COLORS[SNOW_BLUE], COLORS[SNOW])


def sparks(t):
    # thrown from the star as it catches fire
    age = t - (T_STAR + 0.10)
    if not 0 < age < 1.8:
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    g = SPARK
    a, up = g[:, 0] * TAU, 0.25 + 1.0 * g[:, 1]
    speed = 0.55 + 0.75 * g[:, 2]
    v = np.stack([np.cos(a) * speed, up, np.sin(a) * speed], 1)
    head = np.array([0.0, TOP_Y + 0.36, 0.0])
    p = head + v * age - np.array([0, 0.75, 0]) * age ** 2
    s = (1 - age / 1.8) ** 1.2 * (0.7 + 0.5 * g[:, 3])
    return p, s, np.where((g[:, 4] > 0.45)[:, None], COLORS[STAR],
                          np.array([1.0, 0.82, 0.42], np.float32))


# ---------------------------------------------------------------------------------------------
# light, motion, camera

def lights(t):
    key = np.array([0.42, 0.82, 0.52])
    sheen = np.array([0.9 * np.sin(t * 0.45), 0.30, 0.6])
    return key / np.linalg.norm(key), sheen / np.linalg.norm(sheen)


def sway(t):
    # the tree turns gently from side to side
    return 0.22 * np.sin(TAU * t / LOOP) * smooth(t / 1.5)


_FWD = (TARGET - EYE) / np.linalg.norm(TARGET - EYE)
_RIGHT = np.cross(_FWD, [0, 1.0, 0])
_RIGHT /= np.linalg.norm(_RIGHT)
_UP = np.cross(_RIGHT, _FWD)
VIEW = (-_FWD).astype(np.float32)
_RIGHT, _UP, _FWD = (v.astype(np.float32) for v in (_RIGHT, _UP, _FWD))
EYE32 = EYE.astype(np.float32)


def look(P):
    rel = P - EYE32
    return rel @ _RIGHT, rel @ _UP, rel @ _FWD


def scene(t):
    v = STATIC
    yaw = sway(t)
    Ry = spin_y(yaw)
    P = v["P"]
    grow = t < GROWN + FLASH
    if grow:
        age = t - v["appear"]
        pf = np.flatnonzero(v["pop"] & (age >= 0) & (age < FLASH))
        if len(pf):
            g = smooth(age[pf] / 0.2) * (1 + 0.35 * np.clip(1 - age[pf] / FLASH, 0, 1) ** 2)
            P = P.copy()
            P[pf] = v["centre"][pf] + (P[pf] - v["centre"][pf]) * g[:, None].astype(np.float32)
    P = P @ Ry.T
    N = v["N"] @ Ry.T
    K = v["K"]
    T = v["T"].copy()
    rgb = v["tone"].copy()

    key, sheen = (x.astype(np.float32) for x in lights(t))
    half_k = ((key + VIEW) / np.linalg.norm(key + VIEW)).astype(np.float32)
    half_s = ((sheen + VIEW) / np.linalg.norm(sheen + VIEW)).astype(np.float32)
    back = N @ VIEW < 0
    side = np.where(back, np.float32(-1), np.float32(1))
    diff = np.clip(side * (N @ key), 0, 1)
    a2 = np.clip(side * (N @ half_k), 0, 1) ** 2
    b2 = np.clip(side * (N @ half_s), 0, 1) ** 2
    a8, b8 = (a2 * a2) ** 2, (b2 * b2) ** 2
    shade = (0.58 + 0.62 * diff + v["gloss"] * (a8 * a8 * a8 + 0.7 * b8 * b8 * b2)) * v["gain"]
    # a little cross-light, so the tree has a bright side and a shaded one
    shade *= np.clip(1.0 + 0.20 * (P @ _RIGHT) / 1.1, 0.80, 1.20)

    # the star twinkles once it is lit
    if t > T_STAR:
        st = np.flatnonzero(T == STAR)
        shade[st] *= 1.0 + 0.16 * np.sin(t * 5.0) ** 2

    # a rose flashes white as it starts to open; so does each new bauble, parcel and the star
    if grow:
        fresh = ~v["pop"] & (age >= 0) & (age < 0.14)
        T[fresh], shade[fresh], rgb[fresh] = GLINT, 1.15, WHITE
        hot = v["pop"] & (age >= 0) & (age < FLASH)
        f = (np.clip(1 - age[hot] / FLASH, 0, 1) ** 2).astype(np.float32)
        rgb[hot] += (WHITE - rgb[hot]) * f[:, None]
        shade[hot] += 1.25 * f
        seen = age >= 0
        P, shade, rgb, T, K = P[seen], shade[seen], rgb[seen], T[seen], K[seen]

    bp, bs, bcol = bulbs(t)
    snp, sns, sncol = snow(t)
    spp, sps, spcol = sparks(t)
    bp = bp @ Ry.T
    extra = np.concatenate([bp, snp, spp])
    fx_shade = np.concatenate([bs, sns, sps])
    fx_tag = np.concatenate([np.full(len(bp), 1), np.full(len(snp), 0), np.full(len(spp), 2)])
    fx_rgb = np.concatenate([bcol, sncol, spcol])

    info = {"grown": float(np.clip((t - T_TREE[0]) / (T_TREE[1] - T_TREE[0]), 0, 1)),
            "balls": int((t >= BALL_START).sum()), "bulbs": int((t >= BULB_APPEAR).sum()),
            "snow": int(len(snp))}
    return (P, shade, rgb, T, K), (extra, fx_shade, fx_tag, fx_rgb), info


# points -> letters

def outline_letter(gx, gy):
    ax, ay = np.abs(gx), np.abs(gy)
    c = np.where(ax > 2 * ay, C("|"), np.where(ay > 2 * ax, np.where(gy > 0, C("_"), C("-")),
                 np.where(gx * gy > 0, C("/"), C("\\"))))
    return np.where((ax + ay) == 0, C("+"), c)


def rays(char, col, cy, cx, reach, rgb, strength=1.0, over=1.0):
    # a four-pointed flare: long arms sideways and up/down, short diagonals
    rows, cols = char.shape
    for dist in range(1, reach + 1):
        fade = (1 - dist / (reach + 1)) * strength
        arms = [(0, 1, "-"), (0, -1, "-"), (1, 0, "|"), (-1, 0, "|")]
        if dist <= reach // 2:
            arms += [(1, 1, "\\"), (-1, -1, "\\"), (1, -1, "/"), (-1, 1, "/")]
        for dy, dx, stroke in arms:
            yy, xx = cy + dy * dist, cx + dx * dist * (2 if dy == 0 else 1)
            k = fade * (0.7 if dx and dy else 1.0)
            if 0 <= yy < rows and 0 <= xx < cols and col[yy, xx].max() < k * over:
                char[yy, xx], col[yy, xx] = C(stroke), rgb * k
    if 0 <= cy < rows and 0 <= cx < cols:
        char[cy, cx], col[cy, cx] = C("*"), rgb * 1.3 * strength


def frame(t, cols, rows, cw, ch):
    (P, shade, rgb, T, K), (fx_p, fx_s, fx_t, fx_rgb), info = scene(t)
    aspect = (cols * cw) / (rows * ch)
    x, y, z = look(P)
    zz = np.maximum(z, 0.2)
    sx = (0.5 + x / zz * FOCAL * 0.5 / aspect) * cols
    sy = (MIDDLE - y / zz * FOCAL * 0.5) * rows
    order = np.argsort(-z)
    ix, iy = sx[order].astype(int), sy[order].astype(int)
    ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows) & (z[order] > 0.2)
    o = order[ok]
    ix, iy = ix[ok], iy[ok]

    part = np.full((rows, cols), -1)
    depth = np.full((rows, cols), 99.0)
    lum = np.zeros((rows, cols))
    tag = np.zeros((rows, cols), int)
    tint = np.zeros((rows, cols, 3), np.float32)
    part[iy, ix], depth[iy, ix], lum[iy, ix], tag[iy, ix], tint[iy, ix] = K[o], z[o], shade[o], T[o], rgb[o]
    body = part >= 0

    # outlines: next to empty space, or to another part further back
    pp = np.pad(part, 1, constant_values=-1)
    pd = np.pad(depth, 1, constant_values=99)
    edge = np.zeros_like(body)
    gx, gy = np.zeros((rows, cols)), np.zeros((rows, cols))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            nb = pp[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx]
            nz = pd[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx]
            other = (nb // 100 != part // 100) & (nb >= 1000) & (part >= 1000) & (nz >= depth)
            out = body & ((nb < 0) | ((nb != part) & (nz > depth + FOLD)) | other)
            gx += dx * out
            gy += dy * out
            if dx == 0 or dy == 0:
                edge |= out

    # letters: outline strokes, the inside filled by brightness and changing slowly
    r, c = np.mgrid[0:rows, 0:cols]
    h = ((r // 2) * 7919 + (c // 3) * 104729) % 9973
    level = np.clip(((lum - 0.32) * 2.4).astype(int), 0, 3)
    level = np.where(body & (tag == NEEDLE), np.minimum(level, 1), level)
    inside = FILL[level, (h + (t * 2 + h % 11).astype(int)) % 12]
    sparkle = body & (tag == GLINT)
    char = np.where(body, np.where(edge, outline_letter(gx, gy), inside), 0)
    char = np.where(sparkle, C("*"), char)

    # the star reads best as a clean outline: a hollow, faintly lit star with a bright rim
    star_body = body & (tag == STAR)
    star_in = np.zeros_like(star_body)
    if star_body.any():
        star_in = star_body & ~edge
        char = np.where(star_in, C("."), char)
        halo = np.zeros_like(star_body)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                halo |= np.roll(np.roll(star_body, dy, 0), dx, 1)
        halo &= ~star_body
        char = np.where(halo, 0, char)
    edge_light = np.where((tag == NEEDLE) | (tag == NEEDLE_LIT), np.clip(0.55 * lum + 0.25, 0.35, 1.0),
                          np.maximum(0.8, lum))
    light = np.where(body, np.where(edge, edge_light, np.clip(0.34 + 0.52 * lum / 0.9, 0.34, 1.10)), 0)
    light = np.where(star_in, 0.40, light)
    light = np.where(edge & (tag == STAR), 1.45, light)
    col = tint * light[..., None]
    steady = body.copy()
    seen_body = body

    # particles
    if len(fx_p):
        px, py, pz = look(fx_p)
        qx = ((0.5 + px / np.maximum(pz, 0.2) * FOCAL * 0.5 / aspect) * cols).astype(int)
        qy = ((MIDDLE - py / np.maximum(pz, 0.2) * FOCAL * 0.5) * rows).astype(int)
        m = (qx >= 0) & (qx < cols) & (qy >= 0) & (qy < rows) & (fx_s > 0.03) & (pz > 0.2)
        qx, qy, pz, s, k, pc = qx[m], qy[m], pz[m], fx_s[m], fx_t[m], fx_rgb[m]
        behind = seen_body[np.clip(qy, 0, rows - 1), np.clip(qx, 0, cols - 1)] & \
            (pz > depth[np.clip(qy, 0, rows - 1), np.clip(qx, 0, cols - 1)] + 0.02)
        keep = ~behind
        qx, qy, s, k, pc = qx[keep], qy[keep], s[keep], k[keep], pc[keep]
        star_rays = []
        for j in range(len(qx)):
            if k[j] == 2:                                           # a spark
                char[qy[j], qx[j]] = C("*") if s[j] > 0.45 else C("+")
                col[qy[j], qx[j]] = pc[j] * min(1.3, 0.5 + s[j])
                steady[qy[j], qx[j]] = False
            elif k[j] == 1:                                         # a bulb
                char[qy[j], qx[j]] = C("*") if s[j] > 1.0 else (C("o") if s[j] > 0.5 else C("+"))
                col[qy[j], qx[j]] = pc[j] * min(1.25, 0.35 + s[j])
                steady[qy[j], qx[j]] = True
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = qy[j] + dy, qx[j] + dx
                    if 0 <= yy < rows and 0 <= xx < cols and (not body[yy, xx] or lum[yy, xx] < 0.62) and col[yy, xx].max() < 0.18:
                        char[yy, xx] = C(".")
                        col[yy, xx] = pc[j] * 0.38
                        steady[yy, xx] = True
                if s[j] > 0.9:
                    star_rays.append((qy[j], qx[j], s[j], pc[j]))
            else:                                                   # a snowflake
                char[qy[j], qx[j]] = C("*") if s[j] > 0.8 else (C("+") if s[j] > 0.45 else C("."))
                col[qy[j], qx[j]] = pc[j] * (0.3 + 0.7 * s[j])
                steady[qy[j], qx[j]] = False
        for cy, cx, s_, rgb_ in sorted(star_rays, key=lambda q: -q[2])[:10]:
            rays(char, col, int(cy), int(cx), int(1 + 2.5 * min(s_, 1.3) ** 2), rgb_, 0.30 + 0.30 * s_)

    # the star: rays that breathe once it has caught fire
    if t > T_STAR:
        hx, hy, hz = look(np.array([[0.0, TOP_Y + 0.36, 0.0]]))
        cx = int((0.5 + hx[0] / hz[0] * FOCAL * 0.5 / aspect) * cols)
        cy = int((MIDDLE - hy[0] / hz[0] * FOCAL * 0.5) * rows)
        if t < T_STAR + 1.2:
            g = 1 - (t - T_STAR) / 1.2
            rays(char, col, cy, cx, int(3 + 8 * g), WHITE, 0.6 + 0.8 * g)
        else:
            ph = (t - T_STAR - 1.2) % 4.0
            g = float(np.exp(-((ph - 0.25) / 0.22) ** 2))
            if g > 0.05:
                rays(char, col, cy, cx, int(1 + 3 * g), COLORS[STAR], 0.30 + 0.45 * g)

    # spangles on the brightest glints
    score = np.where(sparkle, lum + (h % 7) * 0.01, 0)
    for n, idx in enumerate(np.argsort(score.ravel())[::-1][:3]):
        if score.ravel()[idx] <= 0:
            break
        cy, cx = divmod(int(idx), cols)
        rays(char, col, cy, cx, int(2 + 3 * (0.5 + 0.5 * np.sin(t * 5 + n * 2.1))), WHITE, 0.8, over=2.0)

    info["message"] = message(char, col, t, cw, ch) if t > T_MSG else None
    info["letters"] = int((char > 0).sum())
    return char, col, steady, info


WARM = np.array([1.0, 0.95, 0.82], np.float32)
MERRY_RED = np.array([1.0, 0.28, 0.34], np.float32)
PINE = np.array([0.42, 0.95, 0.60], np.float32)
MESSAGE_PX = 18


def message_px(width):
    return max(8, round(MESSAGE_PX * width / 792))


def message(char, col, t, cw, ch):
    # "M E R R Y   C H R I S T M A S" typed out under the tree: red for the greeting, pine green
    # for the wish, warm white while a letter is new. Its letters are bigger than the grid's, so
    # they are drawn by the Renderer; here the grid under each one is cleared
    rows, cols = char.shape
    _, y_, z_ = look(np.array([[0.0, BASE_Y - 0.64, 0.0]], np.float32))
    row = int((MIDDLE - y_[0] / z_[0] * FOCAL * 0.5) * rows)
    size = message_px(cols * cw)
    advance = 0.6 * size
    x0 = (cols * cw - advance * len(MESSAGE)) / 2
    shown = min(len(MESSAGE), int((t - T_MSG) * 10))
    split = MESSAGE.index("C")
    reach = max(1, int(np.ceil(0.58 * size / ch)))
    band = slice(max(row - reach, 0), min(row + reach + 1, rows))
    letters = []
    for i, letter in enumerate(MESSAGE[:shown]):
        if letter == " ":
            continue
        new = np.clip(1 - ((t - T_MSG) * 10 - i) / 3, 0, 1)
        shimmer = np.clip(np.sin(t * 2.4 - i * 0.5), 0, 1) ** 6
        base = MERRY_RED if i < split else PINE
        tint = base + (WARM - base) * max(new, 0.75 * shimmer)
        cx = x0 + advance * (i + 0.5)
        letters.append((letter, cx, np.clip(tint, 0, 1)))
        c0 = max(int((cx - 0.42 * size) // cw), 0)
        c1 = min(int((cx + 0.42 * size) // cw) + 1, cols)
        char[band, c0:c1] = 0                      # clear only the grid under the letters
        col[band, c0:c1] = 0
    return {"size": size, "y": (row + 0.5) * ch, "letters": letters}


# ---------------------------------------------------------------------------------------------
# window

TITLE = "Christmas tree in ASCII"
CW, CH, FONT_PX = 7, 10, 11
COLS, ROWS = 113, 105
W, H = COLS * CW, ROWS * CH
HINT = "Space pause · R restart · S screenshot · V video · Esc quit"
DIM, ACCENT, SIGN = (110, 104, 120), (255, 214, 140), (176, 168, 196)
TRAIL = 0.45
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"
GREYS = [(i, i, i) for i in range(256)]
_ATLAS = {}


def glyph_atlas(cw, ch, font_px):
    if (cw, ch, font_px) not in _ATLAS:
        pygame.font.init()
        font = pygame.font.SysFont(MONO, font_px)
        cells = []
        for code in range(32, 127):
            s = pygame.Surface((cw, ch))
            g = font.render(chr(code), True, (255, 255, 255))
            s.blit(g, ((cw - g.get_width()) // 2, (ch - g.get_height()) // 2))
            cells.append(pygame.surfarray.array3d(s).transpose(1, 0, 2)[..., 0])
        _ATLAS[cw, ch, font_px] = np.stack(cells)
    return _ATLAS[cw, ch, font_px]


class Renderer:
    def __init__(self, cols=COLS, rows=ROWS, cw=CW, ch=CH, font_px=FONT_PX, hud_px=15, hint=True):
        self.cols, self.rows, self.cw, self.ch = cols, rows, cw, ch
        self.w, self.h = cols * cw, rows * ch
        self.atlas = glyph_atlas(cw, ch, font_px)
        self.art = pygame.Surface((self.w, self.h))
        self.trail = pygame.Surface((self.w, self.h))
        self.out = pygame.Surface((self.w, self.h))
        self.tint = pygame.Surface((cols, rows))
        self.moving, self.still = pygame.Surface((self.w, self.h)), pygame.Surface((self.w, self.h))
        self.small = pygame.font.SysFont(MONO, hud_px)
        self.big = pygame.font.SysFont(MONO, message_px(self.w))
        self.hud_gap = int(hud_px * 1.45)
        self.hint = hint
        self.last_t = None

    @staticmethod
    def blur(src, size, radius):
        small = pygame.transform.smoothscale(src, size)
        if hasattr(pygame.transform, "gaussian_blur"):
            small = pygame.transform.gaussian_blur(small, max(1, radius))
        return small

    def glow(self, src):
        w, h = self.w, self.h
        tight = self.blur(src, (w // 4, h // 4), 2)
        wide = self.blur(tight, (w // 8, h // 8), 3)
        for layer, amount in ((tight, 0.75), (wide, 0.45)):
            layer = pygame.transform.smoothscale(layer, (w, h))
            layer.fill((int(255 * amount),) * 3, special_flags=pygame.BLEND_MULT)
            self.out.blit(layer, (0, 0), special_flags=pygame.BLEND_ADD)

    def tinted(self, col, into):
        pygame.surfarray.blit_array(self.tint, (np.clip(col, 0, 1) * 255).astype(np.uint8).transpose(1, 0, 2))
        pygame.transform.scale(self.tint, (self.w, self.h), into)
        into.blit(self.art, (0, 0), special_flags=pygame.BLEND_MULT)
        return into

    def draw(self, t):
        w, h = self.w, self.h
        char, col, steady, info = frame(t, self.cols, self.rows, self.cw, self.ch)
        grey = pygame.image.frombuffer(self.atlas[char].transpose(0, 2, 1, 3).tobytes(), (w, h), "P")
        grey.set_palette(GREYS)
        self.art.blit(grey, (0, 0))
        if self.last_t is None or t < self.last_t:
            self.trail.fill((0, 0, 0))
        self.last_t = t
        self.trail.fill((int(TRAIL * 255),) * 3, special_flags=pygame.BLEND_MULT)
        self.trail.blit(self.tinted(col * ~steady[..., None], self.moving), (0, 0), special_flags=pygame.BLEND_MAX)
        self.out.blit(self.trail, (0, 0))
        self.out.blit(self.tinted(col * steady[..., None], self.still), (0, 0), special_flags=pygame.BLEND_MAX)
        if info["message"]:
            m = info["message"]
            for letter, x, rgb in m["letters"]:
                g = self.big.render(letter, True, tuple(int(255 * v) for v in rgb))
                self.out.blit(g, (int(x - g.get_width() / 2), int(m["y"] - g.get_height() / 2)))
        self.glow(self.out)
        first = (f"growing {info['grown'] * 100:5.1f}%" if info["grown"] < 1
                 else f"baubles {info['balls']:>2}/{N_BALLS}      lights {info['bulbs']:>2}/{N_BULB}")
        lines = [first, f"letters {info['letters']:,}"]
        if self.hint:
            lines.append(HINT)
        g = self.hud_gap
        for i, text in enumerate(lines):
            self.out.blit(self.small.render(text, True, DIM), (24, h - 18 - g * (len(lines) - i)))
        self.out.blit(self.small.render(CREDIT, True, SIGN), (24, h - 18 - g * (len(lines) + 1) - g // 3))
        tt = self.small.render(f"{'●' if int(t * 2) % 2 == 0 else ' '} t {t:05.2f}s", True, ACCENT)
        self.out.blit(tt, (w - 24 - tt.get_width(), 20))
        return self.out


def export_video(window, renderer, fps=30, seconds=LOOP):
    # render one whole loop on a fixed clock, frame by frame, and write it to an mp4
    try:
        import cv2
    except ImportError:
        print("making a video needs opencv-python:  pip install opencv-python")
        return None
    frames = int(round(seconds * fps))
    w, h = renderer.w, renderer.h
    vw, vh = w + w % 2, h + h % 2                 # some codecs want even dimensions
    path = Path(__file__).with_name(f"tree_{int(time.time())}.mp4")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    if not writer.isOpened():
        print(f"could not open a video writer for {path.name}")
        return None
    win = window.get_size()
    buf = np.zeros((vh, vw, 3), np.uint8)
    bar_y = win[1] - 46
    cancelled = False
    try:
        for i in range(frames):
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    cancelled = True
            if cancelled:
                break
            frame = renderer.draw(i / fps)
            rgb = pygame.surfarray.array3d(frame).transpose(1, 0, 2)
            buf[:h, :w] = rgb[..., ::-1]          # cv2 wants BGR
            writer.write(buf)
            if i % 4 == 0:                         # show how far along the render is
                window.blit(frame if win == (w, h) else pygame.transform.smoothscale(frame, win), (0, 0))
                pygame.draw.rect(window, (20, 24, 22), (24, bar_y, win[0] - 48, 5))
                pygame.draw.rect(window, ACCENT, (24, bar_y, int((win[0] - 48) * (i + 1) / frames), 5))
                tip = renderer.small.render(f"recording video  {100 * (i + 1) // frames}%   Esc to cancel",
                                            True, SIGN)
                window.blit(tip, (24, bar_y - 26))
                pygame.display.flip()
            pygame.event.pump()
    finally:
        writer.release()
    renderer.last_t = None                         # start the trail fresh afterwards
    if cancelled:
        path.unlink(missing_ok=True)
        print("video cancelled")
        return None
    print(f"saved {path.name}")
    return path


def window_size():
    info = pygame.display.Info()
    k = 1.0
    if info.current_w > 0 and info.current_h > 0:
        k = min(1.0, 0.88 * info.current_h / H, 0.95 * info.current_w / W)
    return int(W * k), int(H * k)


def main():
    pygame.init()
    size = window_size()
    window = pygame.display.set_mode(size)
    pygame.display.set_caption(TITLE)
    renderer, clock = Renderer(), pygame.time.Clock()
    if "--video" in sys.argv[1:]:                  # render the loop and exit
        ok = export_video(window, renderer) is not None
        pygame.quit()
        return 0 if ok else 1
    t, paused = 0.0, False
    toast, toast_until = "", 0.0
    screen = renderer.draw(t)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return 0
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                t = 0.0
            if e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                pygame.image.save(screen, str(Path(__file__).with_name(f"tree_{int(time.time())}.png")))
            if e.type == pygame.KEYDOWN and e.key == pygame.K_v:
                path = export_video(window, renderer)
                if path:
                    toast, toast_until = f"saved {path.name}", time.time() + 4.0
                screen = renderer.draw(t)
        dt = clock.tick(30) / 1000
        if not paused:
            t += min(dt, 1 / 15)
            if t > LOOP:
                t = 0.0
            screen = renderer.draw(t)
        window.blit(screen if size == (W, H) else pygame.transform.smoothscale(screen, size), (0, 0))
        if time.time() < toast_until:
            tip = renderer.small.render(toast, True, ACCENT)
            window.blit(tip, (window.get_width() - 24 - tip.get_width(), window.get_height() - 26))
        pygame.display.flip()


if __name__ == "__main__":
    sys.exit(main())
