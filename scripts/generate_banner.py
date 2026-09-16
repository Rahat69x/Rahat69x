#!/usr/bin/env python3
"""
generate_banner.py
Generates dark.svg and light.svg animated GitHub profile banners for Rahat69x.
Implements:
1. Exact head+shoulders cropping and preprocessing (autocontrast + 1.3x contrast + unsharp mask)
2. Background segmentation via GrabCut with hard-cleared error bleed for dark mode
3. 1-bit Floyd-Steinberg dithering in serpentine order
4. Shape-rendering crispEdges path runs
5. Intro shimmer layer with 60 interleaved random groups (~2s fade-in)
6. Main loop layer with ~94 organic drift bands (sigma=4 noise to avoid grid trap)
7. ~900 travellers morphing across Flutter, </>, and Vercel via optimal transport
8. Locked textLength info panel with computed dotted leaders and pulsing LIVE badge
"""

import os
import sys
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
import cv2
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
PHOTO_PATH = r"C:\Users\Rahat\.gemini\antigravity-ide\brain\0feb7f55-42c8-4cbf-b5b3-3f9820292a73\.user_uploaded\media_1789500346184.jpg"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

GRID_W, GRID_H = 300, 340
BANNER_W, BANNER_H = 1180, 610

# Animation timings
INTRO_DUR = 3.2
INTRO_FADE_DUR = 2.0
LOOP_DUR = 14.2
T_PORTRAIT_HOLD = 3.0
T_TRANSITION = 1.3
T_LOGO_HOLD = 2.0

# Normalized keyTimes for 14.2s loop
# 0 -> 3.0 -> 4.3 -> 6.3 -> 7.6 -> 9.6 -> 10.9 -> 12.9 -> 14.2
KEYTIMES = [
    0.0,
    T_PORTRAIT_HOLD / LOOP_DUR,                           # 3.0 / 14.2 = 0.2113
    (T_PORTRAIT_HOLD + T_TRANSITION) / LOOP_DUR,          # 4.3 / 14.2 = 0.3028
    (T_PORTRAIT_HOLD + T_TRANSITION + T_LOGO_HOLD) / LOOP_DUR, # 6.3 / 14.2 = 0.4437
    (T_PORTRAIT_HOLD + 2*T_TRANSITION + T_LOGO_HOLD) / LOOP_DUR, # 7.6 / 14.2 = 0.5352
    (T_PORTRAIT_HOLD + 2*T_TRANSITION + 2*T_LOGO_HOLD) / LOOP_DUR, # 9.6 / 14.2 = 0.6761
    (T_PORTRAIT_HOLD + 3*T_TRANSITION + 2*T_LOGO_HOLD) / LOOP_DUR, # 10.9 / 14.2 = 0.7676
    (T_PORTRAIT_HOLD + 3*T_TRANSITION + 3*T_LOGO_HOLD) / LOOP_DUR, # 12.9 / 14.2 = 0.9085
    1.0
]
KEYTIMES_STR = "; ".join(f"{k:.4f}" for k in KEYTIMES)

# Palette Definitions
THEMES = {
    "dark": {
        "bg": "#0A101F",
        "card_bg": "#0D1527",
        "border": "#1E293B",
        "chrome": "#22D3EE",
        "chrome_subtle": "rgba(34, 211, 238, 0.15)",
        "portrait": "#A78BFA",
        "accent": "#10B981",
        "text_primary": "#F1F5F9",
        "text_secondary": "#94A3B8",
        "text_muted": "#475569",
        "leader": "#334155",
        "live_red": "#EF4444",
        "pill_bg": "rgba(34, 211, 238, 0.12)",
        "pill_border": "#22D3EE",
        "pill_text": "#22D3EE",
    },
    "light": {
        "bg": "#F8FAFC",
        "card_bg": "#FFFFFF",
        "border": "#E2E8F0",
        "chrome": "#0891B2",
        "chrome_subtle": "rgba(8, 145, 178, 0.12)",
        "portrait": "#7C3AED",
        "accent": "#059669",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_muted": "#94A3B8",
        "leader": "#CBD5E1",
        "live_red": "#DC2626",
        "pill_bg": "rgba(8, 145, 178, 0.10)",
        "pill_border": "#0891B2",
        "pill_text": "#0891B2",
    }
}

INFO_ROWS = [
    # Group 1: Identity & Background
    ("Subject", "Mahamudol Hasan Rahat"),
    ("Role", "Full-Stack Developer"),
    ("Origin", "Dhaka, Bangladesh"),
    ("Education", "B.Sc in CSE"),
    ("Status", "CSE Student + Cybersecurity Learner"),
    ("ToolChain", "VS Code · Git · Android Studio · Figma"),
    None, # separator
    # Group 2: Core Competencies
    ("Core.Lang", "C · C++ · Java · Python"),
    ("Core.Frontend", "HTML · CSS · JavaScript"),
    ("Core.Backend", "Java · Python"),
    ("Core.Database", "MySQL"),
    ("Core.Infra", "Git · GitHub · Linux"),
    None, # separator
    # Group 3: Grid Coordinates & Contacts
    ("Grid.Mail", "contact.rahat69x@gmail.com"),
    ("Grid.Portfolio", "rahat69x.dev"),
    ("Grid.LinkedIn", "linkedin.com/in/rahat69x"),
    ("Grid.GitHub", "github.com/Rahat69x"),
    ("Grid.Facebook", "facebook.com/rahat69x"),
]


# ---------------------------------------------------------------------------
# 1. Image Preprocessing & Segmentation
# ---------------------------------------------------------------------------
def load_and_preprocess(image_path):
    print("[1/6] Loading photo and performing head+shoulders crop...")
    im = Image.open(image_path)
    # Head and shoulders crop: (x: [55, 505], y: [360, 870])
    cropped = im.crop((55, 360, 505, 870)).resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)
    
    # GrabCut foreground segmentation
    print("      Computing GrabCut foreground segmentation mask...")
    img_bgr = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
    mask_gc = np.zeros((GRID_H, GRID_W), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask_gc, (35, 12, 230, 327), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype(bool)
    
    # Binary closing and hole filling to retain facial features & hair curls
    struct = ndimage.generate_binary_structure(2, 2)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, structure=struct, iterations=2))

    # Preprocessing as requested:
    # Contrast 1.3x only, autocontrast(cutoff=1) + UnsharpMask(radius=3, percent=140)
    print("      Applying autocontrast(cutoff=1), 1.3x contrast, and UnsharpMask...")
    gray = ImageOps.grayscale(cropped)
    ac = ImageOps.autocontrast(gray, cutoff=1)
    enh = ImageEnhance.Contrast(ac).enhance(1.3)
    unsharp = enh.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    arr = np.array(unsharp).astype(float)
    
    return arr, mask


# ---------------------------------------------------------------------------
# 2. Serpentine Floyd-Steinberg Dithering
# ---------------------------------------------------------------------------
def dither_fs(arr_in, is_dark_mode, mask):
    print(f"      Running serpentine Floyd-Steinberg dither (dark_mode={is_dark_mode})...")
    h, w = arr_in.shape
    arr = arr_in.copy()
    dots = np.zeros((h, w), dtype=np.uint8)
    
    if is_dark_mode:
        # Dark mode: zero out background, hard-clear diffusion bleed
        arr[~mask] = 0.0

    for y in range(h):
        is_even = (y % 2 == 0)
        x_range = range(w) if is_even else range(w - 1, -1, -1)
        direction = 1 if is_even else -1
        
        for x in x_range:
            if is_dark_mode and not mask[y, x]:
                dots[y, x] = 0
                arr[y, x] = 0.0
                continue
                
            old_val = np.clip(arr[y, x], 0.0, 255.0)
            
            if is_dark_mode:
                # Dark mode: dots represent the lit subject (highlights)
                new_val = 255.0 if old_val >= 128.0 else 0.0
                dots[y, x] = 1 if new_val == 255.0 else 0
            else:
                # Light mode: dots represent dark ink (shadows)
                new_val = 0.0 if old_val < 128.0 else 255.0
                dots[y, x] = 1 if new_val == 0.0 else 0
                
            err = old_val - new_val
            
            # Floyd-Steinberg error weights with serpentine direction
            # Neighbor 1: (y, x + direction) -> 7/16
            x1 = x + direction
            if 0 <= x1 < w:
                if not (is_dark_mode and not mask[y, x1]):
                    arr[y, x1] += err * (7.0 / 16.0)
                    
            # Row below: (y + 1)
            if y + 1 < h:
                # Neighbor 2: (y + 1, x - direction) -> 3/16
                x2 = x - direction
                if 0 <= x2 < w:
                    if not (is_dark_mode and not mask[y + 1, x2]):
                        arr[y + 1, x2] += err * (3.0 / 16.0)
                # Neighbor 3: (y + 1, x) -> 5/16
                if not (is_dark_mode and not mask[y + 1, x]):
                    arr[y + 1, x] += err * (5.0 / 16.0)
                # Neighbor 4: (y + 1, x + direction) -> 1/16
                x4 = x + direction
                if 0 <= x4 < w:
                    if not (is_dark_mode and not mask[y + 1, x4]):
                        arr[y + 1, x4] += err * (1.0 / 16.0)
                        
    return dots


def extract_runs(dots):
    """Converts 2D binary dot matrix into horizontal runs (x, y, length)."""
    h, w = dots.shape
    runs = []
    for y in range(h):
        row = dots[y]
        in_run = False
        start_x = 0
        for x in range(w):
            if row[x] == 1:
                if not in_run:
                    in_run = True
                    start_x = x
            else:
                if in_run:
                    runs.append((start_x, y, x - start_x))
                    in_run = False
        if in_run:
            runs.append((start_x, y, w - start_x))
    return runs


def runs_to_svg_path(runs):
    """Formats runs into compact SVG path data with crispEdges."""
    parts = []
    for x, y, l in runs:
        parts.append(f"M{x},{y}h{l}v1h-{l}z")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 3. Intro Shimmer Grouping & Evenness Metric
# ---------------------------------------------------------------------------
def group_intro_runs(runs, num_groups=60):
    print(f"[2/6] Grouping {len(runs)} runs into {num_groups} intro shimmer groups...")
    run_coords = np.array([(rx + rlen/2.0, ry) for rx, ry, rlen in runs])
    sort_idx = np.lexsort((run_coords[:, 0], run_coords[:, 1]))
    
    group_assignments = np.zeros(len(runs), dtype=int)
    group_assignments[sort_idx] = np.arange(len(runs)) % num_groups
    
    # Verify evenness metric (~0.05 good, ~0.7 patchy)
    h_total, _, _ = np.histogram2d(run_coords[:, 0], run_coords[:, 1], bins=[6, 6], range=[[0, GRID_W], [0, GRID_H]])
    p_total = h_total / max(h_total.sum(), 1)
    
    deviations = []
    groups = [[] for _ in range(num_groups)]
    for idx, (rx, ry, rlen) in enumerate(runs):
        g = group_assignments[idx]
        groups[g].append((rx, ry, rlen))
        
    for g in range(num_groups):
        if not groups[g]:
            continue
        g_coords = np.array([(rx + rlen/2.0, ry) for rx, ry, rlen in groups[g]])
        h_g, _, _ = np.histogram2d(g_coords[:, 0], g_coords[:, 1], bins=[6, 6], range=[[0, GRID_W], [0, GRID_H]])
        p_g = h_g / max(h_g.sum(), 1)
        deviations.append(0.5 * np.sum(np.abs(p_g - p_total)))
        
    evenness = np.mean(deviations)
    print(f"      Intro shimmer evenness metric: {evenness:.4f} (target ~0.05-0.12, patchy is ~0.7)")
    return groups, evenness


# ---------------------------------------------------------------------------
# 4. Drift Bands Grouping & Straight-Boundary Metric
# ---------------------------------------------------------------------------
def group_drift_bands(runs, num_bands=94, target_centroid=(155.0, 165.0)):
    print(f"[3/6] Clustering runs into {num_bands} organic drift bands (with sigma=4 noise)...")
    np.random.seed(42)
    
    # Generate 94 well-spaced cluster centers
    gx, gy = np.meshgrid(np.linspace(15, GRID_W - 15, 10), np.linspace(15, GRID_H - 15, 10))
    seeds = np.column_stack([gx.ravel(), gy.ravel()])[:num_bands] + np.random.uniform(-6, 6, (num_bands, 2))
    tree = cKDTree(seeds)
    
    # Run centers with per-dot Gaussian noise sigma=4 to break artificial rectilinear edges
    run_centers = np.array([(rx + rlen/2.0, ry) for rx, ry, rlen in runs])
    noise = np.random.normal(0, 4.0, run_centers.shape)
    noisy_centers = run_centers + noise
    
    _, band_ids = tree.query(noisy_centers)
    
    bands = [[] for _ in range(num_bands)]
    band_points = [[] for _ in range(num_bands)]
    for idx, (rx, ry, rlen) in enumerate(runs):
        b = band_ids[idx]
        bands[b].append((rx, ry, rlen))
        band_points[b].append((rx + rlen/2.0, ry))
        
    # Calculate drift vector for each band: 42% toward first logo centroid
    band_drifts = []
    for b in range(num_bands):
        if band_points[b]:
            centroid = np.mean(band_points[b], axis=0)
            dx = 0.42 * (target_centroid[0] - centroid[0])
            dy = 0.42 * (target_centroid[1] - centroid[1])
        else:
            dx, dy = 0.0, 0.0
        band_drifts.append((dx, dy))
        
    # Verify straight-boundary metric (~0.01 organic, ~0.17 grid)
    ys, xs = np.mgrid[0:GRID_H:2, 0:GRID_W:2]
    grid_pts = np.column_stack([xs.ravel(), ys.ravel()])
    _, grid_labels = tree.query(grid_pts + np.random.normal(0, 4.0, grid_pts.shape))
    grid_mat = grid_labels.reshape((GRID_H // 2, GRID_W // 2))
    diff_h = (grid_mat[:, :-1] != grid_mat[:, 1:])
    diff_v = (grid_mat[:-1, :] != grid_mat[1:, :])
    
    straight_segs = 0
    total_boundary = max(diff_h.sum() + diff_v.sum(), 1)
    for r in range(grid_mat.shape[0]):
        row = diff_h[r]
        cnt = 0
        for val in row:
            if val:
                cnt += 1
            else:
                if cnt >= 4:
                    straight_segs += cnt
                cnt = 0
        if cnt >= 4:
            straight_segs += cnt
            
    straight_metric = straight_segs / total_boundary
    print(f"      Drift straight-boundary metric: {straight_metric:.4f} (target ~0.01 organic, grid is ~0.17)")
    return bands, band_drifts, straight_metric


# ---------------------------------------------------------------------------
# 5. Logo Shapes & Optimal Transport Matching
# ---------------------------------------------------------------------------
def sample_logo_points(num_points=900):
    print(f"[4/6] Sampling {num_points} dots inside Flutter, </>, and Vercel logos...")
    np.random.seed(1337)
    
    # Logo 1: Flutter
    im_flutter = Image.new("L", (GRID_W, GRID_H), 0)
    d1 = ImageDraw.Draw(im_flutter)
    p1_a = [(180, 75), (105, 150), (135, 180), (210, 105)]
    p1_b = [(135, 180), (105, 210), (155, 260), (190, 260), (150, 220), (210, 160)]
    d1.polygon(p1_a, fill=255)
    d1.polygon(p1_b, fill=255)
    ys1, xs1 = np.where(np.array(im_flutter) > 128)
    idx1 = np.random.choice(len(xs1), num_points, replace=False if len(xs1) >= num_points else True)
    flutter_pts = np.column_stack([xs1[idx1], ys1[idx1]]).astype(float)
    
    # Logo 2: </> Code Glyph
    im_code = Image.new("L", (GRID_W, GRID_H), 0)
    d2 = ImageDraw.Draw(im_code)
    d2.line([(100, 110), (60, 170), (100, 230)], fill=255, width=16)
    d2.line([(170, 95), (130, 245)], fill=255, width=16)
    d2.line([(200, 110), (240, 170), (200, 230)], fill=255, width=16)
    ys2, xs2 = np.where(np.array(im_code) > 128)
    idx2 = np.random.choice(len(xs2), num_points, replace=False if len(xs2) >= num_points else True)
    code_pts = np.column_stack([xs2[idx2], ys2[idx2]]).astype(float)
    
    # Logo 3: Vercel Triangle
    im_vercel = Image.new("L", (GRID_W, GRID_H), 0)
    d3 = ImageDraw.Draw(im_vercel)
    triangle = [(150, 85), (65, 255), (235, 255)]
    d3.polygon(triangle, fill=255)
    ys3, xs3 = np.where(np.array(im_vercel) > 128)
    idx3 = np.random.choice(len(xs3), num_points, replace=False if len(xs3) >= num_points else True)
    vercel_pts = np.column_stack([xs3[idx3], ys3[idx3]]).astype(float)
    
    # Optimal Transport Matching via Hungarian linear sum assignment
    print("      Computing Optimal Transport paths between logos...")
    # Flutter -> Code
    cost_fc = np.sum((flutter_pts[:, None, :] - code_pts[None, :, :]) ** 2, axis=2)
    _, col_fc = linear_sum_assignment(cost_fc)
    code_pts_matched = code_pts[col_fc]
    
    # Code -> Vercel
    cost_cv = np.sum((code_pts_matched[:, None, :] - vercel_pts[None, :, :]) ** 2, axis=2)
    _, col_cv = linear_sum_assignment(cost_cv)
    vercel_pts_matched = vercel_pts[col_cv]
    
    mean_dist_1 = np.sqrt(np.sum((flutter_pts - code_pts_matched) ** 2, axis=1)).mean()
    mean_dist_2 = np.sqrt(np.sum((code_pts_matched - vercel_pts_matched) ** 2, axis=1)).mean()
    print(f"      Optimal transport mean travel distance: Flutter->Code {mean_dist_1:.1f}px, Code->Vercel {mean_dist_2:.1f}px")
    
    return flutter_pts, code_pts_matched, vercel_pts_matched


# ---------------------------------------------------------------------------
# 6. Assemble SVG Document
# ---------------------------------------------------------------------------
def build_banner_svg(theme_name, runs, intro_groups, drift_bands, band_drifts,
                     flutter_pts, code_pts, vercel_pts):
    t = THEMES[theme_name]
    print(f"[5/6] Building {theme_name}.svg document...")
    
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {BANNER_W} {BANNER_H}" width="{BANNER_W}" height="{BANNER_H}">')
    svg.append('<defs>')
    svg.append('<style>')
    svg.append(f'''
        .term-title {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; font-weight: 600; fill: {t["text_secondary"]}; }}
        .frame-label {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; font-weight: 700; letter-spacing: 1.5px; fill: {t["chrome"]}; }}
        .header-title {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; font-weight: 700; letter-spacing: 1.5px; fill: {t["chrome"]}; }}
        .live-text {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; font-weight: 700; fill: {t["live_red"]}; }}
        .pill-text {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; font-weight: 600; fill: {t["pill_text"]}; }}
        .info-row {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 14px; }}
        .term-prompt {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; fill: {t["accent"]}; }}
        .cursor {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; fill: {t["accent"]}; }}
    ''')
    svg.append('</style>')
    
    # Clip path for rounded terminal window
    svg.append('<clipPath id="win-clip">')
    svg.append(f'  <rect width="{BANNER_W}" height="{BANNER_H}" rx="12" ry="12"/>')
    svg.append('</clipPath>')
    svg.append('</defs>')
    
    # Terminal Window Container
    svg.append(f'<g clip-path="url(#win-clip)">')
    svg.append(f'  <rect width="{BANNER_W}" height="{BANNER_H}" fill="{t["bg"]}"/>')
    
    # Window Chrome Bar (Header)
    svg.append(f'  <rect width="{BANNER_W}" height="40" fill="{t["card_bg"]}"/>')
    svg.append(f'  <line x1="0" y1="40" x2="{BANNER_W}" y2="40" stroke="{t["border"]}" stroke-width="1"/>')
    
    # Window Controls (Red, Amber, Green)
    svg.append('  <circle cx="24" cy="20" r="6" fill="#EF4444"/>')
    svg.append('  <circle cx="44" cy="20" r="6" fill="#F59E0B"/>')
    svg.append('  <circle cx="64" cy="20" r="6" fill="#10B981"/>')
    
    # Window Title
    svg.append(f'  <text x="86" y="24" class="term-title">profile.sh --live</text>')
    
    # -----------------------------------------------------------------------
    # Left Frame: VISUAL.MAP (~38% width)
    # -----------------------------------------------------------------------
    frame_x, frame_y, frame_w, frame_h = 30, 56, 390, 524
    svg.append(f'  <rect x="{frame_x}" y="{frame_y}" width="{frame_w}" height="{frame_h}" rx="8" fill="{t["card_bg"]}" stroke="{t["border"]}" stroke-width="1"/>')
    svg.append(f'  <text x="{frame_x + 18}" y="{frame_y + 24}" class="frame-label">VISUAL.MAP</text>')
    svg.append(f'  <line x1="{frame_x + 18}" y1="{frame_y + 34}" x2="{frame_x + frame_w - 18}" y2="{frame_y + 34}" stroke="{t["border"]}" stroke-width="1"/>')
    
    # Inner nested SVG for the 300x340 portrait grid
    # Scaled to 350x396.6 centered inside the frame
    inner_x = frame_x + 20
    inner_y = frame_y + 44
    inner_w = 350
    inner_h = int(350 * GRID_H / GRID_W) # 396
    
    svg.append(f'  <svg x="{inner_x}" y="{inner_y}" width="{inner_w}" height="{inner_h}" viewBox="0 0 {GRID_W} {GRID_H}">')
    
    # -----------------------------------------------------------------------
    # LAYER A: Intro Shimmer Layer (plays once for 3.2s, then vanishes)
    # -----------------------------------------------------------------------
    svg.append(f'    <g id="intro-layer">')
    # Intro layer turns off at t=3.2s
    svg.append(f'      <animate attributeName="opacity" from="1" to="0" begin="{INTRO_DUR}s" dur="0.01s" fill="freeze"/>')
    
    for g_idx, g_runs in enumerate(intro_groups):
        if not g_runs:
            continue
        stagger = g_idx * (INTRO_FADE_DUR / len(intro_groups))
        path_data = runs_to_svg_path(g_runs)
        svg.append(f'      <g opacity="0">')
        svg.append(f'        <animate attributeName="opacity" from="0" to="1" begin="{stagger:.3f}s" dur="0.4s" fill="freeze"/>')
        svg.append(f'        <path d="{path_data}" fill="{t["portrait"]}" shape-rendering="crispEdges"/>')
        svg.append(f'      </g>')
    svg.append('    </g>') # end intro-layer
    
    # -----------------------------------------------------------------------
    # LAYER B: Main Loop Portrait (~94 Drift Bands, loops every 14.2s)
    # -----------------------------------------------------------------------
    svg.append(f'    <g id="loop-layer" opacity="0">')
    # Loop layer activates at t=3.2s
    svg.append(f'      <animate attributeName="opacity" from="0" to="1" begin="{INTRO_DUR}s" dur="0.01s" fill="freeze"/>')
    
    # Group bands
    for b_idx, (b_runs, (dx, dy)) in enumerate(zip(drift_bands, band_drifts)):
        if not b_runs:
            continue
        path_data = runs_to_svg_path(b_runs)
        svg.append(f'      <g id="band-{b_idx}">')
        # Drift animation: translate 42% toward first logo centroid while fading
        # keyTimes: 0 -> 3.0s (0) -> 4.3s (dx, dy) -> 12.9s (dx, dy) -> 14.2s (0)
        kt_drift = f"0; {KEYTIMES[1]:.4f}; {KEYTIMES[2]:.4f}; {KEYTIMES[7]:.4f}; 1"
        svg.append(f'        <animateTransform attributeName="transform" type="translate" dur="{LOOP_DUR}s" repeatCount="indefinite" keyTimes="{kt_drift}" values="0 0; 0 0; {dx:.2f} {dy:.2f}; {dx:.2f} {dy:.2f}; 0 0"/>')
        svg.append(f'        <animate attributeName="opacity" dur="{LOOP_DUR}s" repeatCount="indefinite" keyTimes="{kt_drift}" values="1; 1; 0; 0; 1"/>')
        svg.append(f'        <path d="{path_data}" fill="{t["portrait"]}" shape-rendering="crispEdges"/>')
        svg.append(f'      </g>')
    svg.append('    </g>') # end loop-layer
    
    # -----------------------------------------------------------------------
    # LAYER C: Travellers Swarm (~900 dots morphing across 3 logos)
    # -----------------------------------------------------------------------
    svg.append(f'    <g id="travellers-layer">')
    # Opacity keyframes: hidden during portrait hold (0 to 3.0s and 12.9s to 14.2s)
    # 0 -> 3.0s (0) -> 4.3s (1) -> 6.3s (1) -> 7.6s (1) -> 9.6s (1) -> 10.9s (1) -> 12.9s (1) -> 14.2s (0)
    op_values = "0; 0; 1; 1; 1; 1; 1; 1; 0"
    
    for i in range(len(flutter_pts)):
        x1, y1 = flutter_pts[i]
        x2, y2 = code_pts[i]
        x3, y3 = vercel_pts[i]
        
        # cx and cy values across the 8 intervals
        cx_vals = f"{x1:.1f}; {x1:.1f}; {x1:.1f}; {x1:.1f}; {x2:.1f}; {x2:.1f}; {x3:.1f}; {x3:.1f}; {x1:.1f}"
        cy_vals = f"{y1:.1f}; {y1:.1f}; {y1:.1f}; {y1:.1f}; {y2:.1f}; {y2:.1f}; {y3:.1f}; {y3:.1f}; {y1:.1f}"
        
        svg.append(f'      <circle cx="{x1:.1f}" cy="{y1:.1f}" r="1.6" fill="{t["portrait"]}" opacity="0">')
        svg.append(f'        <animate attributeName="cx" dur="{LOOP_DUR}s" repeatCount="indefinite" keyTimes="{KEYTIMES_STR}" values="{cx_vals}"/>')
        svg.append(f'        <animate attributeName="cy" dur="{LOOP_DUR}s" repeatCount="indefinite" keyTimes="{KEYTIMES_STR}" values="{cy_vals}"/>')
        svg.append(f'        <animate attributeName="opacity" dur="{LOOP_DUR}s" repeatCount="indefinite" keyTimes="{KEYTIMES_STR}" values="{op_values}"/>')
        svg.append(f'      </circle>')
    svg.append('    </g>') # end travellers-layer
    
    svg.append('  </svg>') # end inner portrait svg
    
    # Left Frame Bottom Telemetry Readout
    tele_y = inner_y + inner_h + 20
    svg.append(f'  <text x="{frame_x + 20}" y="{tele_y}" font-family="monospace" font-size="11" fill="{t["text_muted"]}">GRID: 300x340</text>')
    svg.append(f'  <text x="{frame_x + 130}" y="{tele_y}" font-family="monospace" font-size="11" fill="{t["text_muted"]}">FS-1BIT: SERP</text>')
    svg.append(f'  <text x="{frame_x + 240}" y="{tele_y}" font-family="monospace" font-size="11" fill="{t["accent"]}">SWARM: 900</text>')

    # -----------------------------------------------------------------------
    # Right Frame: SYSTEM.INFO (~62% width)
    # -----------------------------------------------------------------------
    right_x, right_y, right_w, right_h = 445, 56, 705, 524
    svg.append(f'  <rect x="{right_x}" y="{right_y}" width="{right_w}" height="{right_h}" rx="8" fill="{t["card_bg"]}" stroke="{t["border"]}" stroke-width="1"/>')
    
    # Header: Title + LIVE badge + Handle Pill
    svg.append(f'  <text x="{right_x + 20}" y="{right_y + 24}" class="header-title">SYSTEM.INFO</text>')
    
    # Pulsing LIVE badge
    svg.append(f'  <circle cx="{right_x + 160}" cy="{right_y + 20}" r="4.5" fill="{t["live_red"]}"/>')
    svg.append(f'  <circle cx="{right_x + 160}" cy="{right_y + 20}" r="4.5" fill="{t["live_red"]}"><animate attributeName="opacity" values="1;0.2;1" dur="1.5s" repeatCount="indefinite"/></circle>')
    svg.append(f'  <text x="{right_x + 172}" y="{right_y + 24}" class="live-text">LIVE</text>')
    
    # Pill with handle
    pill_w = 110
    pill_x = right_x + right_w - pill_w - 20
    pill_y = right_y + 8
    svg.append(f'  <rect x="{pill_x}" y="{pill_y}" width="{pill_w}" height="24" rx="12" fill="{t["pill_bg"]}" stroke="{t["pill_border"]}" stroke-width="1"/>')
    svg.append(f'  <text x="{pill_x + pill_w//2}" y="{pill_y + 16}" text-anchor="middle" class="pill-text">Rahat69x</text>')
    
    svg.append(f'  <line x1="{right_x + 20}" y1="{right_y + 36}" x2="{right_x + right_w - 20}" y2="{right_y + 36}" stroke="{t["border"]}" stroke-width="1"/>')

    # Rows with computed dotted leaders and locked width
    row_x = right_x + 20
    row_w = right_w - 40 # 665px
    curr_y = right_y + 60
    TARGET_CHARS = 74
    
    for row in INFO_ROWS:
        if row is None:
            # Separator gap
            curr_y += 10
            svg.append(f'  <line x1="{row_x}" y1="{curr_y - 4}" x2="{row_x + row_w}" y2="{curr_y - 4}" stroke="{t["border"]}" stroke-width="0.8" stroke-dasharray="2,4"/>')
            curr_y += 12
            continue
            
        label, val = row
        # Compute dotted leader length
        needed_dots = TARGET_CHARS - len(label) - len(val) - 2
        leader_str = " " + ("." * max(needed_dots, 3)) + " "
        
        # Distinguish label categories with palette colors
        label_color = t["chrome"] if label.startswith("Grid.") else (t["accent"] if label.startswith("Core.") else t["text_secondary"])
        
        svg.append(f'  <text x="{row_x}" y="{curr_y}" class="info-row" textLength="{row_w}" lengthAdjust="spacingAndGlyphs">')
        svg.append(f'    <tspan fill="{label_color}" font-weight="600">{label}</tspan>')
        svg.append(f'    <tspan fill="{t["leader"]}">{leader_str}</tspan>')
        svg.append(f'    <tspan fill="{t["text_primary"]}">{val}</tspan>')
        svg.append(f'  </text>')
        curr_y += 23 # spacing 23px
        
    # Terminal bottom prompt
    prompt_y = curr_y + 12
    svg.append(f'  <text x="{row_x}" y="{prompt_y}" class="term-prompt">rahat@system:~$ status --live</text>')
    svg.append(f'  <rect x="{row_x + 245}" y="{prompt_y - 12}" width="8" height="14" fill="{t["accent"]}"><animate attributeName="opacity" values="1;0;1" dur="1.0s" repeatCount="indefinite"/></rect>')
    
    svg.append('</g>') # end terminal window container
    svg.append('</svg>')
    
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    print("=================================================================")
    print("  Rahat69x GitHub Profile Banner Generator (Phase 1)             ")
    print("=================================================================")
    
    if not os.path.exists(PHOTO_PATH):
        print(f"Error: Photo not found at {PHOTO_PATH}")
        sys.exit(1)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Preprocess & Segment
    arr_unsharp, mask = load_and_preprocess(PHOTO_PATH)
    
    # 2. Dither
    dots_dark = dither_fs(arr_unsharp, is_dark_mode=True, mask=mask)
    dots_light = dither_fs(arr_unsharp, is_dark_mode=False, mask=mask)
    
    runs_dark = extract_runs(dots_dark)
    runs_light = extract_runs(dots_light)
    print(f"      Extracted dark runs: {len(runs_dark)} (dots: {dots_dark.sum()})")
    print(f"      Extracted light runs: {len(runs_light)} (dots: {dots_light.sum()})")
    
    # 3. Intro Shimmer Grouping
    intro_groups_dark, evenness_dark = group_intro_runs(runs_dark, num_groups=60)
    intro_groups_light, evenness_light = group_intro_runs(runs_light, num_groups=60)
    
    # 4. Drift Bands Grouping
    drift_bands_dark, band_drifts_dark, straight_dark = group_drift_bands(runs_dark, num_bands=94)
    drift_bands_light, band_drifts_light, straight_light = group_drift_bands(runs_light, num_bands=94)
    
    # 5. Logos & Optimal Transport Swarm
    flutter_pts, code_pts, vercel_pts = sample_logo_points(num_points=900)
    
    # 6. Persist source of truth .npy arrays
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "dots_dark.npy"), dots_dark)
    np.save(os.path.join(data_dir, "dots_light.npy"), dots_light)
    np.save(os.path.join(data_dir, "mask.npy"), mask)
    np.save(os.path.join(data_dir, "travellers_flutter.npy"), flutter_pts)
    np.save(os.path.join(data_dir, "travellers_code.npy"), code_pts)
    np.save(os.path.join(data_dir, "travellers_vercel.npy"), vercel_pts)
    print(f"      Persisted source of truth .npy arrays in {data_dir}")
    
    # 7. Build and save dark.svg
    dark_svg_content = build_banner_svg(
        "dark", runs_dark, intro_groups_dark, drift_bands_dark, band_drifts_dark,
        flutter_pts, code_pts, vercel_pts
    )
    dark_svg_path = os.path.join(OUTPUT_DIR, "dark.svg")
    with open(dark_svg_path, "w", encoding="utf-8") as f:
        f.write(dark_svg_content)
    dark_size_kb = os.path.getsize(dark_svg_path) / 1024.0
    print(f"[6/6] Saved {dark_svg_path} ({dark_size_kb:.1f} KB)")
    
    # Build and save light.svg
    light_svg_content = build_banner_svg(
        "light", runs_light, intro_groups_light, drift_bands_light, band_drifts_light,
        flutter_pts, code_pts, vercel_pts
    )
    light_svg_path = os.path.join(OUTPUT_DIR, "light.svg")
    with open(light_svg_path, "w", encoding="utf-8") as f:
        f.write(light_svg_content)
    light_size_kb = os.path.getsize(light_svg_path) / 1024.0
    print(f"      Saved {light_svg_path} ({light_size_kb:.1f} KB)")
    
    print("=================================================================")
    print("  Generation Summary & Verification Metrics:                     ")
    print(f"  - Dark Mode Size: {dark_size_kb:.1f} KB (Target ~900KB-1MB)")
    print(f"  - Light Mode Size: {light_size_kb:.1f} KB")
    print(f"  - Dark Intro Evenness Metric: {evenness_dark:.4f} (Ideal ~0.05-0.10)")
    print(f"  - Dark Drift Straight Metric: {straight_dark:.4f} (Organic ~0.01)")
    print("=================================================================")

if __name__ == "__main__":
    main()
