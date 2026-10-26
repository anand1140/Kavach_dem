#!/usr/bin/env python3
"""
terrain_loader.py - Real DEM & satellite data fetcher for a Ladakh Himalayan pass.

Downloads:
  * AWS Terrarium elevation tile  (default z=14, x=11723, y=6498)
  * Esri World Imagery            (2x2 tiles at z+1 covering the same area -> 512x512)

Writes (into ./static by default):
  ladakh_dem.npy    float32 (60, 60)  elevation, metres above sea level
  ladakh_slope.npy  float32 (60, 60)  slope angle, degrees
  ladakh_sat.jpg    512x512 RGB satellite image

Offline / failure fallback: a synthetic 4800-5400 m ridge is generated in the
exact same formats. Raw tiles are cached in static/cache so reruns are instant.

Usage:
  python terrain_loader.py
  python terrain_loader.py --offline          # force synthetic data
  python terrain_loader.py --refresh          # ignore cached tiles
  python terrain_loader.py --z 14 --x 11723 --y 6498 --grid 60 --out static

Dependencies: numpy, Pillow
"""

import argparse
import io
import math
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ESRI_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
HEADERS = {"User-Agent": "ladakh-terrain-loader/2.0 (educational project)"}
TIMEOUT = 15
RETRIES = 2
SAT_SIZE = 512
DEM_VALID_RANGE = (-500.0, 9000.0)     # anything outside is treated as nodata


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def tile_center_latlon(z, x, y):
    n = 2 ** z
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n))))
    return lat, lon


def metres_per_tile_pixel(z, lat_deg, tile_px=256):
    """Ground size of one tile pixel (Web Mercator, corrected for latitude)."""
    return 40075016.686 * math.cos(math.radians(lat_deg)) / (tile_px * 2 ** z)


# --------------------------------------------------------------------------- #
# Networking with cache
# --------------------------------------------------------------------------- #
def http_get(url, cache_path=None, refresh=False):
    if cache_path and os.path.exists(cache_path) and not refresh:
        with open(cache_path, "rb") as f:
            return f.read()
    last = None
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = r.read()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(data)
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.5 * (attempt + 1))
    raise ConnectionError(f"GET failed for {url}: {last}")


# --------------------------------------------------------------------------- #
# Raster maths
# --------------------------------------------------------------------------- #
def decode_terrarium(png_bytes):
    """Terrarium RGB -> metres ASL:  (R*256 + G + B/256) - 32768."""
    rgb = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=np.float64)
    elev = rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
    return elev.astype(np.float32)


def fill_nodata(dem):
    """Replace out-of-range / NaN cells with the nearest valid value (iterative dilation)."""
    lo, hi = DEM_VALID_RANGE
    bad = ~np.isfinite(dem) | (dem < lo) | (dem > hi)
    if not bad.any():
        return dem, 0
    if bad.all():
        raise ValueError("DEM tile contains no valid elevation values")
    out = dem.copy()
    out[bad] = np.nan
    while np.isnan(out).any():
        p = np.pad(out, 1, mode="edge")
        neigh = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        with np.errstate(all="ignore"):
            avg = np.nanmean(neigh, axis=0)
        mask = np.isnan(out) & ~np.isnan(avg)
        out[mask] = avg[mask]
    return out, int(bad.sum())


def resize_float(arr, size):
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    resample = Image.BOX if size < arr.shape[0] else Image.BICUBIC
    return np.asarray(img.resize((size, size), resample), dtype=np.float32)


def compute_slope(dem, cell_m):
    """Physical slope (degrees) from square cells of cell_m metres."""
    dz_dy, dz_dx = np.gradient(dem.astype(np.float64), cell_m, cell_m)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)


def smooth3(a):
    """Light 3x3 binomial blur to suppress single-pixel DEM noise before slope."""
    k = np.array([1, 2, 1], dtype=np.float64) / 4.0
    p = np.pad(a.astype(np.float64), 1, mode="edge")
    p = k[0] * p[:, :-2] + k[1] * p[:, 1:-1] + k[2] * p[:, 2:]
    p = k[0] * p[:-2] + k[1] * p[1:-1] + k[2] * p[2:]
    return p.astype(np.float32)


def hillshade(dem, cell_m, azimuth=315.0, altitude=45.0):
    az = math.radians(360.0 - azimuth + 90.0)
    alt = math.radians(altitude)
    dz_dy, dz_dx = np.gradient(dem.astype(np.float64), cell_m, cell_m)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(dz_dy, -dz_dx)
    hs = math.sin(alt) * np.cos(slope) + math.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
def fetch_real_dem(z, x, y, cache_dir, refresh):
    path = os.path.join(cache_dir, f"terrarium_{z}_{x}_{y}.png")
    png = http_get(TERRARIUM_URL.format(z=z, x=x, y=y), path, refresh)
    dem = decode_terrarium(png)
    if dem.shape != (256, 256):
        raise ValueError(f"unexpected Terrarium tile shape {dem.shape}")
    return dem


def fetch_real_satellite(z, x, y, cache_dir, refresh):
    """Stitch the 2x2 tiles at z+1 that exactly cover the z tile -> 512x512."""
    canvas = Image.new("RGB", (512, 512))
    for dy in range(2):
        for dx in range(2):
            tx, ty = 2 * x + dx, 2 * y + dy
            path = os.path.join(cache_dir, f"esri_{z + 1}_{tx}_{ty}.jpg")
            data = http_get(ESRI_URL.format(z=z + 1, x=tx, y=ty), path, refresh)
            canvas.paste(Image.open(io.BytesIO(data)).convert("RGB"), (dx * 256, dy * 256))
    arr = np.asarray(canvas)
    if arr.std() < 3.0:   # blank / "no data" placeholder tiles
        raise ValueError("satellite imagery looks blank")
    return canvas if SAT_SIZE == 512 else canvas.resize((SAT_SIZE, SAT_SIZE), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #
def synthetic_dem(size=256, seed=42, lo=4800.0, hi=5400.0):
    """Diagonal ridge with a central saddle plus multi-octave fractal detail."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1.0)
    d = xx * 0.6 + yy * 0.8 - 0.7
    along = -xx * 0.8 + yy * 0.6
    # The saddle (valley pass) sits off the start->base-camp diagonal, so a
    # straight line crosses the steep crest while a slope-aware route detours.
    saddle = 1.0 - 0.60 * np.exp(-((along - 0.40) / 0.16) ** 2)
    base = np.exp(-(d / 0.25) ** 2) * saddle

    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    radius[0, 0] = 1.0

    def octave(power):
        f = np.fft.fft2(rng.standard_normal((size, size))) / radius ** power
        f[0, 0] = 0
        o = np.real(np.fft.ifft2(f))
        return (o - o.min()) / (o.max() - o.min())

    detail = 0.8 * octave(2.4) + 0.2 * octave(1.6)
    h = 0.88 * base + 0.12 * detail
    h = (h - h.min()) / (h.max() - h.min())
    return (lo + (hi - lo) * h).astype(np.float32)


def shaded_satellite(dem256, px_m, lo=None, hi=None):
    """Plausible desert-mountain image from a DEM: hillshade + altitude tint + snow."""
    lo = float(dem256.min()) if lo is None else lo
    hi = float(dem256.max()) if hi is None else hi
    dem = resize_float(dem256, SAT_SIZE)
    hs = hillshade(dem, px_m * 256.0 / SAT_SIZE)
    t = np.clip((dem - lo) / max(hi - lo, 1.0), 0, 1)[..., None]
    low = np.array([150, 125, 95.0])
    mid = np.array([120, 112, 105.0])
    snow = np.array([235, 238, 242.0])
    col = np.where(t < 0.6, low + (mid - low) * (t / 0.6),
                   mid + (snow - mid) * np.clip((t - 0.6) / 0.4, 0, 1))
    col = col * (0.35 + 0.75 * hs[..., None])
    return Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def atomic_save_npy(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def atomic_save_jpg(path, img):
    tmp = path + ".tmp"
    img.save(tmp, format="JPEG", quality=92)
    os.replace(tmp, path)


def build(args):
    os.makedirs(args.out, exist_ok=True)
    cache_dir = os.path.join(args.out, "cache")
    lat, lon = tile_center_latlon(args.z, args.x, args.y)
    px_m = metres_per_tile_pixel(args.z, lat)
    print(f"Tile z={args.z} x={args.x} y={args.y}  centre {lat:.4f}N {lon:.4f}E  (~{px_m:.2f} m/px)")

    source, dem256 = "REAL", None
    if not args.offline:
        try:
            print("Downloading AWS Terrarium DEM ...")
            dem256 = fetch_real_dem(args.z, args.x, args.y, cache_dir, args.refresh)
            dem256, n_bad = fill_nodata(dem256)
            if n_bad:
                print(f"  filled {n_bad} nodata pixels")
        except Exception as e:  # noqa: BLE001
            print(f"  ! DEM unavailable ({e})")
    if dem256 is None:
        print("  -> using synthetic 4800-5400 m ridge (offline mode)")
        dem256, source = synthetic_dem(), "SYNTHETIC"

    # Slope at native DEM resolution (keeps steep faces), then average to GRID.
    slope256 = compute_slope(smooth3(dem256), px_m)
    dem = resize_float(dem256, args.grid)
    slope = resize_float(slope256, args.grid)
    if args.max_slope is not None:
        slope = np.minimum(slope, args.max_slope)

    sat = None
    if source == "REAL":
        try:
            print("Downloading Esri satellite imagery (2x2 tiles) ...")
            sat = fetch_real_satellite(args.z, args.x, args.y, cache_dir, args.refresh)
        except Exception as e:  # noqa: BLE001
            print(f"  ! Satellite unavailable ({e}); rendering shaded image from DEM")
    if sat is None:
        sat = shaded_satellite(dem256, px_m)

    dem_p = os.path.join(args.out, "ladakh_dem.npy")
    slope_p = os.path.join(args.out, "ladakh_slope.npy")
    sat_p = os.path.join(args.out, "ladakh_sat.jpg")
    atomic_save_npy(dem_p, dem.astype(np.float32))
    atomic_save_npy(slope_p, slope.astype(np.float32))
    atomic_save_jpg(sat_p, sat)

    print(f"\nSource : {source}")
    print(f"DEM    : {dem_p}  {dem.shape} {dem.dtype}  {dem.min():.1f}-{dem.max():.1f} m ASL")
    print(f"Slope  : {slope_p}  {slope.shape} {slope.dtype}  {slope.min():.1f}-{slope.max():.1f} deg "
          f"(mean {slope.mean():.1f})")
    print(f"Sat    : {sat_p}  {sat.size[0]}x{sat.size[1]}")
    return source


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ladakh DEM + satellite fetcher")
    p.add_argument("--z", type=int, default=14)
    p.add_argument("--x", type=int, default=11723)
    p.add_argument("--y", type=int, default=6498)
    p.add_argument("--grid", type=int, default=60, help="DEM/slope grid size (default 60)")
    p.add_argument("--out", default="static", help="output directory")
    p.add_argument("--offline", action="store_true", help="skip network, use synthetic terrain")
    p.add_argument("--refresh", action="store_true", help="ignore cached tiles")
    p.add_argument("--max-slope", type=float, default=None,
                   help="optionally clip slope to this many degrees (e.g. 52)")
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
    sys.exit(0)
