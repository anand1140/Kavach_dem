#!/usr/bin/env python3
"""
main.py - FastAPI server + OpenCV ArUco HUD for the Ladakh UGV/UAV simulation.

  GET  /                   -> static/index.html (if present)
  GET  /static/*           -> ladakh_sat.jpg, pcb_schematic.png, index.html, ...
  GET  /api/init_map       -> 60x60 DEM + slope grids, naive_path, resilient_path (once)
  POST /api/attack         -> {attack_mode, jamming_dbm, force_recovery}
  POST /api/reset          -> restart the scenario
  WS   /ws/telemetry       -> 15 Hz telemetry stream incl. 260x260 ArUco HUD (base64 JPEG)

Physics (sim_engine.SimulationEngine) ticks at 10 Hz (100 ms). The stream runs
at 15 Hz, so roughly every third frame repeats the same physics state with a
fresh HUD render; use the `tick` counter in each message to tell them apart.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import json
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont

import sim_engine as se
from sim_engine import CELL_X_M, CELL_Y_M, GRID_N, TICK_S, SimulationEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

STREAM_HZ = 15.0
INSTANT_SPOOF_JUMP_M = 150.0        # GPS fix teleports this far at once
MEACONING_RATE_MPS = 1.2            # slow drag-off rate for GRADUAL_MEACONING
SPOOF_DIRECTION_DEG = 40.0
SPOOF_MAX_OFFSET_M = 400.0


# =========================================================================== #
# Static assets
# =========================================================================== #
def make_pcb_schematic(path: Path):
    """Block-diagram of the anti-jamming RF front end (generated if missing)."""
    W, H = 1000, 460
    img = Image.new("RGB", (W, H), (14, 22, 18))
    d = ImageDraw.Draw(img)
    try:
        f_big = ImageFont.load_default(size=17)
        f = ImageFont.load_default(size=13)
    except TypeError:                       # very old Pillow
        f_big = f = ImageFont.load_default()
    green, amber, grey, white = (60, 220, 120), (255, 190, 60), (120, 140, 130), (225, 235, 230)

    def block(x, y, w, h, title, sub="", col=green):
        d.rounded_rectangle([x, y, x + w, y + h], 8, outline=col, width=2, fill=(20, 34, 28))
        d.text((x + w / 2, y + h / 2 - (8 if sub else 0)), title, fill=white, font=f_big, anchor="mm")
        if sub:
            d.text((x + w / 2, y + h / 2 + 14), sub, fill=grey, font=f, anchor="mm")

    def arrow(x1, y1, x2, y2, col=green):
        d.line([x1, y1, x2, y2], fill=col, width=2)
        a = math.atan2(y2 - y1, x2 - x1)
        for s in (-1, 1):
            d.line([x2, y2, x2 - 10 * math.cos(a + s * 0.45), y2 - 10 * math.sin(a + s * 0.45)], fill=col, width=2)

    d.text((20, 14), "ANTI-JAMMING GNSS FRONT END  (virtual PCB)", fill=green, font=f_big)
    y = 130
    block(30, y, 110, 70, "L1 ANT", "patch + choke")
    block(190, y, 150, 70, "SAW NOTCH", "active < 0.85 V AGC")
    block(390, y, 150, 70, "LNA + AGC", "lna_agc_volts")
    block(590, y, 150, 70, "GNSS RX / ADC", "cn0_db_hz")
    block(790, y, 180, 70, "MCU  (STM32)", "KF + TERCOM fusion")
    for a, b in ((140, 190), (340, 390), (540, 590), (740, 790)):
        arrow(a, y + 35, b, y + 35)

    block(390, 290, 150, 60, "JAM DETECT", "power sensor", amber)
    arrow(465, 290, 465, y + 70, amber)                       # AGC feedback
    d.text((472, 262), "AGC loop", fill=amber, font=f)
    arrow(540, 320, 590, 320 - 0, amber)
    block(790, 290, 180, 60, "GPIO PA14", "RF link select", amber)
    arrow(880, y + 70, 880, 290, amber)
    block(640, 380, 150, 55, "2.4 GHz PRIMARY", "", green)
    block(820, 380, 150, 55, "915 MHz FHSS", "backup", amber)
    arrow(850, 350, 715, 380, green)
    arrow(900, 350, 895, 380, amber)
    d.text((20, 430), "cn0 = max(12, 47 - 0.38 J)    agc = max(0.25, 1.80 - 0.016 J)    J = jamming dBm",
           fill=grey, font=f)
    img.save(path)


def ensure_assets():
    """Create terrain files / satellite image / PCB schematic if they are missing."""
    need = [STATIC_DIR / n for n in ("ladakh_dem.npy", "ladakh_slope.npy", "ladakh_sat.jpg")]
    if not all(p.exists() for p in need):
        import terrain_loader
        terrain_loader.build(terrain_loader.parse_args(["--out", str(STATIC_DIR)]))
    pcb = STATIC_DIR / "pcb_schematic.png"
    if not pcb.exists():
        make_pcb_schematic(pcb)


# =========================================================================== #
# Terrain helpers
# =========================================================================== #
def bilinear(grid: np.ndarray, x: float, y: float) -> float:
    x = min(max(x, 0.0), GRID_N - 1.0)
    y = min(max(y, 0.0), GRID_N - 1.0)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, GRID_N - 1), min(y0 + 1, GRID_N - 1)
    fx, fy = x - x0, y - y0
    return float((grid[y0, x0] * (1 - fx) + grid[y0, x1] * fx) * (1 - fy)
                 + (grid[y1, x0] * (1 - fx) + grid[y1, x1] * fx) * fy)


# =========================================================================== #
# OpenCV ArUco HUD
# =========================================================================== #
class ArucoHUD:
    """
    Renders the UAV's nadir camera over the satellite image, with the UGV (and
    its roof ArUco pad) drawn at its true relative position, then runs a real
    ArUco detection on the rendered frame and overlays the lock.

    The frame is rendered at 520x520 for detection range, then shown at 260x260.
    """
    SENSOR, OUT = 520, 260
    FOV_DEG = 70.0
    MARKER_ID = 23
    MARKER_M, PAD_M = 1.8, 2.4          # black marker / white pad on the roof
    BODY_L, BODY_W = 4.2, 2.6           # UGV hull footprint (m)
    MIN_CAM_H = 6.0                     # lens min focus height (keeps the marker inside the detector's
                                        # adaptive-threshold window; closer than this the pad would fill the frame)

    def __init__(self, sat_path: Path):
        sat = cv2.imread(str(sat_path), cv2.IMREAD_COLOR)
        if sat is None:
            sat = np.full((512, 512, 3), (70, 90, 100), np.uint8)
        self.sat = sat
        self.sat_h, self.sat_w = sat.shape[:2]

        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_px = 225
        if hasattr(cv2.aruco, "generateImageMarker"):
            m = cv2.aruco.generateImageMarker(self.dict, self.MARKER_ID, marker_px)
        else:                                               # OpenCV < 4.7
            m = cv2.aruco.drawMarker(self.dict, self.MARKER_ID, marker_px)
        pad_px = int(round(marker_px * self.PAD_M / self.MARKER_M))
        pad = np.full((pad_px, pad_px), 255, np.uint8)
        o = (pad_px - marker_px) // 2
        pad[o:o + marker_px, o:o + marker_px] = m
        self.pad = pad

        params = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
                  else cv2.aruco.DetectorParameters_create())
        params.minMarkerPerimeterRate = 0.02
        params.adaptiveThreshWinSizeMax = 43
        params.adaptiveThreshWinSizeStep = 10
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.params = params
        self.detector = cv2.aruco.ArucoDetector(self.dict, params) if hasattr(cv2.aruco, "ArucoDetector") else None
        self.rng = np.random.default_rng(1)
        n = self.rng.normal(0, 2.5, (8, self.SENSOR, self.SENSOR, 1))
        self._noise = [np.repeat(np.clip(x, 0, None), 3, axis=2).astype(np.uint8) for x in n]

    # ---- detection (cv2.aruco.detectMarkers) ------------------------------ #
    def _detect(self, gray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)[:3]
        return cv2.aruco.detectMarkers(gray, self.dict, parameters=self.params)

    # ---- rendering -------------------------------------------------------- #
    def render(self, s: dict):
        """s: plain-float snapshot (see Session.snapshot). Returns (b64_jpeg, info)."""
        S, OUT = self.SENSOR, self.OUT
        uav = np.asarray(s["uav"], float)
        ugv = np.asarray(s["ugv"], float)
        fwd_u = np.asarray(s["ugv_fwd"], float)
        agl = float(s["agl"])
        th = math.radians(s["heading"])
        F = np.array([math.sin(th), -math.cos(th)])          # camera "up"   (x east, y south)
        R = np.array([math.cos(th), math.sin(th)])           # camera "right"
        cam_h = max(agl, self.MIN_CAM_H)
        scale = S / (2.0 * cam_h * math.tan(math.radians(self.FOV_DEG / 2)))   # px per metre
        c = S / 2.0

        # --- background: warp satellite image into camera frame ---
        sx, sy = self.sat_w / GRID_N / CELL_X_M, self.sat_h / GRID_N / CELL_Y_M
        Sm = np.array([[sx, 0], [0, sy]])
        Bm = np.stack([R, -F], axis=1)                       # (u,v) -> world directions
        lin = Sm @ Bm / scale
        off = Sm @ (uav - Bm @ np.array([c, c]) / scale) + np.array([0.5 * self.sat_w / GRID_N,
                                                                     0.5 * self.sat_h / GRID_N])
        M = np.hstack([lin, off[:, None]])
        frame = cv2.warpAffine(self.sat, M, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)
        frame = cv2.GaussianBlur(frame, (0, 0), 1.0)

        def to_px(p):
            d = np.asarray(p) - uav
            return np.array([c + scale * float(d @ R), c - scale * float(d @ F)])

        # --- UGV hull + roof pad ---
        ru = np.array([-fwd_u[1], fwd_u[0]])
        hull = np.array([ugv + a * fwd_u * self.BODY_L / 2 + b * ru * self.BODY_W / 2
                         for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        hull_px = np.array([to_px(p) for p in hull], np.int32)
        ugv_px = to_px(ugv)
        if all(abs(v) < 4 * S for v in ugv_px):
            cv2.fillConvexPoly(frame, hull_px, (52, 70, 60), cv2.LINE_AA)
            cv2.polylines(frame, [hull_px], True, (30, 40, 36), 2, cv2.LINE_AA)
            tp = int(round(self.PAD_M * scale))
            if 6 <= tp <= 2 * S:
                pad = cv2.resize(self.pad, (tp, tp), interpolation=cv2.INTER_AREA)
                pad3 = cv2.cvtColor(pad, cv2.COLOR_GRAY2BGR)
                h = tp / 2.0
                colA = np.array([float(ru @ R), -float(ru @ F)])        # pad +x axis in camera px
                colB = np.array([-float(fwd_u @ R), float(fwd_u @ F)])  # pad +y (down) axis
                A = np.stack([colA, colB], axis=1)
                A = np.hstack([A, (ugv_px - A @ np.array([h, h]))[:, None]])
                warped = cv2.warpAffine(pad3, A, (S, S), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                mask = cv2.warpAffine(np.full((tp, tp), 255, np.uint8), A, (S, S), flags=cv2.INTER_NEAREST)
                frame[mask > 0] = warped[mask > 0]

        frame = cv2.add(frame, self._noise[self.rng.integers(0, 8)])      # sensor noise (cheap, pre-generated)

        # --- ArUco detection on the rendered sensor frame ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        hud = cv2.resize(frame, (OUT, OUT), interpolation=cv2.INTER_AREA)
        k = OUT / S
        cx0 = OUT / 2.0
        GREEN, CYAN, AMBER, WHITE = (60, 255, 90), (255, 220, 40), (40, 180, 255), (235, 235, 235)

        info = {"detected": False, "id": None, "dx_px": None, "dy_px": None,
                "dx_m": None, "dy_m": None, "corners": None, "camera_agl_m": round(cam_h, 1),
                "px_per_m": round(scale * k, 2)}
        if ids is not None and self.MARKER_ID in ids.flatten():
            i = int(np.where(ids.flatten() == self.MARKER_ID)[0][0])
            quad = corners[i].reshape(4, 2) * k
            ctr = quad.mean(axis=0)
            dx, dy = float(ctr[0] - cx0), float(ctr[1] - cx0)
            cv2.polylines(hud, [quad.astype(np.int32)], True, GREEN, 2, cv2.LINE_AA)
            x0, y0 = quad.min(axis=0).astype(int)
            x1, y1 = quad.max(axis=0).astype(int)
            cv2.rectangle(hud, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), GREEN, 1, cv2.LINE_AA)
            cv2.circle(hud, tuple(ctr.astype(int)), 3, GREEN, -1, cv2.LINE_AA)
            cv2.arrowedLine(hud, (int(cx0), int(cx0)), tuple(ctr.astype(int)), CYAN, 2, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(hud, f"ID {self.MARKER_ID} LOCK", (max(2, x0 - 4), max(24, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREEN, 1, cv2.LINE_AA)
            cv2.putText(hud, f"dx={dx:+.0f} dy={dy:+.0f} px", (6, OUT - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)
            ppm = scale * k
            info.update(detected=True, id=self.MARKER_ID, dx_px=round(dx, 1), dy_px=round(dy, 1),
                        dx_m=round(dx / ppm, 2), dy_m=round(-dy / ppm, 2),     # +dy_m = marker ahead
                        corners=[[round(float(a), 1), round(float(b), 1)] for a, b in quad])
        else:
            cv2.putText(hud, "SEARCHING: NO MARKER", (6, OUT - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, AMBER, 1,
                        cv2.LINE_AA)
            v = ugv_px - c                                    # edge cue toward the UGV
            if np.linalg.norm(v) > 1:
                u = v / np.linalg.norm(v)
                p1, p0 = cx0 + u * (OUT * 0.46), cx0 + u * (OUT * 0.34)
                cv2.arrowedLine(hud, tuple(p0.astype(int)), tuple(p1.astype(int)), AMBER, 2, cv2.LINE_AA,
                                tipLength=0.4)

        # crosshair + corner brackets + header
        ci = int(cx0)
        cv2.line(hud, (ci - 12, ci), (ci - 4, ci), WHITE, 1, cv2.LINE_AA)
        cv2.line(hud, (ci + 4, ci), (ci + 12, ci), WHITE, 1, cv2.LINE_AA)
        cv2.line(hud, (ci, ci - 12), (ci, ci - 4), WHITE, 1, cv2.LINE_AA)
        cv2.line(hud, (ci, ci + 4), (ci, ci + 12), WHITE, 1, cv2.LINE_AA)
        for (bx, by, sx_, sy_) in ((2, 2, 1, 1), (OUT - 3, 2, -1, 1), (2, OUT - 3, 1, -1), (OUT - 3, OUT - 3, -1, -1)):
            cv2.line(hud, (bx, by), (bx + 14 * sx_, by), GREEN, 1)
            cv2.line(hud, (bx, by), (bx, by + 14 * sy_), GREEN, 1)
        label = {"NOMINAL_SCOUT": "SCOUT", "VIO_HOLD_ISOLATED": "VIO HOLD",
                 "UGV_MOVING_DECK_LANDING": "DECK LANDING"}.get(s["state"], s["state"][:14])
        cv2.putText(hud, f"AGL {agl:4.1f}m  {label}", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, WHITE, 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", hud, [cv2.IMWRITE_JPEG_QUALITY, 72])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
        return b64, info


# =========================================================================== #
# Session: engine + attack state + telemetry assembly
# =========================================================================== #
class AttackRequest(BaseModel):
    attack_mode: Optional[Literal["NONE", "INSTANT_SPOOF", "GRADUAL_MEACONING"]] = None
    jamming_dbm: Optional[float] = Field(default=None, ge=0, le=100)
    force_recovery: Optional[bool] = None


def _r(v, n=3):
    return None if v is None else round(float(v), n)


class Session:
    def __init__(self):
        self.engine = SimulationEngine(static_dir=str(STATIC_DIR))
        self.attack_mode = "NONE"
        self.jamming_dbm = 0.0
        self.tick_count = 0
        self.last_tel = self.engine.tick()
        self.tick_count += 1

    def tick(self):
        self.last_tel = self.engine.tick()
        self.tick_count += 1

    # ---- control ---------------------------------------------------------- #
    def apply_attack(self, req: AttackRequest) -> dict:
        e = self.engine
        if req.jamming_dbm is not None:
            self.jamming_dbm = float(req.jamming_dbm)
            e.set_jamming(self.jamming_dbm)
        if req.attack_mode is not None and req.attack_mode != self.attack_mode:
            self.attack_mode = req.attack_mode
            if req.attack_mode == "NONE":
                e.stop_spoof()
            elif req.attack_mode == "INSTANT_SPOOF":
                e.start_spoof(rate_mps=0.0, direction_deg=SPOOF_DIRECTION_DEG, max_offset_m=SPOOF_MAX_OFFSET_M)
                e.spoof_offset = e.spoof_dir * INSTANT_SPOOF_JUMP_M
            else:  # GRADUAL_MEACONING
                e.stop_spoof()
                e.start_spoof(rate_mps=MEACONING_RATE_MPS, direction_deg=SPOOF_DIRECTION_DEG,
                              max_offset_m=SPOOF_MAX_OFFSET_M)
        forced = False
        if req.force_recovery:
            e.emergency_recovery()
            forced = True
        return {"attack_mode": self.attack_mode, "jamming_dbm": self.jamming_dbm,
                "force_recovery_triggered": forced, "fsm_state": e.fsm.state,
                "detector_status": self.last_tel["detector"]["status"]}

    def reset(self):
        self.engine = SimulationEngine(static_dir=str(STATIC_DIR))
        self.attack_mode, self.jamming_dbm = "NONE", 0.0
        self.last_tel = self.engine.tick()
        self.tick_count = 1

    # ---- init payload ----------------------------------------------------- #
    def init_map(self) -> dict:
        p = self.engine.planner
        path = lambda pp: None if pp is None else [[int(x), int(y)] for x, y in pp]
        return {
            "grid_size": GRID_N,
            "elevation": np.round(self.engine.dem, 1).tolist(),
            "slope": np.round(self.engine.slope, 1).tolist(),
            "naive_path": path(p.naive_path),
            "resilient_path": path(p.resilient_path),
            "start": list(p.start), "base_camp": list(p.goal),
            "gps_mapping": {"lat0": se.LAT0, "lon0": se.LON0, "step_deg": se.DEG_STEP,
                            "formula": "lat = 34.2650 + (60 - y) * 0.00035 ; lon = 77.5900 + x * 0.00035"},
            "cell_size_m": {"x": CELL_X_M, "y": CELL_Y_M},
            "slope_limit_deg": p.slope_limit, "rollover_deg": p.rollover_deg,
            "naive_hazard": p.first_rollover_hazard(p.naive_path) if p.naive_path else None,
            "planner_summary": p.summary(),
            "satellite_url": "/static/ladakh_sat.jpg", "pcb_schematic_url": "/static/pcb_schematic.png",
        }

    # ---- streaming snapshot ------------------------------------------------ #
    def snapshot(self) -> dict:
        """Everything for one telemetry frame, as plain python types (safe to hand to a thread)."""
        e, tel = self.engine, self.last_tel
        dem = e.dem
        ugv, uav = tel["ugv"], tel["uav"]
        gps = e.last_gps_fix
        gx, gy = gps[0] / CELL_X_M, gps[1] / CELL_Y_M
        glat, glon = e.planner.grid_to_gps(gx, gy)
        uav_ground = bilinear(dem, uav["x_true"], uav["y_true"])
        ugv_cell = [int(round(min(max(ugv["x"], 0), GRID_N - 1))), int(round(min(max(ugv["y"], 0), GRID_N - 1)))]
        det = tel["detector"]
        payload = {
            "tick": self.tick_count,
            "t": tel["t"],
            "server_time": round(time.time(), 3),
            "attack_mode": self.attack_mode,
            "jamming_dbm": self.jamming_dbm,
            "ugv": {
                "grid": {"x": _r(ugv["x"]), "y": _r(ugv["y"])}, "cell": ugv_cell,
                "lat": _r(ugv["lat_lon"][0], 6), "lon": _r(ugv["lat_lon"][1], 6),
                "asl_m": _r(bilinear(dem, ugv["x"], ugv["y"]), 1), "slope_deg": _r(ugv["slope_deg"], 1),
                "mode": ugv["mode"], "progress_pct": _r(ugv["progress_pct"], 1),
                "halted": ugv["halted"], "alert": ugv["alert"],
            },
            "uav": {
                "grid": {"x": _r(uav["x_true"]), "y": _r(uav["y_true"])},
                "nav_grid": {"x": _r(uav["x_nav"]), "y": _r(uav["y_nav"])},
                "lat": _r(uav["lat_lon_true"][0], 6), "lon": _r(uav["lat_lon_true"][1], 6),
                "asl_m": _r(uav_ground + uav["agl_m"], 1), "ground_m": _r(uav_ground, 1),
                "agl_m": _r(uav["agl_m"], 1), "heading_deg": _r(uav["heading_deg"], 1),
                "landed": uav["landed"], "nav_error_m": _r(uav["nav_error_m"], 2),
                "dist_to_ugv_m": _r(uav["dist_to_ugv_m"], 1), "nav_source": uav["nav_source"],
                "gps_reported": {"grid": {"x": _r(gx), "y": _r(gy)}, "lat": _r(glat, 6), "lon": _r(glon, 6),
                                 "offset_m": _r(tel["spoof"]["offset_m"], 1)},
            },
            "r_KF": _r(det["r_kf_m"], 3),
            "r_TERCOM": _r(det["r_tercom_m"], 3),
            "detector": {"status": det["status"], "tripped": bool(det["tripped"]),
                         "confirmed": bool(det["confirmed"]), "trigger": det["trigger"],
                         "gps_gain_locked": bool(det["gps_gain_locked"]), "K_norm": _r(det["K_norm"], 4),
                         "thresholds": {"r_KF_m": se.KF_THRESH_M, "r_TERCOM_m": se.TERCOM_THRESH_M}},
            "fsm": tel["fsm"],
            "pcb": {"lna_agc_volts": tel["pcb"]["lna_agc_volts"], "cn0_db_hz": tel["pcb"]["cn0_db_hz"],
                    "gpio_pa14_mux": tel["pcb"]["gpio_pa14_mux"],
                    "saw_notch_active": tel["pcb"]["saw_notch_active"], "jamming_dbm": tel["pcb"]["jamming_dbm"]},
        }
        hud_in = {"uav": e.uav_pos.copy(), "ugv": e.ugv_pos.copy(), "ugv_fwd": e.ugv_heading_vec(),
                  "heading": float(e.heading_deg), "agl": float(e.uav_agl), "state": tel["fsm"]["state"]}
        return {"payload": payload, "hud": hud_in}


# =========================================================================== #
# Hub: engine loop, stream loop, websocket clients
# =========================================================================== #
class Hub:
    def __init__(self):
        self.clients = set()

    async def broadcast(self, text: str):
        if not self.clients:
            return
        clients = list(self.clients)

        async def send(ws):
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=0.5)
                return None
            except Exception:                       # noqa: BLE001  (slow / closed client)
                return ws

        for dead in await asyncio.gather(*(send(c) for c in clients)):
            if dead is not None:
                self.clients.discard(dead)


async def _paced(period: float, step):
    """Run step() every `period` seconds without drift; skip (not queue) missed frames."""
    loop = asyncio.get_running_loop()
    nxt = loop.time()
    while True:
        await step()
        nxt += period
        delay = nxt - loop.time()
        if delay < 0:
            nxt, delay = loop.time(), 0
        await asyncio.sleep(delay)


async def engine_loop(sess: Session):
    async def step():
        sess.tick()
    await _paced(TICK_S, step)


async def stream_loop(sess: Session, hub: Hub, hud: ArucoHUD):
    async def step():
        if not hub.clients:
            return
        snap = sess.snapshot()
        b64, aruco = await asyncio.to_thread(hud.render, snap["hud"])
        msg = snap["payload"]
        msg["aruco"] = aruco
        msg["aruco_hud_b64"] = b64
        await hub.broadcast(json.dumps(msg))
    await _paced(1.0 / STREAM_HZ, step)


# =========================================================================== #
# App
# =========================================================================== #
session: Optional[Session] = None
hub = Hub()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    await asyncio.to_thread(ensure_assets)
    session = Session()
    hud = ArucoHUD(STATIC_DIR / "ladakh_sat.jpg")
    tasks = [asyncio.create_task(engine_loop(session)), asyncio.create_task(stream_loop(session, hub, hud))]
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Ladakh UGV/UAV Resilience Simulator", lifespan=lifespan)


def _need_session() -> Session:
    if session is None:
        raise HTTPException(503, "simulation not ready")
    return session


@app.get("/api/init_map")
async def init_map():
    return _need_session().init_map()


@app.post("/api/attack")
async def attack(req: AttackRequest):
    return _need_session().apply_attack(req)


@app.post("/api/reset")
async def reset():
    _need_session().reset()
    return {"status": "reset"}


@app.post("/api/reposition")
async def api_reposition(data: dict):
    pct = data.get("progress_pct", None)
    target_grid = data.get("target_grid", None)
    
    # If clicked on the map, calculate progress percentage along resilient path
    if target_grid is not None:
        tx, ty = target_grid
        path = session.engine.planner.resilient_path
        if path:
            dists = [math.hypot(pt[0] - tx, pt[1] - ty) for pt in path]
            closest_idx = int(np.argmin(dists))
            pct = (closest_idx / len(path)) * 100.0
            
    if pct is None:
        pct = 0.0
        
    session.engine.reposition(pct)
    return {"status": "success", "progress_pct": pct}


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    try:
        while True:
            await ws.receive_text()                 # keeps the socket open; client messages ignored
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


@app.get("/", include_in_schema=False)
async def index():
    p = STATIC_DIR / "index.html"
    if p.exists():
        return FileResponse(p)
    return HTMLResponse("<h3>Simulator API is running.</h3><p>Put <code>index.html</code> in "
                        "<code>static/</code>. See <a href='/docs'>/docs</a>.</p>")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)