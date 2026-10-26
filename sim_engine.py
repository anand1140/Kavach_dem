#!/usr/bin/env python3
"""
sim_engine.py - Core physics, sensor fusion & PCB engine (100 ms tick).

Loads static/ladakh_dem.npy + static/ladakh_slope.npy (from terrain_loader.py)
and runs four coupled models:

  1. RealTerrainPlanner          naive vs slope-weighted A* UGV routes + GPS mapping
  2. DualLayerSpoofingDetector   Kalman (VIO vs GPS) + TERCOM elevation check
  3. AutonomousRecoveryFSM       NOMINAL_SCOUT -> VIO_HOLD_ISOLATED -> UGV_MOVING_DECK_LANDING
  4. VirtualPCBFrontEnd          jamming dBm -> C/N0, LNA AGC, SAW notch, GPIO mux

SimulationEngine wires them together and exposes tick() -> JSON-safe telemetry.

Conventions
  * Grid index (x, y): x = column (east), y = row (south). Arrays are [y, x].
  * Local metric frame: X_m = x * CELL_X_M (east), Y_m = y * CELL_Y_M (south).
  * GPS:  lat = 34.2650 + (60 - y) * 0.00035,  lon = 77.5900 + x * 0.00035

Run `python sim_engine.py` for a scripted demo.
"""

import heapq
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TICK_S = 0.1                       # 100 ms
GRID_N = 60
START = (5, 5)
BASE_CAMP = (54, 52)

LAT0, LON0, DEG_STEP = 34.2650, 77.5900, 0.00035
_LAT_MID = LAT0 + (GRID_N / 2) * DEG_STEP
CELL_X_M = DEG_STEP * 111_320.0 * math.cos(math.radians(_LAT_MID))   # ~32.2 m
CELL_Y_M = DEG_STEP * 110_920.0                                      # ~38.8 m

SLOPE_LIMIT_DEG = 26.0             # resilient planner: impassable above this
SLOPE_SOFT_DEG = 10.0
SLOPE_PENALTY = 4.5
ROLLOVER_DEG = 30.0                # UGV CRITICAL ROLLOVER HAZARD threshold

KF_THRESH_M = 6.0
TERCOM_THRESH_M = 22.0
ISOLATION_TO_LANDING_S = 2.5
SCOUT_AHEAD_CELLS = 4
SCOUT_AGL_M = 60.0


# --------------------------------------------------------------------------- #
# Terrain loading
# --------------------------------------------------------------------------- #
def load_terrain(static_dir: str = "static") -> Tuple[np.ndarray, np.ndarray]:
    dem = np.load(os.path.join(static_dir, "ladakh_dem.npy")).astype(np.float64)
    slope = np.load(os.path.join(static_dir, "ladakh_slope.npy")).astype(np.float64)
    if dem.shape != (GRID_N, GRID_N) or slope.shape != dem.shape:
        raise ValueError(f"expected {(GRID_N, GRID_N)} grids, got {dem.shape} / {slope.shape}")
    return dem, slope


def _clamp_cell(v: float) -> int:
    return int(min(GRID_N - 1, max(0, round(v))))


# =========================================================================== #
# 1. RealTerrainPlanner
# =========================================================================== #
class RealTerrainPlanner:
    """8-connected A* over the 60x60 grid; naive (Euclid) vs slope-weighted."""

    NEIGHBOURS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    def __init__(self, dem: np.ndarray, slope: np.ndarray,
                 start=START, goal=BASE_CAMP,
                 slope_limit=SLOPE_LIMIT_DEG, rollover_deg=ROLLOVER_DEG):
        self.dem, self.slope = dem, slope
        self._s = slope.tolist()
        self.start, self.goal = tuple(start), tuple(goal)
        self.slope_limit, self.rollover_deg = slope_limit, rollover_deg
        self.naive_path: Optional[List[Tuple[int, int]]] = None
        self.resilient_path: Optional[List[Tuple[int, int]]] = None
        self.naive_cost = self.resilient_cost = None

    # ---- coordinates ---------------------------------------------------- #
    @staticmethod
    def grid_to_gps(x: float, y: float) -> Tuple[float, float]:
        return LAT0 + (GRID_N - y) * DEG_STEP, LON0 + x * DEG_STEP

    def slope_at(self, x: float, y: float) -> float:
        return float(self.slope[_clamp_cell(y), _clamp_cell(x)])

    def elevation_at(self, x: float, y: float) -> float:
        return float(self.dem[_clamp_cell(y), _clamp_cell(x)])

    # ---- step costs ----------------------------------------------------- #
    def _naive_cost(self, u, v, d):
        return d

    def _resilient_cost(self, u, v, d):
        sv = self._s[v[1]][v[0]]
        if sv > self.slope_limit:
            return math.inf
        return d + SLOPE_PENALTY * max(0.0, sv - SLOPE_SOFT_DEG) ** 2

    def _resilient_soft_cost(self, u, v, d):
        """Fallback when a continuous mountain ridge exceeds slope_limit everywhere:
        heavily penalizes steep cells so A* routes through the lowest-angle saddle."""
        sv = self._s[v[1]][v[0]]
        base = d + SLOPE_PENALTY * max(0.0, sv - SLOPE_SOFT_DEG) ** 2
        if sv > self.slope_limit:
            base += 150.0 * (sv - self.slope_limit) ** 2
        return base

    # ---- A* ------------------------------------------------------------- #
    def _astar(self, cost_fn):
        goal, start = self.goal, self.start
        h = lambda p: math.hypot(p[0] - goal[0], p[1] - goal[1])   # admissible
        g = {start: 0.0}
        came: Dict[Tuple[int, int], Tuple[int, int]] = {}
        heap = [(h(start), 0, start)]
        counter, closed = 0, set()
        while heap:
            _, _, u = heapq.heappop(heap)
            if u in closed:
                continue
            if u == goal:
                path = [u]
                while u in came:
                    u = came[u]
                    path.append(u)
                return path[::-1], g[goal]
            closed.add(u)
            for dx, dy in self.NEIGHBOURS:
                v = (u[0] + dx, u[1] + dy)
                if not (0 <= v[0] < GRID_N and 0 <= v[1] < GRID_N) or v in closed:
                    continue
                c = cost_fn(u, v, math.hypot(dx, dy))
                if math.isinf(c):
                    continue
                ng = g[u] + c
                if ng < g.get(v, math.inf):
                    g[v] = ng
                    came[v] = u
                    counter += 1
                    heapq.heappush(heap, (ng + h(v), counter, v))
        return None, math.inf

    def plan(self):
        self.naive_path, self.naive_cost = self._astar(self._naive_cost)
        self.resilient_path, self.resilient_cost = self._astar(self._resilient_cost)
        # If strict 26 deg cutoff is blocked by a continuous ridge, relax or use soft penalty
        if self.resilient_path is None:
            for lim in (28.0, 30.0, 33.0, 36.0):
                self.slope_limit = lim
                self.resilient_path, self.resilient_cost = self._astar(self._resilient_cost)
                if self.resilient_path is not None:
                    break
        if self.resilient_path is None:
            self.resilient_path, self.resilient_cost = self._astar(self._resilient_soft_cost)
        return self.naive_path, self.resilient_path
    # ---- analysis ------------------------------------------------------- #
    def path_slopes(self, path) -> List[float]:
        return [self.slope_at(x, y) for x, y in path]

    def first_rollover_hazard(self, path) -> Optional[dict]:
        """First path cell whose slope exceeds the rollover threshold."""
        for i, (x, y) in enumerate(path):
            s = self.slope_at(x, y)
            if s >= self.rollover_deg:
                lat, lon = self.grid_to_gps(x, y)
                return {"alert": "CRITICAL ROLLOVER HAZARD", "index": i, "cell": (x, y),
                        "slope_deg": round(s, 1), "elevation_m": round(self.elevation_at(x, y), 1),
                        "lat": lat, "lon": lon}
        return None

    def summary(self) -> dict:
        def stats(p, cost):
            if p is None:
                return {"found": False}
            sl = self.path_slopes(p)
            return {"found": True, "cells": len(p), "cost": round(cost, 1),
                    "max_slope_deg": round(max(sl), 1), "mean_slope_deg": round(float(np.mean(sl)), 1)}
        return {"naive": stats(self.naive_path, self.naive_cost),
                "resilient": stats(self.resilient_path, self.resilient_cost),
                "naive_hazard": self.first_rollover_hazard(self.naive_path) if self.naive_path else None}


# =========================================================================== #
# 4. VirtualPCBFrontEnd
# =========================================================================== #
class VirtualPCBFrontEnd:
    """Anti-jamming RF front-end emulation driven by the UI jamming slider."""

    def __init__(self):
        self.state = self.update(0.0)

    def update(self, jamming_dbm: float) -> dict:
        j = min(100.0, max(0.0, float(jamming_dbm)))
        cn0 = max(12.0, 47.0 - 0.38 * j)
        agc = max(0.25, 1.80 - 0.016 * j)
        notch = agc < 0.85
        self.state = {
            "jamming_dbm": j,
            "cn0_db_hz": round(cn0, 2),
            "lna_agc_volts": round(agc, 3),
            "saw_notch_active": notch,
            "gpio_pa14_mux": "915MHz_FHSS_BACKUP" if notch else "2.4GHz_PRIMARY",
        }
        return self.state


# =========================================================================== #
# 2. DualLayerSpoofingDetector
# =========================================================================== #
class DualLayerSpoofingDetector:
    """
    Layer 1: 2-D linear Kalman filter, state [x, y, vx, vy] (metres, m/s).
             Prediction is driven by VIO velocity; GPS is the measurement.
             r_KF = || z_gps - H x_{k|k-1} ||
    Layer 2: TERCOM. h_meas = h_baro - d_agl compared with DEM at the *GPS* cell.
             r_TERCOM = | h_meas - DEM[y_gps, x_gps] |
    Trigger: r_KF > 6 m OR r_TERCOM > 22 m  ->  CONFIRMED SPOOFING ATTACK,
             Kalman GPS gain K = 0, navigation locked on VIO.

    A tripped tick is also *gated* (GPS not fused that tick), so a spoofed fix
    never leaks into the state while confirmation is pending.
    """

    def __init__(self, dem: np.ndarray, dt: float = TICK_S,
                 kf_thresh=KF_THRESH_M, tercom_thresh=TERCOM_THRESH_M,
                 confirm_ticks: int = 2, gps_sigma_nominal: float = 1.0):
        self.dem, self.dt = dem, dt
        self.kf_thresh, self.tercom_thresh = kf_thresh, tercom_thresh
        self.confirm_ticks = max(1, confirm_ticks)
        self.gps_sigma_nominal = gps_sigma_nominal
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.q_pos = 0.03 ** 2          # position process noise (VIO is trusted, so small)
        self.sigma_v_vio = 0.05         # VIO velocity noise (m/s)
        self.Q = np.diag([self.q_pos, self.q_pos, 0.0, 0.0])
        self.reset(np.zeros(2))

    def reset(self, pos_m, vel=(0.0, 0.0)):
        self.x = np.array([pos_m[0], pos_m[1], vel[0], vel[1]], dtype=float)
        self.P = np.diag([4.0, 4.0, 0.25, 0.25])
        self.gps_gain_locked = False
        self.confirmed = False
        self._streak = 0
        self.r_kf = self.r_tercom = 0.0
        self.K_norm = 0.0
        self.trigger = None

    @property
    def pos(self):
        return self.x[:2].copy()

    @property
    def vel(self):
        return self.x[2:].copy()

    def dem_at_m(self, p) -> float:
        return float(self.dem[_clamp_cell(p[1] / CELL_Y_M), _clamp_cell(p[0] / CELL_X_M)])

    def step(self, v_vio, z_gps, h_meas, gps_sigma: float) -> dict:
        dt = self.dt
        v_vio = np.asarray(v_vio, dtype=float)

        # ---- predict: position propagated by VIO velocity ----
        x_pred = self.F @ self.x
        x_pred[:2] = self.x[:2] + v_vio * dt
        x_pred[2:] = v_vio
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # velocity is a *measured input* (VIO), not an estimated state: reset its block
        P_pred[2:, :] = 0.0
        P_pred[:, 2:] = 0.0
        P_pred[2:, 2:] = np.eye(2) * self.sigma_v_vio ** 2
        P_pred[:2, :2] += np.eye(2) * (self.sigma_v_vio * dt) ** 2

        # ---- layer 1: KF innovation residual ----
        innov = np.asarray(z_gps, dtype=float) - self.H @ x_pred
        self.r_kf = float(np.linalg.norm(innov))

        # ---- layer 2: TERCOM ----
        h_gps_dem = self.dem_at_m(z_gps)
        self.r_tercom = abs(h_meas - h_gps_dem)

        # ---- trigger / confirmation ----
        tripped_kf = self.r_kf > self.kf_thresh
        tripped_tc = self.r_tercom > self.tercom_thresh
        tripped = tripped_kf or tripped_tc
        self._streak = self._streak + 1 if tripped else 0
        if tripped and not self.confirmed and self._streak >= self.confirm_ticks:
            self.confirmed = True
            self.gps_gain_locked = True
            self.trigger = ("KF+TERCOM" if tripped_kf and tripped_tc else "KF" if tripped_kf else "TERCOM")

        # ---- update: K = 0 when locked or this tick is gated ----
        if self.gps_gain_locked or tripped:
            K = np.zeros((4, 2))
            self.x, self.P = x_pred, P_pred
        else:
            R = np.eye(2) * gps_sigma ** 2
            S = self.H @ P_pred @ self.H.T + R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            self.x = x_pred + K @ innov
            self.P = (np.eye(4) - K @ self.H) @ P_pred
        self.K_norm = float(np.linalg.norm(K))

        return {"r_kf_m": self.r_kf, "r_tercom_m": self.r_tercom, "h_gps_dem_m": h_gps_dem,
                "tripped": tripped, "confirmed": self.confirmed, "trigger": self.trigger,
                "gps_gain_locked": self.gps_gain_locked, "K_norm": self.K_norm,
                "status": "CONFIRMED SPOOFING ATTACK" if self.confirmed else "NOMINAL"}


# =========================================================================== #
# 3. AutonomousRecoveryFSM
# =========================================================================== #
class AutonomousRecoveryFSM:
    NOMINAL_SCOUT = "NOMINAL_SCOUT"
    VIO_HOLD_ISOLATED = "VIO_HOLD_ISOLATED"
    UGV_MOVING_DECK_LANDING = "UGV_MOVING_DECK_LANDING"

    def __init__(self, isolation_s: float = ISOLATION_TO_LANDING_S):
        self.isolation_s = isolation_s
        self.reset()

    def reset(self):
        self.state = self.NOMINAL_SCOUT
        self.t_isolated: Optional[float] = None
        self.reason = None

    def update(self, t: float, spoof_confirmed: bool) -> str:
        if self.state == self.NOMINAL_SCOUT and spoof_confirmed:
            self.state, self.t_isolated, self.reason = self.VIO_HOLD_ISOLATED, t, "SPOOFING_DETECTED"
        elif (self.state == self.VIO_HOLD_ISOLATED
              and t - self.t_isolated >= self.isolation_s - 1e-9):
            self.state = self.UGV_MOVING_DECK_LANDING
        return self.state

    def emergency_recovery(self, t: float):
        """UI 'Emergency Recovery' button: jump straight to landing."""
        if self.state != self.UGV_MOVING_DECK_LANDING:
            self.state, self.reason = self.UGV_MOVING_DECK_LANDING, "MANUAL_EMERGENCY_RECOVERY"
            if self.t_isolated is None:
                self.t_isolated = t

    def reposition(self, progress_pct: float = 0.0):
        """Teleports the UGV and UAV to a specific progress percentage (0 to 100) along the route."""
        self.ugv_s = max(0.0, min(100.0, progress_pct)) / 100.0 * self._ugv_cum[-1]
        self.ugv_halted = False
        self.ugv_alert = None
        # Reset UAV to hover above the new UGV position
        ugv = self.ugv_pos
        self.uav_pos = ugv + np.array([0.0, -40.0])
        self.uav_vel = np.zeros(2)
        self.uav_agl = SCOUT_AGL_M
        self.uav_landed = False
        self.detector.reset(self.uav_pos)
        self.fsm.reset()

    @property
    def gps_severed(self) -> bool:
        return self.state != self.NOMINAL_SCOUT and self.reason == "SPOOFING_DETECTED"


# =========================================================================== #
# Orchestrator
# =========================================================================== #
class SimulationEngine:
    """UGV + UAV kinematics, sensor models, and the four classes above."""

    def __init__(self, static_dir="static", seed: Optional[int] = 7,
                 ugv_speed_mps: float = 3.0, ugv_mode: str = "resilient",
                 dem: Optional[np.ndarray] = None, slope: Optional[np.ndarray] = None):
        if dem is None or slope is None:
            dem, slope = load_terrain(static_dir)
        self.dem, self.slope = dem, slope
        self.rng = np.random.default_rng(seed)
        self.dt = TICK_S

        self.planner = RealTerrainPlanner(dem, slope)
        self.planner.plan()
        self.pcb = VirtualPCBFrontEnd()
        self.detector = DualLayerSpoofingDetector(dem, self.dt)
        self.fsm = AutonomousRecoveryFSM()

        self.ugv_speed = ugv_speed_mps
        self.ugv_mode = ugv_mode
        self.ugv_path = self.planner.naive_path if ugv_mode == "naive" else self.planner.resilient_path
        if not self.ugv_path:
            raise RuntimeError(f"no {ugv_mode} path from {START} to {BASE_CAMP}: "
                               f"terrain has no route under the slope limit")
        self._ugv_pts = np.array([(x * CELL_X_M, y * CELL_Y_M) for x, y in self.ugv_path])
        seg = np.hypot(*np.diff(self._ugv_pts, axis=0).T)
        self._ugv_cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.ugv_s = 0.0
        self.ugv_alert: Optional[dict] = None
        self.ugv_halted = False

        # UAV truth
        self.t = 0.0
        self.uav_pos = self._path_point(SCOUT_AHEAD_CELLS)
        self.uav_vel = np.zeros(2)
        self.uav_agl = SCOUT_AGL_M
        self.uav_landed = False
        self.heading_deg = 0.0
        self.vio_bias = np.zeros(2)
        self.detector.reset(self.uav_pos)

        # attack / jamming inputs
        self.jamming_dbm = 0.0
        self.spoof_active = False
        self.spoof_rate = 6.0
        self.spoof_dir = np.array([math.cos(math.radians(40)), math.sin(math.radians(40))])
        self.spoof_offset = np.zeros(2)
        self.spoof_max_m = 400.0
        self.couple_jamming_to_gps = False   # True: low C/N0 inflates GPS noise (jamming can then trip the spoof detector)
        self._last = {}
        self.last_gps_fix = self.uav_pos.copy()     # most recent (possibly spoofed) GPS fix, metres

    # ---- UI inputs ------------------------------------------------------ #
    def set_jamming(self, dbm: float):
        self.jamming_dbm = float(dbm)

    def start_spoof(self, rate_mps: float = 6.0, direction_deg: float = 40.0, max_offset_m: float = 400.0):
        self.spoof_active, self.spoof_rate, self.spoof_max_m = True, rate_mps, max_offset_m
        a = math.radians(direction_deg)
        self.spoof_dir = np.array([math.cos(a), math.sin(a)])

    def stop_spoof(self):
        self.spoof_active = False
        self.spoof_offset = np.zeros(2)

    def emergency_recovery(self):
        """UI 'Emergency Recovery' button: jump straight to landing."""
        self.fsm.emergency_recovery(self.t)

    def reposition(self, progress_pct: float = 0.0):
        """Teleports the UGV and UAV to a specific progress percentage (0 to 100) along the route."""
        self.ugv_s = max(0.0, min(100.0, progress_pct)) / 100.0 * self._ugv_cum[-1]
        self.ugv_halted = False
        self.ugv_alert = None
        # Reset UAV to hover above the new UGV position
        ugv = self.ugv_pos
        self.uav_pos = ugv + np.array([0.0, -40.0])
        self.uav_vel = np.zeros(2)
        self.uav_agl = SCOUT_AGL_M
        self.uav_landed = False
        self.detector.reset(self.uav_pos)
        self.fsm.reset()

    # ---- helpers -------------------------------------------------------- #
    def _path_point(self, idx_float: float) -> np.ndarray:
        idx_float = min(max(idx_float, 0.0), len(self._ugv_pts) - 1.0)
        i = int(idx_float)
        j = min(i + 1, len(self._ugv_pts) - 1)
        return self._ugv_pts[i] + (self._ugv_pts[j] - self._ugv_pts[i]) * (idx_float - i)

    def _ugv_idx_float(self) -> float:
        return float(np.interp(self.ugv_s, self._ugv_cum, np.arange(len(self._ugv_cum))))

    @property
    def ugv_pos(self) -> np.ndarray:
        return self._path_point(self._ugv_idx_float())

    def ugv_heading_vec(self) -> np.ndarray:
        """Unit vector (x east, y south) of the UGV's direction of travel."""
        i = self._ugv_idx_float()
        a = self._path_point(i)
        v = self._path_point(i + 0.5) - a
        if np.linalg.norm(v) < 1e-6:
            v = a - self._path_point(i - 0.5)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([0.0, -1.0])

    @staticmethod
    def _to_cell(p) -> Tuple[int, int]:
        return _clamp_cell(p[0] / CELL_X_M), _clamp_cell(p[1] / CELL_Y_M)

    def gps_sigma(self) -> float:
        if not self.couple_jamming_to_gps:
            return self.detector.gps_sigma_nominal
        cn0 = self.pcb.state["cn0_db_hz"]
        return self.detector.gps_sigma_nominal * (1.0 + 0.8 * max(0.0, 40.0 - cn0) / 28.0)

    # ---- one 100 ms tick ------------------------------------------------ #
    def tick(self) -> dict:
        dt = self.dt
        self.t += dt

        # PCB
        pcb = self.pcb.update(self.jamming_dbm)

        # UGV motion along its path; rollover check
        if not self.ugv_halted:
            self.ugv_s = min(self.ugv_s + self.ugv_speed * dt, self._ugv_cum[-1])
            cx, cy = self._to_cell(self.ugv_pos)
            sl = float(self.slope[cy, cx])
            if sl >= self.planner.rollover_deg and self.ugv_alert is None:
                lat, lon = self.planner.grid_to_gps(cx, cy)
                self.ugv_alert = {"alert": "CRITICAL ROLLOVER HAZARD", "cell": (cx, cy),
                                  "slope_deg": round(sl, 1), "t": round(self.t, 1), "lat": lat, "lon": lon}
                self.ugv_halted = True
        ugv = self.ugv_pos

        # spoof injection
        if self.spoof_active:
            n = np.linalg.norm(self.spoof_offset)
            if n < self.spoof_max_m:
                self.spoof_offset = self.spoof_offset + self.spoof_dir * self.spoof_rate * dt

        # ---- sensors ----
        sigma = self.gps_sigma()
        z_gps = self.uav_pos + self.spoof_offset + self.rng.normal(0, sigma, 2)
        self.last_gps_fix = z_gps
        self.vio_bias += self.rng.normal(0, 0.01 * math.sqrt(dt), 2)
        v_vio = self.uav_vel + self.vio_bias + self.rng.normal(0, 0.05, 2)
        ground = float(self.dem[_clamp_cell(self.uav_pos[1] / CELL_Y_M),
                                _clamp_cell(self.uav_pos[0] / CELL_X_M)])
        h_baro = ground + self.uav_agl + self.rng.normal(0, 0.85)     # baro + radar noise
        d_agl = self.uav_agl + self.rng.normal(0, 0.85)               # combine to N(0, 1.2 m)
        h_meas = h_baro - d_agl

        # ---- detector (skipped once landed: UAV is riding the deck) ----
        det = self.detector.step(v_vio, z_gps, h_meas, sigma)

        # ---- FSM ----
        state = self.fsm.update(self.t, det["confirmed"])
        est = self.detector.pos
        ugv_idx = self._ugv_idx_float()

        # ---- guidance ----
        if self.uav_landed:
            self.uav_pos, self.uav_vel, self.uav_agl = ugv.copy(), np.zeros(2), 0.0
            v_des, agl_rate = np.zeros(2), 0.0
        else:
            if state == AutonomousRecoveryFSM.NOMINAL_SCOUT:
                target = self._path_point(ugv_idx + SCOUT_AHEAD_CELLS)
                v_des = np.clip(0.8 * (target - est), -15, 15)
                agl_rate = 0.5 * (SCOUT_AGL_M - self.uav_agl)
            elif state == AutonomousRecoveryFSM.VIO_HOLD_ISOLATED:
                v_des, agl_rate = np.zeros(2), 0.5 * (SCOUT_AGL_M - self.uav_agl)   # hover, hold heading
            else:  # UGV_MOVING_DECK_LANDING: chase live UGV position, descend
                err = ugv - est
                v_des = np.clip(1.2 * err, -18, 18)
                # feed-forward UGV velocity
                if not self.ugv_halted and self.ugv_s < self._ugv_cum[-1]:
                    ahead = self._path_point(ugv_idx + 0.5) - ugv
                    nrm = np.linalg.norm(ahead)
                    if nrm > 1e-6:
                        v_des = np.clip(v_des + ahead / nrm * self.ugv_speed, -20, 20)
                horiz = float(np.linalg.norm(err))
                descend_ok = self.uav_agl > 10.0 or horiz < 3.0
                agl_rate = -4.0 if descend_ok else 0.0

            self.uav_vel += (v_des - self.uav_vel) * dt / 0.5
            self.uav_vel += self.rng.normal(0, 0.15, 2) * math.sqrt(dt)        # gusts
            self.uav_pos = self.uav_pos + self.uav_vel * dt
            self.uav_agl = max(0.0, self.uav_agl + agl_rate * dt)
            if np.linalg.norm(self.detector.vel) > 0.5:
                self.heading_deg = math.degrees(math.atan2(self.detector.vel[0], -self.detector.vel[1])) % 360
            if (state == AutonomousRecoveryFSM.UGV_MOVING_DECK_LANDING and self.uav_agl <= 0.05
                    and np.linalg.norm(self.uav_pos - ugv) < 3.0):
                self.uav_landed = True

        return self._telemetry(pcb, det, state, ugv, sigma)

    # ---- telemetry ------------------------------------------------------ #
    def _telemetry(self, pcb, det, state, ugv, sigma) -> dict:
        ux, uy = ugv[0] / CELL_X_M, ugv[1] / CELL_Y_M
        vx, vy = self.uav_pos[0] / CELL_X_M, self.uav_pos[1] / CELL_Y_M
        ex, ey = self.detector.pos[0] / CELL_X_M, self.detector.pos[1] / CELL_Y_M
        ucx, ucy = self._to_cell(ugv)
        t = {
            "t": round(self.t, 1),
            "ugv": {"mode": self.ugv_mode, "x": ux, "y": uy,
                    "lat_lon": self.planner.grid_to_gps(ux, uy),
                    "elevation_m": float(self.dem[ucy, ucx]), "slope_deg": float(self.slope[ucy, ucx]),
                    "progress_pct": 100 * self.ugv_s / self._ugv_cum[-1],
                    "halted": self.ugv_halted, "alert": self.ugv_alert},
            "uav": {"x_true": vx, "y_true": vy, "x_nav": ex, "y_nav": ey,
                    "lat_lon_true": self.planner.grid_to_gps(vx, vy),
                    "agl_m": self.uav_agl, "heading_deg": self.heading_deg, "landed": self.uav_landed,
                    "nav_error_m": float(np.linalg.norm(self.uav_pos - self.detector.pos)),
                    "dist_to_ugv_m": float(np.linalg.norm(self.uav_pos - ugv)),
                    "nav_source": "VIO_ONLY" if self.detector.gps_gain_locked else "GPS+VIO_KF"},
            "spoof": {"active": self.spoof_active, "offset_m": float(np.linalg.norm(self.spoof_offset))},
            "detector": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in det.items()},
            "gps_sigma_m": sigma,
            "fsm": {"state": state, "reason": self.fsm.reason},
            "pcb": pcb,
        }
        self._last = t
        return t


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _demo():
    eng = SimulationEngine()
    print(json.dumps(eng.planner.summary(), indent=2, default=str))
    last_state = None
    for _ in range(3000):
        if abs(eng.t - 10.0) < 1e-6:
            print(">>> t=10.0s: GPS spoofing attack started")
            eng.start_spoof(rate_mps=6.0)
        tel = eng.tick()
        if tel["fsm"]["state"] != last_state:
            last_state = tel["fsm"]["state"]
            d = tel["detector"]
            print(f"t={tel['t']:5.1f}s  FSM -> {last_state:26s} r_KF={d['r_kf_m']:5.1f} m "
                  f"r_TERCOM={d['r_tercom_m']:5.1f} m  trigger={d['trigger']}")
        if tel["uav"]["landed"]:
            print(f"t={tel['t']:5.1f}s  LANDED on UGV deck, nav error {tel['uav']['nav_error_m']:.1f} m")
            break


if __name__ == "__main__":
    _demo()
