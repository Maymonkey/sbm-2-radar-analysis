"""CAPPI decoding and conservative echo tracking. Pixel coordinates are x/east, y/south."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import math
import re

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

ICT = timezone(timedelta(hours=7))
FILENAME = re.compile(r"^(\d{14})\d{2}dBZ\.cappi\.png$")
MODEL_VERSION = "2.0.0"


def scan_time(path):
    match = FILENAME.fullmatch(Path(path).name)
    if not match:
        raise ValueError(f"Invalid radar filename: {Path(path).name}")
    return datetime.strptime(match[1], "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc).astimezone(ICT)


def load_config(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "config/radar.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    width, height = cfg["image_size"]
    x, y = cfg["target"]["pixel_xy"]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Target pixel lies outside image")
    if cfg["radar"]["km_per_pixel"] <= 0 or cfg["target"]["cell_size_km"] <= 0:
        raise ValueError("Pixel scale and target size must be positive")
    return cfg


def config_digest(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def validate_png(path, cfg):
    if Path(path).stat().st_size > cfg["source"]["max_bytes"]:
        raise ValueError("Radar PNG exceeds maximum file size")
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != tuple(cfg["image_size"]):
            raise ValueError(f"Unexpected image format/dimensions: {path}")
        image.verify()
    with Image.open(path) as image:
        image.load()  # verify() alone does not decode every pixel


def palette_for_image(image, cfg):
    """Use marker boundaries, not text/sample offsets; retain quantization bounds.

    White is indistinguishable from province labels and remains unclassified.
    Persistent coloured echoes are never removed just because they are stationary.
    """
    rgb = np.asarray(image.convert("RGB"))
    if image.size != tuple(cfg["image_size"]):
        raise ValueError("Radar image dimensions changed; recalibrate before analysis")
    legend = cfg["legend"]
    rows = [item["marker_y"] for item in legend["thresholds"]]
    levels = [item["dbz_greater_than"] for item in legend["thresholds"]]
    excluded = {tuple(color) for color in legend["excluded_rgb"]}
    palette = {}
    for band in legend["bands"]:
        color = tuple(band["rgb"])
        top, bottom = band["top_y"], band["bottom_y"]
        if not np.all(rgb[top:bottom, legend["sample_x"]] == color):
            raise ValueError("Radar legend changed; refusing an uncalibrated dBZ conversion")
        if color in excluded:
            continue
        lo = float(np.interp(bottom, rows, levels))
        hi = float(np.interp(top, rows, levels))
        palette[color] = (lo, hi)
    return palette


def masks(cfg):
    width, height = cfg["image_size"]
    yy, xx = np.indices((height, width))
    tx, ty = cfg["target"]["pixel_xy"]
    scale = cfg["radar"]["km_per_pixel"]
    radius = cfg["target"]["analysis_radius_km"] / scale
    half = cfg["target"]["cell_size_km"] / (2 * scale)
    left, top, right, bottom = cfg["map_box"]
    roi = ((xx-tx)**2 + (yy-ty)**2 <= radius**2)
    roi &= (xx >= left) & (xx < right) & (yy >= top) & (yy < bottom)
    cell = roi & (abs(xx-tx) <= half) & (abs(yy-ty) <= half)
    return roi, cell


def decode(image, cfg, roi):
    palette = palette_for_image(image, cfg)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint32)
    packed = rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]
    lower = np.full(roi.shape, np.nan, dtype=np.float32)
    upper = lower.copy()
    for (red, green, blue), (lo, hi) in palette.items():
        hit = roi & (packed == red * 65536 + green * 256 + blue)
        lower[hit], upper[hit] = lo, hi
    return lower, upper


def gap_km(points, cfg):
    target = np.asarray(cfg["target"]["pixel_xy"])
    scale = cfg["radar"]["km_per_pixel"]
    half = cfg["target"]["cell_size_km"] / (2 * scale)
    distances = np.maximum(np.abs(np.asarray(points) - target) - half, 0)
    return float(np.linalg.norm(distances, axis=1).min() * scale)


def components(lower, upper, cfg):
    eligible = np.isfinite(lower) & (lower >= cfg["analysis"]["echo_threshold_dbz"] - 1e-5)
    labels, _ = ndimage.label(eligible, structure=np.ones((3, 3)))
    result = []
    for number, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        yy, xx = np.nonzero(labels[box] == number)
        if len(xx) < cfg["analysis"]["min_component_pixels"]:
            continue
        yy, xx = yy + box[0].start, xx + box[1].start
        points = np.column_stack((xx, yy))
        lo, hi = lower[yy, xx], upper[yy, xx]
        result.append({
            "pixels": points, "centroid": points.mean(axis=0),
            "pixel_count": len(points), "gap_km": gap_km(points, cfg),
            "peak_dbz": float(((lo + hi) / 2).max()),
            "peak_dbz_lower": float(lo.max()), "peak_dbz_upper": float(hi.max()),
            "mean_dbz": float(((lo + hi) / 2).mean()),
            "fingerprint": hashlib.sha256(points.astype("<i4").tobytes()).hexdigest()[:24],
        })
    return result


def direction(vector):
    dx, dy = vector
    if math.hypot(dx, dy) < 1e-9:
        return "", None
    bearing = math.degrees(math.atan2(dx, -dy)) % 360
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((bearing + 22.5) // 45) % 8], bearing


def approximate_latlon(point, cfg):
    """Inverse spherical azimuthal equidistant mapping, pending source georeference validation."""
    radar = cfg["radar"]
    east, south = (np.asarray(point) - radar["pixel_xy"]) * radar["km_per_pixel"]
    distance = math.hypot(east, south) / 6371.0088
    bearing = math.atan2(east, -south)
    lat, lon = math.radians(radar["latitude"]), math.radians(radar["longitude"])
    lat2 = math.asin(math.sin(lat)*math.cos(distance) + math.cos(lat)*math.sin(distance)*math.cos(bearing))
    lon2 = lon + math.atan2(math.sin(bearing)*math.sin(distance)*math.cos(lat),
                           math.cos(distance)-math.sin(lat)*math.sin(lat2))
    return round(math.degrees(lat2), 6), round(math.degrees(lon2), 6)


def shape_arrival_minutes(points, velocity, cfg):
    """Earliest intersection of a translated echo footprint with the 5 km square.

    Exact ray/box intersection for pixel centres; None means the path misses the cell.
    Raster resolution limits this estimate. No storm growth is extrapolated.
    """
    points = np.asarray(points, dtype=float)
    target = np.asarray(cfg["target"]["pixel_xy"])
    half = cfg["target"]["cell_size_km"] / (2 * cfg["radar"]["km_per_pixel"])
    enter, leave = np.zeros(len(points)), np.full(len(points), np.inf)
    for axis in (0, 1):
        speed = velocity[axis]
        if abs(speed) < 1e-9:
            outside = abs(points[:, axis] - target[axis]) > half
            leave[outside] = -1
        else:
            a = (target[axis] - half - points[:, axis]) / speed
            b = (target[axis] + half - points[:, axis]) / speed
            enter = np.maximum(enter, np.minimum(a, b))
            leave = np.minimum(leave, np.maximum(a, b))
    good = leave >= enter
    return float(enter[good].min()) if np.any(good) else None


def overlap_iou(previous, current, shift):
    old = {tuple(p) for p in (previous + np.rint(shift).astype(int))}
    new = {tuple(p) for p in current}
    intersection = len(old & new)
    return intersection / (len(old) + len(new) - intersection)


def associate(frames, cfg):
    """Global one-to-one assignment using predicted location, shape overlap, and size.

    A split/merge is not treated as a confident velocity measurement. Candidate
    ambiguity and size changes remain in each observation for later audit.
    """
    tracks = []
    scale = cfg["radar"]["km_per_pixel"]
    settings = cfg["analysis"]
    for frame_index, frame in enumerate(frames):
        current = frame["components"]
        candidates = []
        for track in tracks:
            prev = track["observations"][-1]
            dt = (frame["time"] - prev["time"]).total_seconds() / 60
            if 0 < dt <= settings["max_reconnect_minutes"]:
                candidates.append((track, dt))
        count_old, count_new = len(candidates), len(current)
        costs = np.full((count_old, count_new + count_old), 1.8)
        details = {}
        for i, (track, dt) in enumerate(candidates):
            obs = track["observations"]
            prev = obs[-1]
            shift = np.zeros(2)
            if len(obs) >= 2:
                prev_dt = (prev["time"] - obs[-2]["time"]).total_seconds() / 60
                shift = (prev["centroid"] - obs[-2]["centroid"]) / prev_dt * dt
                if np.linalg.norm(shift) * scale / dt * 60 > settings["max_speed_kmh"]:
                    shift = np.zeros(2)
            limit = settings["max_speed_kmh"] * dt / 60 + 1.2
            for j, item in enumerate(current):
                displacement = np.linalg.norm(item["centroid"] - prev["centroid"]) * scale
                ratio = item["pixel_count"] / prev["pixel_count"]
                if displacement > limit or not 1/3 <= ratio <= 3:
                    costs[i, j] = 1e6
                    continue
                residual = np.linalg.norm(item["centroid"] - prev["centroid"] - shift) * scale
                overlap = overlap_iou(prev["pixels"], item["pixels"], shift)
                costs[i, j] = 0.6 * residual / limit + 0.8 * (1-overlap) + 0.3 * abs(math.log(ratio))
                details[i, j] = {"match_cost": float(costs[i, j]),
                                 "match_iou": overlap, "area_ratio": ratio}
        used = set()
        rows, cols = linear_sum_assignment(costs) if count_old else ([], [])
        for i, j in zip(rows, cols):
            if j >= count_new or costs[i, j] >= 1.8:
                continue
            alternative_row = np.delete(costs[i, :count_new], j)
            alternative_col = np.delete(costs[:, j], i)
            alternative = min(alternative_row.min(initial=np.inf), alternative_col.min(initial=np.inf))
            detail = details[i, j]
            detail["possible_split_or_merge"] = (
                sum(d["match_iou"] > 0.05 for (r, _), d in details.items() if r == i) > 1
                or sum(d["match_iou"] > 0.05 for (_, c), d in details.items() if c == j) > 1)
            detail["match_ambiguous"] = bool(
                detail["possible_split_or_merge"] or
                alternative - costs[i, j] < settings["ambiguity_cost_margin"]
                or not 0.5 <= detail["area_ratio"] <= 2)
            candidates[i][0]["observations"].append({**current[j], **detail,
                "time": frame["time"], "file": frame["file"], "frame_index": frame_index})
            used.add(j)
        for j, item in enumerate(current):
            if j not in used:
                tracks.append({"observations": [{**item, "time": frame["time"],
                    "file": frame["file"], "frame_index": frame_index,
                    "match_cost": None, "match_iou": None, "area_ratio": None,
                    "match_ambiguous": False, "possible_split_or_merge": False}]})
    return tracks


def observation_key(obs):
    return obs["time"].isoformat() + ":" + obs["fingerprint"]


def assign_ids(tracks, previous_state, cfg):
    """Reuse IDs through matching observations in the overlapping rolling window."""
    previous_state = previous_state if isinstance(previous_state, dict) else {}
    compatible = (previous_state.get("model_version") == MODEL_VERSION
                  and previous_state.get("config_sha256") == config_digest(cfg))
    mapping = previous_state.get("observations", {}) if compatible else {}
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        mapping = {}
    used = set()
    new_mapping = {}
    for track in tracks:
        votes = [mapping[observation_key(o)] for o in track["observations"]
                 if observation_key(o) in mapping]
        available = sorted(set(votes) - used, key=lambda x: (-votes.count(x), x))
        track["identity_conflict"] = len(set(votes)) > 1 or (bool(votes) and not available)
        identifier = available[0] if available else "T" + hashlib.sha256(
            observation_key(track["observations"][0]).encode()).hexdigest()[:12]
        if identifier in used:
            identifier += "-" + str(len(used))
        track["track_id"] = identifier
        used.add(identifier)
        for obs in track["observations"]:
            new_mapping[observation_key(obs)] = identifier
    return {"schema_version": 1, "model_version": MODEL_VERSION,
            "config_sha256": config_digest(cfg), "observations": new_mapping}


def assess_track(track, latest_frame_index, cfg):
    obs = track["observations"]
    latest = obs[-1]
    result = {"track_id": track["track_id"], "status": "building_track",
              "reason": "fewer_than_three_consecutive_scans", "direction": "",
              "target_alignment_deg": None, "speed_kmh": None, "closing_speed_kmh": None,
              "eta_min": None, "fit_residual_km": None, "velocity": None}
    if latest["frame_index"] != latest_frame_index:
        result.update(status="ended", reason="not_seen_in_latest_scan")
        return result
    if latest["gap_km"] <= 1e-6:
        result.update(status="echo_at_target", reason="observed_echo_intersects_target", eta_min=0)
        return result
    recent = [latest]
    for previous in reversed(obs[:-1]):
        dt = (recent[0]["time"] - previous["time"]).total_seconds() / 60
        if (recent[0]["frame_index"] != previous["frame_index"] + 1
                or not 0 < dt <= cfg["analysis"]["max_scan_interval_minutes"]):
            break
        recent.insert(0, previous)
        if len(recent) == 5:
            break
    if len(recent) < 3:
        return result
    if track["identity_conflict"] or any(o["match_ambiguous"] for o in recent[1:]):
        result.update(status="uncertain_motion", reason="ambiguous_match_or_shape_change")
        return result
    times = np.array([(o["time"]-recent[0]["time"]).total_seconds()/60 for o in recent])
    positions = np.array([o["centroid"] for o in recent])
    design = np.column_stack((times, np.ones(len(times))))
    fit = np.linalg.lstsq(design, positions, rcond=None)[0]
    velocity = fit[0]
    scale = cfg["radar"]["km_per_pixel"]
    residual = float(np.sqrt(np.mean(np.sum((positions-design@fit)**2, axis=1))) * scale)
    speed = float(np.linalg.norm(velocity) * scale * 60)
    target = np.asarray(cfg["target"]["pixel_xy"]) - latest["centroid"]
    norm = float(np.linalg.norm(target))
    closing = float(np.dot(velocity, target) / norm * scale * 60) if norm else 0.0
    alignment = math.degrees(math.acos(float(np.clip(closing / speed, -1, 1)))) if speed > 1e-6 else None
    result.update(direction=direction(velocity)[0], speed_kmh=round(speed, 1),
                  closing_speed_kmh=round(closing, 1), target_alignment_deg=round(alignment, 1) if alignment is not None else None,
                  fit_residual_km=round(residual, 2), velocity=velocity)
    if residual > cfg["analysis"]["max_fit_residual_km"] or speed > cfg["analysis"]["max_speed_kmh"]:
        result.update(status="uncertain_motion", reason="inconsistent_motion_fit")
        return result
    if speed < 1:
        result.update(status="stationary", reason="motion_below_resolution")
        return result
    arrival = shape_arrival_minutes(latest["pixels"], velocity, cfg)
    if arrival is None:
        result.update(status="moving_away" if closing <= 0 else "passing_target",
                      reason="projected_footprint_misses_target")
    elif arrival > cfg["analysis"]["forecast_horizon_minutes"]:
        result.update(status="approaching_outside_horizon", reason="intersection_beyond_60_minutes")
    else:
        gaps = [o["gap_km"] for o in recent[-3:]]
        if any(b > a + 0.3 for a, b in zip(gaps, gaps[1:])) or gaps[0]-gaps[-1] < 0.6:
            result.update(status="uncertain_motion", reason="edge_not_consistently_approaching")
        else:
            result.update(status="inbound", reason="projected_footprint_intersects_target",
                          eta_min=round(arrival, 1))
    return result
