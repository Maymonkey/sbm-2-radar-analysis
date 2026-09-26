from array import array
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import math
import re

from PIL import Image, ImageDraw


# ---------- SBM-2 / radar configuration ----------
IMAGE_DIR = Path("radar_images")
SBM2_XY = (366, 321)
KM_PER_PIXEL = 0.6
ANALYSIS_RADIUS_KM = 100
TARGET_CELL_KM = 5
DBZ_THRESHOLD = 20
MIN_COMPONENT_PIXELS = 5
MAX_TRACK_SPEED_KMH = 100

MAP_MAX_X = 801
LEGEND_SAMPLE_X = 850
UNKNOWN_DBZ = -32768
ICT = timezone(timedelta(hours=7))

# Legend sample rows and their approximate dBZ values.
# These are the anchors previously sampled from the Sattahip CAPPI legend.
LEGEND_ANCHORS = [
    (81, 72), (101, 70), (121, 68), (140, 64), (160, 60),
    (180, 56), (200, 54), (219, 50), (239, 48), (259, 42),
    (278, 40), (298, 36), (318, 34), (338, 28), (357, 26),
    (377, 22), (397, 20), (416, 16), (436, 12), (456, 8),
]


def filename_time_ict(path):
    match = re.search(r"(\d{14})\d{2}dBZ", path.name)
    if not match:
        raise ValueError(f"Cannot read timestamp from filename: {path.name}")

    time_utc = datetime.strptime(
        match.group(1), "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)

    return time_utc.astimezone(ICT)


def legend_dbz_at_y(y):
    if y <= LEGEND_ANCHORS[0][0]:
        return float(LEGEND_ANCHORS[0][1])
    if y >= LEGEND_ANCHORS[-1][0]:
        return float(LEGEND_ANCHORS[-1][1])

    for (y1, dbz1), (y2, dbz2) in zip(
        LEGEND_ANCHORS, LEGEND_ANCHORS[1:]
    ):
        if y1 <= y <= y2:
            fraction = (y - y1) / (y2 - y1)
            return dbz1 + fraction * (dbz2 - dbz1)

    return float(LEGEND_ANCHORS[-1][1])


def build_palette(image):
    """Sample the color legend and map exact RGB values to approximate dBZ."""
    samples = {}

    for y in range(81, 457):
        rgb = image.convert("RGB").getpixel((LEGEND_SAMPLE_X, y))
        dbz = legend_dbz_at_y(y)
        samples.setdefault(rgb, []).append(dbz)

    return {
        rgb: sum(values) / len(values)
        for rgb, values in samples.items()
    }


def build_analysis_pixels(width, height):
    """Return map pixels inside the 100 km SBM-2 circle."""
    center_x, center_y = SBM2_XY
    radius_px = ANALYSIS_RADIUS_KM / KM_PER_PIXEL

    min_x = max(0, int(center_x - radius_px))
    max_x = min(MAP_MAX_X, width - 1, int(center_x + radius_px))
    min_y = max(0, int(center_y - radius_px))
    max_y = min(height - 1, int(center_y + radius_px))

    pixels = []
    radius_sq = radius_px * radius_px

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius_sq:
                pixels.append((x, y, y * width + x))

    return pixels


def build_dbz_grid(image, palette, analysis_pixels):
    """Convert exact CAPPI legend colors into approximate dBZ values."""
    width, height = image.size
    grid = array("h", [UNKNOWN_DBZ]) * (width * height)
    rgb_image = image.convert("RGB")
    image_pixels = rgb_image.load()

    for x, y, index in analysis_pixels:
        rgb = image_pixels[x, y]
        if rgb in palette:
            grid[index] = round(palette[rgb] * 10)

    return grid


def find_components(grid, width, analysis_pixels, excluded_labels):
    """Find 8-connected >=20 dBZ pixel groups and retain their shapes."""
    eligible = bytearray(width * (len(grid) // width))

    for x, y, index in analysis_pixels:
        if index in excluded_labels:
            continue
        if grid[index] != UNKNOWN_DBZ and grid[index] >= DBZ_THRESHOLD * 10:
            eligible[index] = 1

    components = []

    for x, y, start_index in analysis_pixels:
        if not eligible[start_index]:
            continue

        eligible[start_index] = 0
        queue = deque([start_index])
        points = []
        dbz_values = []

        while queue:
            index = queue.popleft()
            px = index % width
            py = index // width
            points.append((px, py))
            dbz_values.append(grid[index] / 10)

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx = px + dx
                    ny = py + dy
                    if nx < 0 or nx > MAP_MAX_X or ny < 0:
                        continue

                    neighbor = ny * width + nx
                    if eligible[neighbor]:
                        eligible[neighbor] = 0
                        queue.append(neighbor)

        if len(points) < MIN_COMPONENT_PIXELS:
            continue

        components.append({
            "pixels": points,
            "pixel_count": len(points),
            "area_km2_approx": round(
                len(points) * KM_PER_PIXEL * KM_PER_PIXEL, 2
            ),
            "centroid_x": sum(p[0] for p in points) / len(points),
            "centroid_y": sum(p[1] for p in points) / len(points),
            "max_estimated_dbz": max(dbz_values),
            "mean_estimated_dbz": sum(dbz_values) / len(dbz_values),
        })

    return components


def distance_to_target_cell_km(points):
    """Distance from a cloud's edge to the 5 km SBM-2 target cell."""
    target_x, target_y = SBM2_XY
    half_cell_pixels = TARGET_CELL_KM / (2 * KM_PER_PIXEL)
    nearest = float("inf")

    for x, y in points:
        dx = max(abs(x - target_x) - half_cell_pixels, 0)
        dy = max(abs(y - target_y) - half_cell_pixels, 0)
        nearest = min(nearest, math.hypot(dx, dy) * KM_PER_PIXEL)

    return nearest


def direction_from_sbm2(x, y):
    """Compass direction from SBM-2 to a component centroid."""
    dx = x - SBM2_XY[0]
    north = SBM2_XY[1] - y
    bearing = math.degrees(math.atan2(dx, north)) % 360
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(bearing / 45) % 8], bearing


def motion_direction(dx, dy):
    """Compass direction of movement in image coordinates."""
    bearing = math.degrees(math.atan2(dx, -dy)) % 360
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(bearing / 45) % 8]


def make_tracks(frames):
    """Associate cloud components between scans and identify inbound tracks."""
    tracks = {}
    next_id = 1

    for frame_index, frame in enumerate(frames):
        current_time = frame["time_ict"]
        components = frame["components"]
        possible_matches = []

        for track_id, track in tracks.items():
            previous = track["observations"][-1]
            elapsed_min = (
                current_time - previous["time_ict"]
            ).total_seconds() / 60

            if elapsed_min <= 0 or elapsed_min > 12:
                continue

            max_distance_km = (
                MAX_TRACK_SPEED_KMH * elapsed_min / 60 + 1.2
            )

            for component_index, component in enumerate(components):
                distance_km = math.hypot(
                    component["centroid_x"] - previous["centroid_x"],
                    component["centroid_y"] - previous["centroid_y"],
                ) * KM_PER_PIXEL

                if distance_km <= max_distance_km:
                    possible_matches.append(
                        (distance_km, track_id, component_index)
                    )

        possible_matches.sort()
        used_tracks = set()
        used_components = set()

        for _, track_id, component_index in possible_matches:
            if track_id in used_tracks or component_index in used_components:
                continue

            component = components[component_index]
            tracks[track_id]["observations"].append({
                **component,
                "time_ict": current_time,
                "frame_index": frame_index,
                "gap_km": distance_to_target_cell_km(
                    component["pixels"]
                ),
            })
            used_tracks.add(track_id)
            used_components.add(component_index)

        for component_index, component in enumerate(components):
            if component_index in used_components:
                continue

            track_id = f"T{next_id:03d}"
            next_id += 1

            tracks[track_id] = {
                "observations": [{
                    **component,
                    "time_ict": current_time,
                    "frame_index": frame_index,
                    "gap_km": distance_to_target_cell_km(
                        component["pixels"]
                    ),
                }]
            }

    summaries = []

    for track_id, track in tracks.items():
        observations = track["observations"]
        latest = observations[-1]
        gap_km = latest["gap_km"]

        result = {
            "track_id": track_id,
            "observations": observations,
            "latest": latest,
            "status": "building_track",
            "direction": "",
            "speed_kmh": "",
            "eta_min": "",
            "gap_km": round(gap_km, 1),
        }

        if gap_km <= 0.01:
            result["status"] = "echo_at_target"
            result["eta_min"] = 0
            summaries.append(result)
            continue

        recent = observations[-3:]
        has_three_consecutive_scans = (
            len(recent) == 3
            and recent[1]["frame_index"] == recent[0]["frame_index"] + 1
            and recent[2]["frame_index"] == recent[1]["frame_index"] + 1
        )

        if has_three_consecutive_scans:
            gaps = [observation["gap_km"] for observation in recent]
            steadily_closing = all(
                gaps[i] - gaps[i + 1] >= 0.6
                for i in range(2)
            )

            first = recent[0]
            last = recent[-1]
            elapsed_min = (
                last["time_ict"] - first["time_ict"]
            ).total_seconds() / 60

            if elapsed_min > 0:
                closing_speed = (
                    (first["gap_km"] - last["gap_km"])
                    / elapsed_min * 60
                )

                dx = last["centroid_x"] - first["centroid_x"]
                dy = last["centroid_y"] - first["centroid_y"]
                result["direction"] = motion_direction(dx, dy)

                if steadily_closing and closing_speed > 0:
                    result["status"] = "inbound"
                    result["speed_kmh"] = round(closing_speed, 1)
                    result["eta_min"] = round(
                        gap_km / closing_speed * 60
                    )

        summaries.append(result)

    return summaries


def save_outputs(frames, summaries, circle_pixels, excluded_labels):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Per-scan statistics around SBM-2
    with (IMAGE_DIR / "dbz-timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "file", "time_utc", "time_ict", "radius_km",
            "circle_pixels", "circle_ge20_pixels", "circle_ge20_percent",
            "cell_pixels", "cell_ge20_pixels", "cell_ge20_percent",
            "cell_max_estimated_dbz", "components_ge20_min5pixels",
            "nearest_component_distance_km",
            "nearest_component_direction",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for frame in frames:
            grid = frame["grid"]
            image = frame["image"]
            width, height = image.size
            circle_ge20 = sum(
                1 for _, _, index in circle_pixels
                if index not in excluded_labels
                and grid[index] != UNKNOWN_DBZ
                and grid[index] >= DBZ_THRESHOLD * 10
            )

            target_x, target_y = SBM2_XY
            cell_points = []
            for y in range(target_y - 4, target_y + 5):
                for x in range(target_x - 4, target_x + 5):
                    if 0 <= x <= MAP_MAX_X and 0 <= y < height:
                        cell_points.append(y * width + x)

            cell_ge20 = sum(
                1 for index in cell_points
                if index not in excluded_labels
                and grid[index] != UNKNOWN_DBZ
                and grid[index] >= DBZ_THRESHOLD * 10
            )

            cell_values = [
                grid[index] / 10 for index in cell_points
                if index not in excluded_labels
                and grid[index] != UNKNOWN_DBZ
            ]
            cell_max = max(cell_values) if cell_values else ""

            nearest = sorted(
                frame["components"],
                key=lambda c: math.hypot(
                    c["centroid_x"] - target_x,
                    c["centroid_y"] - target_y,
                ),
            )
            if nearest:
                nearest_component = nearest[0]
                nearest_km = round(
                    math.hypot(
                        nearest_component["centroid_x"] - target_x,
                        nearest_component["centroid_y"] - target_y,
                    ) * KM_PER_PIXEL,
                    1,
                )
                nearest_direction, _ = direction_from_sbm2(
                    nearest_component["centroid_x"],
                    nearest_component["centroid_y"],
                )
            else:
                nearest_km = ""
                nearest_direction = ""

            circle_percent = (
                circle_ge20 / len(circle_pixels) * 100
                if circle_pixels else 0
            )
            cell_percent = (
                cell_ge20 / len(cell_points) * 100
                if cell_points else 0
            )

            time_utc = frame["time_ict"].astimezone(
                timezone.utc
            ).isoformat()

            writer.writerow({
                "file": frame["file"],
                "time_utc": time_utc,
                "time_ict": frame["time_ict"].isoformat(),
                "radius_km": ANALYSIS_RADIUS_KM,
                "circle_pixels": len(circle_pixels),
                "circle_ge20_pixels": circle_ge20,
                "circle_ge20_percent": round(circle_percent, 2),
                "cell_pixels": len(cell_points),
                "cell_ge20_pixels": cell_ge20,
                "cell_ge20_percent": round(cell_percent, 2),
                "cell_max_estimated_dbz": cell_max,
                "components_ge20_min5pixels": len(frame["components"]),
                "nearest_component_distance_km": nearest_km,
                "nearest_component_direction": nearest_direction,
            })

    # Every detected echo group, with its pixel footprint retained internally
    with (IMAGE_DIR / "echo-components.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "file", "time_ict", "pixel_count", "area_km2_approx",
            "centroid_x", "centroid_y", "distance_from_sbm2_km",
            "direction_from_sbm2", "max_estimated_dbz",
            "mean_estimated_dbz",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for frame in frames:
            for component in frame["components"]:
                direction, _ = direction_from_sbm2(
                    component["centroid_x"], component["centroid_y"]
                )
                distance = math.hypot(
                    component["centroid_x"] - SBM2_XY[0],
                    component["centroid_y"] - SBM2_XY[1],
                ) * KM_PER_PIXEL

                writer.writerow({
                    "file": frame["file"],
                    "time_ict": frame["time_ict"].isoformat(),
                    "pixel_count": component["pixel_count"],
                    "area_km2_approx": component["area_km2_approx"],
                    "centroid_x": round(component["centroid_x"], 2),
                    "centroid_y": round(component["centroid_y"], 2),
                    "distance_from_sbm2_km": round(distance, 1),
                    "direction_from_sbm2": direction,
                    "max_estimated_dbz": round(
                        component["max_estimated_dbz"], 1
                    ),
                    "mean_estimated_dbz": round(
                        component["mean_estimated_dbz"], 1
                    ),
                })

    # Tracked clouds and inbound estimates
    with (IMAGE_DIR / "cloud-tracks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "track_id", "status", "start_time_ict", "latest_time_ict",
            "direction", "speed_kmh", "distance_to_cell_km",
            "eta_min", "peak_dbz", "pixel_count",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for item in summaries:
            observations = item["observations"]
            latest = item["latest"]
            writer.writerow({
                "track_id": item["track_id"],
                "status": item["status"],
                "start_time_ict": observations[0]["time_ict"].isoformat(),
                "latest_time_ict": latest["time_ict"].isoformat(),
                "direction": item["direction"],
                "speed_kmh": item["speed_kmh"],
                "distance_to_cell_km": item["gap_km"],
                "eta_min": item["eta_min"],
                "peak_dbz": round(latest["max_estimated_dbz"], 1),
                "pixel_count": latest["pixel_count"],
            })

    # Latest image with inbound Cloud Shapes and motion arrows
    latest_frame = frames[-1]
    base = latest_frame["image"].convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    inbound_count = 0

    for item in summaries:
        latest = item["latest"]

        if latest["frame_index"] != len(frames) - 1:
            continue
        if item["status"] not in ("inbound", "echo_at_target"):
            continue

        inbound_count += 1
        points = set(latest["pixels"])
        fill = (255, 0, 255, 95)
        edge = (255, 0, 255, 255)

        for x, y in points:
            overlay.putpixel((x, y), fill)

        for x, y in points:
            if any(
                (x + dx, y + dy) not in points
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                draw.point((x, y), fill=edge)

        observations = item["observations"]
        if item["status"] == "inbound" and len(observations) >= 2:
            previous = observations[-2]
            dx = latest["centroid_x"] - previous["centroid_x"]
            dy = latest["centroid_y"] - previous["centroid_y"]
            length = math.hypot(dx, dy) or 1
            end_x = latest["centroid_x"] + dx / length * 18
            end_y = latest["centroid_y"] + dy / length * 18

            draw.line(
                (
                    previous["centroid_x"], previous["centroid_y"],
                    end_x, end_y,
                ),
                fill=edge,
                width=3,
            )

        if item["status"] == "inbound":
            label = (
                f"{item['track_id']} -> SBM-2 "
                f"{item['direction']} ETA {item['eta_min']} min"
            )
        else:
            label = f"{item['track_id']} echo at target"

        draw.text(
            (latest["centroid_x"] + 5, latest["centroid_y"] + 5),
            label,
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    # Draw the 100 km analysis circle and 5 km target cell
    target_x, target_y = SBM2_XY
    radius_px = ANALYSIS_RADIUS_KM / KM_PER_PIXEL
    half_cell_px = TARGET_CELL_KM / (2 * KM_PER_PIXEL)

    draw.ellipse(
        (
            target_x - radius_px, target_y - radius_px,
            target_x + radius_px, target_y + radius_px,
        ),
        outline=(255, 0, 255, 220),
        width=2,
    )
    draw.rectangle(
        (
            target_x - half_cell_px, target_y - half_cell_px,
            target_x + half_cell_px, target_y + half_cell_px,
        ),
        outline=(0, 255, 255, 255),
        width=2,
    )
    draw.text(
        (target_x + 8, target_y - 18),
        "SBM-2 5 km cell",
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )

    if inbound_count == 0:
        draw.text(
            (15, 15),
            "No confirmed inbound cloud track in latest scan",
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    Image.alpha_composite(base, overlay).convert("RGB").save(
        IMAGE_DIR / "diagnostic-cloud-shapes.png"
    )

    print(f"Excluded stable province-label pixels: {len(excluded_labels)}")
    print("Filename times are UTC; reported times are ICT (UTC+07:00).")
    print(f"Analysis radius: {ANALYSIS_RADIUS_KM} km")
    print(f"Latest scan ICT: {latest_frame['time_ict'].isoformat()}")
    print(f"Inbound clouds in latest scan: {inbound_count}")
    print(f"Saved: {IMAGE_DIR / 'dbz-timeseries.csv'}")
    print(f"Saved: {IMAGE_DIR / 'echo-components.csv'}")
    print(f"Saved: {IMAGE_DIR / 'cloud-tracks.csv'}")
    print(f"Saved: {IMAGE_DIR / 'diagnostic-cloud-shapes.png'}")


def main():
    image_files = list(IMAGE_DIR.glob("*dBZ.cappi.png"))
    image_files.sort(key=filename_time_ict)
    image_files = image_files[-10:]

    if len(image_files) < 3:
        raise RuntimeError(
            f"Need at least 3 radar PNG images; found {len(image_files)}"
        )

    first_image = Image.open(image_files[-1]).convert("RGB")
    palette = build_palette(first_image)
    width, height = first_image.size
    circle_pixels = build_analysis_pixels(width, height)

    frames = []

    for path in image_files:
        image = Image.open(path).convert("RGB")
        if image.size != (width, height):
            raise RuntimeError(
                f"Image dimensions differ: {path.name} is {image.size}"
            )

        grid = build_dbz_grid(image, palette, circle_pixels)
        frames.append({
            "file": path.name,
            "time_ict": filename_time_ict(path),
            "image": image,
            "grid": grid,
        })

    # Stable >=35 dBZ pixels across every scan are treated as static labels.
    stable_labels = None
    for frame in frames:
        high_pixels = {
            index
            for _, _, index in circle_pixels
            if frame["grid"][index] != UNKNOWN_DBZ
            and frame["grid"][index] >= 350
        }
        stable_labels = (
            high_pixels if stable_labels is None
            else stable_labels.intersection(high_pixels)
        )

    stable_labels = stable_labels or set()

    # Detect shapes for each of the ten scans.
    for frame in frames:
        frame["components"] = find_components(
            frame["grid"],
            width,
            circle_pixels,
            stable_labels,
        )

        for component in frame["components"]:
            component["gap_km"] = distance_to_target_cell_km(
                component["pixels"]
            )

    summaries = make_tracks(frames)
    save_outputs(frames, summaries, circle_pixels, stable_labels)


if __name__ == "__main__":
    main()
