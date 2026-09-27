"""Analyse the validated latest scans and export an auditable, local result bundle."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from radar_core import (
    ICT, MODEL_VERSION, approximate_latlon, assess_track, assign_ids, associate,
    components, config_digest, decode, direction, load_config, masks, scan_time,
    validate_png, write_json,
)


def csv_output(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value, digits=2):
    return round(float(value), digits) if value is not None else None


def frame_quality(frames, cfg, now):
    intervals = [(b['time']-a['time']).total_seconds()/60 for a, b in zip(frames, frames[1:])]
    age = (now-frames[-1]['time']).total_seconds()/60
    issues = []
    if len(frames) < cfg['analysis']['frames']:
        issues.append('fewer_than_10_frames')
    if any(dt > cfg['analysis']['max_scan_interval_minutes'] for dt in intervals):
        issues.append('missing_scans_in_history')
    if age > cfg['analysis']['max_frame_age_minutes']:
        issues.append('stale_latest_scan')
    if age < -2:
        issues.append('scan_timestamp_in_future')
    return {'forecast_allowed': not issues, 'issues': issues,
            'frame_count': len(frames), 'scan_age_minutes': round(age, 1),
            'scan_intervals_minutes': [round(dt, 2) for dt in intervals]}


def read_frames(image_dir, cfg):
    paths = sorted(image_dir.glob('*dBZ.cappi.png'), key=scan_time)[-cfg['analysis']['frames']:]
    if not paths:
        raise ValueError('No raw CAPPI images found')
    if len({scan_time(path) for path in paths}) != len(paths):
        raise ValueError('Multiple images have the same scan timestamp')
    roi, cell = masks(cfg)
    frames = []
    for path in paths:
        validate_png(path, cfg)
        with Image.open(path) as source:
            image = source.convert('RGB')
        lower, upper = decode(image, cfg, roi)
        groups = components(lower, upper, cfg)
        hits = np.isfinite(lower) & (lower >= cfg['analysis']['echo_threshold_dbz'] - 1e-5)
        values = ((lower + upper)/2)[cell & np.isfinite(lower)]
        frames.append({'file': path.name, 'time': scan_time(path), 'image': image,
                       'components': groups, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'circle_pixels': int(roi.sum()), 'circle_ge20_pixels': int((hits & roi).sum()),
                       'cell_pixels': int(cell.sum()), 'cell_ge20_pixels': int((hits & cell).sum()),
                       'cell_classified_pixels': int((cell & np.isfinite(lower)).sum()),
                       'cell_max_estimated_dbz': number(values.max(), 1) if len(values) else None})
    return frames


def export_tables(destination, frames, tracks, cfg, quality):
    frame_fields = ['file', 'time_ict', 'sha256', 'radius_km', 'circle_pixels', 'circle_ge20_pixels',
                    'circle_ge20_percent', 'cell_pixels', 'cell_ge20_pixels', 'cell_ge20_percent',
                    'cell_classified_pixels', 'cell_max_estimated_dbz', 'components_ge20_min5pixels']
    frame_rows = []
    for frame in frames:
        row = {key: frame[key] for key in ('file', 'sha256', 'circle_pixels', 'circle_ge20_pixels',
               'cell_pixels', 'cell_ge20_pixels', 'cell_classified_pixels', 'cell_max_estimated_dbz')}
        row.update(time_ict=frame['time'].isoformat(), radius_km=cfg['target']['analysis_radius_km'],
                   circle_ge20_percent=round(100*frame['circle_ge20_pixels']/frame['circle_pixels'], 2),
                   cell_ge20_percent=round(100*frame['cell_ge20_pixels']/frame['cell_pixels'], 2),
                   components_ge20_min5pixels=len(frame['components']))
        frame_rows.append(row)
    csv_output(destination/'dbz-timeseries.csv', frame_fields, frame_rows)

    summary_rows, observation_rows = [], []
    for track in tracks:
        obs = track['observations']
        latest = obs[-1]
        assessment = track['assessment']
        summary_rows.append({
            'track_id': track['track_id'], 'status': assessment['status'], 'reason': assessment['reason'],
            'start_time_ict': obs[0]['time'].isoformat(), 'latest_time_ict': latest['time'].isoformat(),
            'is_latest_scan': latest['frame_index'] == len(frames)-1,
            'observation_count': len(obs), 'direction': assessment['direction'],
            'target_alignment_deg': assessment['target_alignment_deg'], 'speed_kmh': assessment['speed_kmh'],
            'closing_speed_kmh': assessment['closing_speed_kmh'], 'distance_to_cell_km': number(latest['gap_km']),
            'eta_min': assessment['eta_min'], 'peak_dbz': number(latest['peak_dbz'], 1),
            'peak_dbz_lower': number(latest['peak_dbz_lower'], 1),
            'peak_dbz_upper': number(latest['peak_dbz_upper'], 1), 'pixel_count': latest['pixel_count'],
            'centroid_x': number(latest['centroid'][0]), 'centroid_y': number(latest['centroid'][1]),
            'fit_residual_km': assessment['fit_residual_km'],
            'identity_conflict': track['identity_conflict'], 'forecast_data_valid': quality['forecast_allowed'],
        })
        for index, item in enumerate(obs):
            partial = assess_track({**track, 'observations': obs[:index+1]}, item['frame_index'], cfg)
            lat, lon = approximate_latlon(item['centroid'], cfg)
            prev = obs[index-1] if index else None
            dt = (item['time']-prev['time']).total_seconds()/60 if prev else None
            shift = item['centroid']-prev['centroid'] if prev else None
            observation_rows.append({
                'track_id': track['track_id'], 'file': item['file'], 'time_ict': item['time'].isoformat(),
                'frame_index': item['frame_index'], 'component_fingerprint': item['fingerprint'],
                'centroid_x': number(item['centroid'][0]), 'centroid_y': number(item['centroid'][1]),
                'latitude_approx': lat, 'longitude_approx': lon,
                'distance_to_cell_km': number(item['gap_km']), 'pixel_count': item['pixel_count'],
                'area_km2_approx': number(item['pixel_count']*cfg['radar']['km_per_pixel']**2),
                'peak_dbz': number(item['peak_dbz'], 1), 'peak_dbz_lower': number(item['peak_dbz_lower'], 1),
                'peak_dbz_upper': number(item['peak_dbz_upper'], 1), 'mean_dbz': number(item['mean_dbz'], 1),
                'elapsed_minutes': number(dt),
                'step_direction': direction(shift)[0] if prev else '',
                'step_speed_kmh': number(np.linalg.norm(shift)*cfg['radar']['km_per_pixel']/dt*60) if prev else None,
                'edge_closing_speed_kmh': number((prev['gap_km']-item['gap_km'])/dt*60) if prev else None,
                'match_cost': number(item['match_cost'], 4), 'predicted_shape_iou': number(item['match_iou'], 4),
                'area_ratio': number(item['area_ratio']), 'match_ambiguous': item['match_ambiguous'],
                'possible_split_or_merge': item['possible_split_or_merge'],
                'identity_conflict': track['identity_conflict'], 'status_at_scan': partial['status'],
                'reason_at_scan': partial['reason'], 'model_eta_min_at_scan': partial['eta_min'],
            })
    summary_fields = ['track_id', 'status', 'reason', 'start_time_ict', 'latest_time_ict', 'is_latest_scan',
                      'observation_count', 'direction', 'target_alignment_deg', 'speed_kmh', 'closing_speed_kmh',
                      'distance_to_cell_km', 'eta_min', 'peak_dbz', 'peak_dbz_lower', 'peak_dbz_upper', 'pixel_count',
                      'centroid_x', 'centroid_y', 'fit_residual_km', 'identity_conflict', 'forecast_data_valid']
    observation_fields = ['track_id', 'file', 'time_ict', 'frame_index', 'component_fingerprint', 'centroid_x',
                          'centroid_y', 'latitude_approx', 'longitude_approx', 'distance_to_cell_km', 'pixel_count',
                          'area_km2_approx', 'peak_dbz', 'peak_dbz_lower', 'peak_dbz_upper', 'mean_dbz',
                          'elapsed_minutes', 'step_direction', 'step_speed_kmh', 'edge_closing_speed_kmh',
                          'match_cost', 'predicted_shape_iou', 'area_ratio', 'match_ambiguous', 'possible_split_or_merge', 'identity_conflict',
                          'status_at_scan', 'reason_at_scan', 'model_eta_min_at_scan']
    csv_output(destination/'cloud-tracks.csv', summary_fields, summary_rows)
    csv_output(destination/'cloud-track-observations.csv', observation_fields, observation_rows)
    return summary_rows


def summary_result(frames, rows, cfg, quality, now, replay):
    latest = frames[-1]
    current = [row for row in rows if row['is_latest_scan']]
    inbound = [row for row in current if row['status'] == 'inbound']
    at_target = [row for row in current if row['status'] == 'echo_at_target']
    if not quality['forecast_allowed']:
        status = 'stale_data' if 'stale_latest_scan' in quality['issues'] else 'insufficient_data'
        message = 'ข้อมูลเรดาร์เก่าหรือไม่ต่อเนื่องเพียงพอ จึงยังประเมินการมาถึงไม่ได้'
    elif at_target:
        status, message = 'echo_at_target', 'ตรวจพบกลุ่มสัญญาณเรดาร์ถึงพื้นที่ SBM-2 แล้ว'
    elif inbound:
        status, message = 'inbound_echo', 'พบกลุ่มสัญญาณเรดาร์ที่แนวเคลื่อนที่คาดว่าจะผ่านพื้นที่ SBM-2 ภายใน 60 นาที'
    elif any(row['status'] in ('building_track', 'uncertain_motion') for row in current):
        status, message = 'uncertain_motion', 'ยังมีบางกลุ่มที่ติดตามทิศทางได้ไม่แน่นอน จึงยังยืนยันการมาถึงไม่ได้'
    else:
        status, message = 'no_confirmed_inbound_echo', 'ยังไม่พบกลุ่มสัญญาณเรดาร์ที่ผ่านเกณฑ์มุ่งเข้าพื้นที่ SBM-2 ภายใน 60 นาที'
    nearest = min(current, key=lambda row: row['distance_to_cell_km'], default=None)
    generated_at = now.astimezone(ICT)
    observed_at = latest['time']
    max_age = cfg['analysis']['max_frame_age_minutes']
    classified = latest['cell_classified_pixels']
    total = latest['cell_pixels']
    coverage = classified / total if total else 0.0
    if latest['cell_ge20_pixels'] > 0:
        echo_state = 'echo_detected'
    elif coverage >= 0.8:
        echo_state = 'no_ge20_echo_observed'
    else:
        echo_state = 'unknown'

    if status == 'stale_data':
        public_message = 'ข้อมูลเรดาร์หมดอายุแล้ว กรุณารอผลวิเคราะห์รอบใหม่'
    elif status == 'insufficient_data':
        public_message = 'ภาพเรดาร์ยังไม่ครบหรือขาดช่วง จึงยังสรุปสถานะไม่ได้'
    elif status == 'echo_at_target':
        public_message = 'ตรวจพบสัญญาณสะท้อนเรดาร์ในพื้นที่ SBM-2; สัญญาณนี้ยังไม่ยืนยันว่ามีฝนถึงพื้น'
    elif status == 'inbound_echo':
        eta = min(row['eta_min'] for row in inbound if row['eta_min'] is not None)
        public_message = f'พบกลุ่มสัญญาณสะท้อนเรดาร์ที่คาดว่าเคลื่อนเข้าพื้นที่ SBM-2 ในประมาณ {eta} นาที (ผลทดลอง)'
    elif status == 'uncertain_motion':
        public_message = 'พบกลุ่มสัญญาณเรดาร์บางส่วน แต่ทิศทางยังไม่ชัด จึงยังประเมินเวลาถึงไม่ได้'
    else:
        public_message = 'ยังไม่พบกลุ่มสัญญาณเรดาร์ที่ผ่านเกณฑ์และคาดว่าจะเคลื่อนเข้าพื้นที่ SBM-2 ภายใน 60 นาที; ไม่ได้ยืนยันว่าจะไม่มีฝน'

    public_arrivals = []
    if quality['forecast_allowed']:
        for row in sorted(inbound, key=lambda item: item['eta_min'] if item['eta_min'] is not None else float('inf')):
            public_arrivals.append({
                'arrival_in_minutes_estimate': row['eta_min'],
                'direction_of_motion': row['direction'],
                'distance_to_target_km': row['distance_to_cell_km'],
                'reflectivity_dbz_estimate': {
                    'lower': row['peak_dbz_lower'], 'upper': row['peak_dbz_upper']},
                'observed_at_ict': row['latest_time_ict'],
            })

    public_status = status
    return {'schema_version': 2, 'api_version': '1.0', 'model_version': MODEL_VERSION,
            'mode': 'historical_replay' if replay else 'live',
            'generated_at_ict': now.astimezone(ICT).isoformat(), 'observed_at_ict': latest['time'].isoformat(),
            'status': status, 'message_th': message, 'forecast_horizon_minutes': cfg['analysis']['forecast_horizon_minutes'],
            'user_summary': {
                'status': public_status,
                'message_th': public_message,
                'updated_at_ict': generated_at.isoformat(),
                'observed_at_ict': observed_at.isoformat(),
                'freshness': {
                    'state_at_generation': 'stale' if 'stale_latest_scan' in quality['issues'] else
                        ('degraded' if quality['issues'] else 'fresh'),
                    'age_minutes_at_generation': quality['scan_age_minutes'],
                    'valid_until_ict': (observed_at + timedelta(minutes=max_age)).isoformat(),
                    'max_age_minutes': max_age,
                    'note_th': 'ผู้ใช้ API ต้องเทียบเวลาปัจจุบันกับ valid_until_ict ทุกครั้ง เพราะ JSON เป็น snapshot',
                },
                'target': {'name': cfg['target']['name'], 'latitude': cfg['target']['latitude'],
                           'longitude': cfg['target']['longitude'],
                           'analysis_cell_size_km': cfg['target']['cell_size_km']},
                'target_echo': {
                    'state': echo_state,
                    'reflectivity_dbz_estimate': latest['cell_max_estimated_dbz'],
                    'classified_pixels': classified,
                    'total_pixels': total,
                    'coverage_percent': round(100 * coverage, 1),
                },
                'forecast': {
                    'available': bool(quality['forecast_allowed'] and status != 'uncertain_motion'),
                    'horizon_minutes': cfg['analysis']['forecast_horizon_minutes'],
                    'arrivals': public_arrivals,
                    'validation': 'experimental_not_operationally_validated',
                },
                'data_quality': {
                    'state': 'stale' if 'stale_latest_scan' in quality['issues'] else
                        ('degraded' if quality['issues'] else 'fresh'),
                    'images_used': quality['frame_count'],
                    'scan_age_minutes_at_generation': quality['scan_age_minutes'],
                    'issues': quality['issues'],
                },
            },
            'quality': quality, 'target': cfg['target'],
            'target_observation': {'cell_ge20_pixels': latest['cell_ge20_pixels'],
                'cell_total_pixels': latest['cell_pixels'], 'cell_classified_pixels': latest['cell_classified_pixels'],
                'max_estimated_dbz': latest['cell_max_estimated_dbz'], 'qualifying_echo_at_target': bool(at_target),
                'echo_state': echo_state, 'coverage_percent': round(100 * coverage, 1)},
            'latest_group_count': len(current), 'nearest_group': nearest,
            'inbound_groups': inbound if quality['forecast_allowed'] else [],
            'validation': {'operationally_validated': False,
                'georeferencing_verified': cfg['georeferencing_verified'],
                'method': 'constant-velocity translation of observed echo pixel footprints',
                'limitations': ['PNG palette estimates, not raw radar reflectivity',
                    'white labels/extreme white echoes and non-palette pixels are unclassified',
                    'unclassified pixels do not mean zero rain',
                    'echoes aloft do not prove surface rainfall, wind gusts, or lightning',
                    'storm growth, decay, splitting, and merging are not forecast',
                    'no calibrated probability or arrival-time error interval yet']}}


def draw_diagnostic(path, frame, tracks, cfg, quality):
    base = frame['image'].convert('RGBA')
    overlay = Image.new('RGBA', base.size)
    draw = ImageDraw.Draw(overlay)
    for track in tracks:
        item, assessment = track['observations'][-1], track['assessment']
        if item['time'] != frame['time']:
            continue
        alert = assessment['status'] in ('inbound', 'echo_at_target') and quality['forecast_allowed']
        color = (255, 0, 255, 230) if alert else (0, 220, 210, 180)
        points = {tuple(p) for p in item['pixels']}
        for x, y in points:
            if any((x+dx, y+dy) not in points for dx, dy in ((1,0),(-1,0),(0,1),(0,-1))):
                draw.point((int(x), int(y)), fill=color)
        if alert:
            route = [tuple(o['centroid']) for o in track['observations']]
            if len(route) >= 2:
                draw.line(route, fill=(255,220,0,255), width=2)
            velocity = assessment['velocity']
            if velocity is not None:
                start = item['centroid']
                end = start + velocity * min(assessment['eta_min'], 10)
                draw.line([tuple(start), tuple(end)], fill=color, width=3)
                bearing = np.arctan2(end[1]-start[1], end[0]-start[0])
                for angle in (bearing+2.6, bearing-2.6):
                    wing = end + 7*np.array([np.cos(angle),np.sin(angle)])
                    draw.line([tuple(end),tuple(wing)], fill=color, width=2)
            label = f"{track['track_id'][:7]} {assessment['status']} ETA {assessment['eta_min']} min"
            draw.text(tuple(item['centroid']+[5,5]), label, fill='white', stroke_width=2, stroke_fill='black')
    tx, ty = cfg['target']['pixel_xy']
    scale = cfg['radar']['km_per_pixel']
    radius = cfg['target']['analysis_radius_km']/scale
    half = cfg['target']['cell_size_km']/(2*scale)
    draw.ellipse((tx-radius,ty-radius,tx+radius,ty+radius), outline=(255,0,255,230), width=2)
    draw.rectangle((tx-half,ty-half,tx+half,ty+half), outline='cyan', width=2)
    draw.text((tx+7,ty-15), 'SBM-2 5 km cell', fill='white', stroke_width=2, stroke_fill='black')
    draw.rectangle((0,0,800,23), fill=(0,0,0,230))
    banner = frame['time'].strftime('Scan %Y-%m-%d %H:%M:%S ICT')
    inbound_count = sum(t['assessment']['status'] == 'inbound' for t in tracks)
    banner += f' | inbound={inbound_count} | experimental' if quality['forecast_allowed'] else ' | DATA NOT VALID FOR FORECAST'
    draw.text((8,5), banner, fill='white')
    Image.alpha_composite(base, overlay).convert('RGB').save(path)


def run(image_dir, cfg, now=None, replay=False):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('Analysis clock must have a timezone')
    frames = read_frames(image_dir, cfg)
    quality = frame_quality(frames, cfg, now)
    old_state = None
    state_path = image_dir/'track-state.json'
    if state_path.exists():
        try:
            old_state = json.loads(state_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            print('WARNING: unreadable previous track state; new IDs will be allocated')
    tracks = associate(frames, cfg)
    state = assign_ids(tracks, old_state, cfg)
    for track in tracks:
        assessment = assess_track(track, len(frames)-1, cfg)
        if not quality['forecast_allowed'] and assessment['status'] == 'inbound':
            assessment.update(status='insufficient_data', reason='history_or_freshness_check_failed', eta_min=None)
        track['assessment'] = assessment
    # Publish files only after decoding and analysis of all inputs succeed.
    with tempfile.TemporaryDirectory(prefix='radar-output-', dir=image_dir.parent) as tmp:
        destination = Path(tmp)
        rows = export_tables(destination, frames, tracks, cfg, quality)
        result = summary_result(frames, rows, cfg, quality, now, replay)
        write_json(destination/'latest-status.json', result)
        write_json(destination/'track-state.json', state)
        write_json(destination/'analysis-manifest.json', {
            'model_version': MODEL_VERSION, 'config_sha256': config_digest(cfg),
            'frames': [{'file': f['file'], 'time_ict': f['time'].isoformat(), 'sha256': f['sha256']} for f in frames]})
        draw_diagnostic(destination/'diagnostic-cloud-shapes.png', frames[-1], tracks, cfg, quality)
        for output in destination.iterdir():
            shutil.move(str(output), image_dir/output.name)
    print(f"Latest scan ICT: {frames[-1]['time'].isoformat()}")
    print(f"Frames: {len(frames)} | Current groups: {result['latest_group_count']} | Status: {result['status']}")
    print(result['message_th'])
    print('Saved: cloud-tracks.csv, cloud-track-observations.csv, dbz-timeseries.csv, latest-status.json')
    print('Saved: track-state.json, analysis-manifest.json, diagnostic-cloud-shapes.png')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=Path, default=Path('radar_images'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--as-of', help='Timezone-aware timestamp for historical replay only')
    args = parser.parse_args()
    cfg = load_config(args.config)
    now = datetime.fromisoformat(args.as_of) if args.as_of else None
    result = run(args.image_dir, cfg, now=now, replay=bool(args.as_of))
    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, 'a', encoding='utf-8') as file:
            file.write('## SBM-2 Radar Analysis\n\n')
            file.write(f"- ภาพล่าสุด: {result['observed_at_ict']}\n")
            file.write(f"- สถานะ: `{result['status']}`\n")
            file.write(f"- จำนวนภาพ: {result['quality']['frame_count']}\n")
            file.write(f"- กลุ่มมุ่งเข้าภายใน 60 นาที: {len(result['inbound_groups'])}\n\n")
            file.write(result['message_th'] + '\n\n')
            file.write('ผลทดลองจาก echo เรดาร์ ยังไม่ผ่านการวัดความแม่นยำเทียบฝนจริง\n')


if __name__ == '__main__':
    main()
