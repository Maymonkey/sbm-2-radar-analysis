"""Regression tests for false arrivals, midnight rollover, masked labels and stale data."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from radar_core import (ICT, associate, assign_ids, assess_track, components, decode,
                        gap_km, load_config, masks, scan_time, shape_arrival_minutes,
                        validate_png)
from analyze_radar import frame_quality, run as analyze
from download_radar import safe_image_url, parse_index, fetch, run as download
from artifact_store import eligible, prune

CFG = load_config()
BASE = datetime(2026, 9, 27, 0, 0, 3, tzinfo=ICT)


def component(x, y):
    points = np.array([(x+dx, y+dy) for dy in range(-1,2) for dx in range(-1,2)])
    return {'pixels': points, 'centroid': points.mean(axis=0), 'pixel_count': len(points),
            'gap_km': gap_km(points, CFG), 'peak_dbz': 30., 'peak_dbz_lower': 29.,
            'peak_dbz_upper': 31., 'mean_dbz': 30., 'fingerprint': f'{x}:{y}'}


def frames_for(locations, gap=6):
    return [{'time': BASE+timedelta(minutes=i*gap), 'file': f'frame{i}',
             'components': [component(x,y)]} for i,(x,y) in enumerate(locations)]


def evaluate(frames):
    tracks = associate(frames, CFG)
    assign_ids(tracks, None, CFG)
    return tracks, [assess_track(t, len(frames)-1, CFG) for t in tracks]


def make_image(path, points=(), color=(51,255,0)):
    im = Image.new('RGB', tuple(CFG['image_size']), (128,192,255))
    for band in CFG['legend']['bands']:
        for y in range(band['top_y'], band['bottom_y']):
            im.putpixel((CFG['legend']['sample_x'], y), tuple(band['rgb']))
    for point in points:
        im.putpixel(tuple(point), color)
    im.save(path)


class GeometryTests(unittest.TestCase):
    def test_head_on_shape_arrival(self):
        # East of target, travelling west at one pixel per minute.
        eta = shape_arrival_minutes(np.array([[386,321]]), np.array([-1.,0.]), CFG)
        self.assertAlmostEqual(eta, 20-5/(2*.6))

    def test_near_heading_can_miss_target(self):
        # Just 26 degrees off the target bearing, but the small shape passes north.
        self.assertIsNone(shape_arrival_minutes([[386,311]], [-1.,0.], CFG))

    def test_shape_edge_can_hit_when_centroid_misses(self):
        self.assertIsNotNone(shape_arrival_minutes([[386,311],[386,320]], [-1.,0.], CFG))

    def test_away_stationary_and_at_target(self):
        self.assertIsNone(shape_arrival_minutes([[386,321]], [1.,0.], CFG))
        self.assertIsNone(shape_arrival_minutes([[386,321]], [0.,0.], CFG))
        self.assertEqual(shape_arrival_minutes([[366,321]], [0.,0.], CFG), 0)


class TrackingTests(unittest.TestCase):
    def test_inbound_with_motion_speed(self):
        tracks, results = evaluate(frames_for([(406,321),(400,321),(394,321)]))
        self.assertEqual(len(tracks), 1)
        self.assertEqual(results[0]['status'], 'inbound')
        self.assertEqual(results[0]['speed_kmh'], 36)
        self.assertAlmostEqual(results[0]['eta_min'], 22.8, places=1)

    def test_heading_toward_but_trajectory_misses(self):
        _, results = evaluate(frames_for([(406,309),(400,309),(394,309)]))
        self.assertEqual(results[0]['status'], 'passing_target')
        self.assertIsNone(results[0]['eta_min'])

    def test_long_range_has_no_precise_eta(self):
        _, results = evaluate(frames_for([(470,321),(464,321),(458,321)]))
        self.assertEqual(results[0]['status'], 'approaching_outside_horizon')
        self.assertIsNone(results[0]['eta_min'])

    def test_old_inbound_track_is_ended(self):
        frames = frames_for([(406,321),(400,321),(394,321)])
        frames.append({'time': BASE+timedelta(minutes=18), 'file':'last','components':[]})
        _, results = evaluate(frames)
        self.assertEqual(results[0]['status'], 'ended')
        self.assertIsNone(results[0]['eta_min'])

    def test_real_scan_gap_blocks_motion(self):
        _, results = evaluate(frames_for([(406,321),(400,321),(394,321)], gap=12))
        self.assertNotEqual(results[0]['status'], 'inbound')

    def test_ambiguous_match_blocks_eta(self):
        tracks, _ = evaluate(frames_for([(406,321),(400,321),(394,321)]))
        tracks[0]['observations'][1]['match_ambiguous'] = True
        result = assess_track(tracks[0], 2, CFG)
        self.assertEqual(result['status'], 'uncertain_motion')
        self.assertIsNone(result['eta_min'])

    def test_rolling_window_preserves_id(self):
        frames = frames_for([(412,321),(406,321),(400,321),(394,321)])
        first = associate(frames[:3], CFG)
        state = assign_ids(first, None, CFG)
        second = associate(frames[1:], CFG)
        assign_ids(second, state, CFG)
        self.assertEqual(first[0]['track_id'], second[0]['track_id'])

    def test_invalid_state_does_not_break_analysis(self):
        tracks = associate(frames_for([(406,321),(400,321),(394,321)]), CFG)
        assign_ids(tracks, ["invalid state"], CFG)
        self.assertTrue(tracks[0]['track_id'].startswith('T'))

    def test_crossing_tracks_follow_predictions(self):
        frames = []
        for i, groups in enumerate([[(330,300),(354,310)],[(336,300),(348,310)],[(342,300),(342,310)],[(348,300),(336,310)]]):
            frames.append({'time':BASE+timedelta(minutes=i*6), 'file':str(i),
                           'components':[component(*p) for p in groups]})
        tracks = associate(frames, CFG)
        self.assertEqual(len(tracks), 2)
        self.assertEqual([o['centroid'][0] for o in tracks[0]['observations']], [330,336,342,348])


class DecoderTests(unittest.TestCase):
    def test_labels_excluded_stationary_coloured_echo_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'2026092617000300dBZ.cappi.png'
            points = [(366+x,321+y) for x in range(3) for y in range(3)]
            make_image(path, points, color=(255,51,0))
            with Image.open(path) as src:
                im=src.convert('RGB')
            im.putpixel((370,321),(255,255,255))
            roi,_=masks(CFG)
            for _ in range(10):
                lo,hi=decode(im,CFG,roi)
                self.assertTrue(np.isnan(lo[321,370]))
                self.assertEqual(len(components(lo,hi,CFG)),1)
                self.assertGreater(lo[321,366],35)

    def test_threshold_marker_is_bin_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'2026092617000300dBZ.cappi.png'
            make_image(path, [(366,321)])
            with Image.open(path) as im:
                lo,hi=decode(im,CFG,masks(CFG)[0])
            self.assertEqual(lo[321,366],20)
            self.assertEqual(hi[321,366],21)

    def test_changed_legend_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'2026092617000300dBZ.cappi.png'
            make_image(path)
            with Image.open(path) as src:
                im=src.convert('RGB')
            im.putpixel((850,100),(1,2,3))
            with self.assertRaises(ValueError): decode(im,CFG,masks(CFG)[0])

    def test_component_at_image_edge_no_index_overflow(self):
        lo=np.full((800,1076),np.nan);hi=lo.copy()
        lo[798:800,799:802]=30;hi[798:800,799:802]=32
        self.assertEqual(len(components(lo,hi,CFG)),1)

    def test_corrupt_png_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.png';path.write_bytes(b'\x89PNG\r\n\x1a\n')
            with self.assertRaises(OSError): validate_png(path,CFG)


class QualityTests(unittest.TestCase):
    def test_filename_utc_to_ict_midnight(self):
        self.assertEqual(scan_time('2026092617000300dBZ.cappi.png'), BASE)

    def test_warmup_stale_future_and_gaps(self):
        frames=frames_for([(400,300)]*10)
        self.assertTrue(frame_quality(frames,CFG,BASE+timedelta(minutes=56))['forecast_allowed'])
        self.assertFalse(frame_quality(frames[:1],CFG,BASE)['forecast_allowed'])
        self.assertFalse(frame_quality(frames,CFG,BASE+timedelta(hours=3))['forecast_allowed'])
        self.assertFalse(frame_quality(frames,CFG,BASE)['forecast_allowed'])
        frames[5]['time']+=timedelta(minutes=6)
        self.assertIn('missing_scans_in_history',frame_quality(frames,CFG,BASE+timedelta(minutes=56))['issues'])


class DownloadTests(unittest.TestCase):
    def test_url_normalization_and_rejection(self):
        url='http://file.royalrain.go.th//opendata//radar_data//cappi//sattahip//2026092617000300dBZ.cappi.png'
        self.assertTrue(safe_image_url(url,CFG).startswith('https://file.royalrain.go.th/opendata/'))
        for bad in ['https://evil.example/x.png','file:///etc/passwd',url+'?x=y',url.replace('sattahip','other'),url.replace('202609','202699')]:
            with self.assertRaises(ValueError): safe_image_url(bad,CFG)

    def test_api_duplicates_and_empty_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.json'
            entry={'url':'http://file.royalrain.go.th/opendata/radar_data/cappi/sattahip/2026092617000300dBZ.cappi.png'}
            path.write_text(json.dumps({'rc':0,'data':[entry,entry]}))
            self.assertEqual(len(parse_index(path,CFG)),1)
            path.write_text(json.dumps({'rc':0,'data':[]}))
            with self.assertRaises(ValueError): parse_index(path,CFG)

    def test_timeout_retry_never_destroys_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'radar.png';target.write_bytes(b'existing valid bytes')
            with patch('download_radar.subprocess.run',side_effect=subprocess.TimeoutExpired('curl',130)) as mocked, patch('download_radar.time.sleep'):
                with self.assertRaises(RuntimeError): fetch('https://example.invalid',target,CFG,lambda p:None)
            self.assertEqual(mocked.call_count,3)
            self.assertEqual(target.read_bytes(),b'existing valid bytes')
            self.assertFalse(target.with_suffix('.png.part').exists())

    def test_failed_batch_preserves_cached_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'images';root.mkdir()
            old=root/'2026092616540300dBZ.cappi.png'
            make_image(old)
            old_bytes=old.read_bytes()
            def fake_fetch(url,target,cfg,validator):
                if target.suffix=='.json':
                    target.write_text(json.dumps({'rc':0,'data':[{'url':
                        'https://file.royalrain.go.th/opendata/radar_data/cappi/sattahip/2026092617000300dBZ.cappi.png'}]}))
                else:
                    target.write_bytes(b'partial')
                    raise RuntimeError('upstream unavailable')
            with patch('download_radar.fetch',side_effect=fake_fetch):
                with self.assertRaises(RuntimeError): download(root,CFG,now=BASE)
            self.assertEqual(old.read_bytes(),old_bytes)
            self.assertEqual(len(list(root.glob('*dBZ.cappi.png'))),1)

    def test_midnight_merge_keeps_ten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'images';root.mkdir()
            # Previous nine scans through 23:54 ICT plus the new 00:00 scan.
            for i in range(10):
                t=BASE-timedelta(minutes=(10-i)*6)
                name=t.astimezone(timezone.utc).strftime('%Y%m%d%H%M%S')+'00dBZ.cappi.png'
                make_image(root/name)
            new='2026092617000300dBZ.cappi.png'
            def fake_fetch(url,target,cfg,validator):
                if target.suffix=='.json':
                    target.write_text(json.dumps({'rc':0,'data':[{'url':cfg['source']['path_prefix']+new}]}))
                    payload=json.loads(target.read_text());payload['data'][0]['url']='https://file.royalrain.go.th'+payload['data'][0]['url']
                    target.write_text(json.dumps(payload))
                else:make_image(target)
                validator(target)
            with patch('download_radar.fetch',side_effect=fake_fetch): download(root,CFG,now=BASE)
            paths=sorted(root.glob('*dBZ.cappi.png'))
            self.assertEqual(len(paths),10)
            self.assertEqual(paths[-1].name,new)
            self.assertEqual(scan_time(paths[0]),BASE-timedelta(minutes=54))


class ArtifactTests(unittest.TestCase):
    def test_only_same_branch_eligible(self):
        item={'workflow_run':{'head_branch':'main','id':10}}
        self.assertTrue(eligible(item,'main'))
        self.assertTrue(eligible(item,'main'))
        self.assertFalse(eligible(item,'feature'))

    def test_cleanup_does_not_delete_newer_or_other_branch(self):
        def item(i,branch='main'):
            return {'id':i,'created_at':f'2026-09-27T00:0{i}:00Z','workflow_run':{'head_branch':branch,'id':i}}
        with patch('artifact_store.artifacts',return_value=[item(1),item(2,'other'),item(3),item(4)]),patch('artifact_store.correct_workflow',return_value=True),patch('artifact_store.gh') as gh:
            prune('owner/repo','main',3,3)
            gh.assert_called_once_with('api','--method','DELETE','repos/owner/repo/actions/artifacts/1')


class EndToEndTests(unittest.TestCase):
    def test_ten_pngs_export_history_and_current_only_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(10):
                t=BASE+timedelta(minutes=6*i)
                name=t.astimezone(timezone.utc).strftime('%Y%m%d%H%M%S')+'00dBZ.cappi.png'
                x=450-6*i
                make_image(root/name,[(x+dx,321+dy) for dx in range(3) for dy in range(3)])
            result=analyze(root,CFG,now=BASE+timedelta(minutes=56),replay=True)
            self.assertEqual(result['status'],'inbound_echo')
            self.assertTrue(result['quality']['forecast_allowed'])
            self.assertEqual(len(result['inbound_groups']),1)
            self.assertTrue(result['user_summary']['forecast']['available'])
            self.assertEqual(len(result['user_summary']['forecast']['arrivals']),1)
            self.assertEqual(result['user_summary']['target_echo']['state'],'unknown')
            self.assertEqual(result['user_summary']['target_echo']['coverage_percent'],0.0)
            self.assertEqual(result['user_summary']['freshness']['valid_until_ict'],
                             (BASE+timedelta(minutes=72)).isoformat())
            with (root/'cloud-track-observations.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),10)
            self.assertEqual(len({r['track_id'] for r in rows}),1)
            self.assertTrue((root/'diagnostic-cloud-shapes.png').exists())
            stale=analyze(root,CFG,now=BASE+timedelta(hours=5))
            self.assertEqual(stale['status'],'stale_data')
            self.assertEqual(stale['inbound_groups'],[])

    def test_one_frame_is_insufficient_not_no_rain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_image(root/'2026092617000300dBZ.cappi.png')
            result=analyze(root,CFG,now=BASE)
            self.assertEqual(result['status'],'insufficient_data')
            self.assertEqual(result['inbound_groups'],[])
            self.assertEqual(result['user_summary']['target_echo']['state'],'unknown')
            self.assertEqual(result['user_summary']['data_quality']['state'],'degraded')


if __name__=='__main__': unittest.main()
