"""Fetch bounded HTTPS input; merge the previous artifact across the ICT daily reset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit

from radar_core import FILENAME, ICT, load_config, scan_time, validate_png, write_json


def safe_image_url(url, cfg):
    if not isinstance(url, str):
        raise ValueError('Missing image URL')
    parsed = urlsplit(url)
    source = cfg['source']
    if (parsed.scheme not in ('http', 'https') or parsed.hostname != source['host']
            or parsed.username or parsed.password or parsed.port not in (None, 80, 443)
            or parsed.query or parsed.fragment):
        raise ValueError('Image URL is not on the configured radar source')
    path = '/' + '/'.join(part for part in parsed.path.split('/') if part)
    prefix = source['path_prefix']
    if not path.startswith(prefix) or not FILENAME.fullmatch(path[len(prefix):]):
        raise ValueError('Unexpected radar image URL path')
    scan_time(path)  # validates the calendar date as well as the filename shape
    return urlunsplit(('https', source['host'], path, '', ''))


def fetch(url, target, cfg, validator):
    """Total timeout per attempt covers slow streaming too; no uncontrolled redirects."""
    source = cfg['source']
    partial = target.with_suffix(target.suffix + '.part')
    error = 'unknown error'
    try:
        for attempt in range(source['attempts']):
            try:
                result = subprocess.run([
                    'curl', '--fail', '--silent', '--show-error', '--proto', '=https',
                    '--connect-timeout', str(source['connect_timeout_seconds']),
                    '--max-time', str(source['request_timeout_seconds']),
                    '--max-filesize', str(source['max_bytes']),
                    '--write-out', '%{http_code}', '--output', str(partial), url,
                ], capture_output=True, text=True, timeout=source['request_timeout_seconds'] + 10)
                if result.returncode or result.stdout.strip() != '200':
                    raise ValueError(f'HTTP {result.stdout.strip()}: {result.stderr.strip()[:200]}')
                validator(partial)
                partial.replace(target)
                return
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                error = str(exc)
                partial.unlink(missing_ok=True)
                if attempt + 1 < source['attempts']:
                    time.sleep(3 * (attempt + 1))
        raise RuntimeError(f'Failed after {source["attempts"]} attempts: {target.name}: {error}')
    finally:
        partial.unlink(missing_ok=True)


def parse_index(path, cfg):
    if path.stat().st_size > cfg['source']['max_bytes']:
        raise ValueError('Index is too large')
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict) or payload.get('rc') != 0 or not isinstance(payload.get('data'), list):
        raise ValueError('Unexpected radar API response')
    images = {}
    for record in payload['data']:
        if not isinstance(record, dict):
            raise ValueError('Invalid image record')
        url = safe_image_url(record.get('url'), cfg)
        filename = Path(urlsplit(url).path).name
        images[filename] = url
    if not images:
        raise ValueError('Radar API returned an empty image list')
    if len({scan_time(name) for name in images}) != len(images):
        raise ValueError('Ambiguous image variants for the same scan timestamp')
    return images


def run(output, cfg, now=None):
    now = now or datetime.now(timezone.utc)
    output.mkdir(parents=True, exist_ok=True)
    # Staging prevents failed/partial downloads from replacing the previous valid set.
    with tempfile.TemporaryDirectory(prefix='radar-download-', dir=output.parent) as tmp:
        stage = Path(tmp)
        index = stage/'index.json'
        fetch(cfg['source']['index_url'], index, cfg, lambda p: parse_index(p, cfg))
        images = parse_index(index, cfg)
        if any(scan_time(name) > now + timedelta(minutes=2) for name in images):
            raise ValueError("Source returned a future scan timestamp")
        cached = {}
        for path in output.glob('*dBZ.cappi.png'):
            try:
                if scan_time(path) > now + timedelta(minutes=2):
                    raise ValueError("Cached image timestamp is in the future")
                validate_png(path, cfg)
                cached[path.name] = path
            except (ValueError, OSError):
                print(f'Ignoring invalid cached image: {path.name}')
        selected = sorted(set(images) | set(cached), key=scan_time)[-cfg['analysis']['frames']:]
        newest = sorted(images, key=scan_time)[-2:]

        def download_one(filename):
            dest = stage/filename
            if filename in cached and (filename not in newest or filename not in images):
                shutil.copyfile(cached[filename], dest)
                origin = 'previous_artifact'
            else:
                fetch(images[filename], dest, cfg, lambda p: validate_png(p, cfg))
                origin = 'source'
            print(f'{scan_time(filename).isoformat()} | {origin} | {filename}')
            return {'file': filename, 'time_ict': scan_time(filename).isoformat(),
                    'origin': origin, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}

        with ThreadPoolExecutor(max_workers=cfg['source']['workers']) as pool:
            records = list(pool.map(download_one, selected))
        # All selected inputs have validated; now advance the rolling window.
        for filename in selected:
            (stage/filename).replace(output/filename)
        for path in output.glob('*dBZ.cappi.png'):
            if path.name not in selected:
                path.unlink()
        write_json(output/'download-manifest.json', {
            'schema_version': 1, 'fetched_at_ict': datetime.now(timezone.utc).astimezone(ICT).isoformat(),
            'api_records': len(images), 'retained_frames': len(selected), 'images': records})
        print(f'Kept {len(selected)} frames; fewer than 10 is a warm-up state, not a no-rain forecast.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=Path, default=Path('radar_images'))
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    run(args.image_dir, load_config(args.config))


if __name__ == '__main__':
    main()
