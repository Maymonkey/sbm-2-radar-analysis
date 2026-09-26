"""Restore radar state and prune only older matching artifacts after a successful upload."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

from radar_core import FILENAME, load_config, scan_time, validate_png

ARTIFACT = 'sattahip-radar-latest-10'
ARTIFACT_PATTERN = re.compile(r'^sattahip-radar-latest-10(?:-\d+-\d+)?$')
WORKFLOW = '.github/workflows/check-source.yml'


def gh(*args):
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True, timeout=180).stdout


def artifacts(repo):
    pages = json.loads(gh('api', '--paginate', '--slurp',
        f'repos/{repo}/actions/artifacts?per_page=100'))
    return [item for page in pages for item in page['artifacts'] if ARTIFACT_PATTERN.fullmatch(item['name']) and not item['expired']]


def eligible(item, branch):
    return (item.get('workflow_run') or {}).get('head_branch') == branch


def correct_workflow(repo, run_id):
    run = json.loads(gh('api', f'repos/{repo}/actions/runs/{run_id}'))
    return run.get('path', '').split('@')[0] == WORKFLOW


def restore(repo, branch, current_run, output, cfg):
    output.mkdir(parents=True, exist_ok=True)
    candidates = sorted((a for a in artifacts(repo) if eligible(a, branch)),
                        key=lambda a: a['created_at'], reverse=True)
    for item in candidates[:5]:
        run_id = item['workflow_run']['id']
        if not correct_workflow(repo, run_id):
            continue
        try:
            with tempfile.TemporaryDirectory(prefix='radar-restore-') as tmp:
                gh('run', 'download', str(run_id), '--repo', repo, '--name', item['name'], '--dir', tmp)
                valid = []
                for path in Path(tmp).rglob('*dBZ.cappi.png'):
                    if path.is_symlink() or not FILENAME.fullmatch(path.name):
                        continue
                    scan_time(path)
                    validate_png(path, cfg)
                    valid.append(path)
                if not valid:
                    continue
                for path in sorted(valid, key=scan_time)[-cfg['analysis']['frames']:]:
                    shutil.copyfile(path, output/path.name)
                state = Path(tmp)/'track-state.json'
                if state.is_file() and not state.is_symlink() and state.stat().st_size < 4_000_000:
                    try:
                        json.loads(state.read_text(encoding='utf-8'))
                        shutil.copyfile(state, output/state.name)
                    except ValueError:
                        print('WARNING: ignoring invalid previous track-state JSON')
                print(f'Restored {min(len(valid), cfg["analysis"]["frames"])} scans from run {run_id}')
                return
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            print(f'WARNING: unable to restore artifact {item["id"]}: {exc}')
    print('No usable previous artifact. Starting in warm-up mode.')


def prune(repo, branch, current_run, current_id):
    items = artifacts(repo)
    current = next((a for a in items if str(a['id']) == str(current_id)
                    and str(a['workflow_run']['id']) == str(current_run)), None)
    if current is None:
        raise ValueError('New artifact was not verified; retaining older artifacts')
    for item in items:
        if (eligible(item, branch) and str(item['id']) != str(current_id)
                and item['created_at'] < current['created_at']
                and correct_workflow(repo, item['workflow_run']['id'])):
            gh('api', '--method', 'DELETE', f'repos/{repo}/actions/artifacts/{item["id"]}')
            print(f'Deleted older artifact: {item["id"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['restore', 'prune'])
    parser.add_argument('--image-dir', type=Path, default=Path('radar_images'))
    args = parser.parse_args()
    repo, branch, run_id = (os.environ[k] for k in ('GITHUB_REPOSITORY', 'GITHUB_REF_NAME', 'GITHUB_RUN_ID'))
    if args.action == 'restore':
        restore(repo, branch, run_id, args.image_dir, load_config())
    else:
        try:
            prune(repo, branch, run_id, os.environ['NEW_ARTIFACT_ID'])
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            # The new bundle is already safe; a cleanup error must not invalidate it.
            print(f'::warning::Artifact cleanup failed; older artifacts retained: {exc}')


if __name__ == '__main__':
    main()
