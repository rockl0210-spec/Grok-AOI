"""End-to-end smoke test on the four sample frames (needs the scientific stack)."""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures'


def test_sample_frames_end_to_end(tmp_path):
    assert (FIXTURES / 'Edge1' / '000' / '01.png').is_file()
    assert (FIXTURES / 'Side' / '000' / '02.png').is_file()
    out = tmp_path / 'out'
    cmd = [sys.executable, str(ROOT / 'grok_aoi.py'),
           '--db', str(FIXTURES), '--out', str(out),
           '--folders', 'Edge1/000', 'Side/000',
           '--nproc', '2', '--skip-eval']
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f'exit {proc.returncode}\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}')
    assert (out / 'final' / 'defects_per_frame.csv').is_file()
    assert (out / 'final' / 'final_defects.csv').is_file()
    assert (out / 'final' / 'final_defects_unique.csv').is_file()
    assert (out / 'final' / 'overlays' / 'Edge1_000' / 'contact_sheet.jpg').is_file()
    assert (out / 'final' / 'overlays' / 'Side_000' / 'contact_sheet.jpg').is_file()
    html = (out / 'report' / 'report.html').read_text(encoding='utf-8')
    assert '落塵' in html and 'L2' in html and 'nuisance' in html
    assert (out / 'report' / 'report.md').is_file()
    assert (out / 'report' / 'summary.json').is_file()
    summary = (out / 'filter' / 'summary.json').read_text(encoding='utf-8')
    assert 'rule_b' in summary or 'rule (b)' in summary or 'side_T2' in summary


def test_rule_b_defaults_off():
    from grok_aoi.filter import rules42
    assert rules42.OPT0['rule_b'] == 0
    assert rules42.OPT0['sideT2'] == 2
    rules42.apply_dataset([1, 2], [3], {'enabled': False, 'folder': 'Side_270', 'frame': 9,
                                        'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'max_area': 10})
    assert rules42.ANOM_EDGE == {1, 2}
    assert rules42.ANOM_SIDE == {3}
    assert rules42.E_STRIP['enabled'] is False
    rules42.apply_dataset([0, 45], [0, 15], {'enabled': True, 'folder': 'Side_270', 'frame': 15,
                                             'x0': 1380, 'x1': 1500, 'y0': 780, 'y1': 1460, 'max_area': 200})
    assert rules42.ANOM_EDGE == {0, 45}


if __name__ == '__main__':
    test_rule_b_defaults_off()
    print('rule_b default ok')
    with tempfile.TemporaryDirectory() as d:
        test_sample_frames_end_to_end(Path(d))
    print('smoke ok')
