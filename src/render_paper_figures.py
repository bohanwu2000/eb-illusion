"""Redraw all paper figures from bundled numerical results without rerunning simulations."""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure-dir', type=Path, default=repo/'figures')
    args = parser.parse_args()
    out = args.figure_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EB_ILLUSION_FIGDIR=str(out), MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    inputs = sorted(p for p in (repo/'results').rglob('*') if p.is_file())
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}

    def run(name, *arguments):
        subprocess.run([sys.executable, str(repo/'src'/name), *map(str, arguments)],
                       check=True, env=env, cwd=repo)

    with tempfile.TemporaryDirectory(prefix='illusion-render-') as scratch:
        tmp = Path(scratch)
        run('main_simulation_figure.py', '--from-results', repo/'results/simulations/main.npz')
        run('simulation_restructure.py', '--plot-only', '--summary', tmp/'finite_studies.json')
        run('nonparametric_npmle.py', '--from-results', repo/'results/nonparametric_npmle.json')
        run('referee_simulations.py', '--plot-only')
        run('referee_simulations.py', '--experiment', 'sensitivity', '--plot-only')
        run('realdata_figure.py')
        run('don_error_distribution.py', '--baseline',
            repo/'results/don-top-districts/district_estimates.csv', '--posterior-unit', '89',
            '--benchmark', 'full', '--output', tmp/'michigan')
        for extension in ('pdf', 'png'):
            shutil.copy2(tmp/'michigan'/f'full_data_error_distribution.{extension}',
                         out/f'don_small_rct.{extension}')
        run('nickerson_analysis.py', '--from-results', repo/'results/nickerson/site_estimates.csv',
            '--figure-dir', out)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    if before != after:
        raise RuntimeError('A bundled numerical input changed during rendering.')
    print('All paper figures saved to', out)
    print('Bundled numerical inputs are unchanged.')


if __name__ == '__main__':
    main()
