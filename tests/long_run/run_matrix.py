#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_int_list(s, default):
    if s is None:
        return default
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    return [int(x) for x in str(s).split(',') if x != '']


def parse_float_list(s, default):
    if s is None:
        return default
    if isinstance(s, (list, tuple)):
        return [float(x) for x in s]
    return [float(x) for x in str(s).split(',') if x != '']


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _supports_color() -> bool:
    try:
        return sys.stdout.isatty() and os.environ.get('TERM', '') not in ('', 'dumb')
    except Exception:
        return False


class _Colors:
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    CYAN = '\033[36m'
    RESET = '\033[0m'


def colorize(text: str, color: str) -> str:
    if _supports_color():
        return f"{color}{text}{_Colors.RESET}"
    return text


def ok_text(ok: bool | None) -> str:
    if ok is None:
        return colorize('SKIP', _Colors.YELLOW)
    return colorize('OK', _Colors.GREEN) if ok else colorize('FAIL', _Colors.RED)


def run_cmd(cmd, cwd: Path = None, log_file: Path | None = None, env=None) -> tuple[int, float, str]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
        out = proc.stdout or ''
    except Exception as e:
        out = f"EXCEPTION: {e}\n"
        proc = subprocess.CompletedProcess(cmd, returncode=1)
    dt = time.time() - t0
    if log_file is not None:
        try:
            log_file.write_text(out)
        except Exception:
            pass
    return proc.returncode, dt, out


def configure_and_build(build_dir: Path, targets: list[str], jobs: int | None, cmake_extra: list[str]):
    print("Configuring CMake (MPI + demos)…")
    ensure_dir(build_dir)
    cfg_cmd = [
        'cmake',
        '-DWITH_MPI=ON',
        '-DBUILD_DEMOS=ON',
        *cmake_extra,
        '..',
    ]
    code, dt, out = run_cmd(cfg_cmd, cwd=build_dir)
    if code != 0:
        print(out)
        raise RuntimeError("CMake configuration failed")
    print(f"Configured in {dt:.1f}s")

    print("Building targets…")
    build_cmd = ['cmake', '--build', '.', '--target', *targets]
    if jobs and jobs > 0:
        build_cmd += ['-j', str(jobs)]
    code, dt, out = run_cmd(build_cmd, cwd=build_dir)
    if code != 0:
        print(out)
        raise RuntimeError("Build failed")
    print(f"Built in {dt:.1f}s")


def find_binary(build_dir: Path, name: str) -> Path | None:
    # Search under build/ for an executable named `name`
    for p in build_dir.rglob(name):
        try:
            if os.access(p, os.X_OK) and p.is_file():
                return p
        except Exception:
            continue
    return None


def list_h5_files(dir_path: Path, base_prefix: str) -> list[Path]:
    # Return all .h5 files whose stem starts with base_prefix
    files = []
    for f in dir_path.glob('*.h5'):
        if f.stem.startswith(base_prefix):
            files.append(f)
    return sorted(files)


def h5_fields_have_nan(h5_path: Path) -> bool:
    import h5py
    import numpy as np
    with h5py.File(h5_path, 'r') as h5:
        if 'mesh' not in h5:
            return False
        mesh = h5['mesh']
        def iter_field_arrays(group):
            if 'fields' in group:
                for key in group['fields'].keys():
                    yield group['fields'][key][:]
            for k in group.keys():
                if k.startswith('rank_') and isinstance(group[k], h5py.Group):
                    if 'fields' in group[k]:
                        for key in group[k]['fields'].keys():
                            yield group[k]['fields'][key][:]

        for arr in iter_field_arrays(mesh):
            if np.issubdtype(arr.dtype, np.floating):
                if np.isnan(arr).any():
                    return True
    return False


def compare_seq_vs_mpi(compare_py: Path, seq_file: Path, mpi_file: Path) -> tuple[bool, str]:
    # Calls python/compare.py; returns (equal, output)
    cmd = [sys.executable, str(compare_py), seq_file.with_suffix('').as_posix(), mpi_file.with_suffix('').as_posix()]
    code, dt, out = run_cmd(cmd)
    ok = (code == 0) and ('are the same' in out)
    return ok, out


def build_run_tag(min_level: int, max_level: int, tf: float) -> str:
    tf_str = ("%g" % tf).replace('.', 'p')
    return f"lmin{min_level}_lmax{max_level}_Tf{tf_str}"


def run_one(demo_name: str,
            binary: Path,
            np_value: int,
            min_level: int,
            max_level: int,
            tf: float,
            nfiles: int,
            timers: bool,
            out_dir: Path,
            base_filename: str) -> dict:
    ensure_dir(out_dir)
    log_path = out_dir / 'run.log'

    cmd = ['mpirun', '-np', str(np_value), str(binary)]
    # Shared options
    cmd += [
        '--min-level', str(min_level),
        '--max-level', str(max_level),
        '--Tf', str(tf),
        '--nfiles', str(nfiles),
        '--path', str(out_dir),
        '--filename', base_filename,
    ]
    if timers:
        cmd += ['--timers']

    t0 = time.time()
    code, dt, out = run_cmd(cmd, log_file=log_path)
    duration = time.time() - t0

    status = {
        'returncode': code,
        'duration_sec': duration,
        'cmd': cmd,
        'log_path': str(log_path),
        'nan_in_files': False,
        'h5_files': [],
        'h5_count': 0,
        'fail_reasons': [],
    }

    # Collect H5 files and check NaNs
    mpi_suffix = f"{base_filename}_size_{np_value}" if np_value >= 2 else base_filename
    h5_files = list_h5_files(out_dir, mpi_suffix)
    status['h5_files'] = [str(p) for p in h5_files]
    status['h5_count'] = len(h5_files)
    any_nan = False
    for f in h5_files:
        try:
            if h5_fields_have_nan(f):
                any_nan = True
                break
        except Exception:
            # Failure to read => treat as error
            any_nan = True
            break
    status['nan_in_files'] = any_nan
    # Reasons
    reasons = []
    if code != 0:
        reasons.append(f"rc={code}")
    if len(h5_files) == 0:
        reasons.append("no_h5")
    if any_nan:
        reasons.append("nan")
    status['fail_reasons'] = reasons
    return status


def main():
    parser = argparse.ArgumentParser(description='Run long-run matrix for Samurai demos (FiniteVolume).')
    parser.add_argument('--build-dir', type=str, default='build')
    parser.add_argument('--skip-compile', action='store_true')
    parser.add_argument('--cmake-extra', type=str, default='')
    parser.add_argument('--jobs', type=str, default='auto', help='Parallel build jobs (int or auto)')
    parser.add_argument('--targets', type=str, default='finite-volume-advection-2d,finite-volume-burgers')
    parser.add_argument('--demos', type=str, default='advection_2d,burgers', help='Comma list: advection_2d,burgers')
    parser.add_argument('--np', type=str, default='1,2', help='Comma list of MPI sizes')
    parser.add_argument('--nfiles', type=int, default=4)
    parser.add_argument('--timers', action='store_true')
    parser.add_argument('--runs-root', type=str, default='long_run/runs')
    parser.add_argument('--session', type=str, default='', help='Optional session tag; default is timestamp')
    parser.add_argument('--no-compare', action='store_true', help='Disable MPI vs seq comparison')

    # Advection params
    parser.add_argument('--advection-min', type=str, default='3,4')
    parser.add_argument('--advection-max', type=str, default='5,6')
    parser.add_argument('--advection-tf', type=str, default='0.1,0.2')

    # Burgers params
    parser.add_argument('--burgers-min', type=str, default='0,1')
    parser.add_argument('--burgers-max', type=str, default='2,3')
    parser.add_argument('--burgers-tf', type=str, default='0.1,0.2')

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = (repo_root / args.build_dir).resolve()
    runs_root = (repo_root / args.runs_root).resolve()
    # Build a session directory to avoid clobbering previous runs
    session_tag = args.session.strip() or time.strftime('%Y%m%d-%H%M%S')
    session_root = ensure_dir(runs_root / session_tag)
    print(f"Runs session directory: {session_root}")

    targets = [t for t in args.targets.split(',') if t]
    demos = [d.strip() for d in args.demos.split(',') if d.strip()]
    np_list = parse_int_list(args.np, default=[1, 2])

    advec_min = parse_int_list(args.advection_min, default=[3, 4])
    advec_max = parse_int_list(args.advection_max, default=[5, 6])
    advec_tf = parse_float_list(args.advection_tf, default=[0.1, 0.2])

    burg_min = parse_int_list(args.burgers_min, default=[0, 1])
    burg_max = parse_int_list(args.burgers_max, default=[2, 3])
    burg_tf = parse_float_list(args.burgers_tf, default=[0.1, 0.2])

    jobs = None
    if str(args.jobs).lower() != 'auto':
        try:
            jobs = int(args.jobs)
        except Exception:
            jobs = None

    cmake_extra = [x for x in args.cmake_extra.split() if x]

    if not args.skip_compile:
        configure_and_build(build_dir, targets, jobs, cmake_extra)
    else:
        print("Skipping compile as requested.")

    # Locate demo binaries
    binaries: dict[str, Path] = {}
    name_map = {
        'advection_2d': 'finite-volume-advection-2d',
        'burgers': 'finite-volume-burgers',
    }
    for demo in demos:
        if demo not in name_map:
            print(f"Unknown demo: {demo}. Skipping.")
            continue
        exe_name = name_map[demo]
        exe_path = find_binary(build_dir, exe_name)
        if not exe_path:
            # Try legacy path hint
            hint = build_dir / 'demosFiniteVolume' / exe_name
            if hint.exists():
                exe_path = hint
        if not exe_path:
            raise FileNotFoundError(f"Could not locate executable {exe_name} under {build_dir}")
        binaries[demo] = exe_path

    # Ensure compare.py is available
    compare_py = repo_root / 'python' / 'compare.py'
    if not compare_py.exists():
        print("WARNING: python/compare.py not found; MPI vs sequential compare disabled.")
        args.no_compare = True

    # Build parameter grids per demo
    grids = {}
    if 'advection_2d' in demos:
        combos = [(mn, mx, tf) for mn in advec_min for mx in advec_max for tf in advec_tf if mx >= mn]
        grids['advection_2d'] = combos
    if 'burgers' in demos:
        combos = [(mn, mx, tf) for mn in burg_min for mx in burg_max for tf in burg_tf if mx >= mn]
        grids['burgers'] = combos

    total_runs = sum(len(grids.get(d, [])) for d in demos)
    if total_runs == 0:
        print("No runs to schedule. Check your parameters.")
        return

    overall_results = []
    run_idx = 0
    for demo in demos:
        binary = binaries[demo]
        param_grid = grids.get(demo, [])
        for (mn, mx, tf) in param_grid:
            run_idx += 1
            tag = build_run_tag(mn, mx, tf)
            print(f"[{run_idx}/{total_runs}] {demo} {tag} — {colorize('preparing…', _Colors.CYAN)}")

            # Safety guard: skip invalid combos
            if mx < mn:
                print(f"    {colorize('skipping', _Colors.YELLOW)}: max_level({mx}) < min_level({mn})")
                overall_results.append({
                    'demo': demo,
                    'tag': tag,
                    'np': None,
                    'run_ok': False,
                    'compare_ok': None,
                    'seq_ok': False,
                    'seq_log': None,
                    'mpi_log': None,
                    'seq_cmd': None,
                    'mpi_cmd': None,
                    'skipped': True,
                    'skip_reason': 'max<min',
                })
                continue

            demo_root = ensure_dir(session_root / demo / tag)
            base_filename = f"{demo}_{tag}"

            # 1) Sequential run (np=1)
            seq_dir = ensure_dir(demo_root / 'np1')
            print(f"    np=1 {colorize('running…', _Colors.CYAN)}")
            seq_status = run_one(demo, binary, 1, mn, mx, tf, args.nfiles, args.timers, seq_dir, base_filename)
            seq_ok = (seq_status['returncode'] == 0) and (not seq_status['nan_in_files']) and (len(seq_status['h5_files']) > 0)
            if seq_ok:
                print(f"    np=1 {ok_text(seq_ok)} ({seq_status['duration_sec']:.1f}s) [h5={seq_status['h5_count']}]")
            else:
                reason = ','.join(seq_status.get('fail_reasons', [])) or 'unknown'
                print(f"    np=1 {ok_text(seq_ok)} ({seq_status['duration_sec']:.1f}s) [{reason}]")

            # 2) MPI runs (np>=2)
            for npv in [n for n in np_list if n >= 2]:
                mpi_dir = ensure_dir(demo_root / f"np{npv}")
                print(f"    np={npv} {colorize('running…', _Colors.CYAN)}")
                mpi_status = run_one(demo, binary, npv, mn, mx, tf, args.nfiles, args.timers, mpi_dir, base_filename)
                mpi_ok = (mpi_status['returncode'] == 0) and (not mpi_status['nan_in_files']) and (len(mpi_status['h5_files']) > 0)

                cmp_ok = None
                cmp_msg = ''
                if mpi_ok and seq_ok and not args.no_compare:
                    # Compare every suffix present in seq that also exists for mpi
                    try:
                        seq_files = [Path(p) for p in seq_status['h5_files']]
                        # Build a map from suffix to path
                        seq_map = {}
                        for sf in seq_files:
                            # suffix is the part after base_filename
                            suf = sf.stem[len(base_filename):]
                            # Normalize optional _size_1 when compiled with MPI
                            if suf.startswith("_size_"):
                                # Strip leading _size_<int>
                                rest = suf.split('_', 3)  # ['', 'size', '<n>', rest]
                                if len(rest) >= 3 and rest[1] == 'size':
                                    # reconstruct remainder starting at index 3 if exists
                                    suf = '' if len(rest) < 4 else ('_' + rest[3])
                            seq_map[suf] = sf

                        mpi_files = [Path(p) for p in mpi_status['h5_files']]
                        # mpi base is base_filename + _size_<npv>
                        mpi_prefix = f"{base_filename}_size_{npv}"
                        mpi_map = {}
                        for mf in mpi_files:
                            suf = mf.stem[len(mpi_prefix):]
                            mpi_map[suf] = mf

                        # Intersect suffixes
                        common_suffixes = sorted(set(seq_map.keys()) & set(mpi_map.keys()))
                        if len(common_suffixes) == 0:
                            cmp_ok = False
                            cmp_msg = 'No common files to compare'
                        else:
                            all_equal = True
                            details = []
                            for suf in common_suffixes:
                                ok, out = compare_seq_vs_mpi(compare_py, seq_map[suf], mpi_map[suf])
                                details.append((suf, ok))
                                if not ok:
                                    all_equal = False
                                    # keep going to report more mismatches
                            cmp_ok = all_equal
                            cmp_msg = json.dumps(details)
                    except Exception as e:
                        cmp_ok = False
                        cmp_msg = f"compare error: {e}"

                cmp_disp = ok_text(cmp_ok)
                # detail reason for compare
                cmp_detail = ''
                if cmp_ok is None:
                    if not mpi_ok:
                        cmp_detail = ' [mpi_failed]'
                    elif not seq_ok:
                        cmp_detail = ' [seq_failed]'
                    elif args.no_compare:
                        cmp_detail = ' [disabled]'
                elif cmp_ok is False:
                    # try to provide a short detail
                    short = ''
                    if isinstance(cmp_msg, str) and cmp_msg.startswith('No common files'):
                        short = 'no_common'
                    else:
                        try:
                            details = json.loads(cmp_msg)
                            # find first failing suffix
                            fail = next((suf for (suf, ok) in details if not ok), None)
                            if fail is not None:
                                short = f"mismatch:{fail or '(root)'}"
                        except Exception:
                            short = ''
                    if short:
                        cmp_detail = f" [{short}]"

                if mpi_ok:
                    print(f"    np={npv} {ok_text(mpi_ok)} ({mpi_status['duration_sec']:.1f}s) [h5={mpi_status['h5_count']}]" \
                          f"; compare={cmp_disp}{cmp_detail}")
                else:
                    reason = ','.join(mpi_status.get('fail_reasons', [])) or 'unknown'
                    print(f"    np={npv} {ok_text(mpi_ok)} ({mpi_status['duration_sec']:.1f}s) [{reason}]" \
                          f"; compare={cmp_disp}{cmp_detail}")

                overall_results.append({
                    'demo': demo,
                    'tag': tag,
                    'np': npv,
                    'run_ok': mpi_ok,
                    'compare_ok': cmp_ok,
                    'seq_ok': seq_ok,
                    'seq_log': seq_status['log_path'],
                    'mpi_log': mpi_status['log_path'],
                    'seq_cmd': ' '.join(seq_status.get('cmd', [])) if isinstance(seq_status.get('cmd'), list) else str(seq_status.get('cmd')),
                    'mpi_cmd': ' '.join(mpi_status.get('cmd', [])) if isinstance(mpi_status.get('cmd'), list) else str(mpi_status.get('cmd')),
                })

            # Record seq outcome too
            overall_results.append({
                'demo': demo,
                'tag': tag,
                'np': 1,
                'run_ok': seq_ok,
                'compare_ok': None,
                'seq_ok': seq_ok,
                'seq_log': seq_status['log_path'],
                'mpi_log': None,
                'seq_cmd': ' '.join(seq_status.get('cmd', [])) if isinstance(seq_status.get('cmd'), list) else str(seq_status.get('cmd')),
                'mpi_cmd': None,
            })

    # Summary
    total = len(overall_results)
    run_ok = sum(1 for r in overall_results if r.get('run_ok'))
    cmp_checked = [r for r in overall_results if r['compare_ok'] is not None]
    cmp_ok = sum(1 for r in cmp_checked if r['compare_ok'])
    cmp_total = len(cmp_checked)
    skipped = sum(1 for r in overall_results if r.get('skipped'))

    # Breakdown seq vs mpi
    seq_entries = [r for r in overall_results if r.get('np') == 1]
    mpi_entries = [r for r in overall_results if isinstance(r.get('np'), int) and r.get('np') >= 2]
    seq_total = len(seq_entries)
    mpi_total = len(mpi_entries)
    seq_ok_cnt = sum(1 for r in seq_entries if r.get('run_ok'))
    mpi_ok_cnt = sum(1 for r in mpi_entries if r.get('run_ok'))

    print("\nSummary")
    runs_all_ok = (run_ok + skipped == total) and (total > 0)
    print(f"- Runs: {run_ok}/{total} " + (colorize('OK', _Colors.GREEN) if runs_all_ok else colorize('FAIL', _Colors.RED))
          + (f", Skipped: {skipped}" if skipped else ""))
    print(f"- Sequential: {seq_ok_cnt}/{seq_total} " + (colorize('OK', _Colors.GREEN) if seq_ok_cnt==seq_total else colorize('FAIL', _Colors.RED)))
    print(f"- MPI: {mpi_ok_cnt}/{mpi_total} " + (colorize('OK', _Colors.GREEN) if mpi_ok_cnt==mpi_total else colorize('FAIL', _Colors.RED)))
    if cmp_total:
        cmp_all_ok = (cmp_ok == cmp_total)
        print(f"- MPI vs seq: {cmp_ok}/{cmp_total} " + (colorize('OK', _Colors.GREEN) if cmp_all_ok else colorize('FAIL', _Colors.RED)))
    else:
        print("- MPI vs seq: SKIPPED")

    # Save machine-readable summary
    summary_path = ensure_dir(session_root) / 'summary.json'
    try:
        summary_path.write_text(json.dumps(overall_results, indent=2))
        print(f"- Wrote {summary_path}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
