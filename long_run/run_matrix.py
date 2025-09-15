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
    }

    # Collect H5 files and check NaNs
    mpi_suffix = f"{base_filename}_size_{np_value}" if np_value >= 2 else base_filename
    h5_files = list_h5_files(out_dir, mpi_suffix)
    status['h5_files'] = [str(p) for p in h5_files]
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
            print(f"[{run_idx}/{total_runs}] {demo} {tag} — preparing…")

            # Safety guard: skip invalid combos
            if mx < mn:
                print(f"    skipping: max_level({mx}) < min_level({mn})")
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
            print(f"    np=1 running…")
            seq_status = run_one(demo, binary, 1, mn, mx, tf, args.nfiles, args.timers, seq_dir, base_filename)
            seq_ok = (seq_status['returncode'] == 0) and (not seq_status['nan_in_files']) and (len(seq_status['h5_files']) > 0)
            print(f"    np=1 {'OK' if seq_ok else 'FAIL'} ({seq_status['duration_sec']:.1f}s)")

            # 2) MPI runs (np>=2)
            for npv in [n for n in np_list if n >= 2]:
                mpi_dir = ensure_dir(demo_root / f"np{npv}")
                print(f"    np={npv} running…")
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

                print(f"    np={npv} {'OK' if mpi_ok else 'FAIL'} ({mpi_status['duration_sec']:.1f}s)"
                      + ("; compare=" + ("OK" if cmp_ok else ("FAIL" if cmp_ok is not None else "SKIP"))) )

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

    print("\nSummary")
    print(f"- Runs: {run_ok}/{total} OK" + (f", Skipped: {skipped}" if skipped else ""))
    if cmp_total:
        print(f"- MPI vs seq: {cmp_ok}/{cmp_total} OK")
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
