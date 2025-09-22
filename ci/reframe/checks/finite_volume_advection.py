"""ReFrame test driving the Samurai finite-volume advection 2D demo."""

from __future__ import annotations

import os
from pathlib import Path

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import parameter, variable
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_level_variants() -> list[tuple[str, int | None, int | None]]:
    spec = os.getenv('SAMURAI_LEVEL_VARIANTS')
    variants: list[tuple[str, int | None, int | None]] = []
    if spec:
        for raw in spec.split(','):
            token = raw.strip()
            if not token:
                continue

            parts = token.split(':')
            if len(parts) != 2:
                raise ValueError(
                    'SAMURAI_LEVEL_VARIANTS entries must be "min:max" pairs'
                )

            def _parse(value: str) -> int | None:
                value = value.strip().lower()
                if value in {'', 'none', '-'}:
                    return None

                return int(value)

            min_lvl = _parse(parts[0])
            max_lvl = _parse(parts[1])
            label = (
                f"min{min_lvl if min_lvl is not None else 'x'}"
                f"_max{max_lvl if max_lvl is not None else 'x'}"
            )
            variants.append((label, min_lvl, max_lvl))

    if not variants:
        variants = [
            ('default', None, None),
            ('min7_max7', 7, 7),
            ('min8_max8', 8, 8),
            ('min9_max9', 9, 9),
            ('min5_max8', 5, 8),
        ]

    return variants


LEVEL_VARIANTS = _load_level_variants()


DEMO_CONFIGS = {
    'finite-volume-advection-2d': {
        'target': 'finite-volume-advection-2d',
        'default_tf': 0.01,
        'default_nfiles': 1,
        'extra_opts': ['--timers'],
        'default_min_level': None,
        'default_max_level': None,
        'sanity': r'iteration 0',
        'perf_regex': (
            r'\btotal runtime\s+(?P<min>\S+)\s+\[\s*\d+\s*]\s+'
            r'\S+\s+\[\s*\d+\s*]\s+(?P<avg>\S+)'
        ),
        'reference': {
            1: (3.0, None, 0.80, 's'),
            2: (3.3, None, 0.80, 's'),
        },
    },
    'finite-volume-burgers': {
        'target': 'finite-volume-burgers',
        'default_tf': 0.05,
        'default_nfiles': 1,
        'extra_opts': [],
        'default_min_level': None,
        'default_max_level': None,
        'sanity': r'iteration 0',
        'perf_regex': None,
        'reference': None,
    },
}


@rfm.simple_test
class FiniteVolumeDemoTest(rfm.RegressionTest):
    """Build and run selected Samurai finite-volume demos with ReFrame."""

    demo = parameter(sorted(DEMO_CONFIGS.keys()), fmt=lambda x: x.replace('finite-volume-', ''))
    mpi_ranks = parameter([1, 2], fmt=lambda x: f'{x}ranks')
    level_variant = parameter(LEVEL_VARIANTS, fmt=lambda x: x[0])
    final_time = variable(float, value=-1.0)
    nfiles = variable(int, value=-1)

    valid_systems = ['local:default']
    valid_prog_environs = ['builtin']
    sourcesdir = None
    build_system = 'CMake'

    def __init__(self) -> None:
        cfg = DEMO_CONFIGS[self.demo]

        self.descr = f'Samurai {self.demo} demo ({self.mpi_ranks} MPI rank(s))'
        self.tags = {'benchmark', 'samurai'}
        self.maintainers = ['samurai-devs']
        self.time_limit = '1h'

        build_dir = Path('build') / 'reframe'
        self.build_system.builddir = str(build_dir)
        self.build_system.srcdir = str(PROJECT_ROOT)
        self.build_system.configuredir = str(PROJECT_ROOT)
        self.build_system.config_opts = ['-DWITH_MPI=ON']
        self.build_system.cxxflags = ['-mtune=native', '-march=native',
                                      '-O3', '-g']
        self.build_system.make_opts = [cfg['target']]
        self.build_system.max_concurrency = 1

        self.executable = os.path.join(str(build_dir), 'demos', 'FiniteVolume',
                                       cfg['target'])

        tf_value = self.final_time if self.final_time >= 0.0 else cfg['default_tf']
        nfiles_value = self.nfiles if self.nfiles >= 0 else cfg['default_nfiles']

        run_opts = []
        if tf_value is not None:
            run_opts += ['--Tf', str(tf_value)]

        if nfiles_value is not None:
            run_opts += ['--nfiles', str(nfiles_value)]

        run_opts += cfg.get('extra_opts', [])

        _, min_lvl, max_lvl = self.level_variant
        if min_lvl is None:
            min_lvl = cfg.get('default_min_level')

        if max_lvl is None:
            max_lvl = cfg.get('default_max_level')

        if min_lvl is not None:
            run_opts += ['--min-level', str(min_lvl)]

        if max_lvl is not None:
            run_opts += ['--max-level', str(max_lvl)]

        self.executable_opts = run_opts

        self.num_tasks = self.mpi_ranks
        self.num_tasks_per_node = self.mpi_ranks

        self.sanity_patterns = sn.assert_found(cfg['sanity'], self.stdout)

        if cfg['perf_regex'] is not None:
            self.perf_patterns = {
                'total_runtime_s': sn.extractsingle(
                    cfg['perf_regex'], self.stdout, 'avg', float
                )
            }
            ref_map = cfg['reference']
            if isinstance(ref_map, dict):
                perf_ref = ref_map[self.mpi_ranks]
            else:
                perf_ref = ref_map
            self.reference = {
                '*': {
                    'total_runtime_s': perf_ref,
                }
            }
        else:
            self.perf_patterns = {}
            self.reference = {}
