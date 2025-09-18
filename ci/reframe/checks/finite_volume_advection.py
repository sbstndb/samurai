"""ReFrame test driving the Samurai finite-volume advection 2D demo."""

from __future__ import annotations

import os
from pathlib import Path

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import parameter, variable
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEMO_CONFIGS = {
    'finite-volume-advection-2d': {
        'target': 'finite-volume-advection-2d',
        'default_tf': 0.01,
        'default_nfiles': 1,
        'extra_opts': ['--timers'],
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
