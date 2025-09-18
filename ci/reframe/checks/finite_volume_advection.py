"""ReFrame test driving the Samurai finite-volume advection 2D demo."""

from __future__ import annotations

import os
from pathlib import Path

import reframe as rfm
import reframe.utility.sanity as sn

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@rfm.simple_test
class FiniteVolumeAdvection2DTest(rfm.RegressionTest):
    """Build and run the finite-volume advection 2D demo with ReFrame."""

    descr = 'Samurai finite-volume advection 2D benchmark'
    valid_systems = ['local:default']
    valid_prog_environs = ['builtin']
    sourcesdir = None
    build_system = 'CMake'

    def __init__(self) -> None:
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
        self.build_system.make_opts = ['finite-volume-advection-2d']
        self.build_system.max_concurrency = 1
        self.executable = os.path.join(str(build_dir), 'demos', 'FiniteVolume',
                                       'finite-volume-advection-2d')
        self.executable_opts = ['--Tf', '0.01', '--timers', '--nfiles', '1']
        self.sanity_patterns = sn.assert_found(
            r'total runtime', self.stdout)
        self.perf_patterns = {
            'total_runtime_s': sn.extractsingle(
                r'\btotal runtime\s+(?P<min>\S+)\s+\[\s*\d+\s*]\s+'
                r'\S+\s+\[\s*\d+\s*]\s+(?P<avg>\S+)',
                self.stdout, 'avg', float),
        }
        self.reference = {
            '*': {
                'total_runtime_s': (1.8, None, 0.50, 's'),
            }
        }
