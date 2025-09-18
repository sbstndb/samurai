"""ReFrame test driving the Samurai finite-volume advection 2D demo."""

from __future__ import annotations

import os
from pathlib import Path

import reframe as rfm
import reframe.utility.sanity as sn

SPACK_SETUP = Path.home() / 'spack' / 'share' / 'spack' / 'setup-env.sh'
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
        self.env_vars = {
            'SAMURAI_CLI11_SPEC': os.getenv('SAMURAI_CLI11_SPEC', 'cli11@2.3.2'),
            'SAMURAI_SPACK_SPEC': os.getenv('SAMURAI_SPACK_SPEC', 'samurai@0.26.1'),
        }
        build_dir = Path('build') / 'reframe'
        self.build_system.builddir = str(build_dir)
        self.build_system.srcdir = str(PROJECT_ROOT)
        self.build_system.configuredir = str(PROJECT_ROOT)
        self.build_system.config_opts = [
            '-DWITH_MPI=ON',
            '-DSAMURAI_BUILD_TESTS=OFF',
            '-DSAMURAI_BUILD_BENCHMARKS=OFF',
            '-DSAMURAI_BUILD_DOC=OFF',
            '-DSAMURAI_BUILD_SHARED=ON',
            '-DCLI11_ROOT=${CLI11_ROOT}',
        ]
        self.build_system.cxxflags = ['-mtune=native', '-march=native',
                                      '-O3', '-g']
        self.build_system.make_opts = ['finite-volume-advection-2d']
        self.build_system.max_concurrency = 1
        self.prebuild_cmds = self._spack_setup_cmds()
        self.prerun_cmds = self._spack_setup_cmds()
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

    @staticmethod
    def _spack_setup_cmds() -> list[str]:
        setup_cmds = [
            f'. {SPACK_SETUP}',
            'spack load ${SAMURAI_CLI11_SPEC}',
            'spack load ${SAMURAI_SPACK_SPEC}',
            'export CLI11_ROOT=$(spack location -i ${SAMURAI_CLI11_SPEC})',
        ]
        return setup_cmds
