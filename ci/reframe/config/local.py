"""ReFrame site configuration for running Samurai benchmarks locally."""

from __future__ import annotations

import os

site_configuration = {
    'systems': [
        {
            'name': 'local',
            'descr': 'Single-node workstation',
            'hostnames': ['.*'],
            'modules_system': 'nomod',
            'prefix': 'ci/reframe',
            'partitions': [
                {
                    'name': 'default',
                    'descr': 'Local execution',
                    'scheduler': 'local',
                    'launcher': 'mpirun',
                    'environs': ['builtin'],
                    'max_jobs': 1,
                }
            ],
        }
    ],
    'environments': [
        {
            'name': 'builtin',
            'cc': os.getenv('CC', 'cc'),
            'cxx': os.getenv('CXX', 'c++'),
            'ftn': os.getenv('FC', ''),
        }
    ],
    'general': [
        {
            'check_search_path': ['ci/reframe/checks'],
            'check_search_recursive': True,
        }
    ],
    'logging': [
        {
            'handlers': [
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s',
                }
            ]
        }
    ],
}
