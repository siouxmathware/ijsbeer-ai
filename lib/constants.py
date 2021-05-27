"""
This module provides common constants to various other (sub)modules.

In particular the location of the data is defined here in a single location. The definition is always used dynamically
so that it may be overridden by tests, which are performed on smaller datasets.
"""

import os.path as op

DATA_DIR = op.join(op.dirname(__file__), '..', 'pipeline_data')
