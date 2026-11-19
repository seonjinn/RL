"""Installs gdn_metadata_reuse in every Python process that has this directory on PYTHONPATH."""

import importlib.util
import os
import sys

import gdn_metadata_reuse

gdn_metadata_reuse.install()

# Python runs only the first sitecustomize on sys.path, so run the next one as well
_here = os.path.dirname(os.path.abspath(__file__))
for _entry in sys.path:
    _candidate = os.path.join(os.path.abspath(_entry or "."), "sitecustomize.py")
    if os.path.dirname(_candidate) == _here or not os.path.isfile(_candidate):
        continue
    _spec = importlib.util.spec_from_file_location("_next_sitecustomize", _candidate)
    if _spec is not None and _spec.loader is not None:
        _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
    break
