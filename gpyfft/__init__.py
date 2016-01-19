from __future__ import absolute_import

try:
    from ._version import __date__ as date
    from ._version import version, version_info, hexversion, strictversion
except ImportError:
    raise RuntimeError("Do NOT use gpyfft from its sources: build it and use the built version")

from .gpyfftlib import GpyFFT, GpyFFT_Error, Plan
from .fft import *
