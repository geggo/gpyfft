from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

import os.path

AMDFFT = r'C:\Program Files (x86)\AMD\clAmdFft'
AMDAPP = r'C:\Program Files (x86)\AMD APP'

setup(
    name = 'Gpyfft',
    version = '0.0',
    description = '',
    cmdclass = {'build_ext': build_ext},
    #packages = [''],
    ext_modules = [Extension("gpyfft",
                             ["gpyfft.pyx"],
                             include_dirs = [".",
                                             os.path.join(AMDFFT,'include'),
                                             os.path.join(AMDAPP,'include'),
                             #                np.get_include()
                                             ],
                             extra_compile_args = [],
                             extra_link_args = [],
                             libraries = ['clAmdFft.Runtime'],
                             library_dirs = [os.path.join(AMDFFT, 'lib64/import')]
                             )
                   ],
    )

#see setup.py for EOS
