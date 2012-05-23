from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

import os.path

#LIBRAW = 'LibRaw-0.14.0-Alpha3'

setup(
    name = 'Gpyfft',
    version = '0.0',
    description = '',
    cmdclass = {'build_ext': build_ext},
    #packages = [''],
    ext_modules = [Extension("_gpyfft",
                             ["gpyfft.pyx"],
                             #include_dirs = [".",
                             #                os.path.join(LIBRAW,'libraw'),
                             #                np.get_include()],
                             extra_compile_args = [],
                             extra_link_args = [],
                             #libraries = ['raw_r', 'stdc++'],
                             #library_dirs = [os.path.join(LIBRAW, 'lib')]
                             )
                   ],
    )

#see setup.py for EOS
