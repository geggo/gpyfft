import os.path, platform
from setuptools import setup, Extension
from Cython.Distutils import build_ext

CLFFT_DIR = r'/Users/gregor/Devel/clFFT'

import Cython.Compiler.Options
Cython.Compiler.Options.generate_cleanup_code = 2

setup(
    name = 'Gpyfft',
    version = '0.2',
    description = 'A Python wrapper for the OpenCL FFT library clFFT by AMD',
    url = r"https://github.com/geggo/gpyfft",
    maintainer='Gregor Thalhammer',
    maintainer_email = 'gregor.thalhammer@gmail.com',
    license = 'LGPL',
    cmdclass = {'build_ext': build_ext},
    packages = ['gpyfft'],
    ext_modules = [Extension("gpyfft.gpyfftlib",
                             [os.path.join('gpyfft', 'gpyfftlib.pyx')],
                             include_dirs = [".",
                                             os.path.join(CLFFT_DIR,'src', 'include'),
                                             ],
                             extra_compile_args = [],
                             extra_link_args = [],
                             libraries = ['clFFT'],
                             library_dirs = [os.path.join(CLFFT_DIR, 'src', 'library')]
                             )
                   ],
    )

