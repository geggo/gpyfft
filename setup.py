import os, os.path, platform
from setuptools import setup, Extension
from Cython.Distutils import build_ext

CLFFT_DIR = r'C:\Users\lab\Devel\clFFT'

CL_INCL_DIRs = []
if 'Windows' in platform.system():
    CL_DIR = os.getenv('AMDAPPSDKROOT')
    CL_INCL_DIRS = [os.path.join(CL_DIR, 'include')]

import Cython.Compiler.Options
Cython.Compiler.Options.generate_cleanup_code = 2

#TODO: see https://github.com/matthew-brett/du-cy-numpy

setup(
    name = 'Gpyfft',
    version = '0.2.1',
    description = 'A Python wrapper for the OpenCL FFT library clFFT by AMD',
    url = r"https://github.com/geggo/gpyfft",
    maintainer='Gregor Thalhammer',
    maintainer_email = 'gregor.thalhammer@gmail.com',
    license = 'LGPL',
    cmdclass = {'build_ext': build_ext},
    packages = ['gpyfft'],
    ext_modules = [Extension("gpyfft.gpyfftlib",
                             [os.path.join('gpyfft', 'gpyfftlib.pyx')],
                             include_dirs = [#".",
                                             os.path.join(CLFFT_DIR,'src', 'include'),
                                             ] + CL_INCL_DIRS,
                             extra_compile_args = [],
                             extra_link_args = [],
                             libraries = ['clFFT'],
                             library_dirs = [os.path.join(CLFFT_DIR, 'src', 'library'),
                                             ],
                             )
                   ],
    )

