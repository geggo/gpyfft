import os.path, platform
from setuptools import setup, Extension
from Cython.Distutils import build_ext

AMDFFT = r'C:\Program Files (x86)\AMD\clAmdFft'
AMDAPP = r'C:\Program Files (x86)\AMD APP'

bits,foo = platform.architecture()
if bits == '64bit':
    AMDFFTLIBDIR = os.path.join(AMDFFT, 'lib64')
else:
    AMDFFTLIBDIR = os.path.join(AMDFFT, 'lib32')

setup(
    name = 'Gpyfft',
    version = '0.1',
    description = 'A Python wrapper for the OpenCL FFT library APPML/clAmdFft from AMD',
    url = r"https://github.com/geggo/gpyfft",
    maintainer='Gregor Thalhammer',
    maintainer_email = 'gregor.thalhammer@gmail.com',
    license = 'LGPL',
    cmdclass = {'build_ext': build_ext},
    packages = ['gpyfft'],
    ext_modules = [Extension("gpyfft.gpyfftlib",
                             ["gpyfft\gpyfftlib.pyx"],
                             include_dirs = [".",
                                             os.path.join(AMDFFT,'include'),
                                             os.path.join(AMDAPP,'include'),
                                             ],
                             extra_compile_args = [],
                             extra_link_args = [],
                             libraries = ['clAmdFft.Runtime'],
                             library_dirs = [os.path.join(AMDFFTLIBDIR, 'import')]
                             )
                   ],
    )

