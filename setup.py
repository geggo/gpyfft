import os, os.path, platform
from setuptools import setup, Extension
from Cython.Build import cythonize

CLFFT_DIR = r'/Users/gregor/Devel/clFFT'

CL_INCL_DIRS = []
if 'Windows' in platform.system():
    CL_DIR = os.getenv('AMDAPPSDKROOT')
    CL_INCL_DIRS = [os.path.join(CL_DIR, 'include')]

import Cython.Compiler.Options
Cython.Compiler.Options.generate_cleanup_code = 2

#TODO: see https://github.com/matthew-brett/du-cy-numpy

extensions = [
    Extension("gpyfft.gpyfftlib",
              [os.path.join('gpyfft', 'gpyfftlib.pyx')],
              include_dirs = [os.path.join(CLFFT_DIR,'src', 'include'),] + CL_INCL_DIRS,
              extra_compile_args = [],
              extra_link_args = [],
              libraries = ['clFFT'],
              library_dirs = [os.path.join(CLFFT_DIR, 'src', 'library'),],
              )
    ]

def copy_clfftdll_to_package():
    import shutil
    shutil.copy(
        os.path.join(CLFFT_DIR, 'src', 'staging', 'clFFT.dll'),
        'gpyfft')
    
    shutil.copy(
        os.path.join(CLFFT_DIR, 'src', 'staging', 'StatTimer.dll'),
        'gpyfft')
    print "copied clFFT.dll, StatTimer.dll"
                 
package_data = {}
if 'Windows' in platform.system():
    copy_clfftdll_to_package()
    package_data.update({'gpyfft': ['clFFT.dll', 'StatTimer.dll']},)

setup(
    name = 'Gpyfft',
    version = '0.2.1',
    description = 'A Python wrapper for the OpenCL FFT library clFFT by AMD',
    url = r"https://github.com/geggo/gpyfft",
    maintainer='Gregor Thalhammer',
    maintainer_email = 'gregor.thalhammer@gmail.com',
    license = 'LGPL',
    packages = ['gpyfft'],
    ext_modules = cythonize(extensions),
    package_data = package_data
    )

