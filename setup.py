import os
import platform
from setuptools import setup, Extension
from setuptools.command.build_py import build_py as _build_py
from distutils.util import convert_path
from Cython.Build import cythonize

CLFFT_DIR = r'/Users/gregor/Devel/clFFT'

CL_INCL_DIRS = []
if 'Windows' in platform.system():
    CL_DIR = os.getenv('AMDAPPSDKROOT')
    CL_INCL_DIRS = [os.path.join(CL_DIR, 'include')]

import Cython.Compiler.Options
Cython.Compiler.Options.generate_cleanup_code = 2

extensions = [
    Extension("gpyfft.gpyfftlib",
              [os.path.join('gpyfft', 'gpyfftlib.pyx')],
              include_dirs=[os.path.join(CLFFT_DIR, 'src', 'include'), ] + CL_INCL_DIRS,
              extra_compile_args=[],
              extra_link_args=[],
              libraries=['clFFT'],
              library_dirs=[os.path.join(CLFFT_DIR, 'src', 'library'), ],
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
    print("copied clFFT.dll, StatTimer.dll")

package_data = {}
if 'Windows' in platform.system():
    copy_clfftdll_to_package()
    package_data.update({'gpyfft': ['clFFT.dll', 'StatTimer.dll']},)


def get_version():
    main_ns = {}
    version_path = convert_path('gpyfft/version.py')
    with open(version_path) as version_file:
        exec(version_file.read(), main_ns)
    version = main_ns['__version__']
    return version


def get_readme():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirname, "README.rst"), "r") as fp:
        long_description = fp.read()
    return long_description


install_requires = ["numpy", "pyopencl"]
setup_requires = ["numpy", "cython"]


setup(
    name='gpyfft',
    version=get_version(),
    description='A Python wrapper for the OpenCL FFT library clFFT',
    long_description=get_readme(),
    url=r"https://github.com/geggo/gpyfft",
    maintainer='Gregor Thalhammer',
    maintainer_email='gregor.thalhammer@gmail.com',
    license='LGPL',
    packages=['gpyfft', "gpyfft.test"],
    ext_modules=cythonize(extensions),
    package_data=package_data,
    install_requires=install_requires,
    setup_requires=setup_requires,
    )
