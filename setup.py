import os
import platform
from setuptools import setup, Extension
from distutils.util import convert_path
from Cython.Build import cythonize

system = platform.system()

## paths settings
# Linux
if 'Linux' in system:
    CLFFT_DIR = r'/home/gregor/devel/clFFT'
    CLFFT_LIB_DIRS = [r'/usr/local/lib64']
    CLFFT_INCL_DIRS = [os.path.join(CLFFT_DIR, 'src', 'include'), ]
    CL_INCL_DIRS = ['/opt/AMDAPPSDK-3.0/include']
    EXTRA_COMPILE_ARGS = []
    EXTRA_LINK_ARGS = []

#Windows
elif 'Windows' in system:
    CLFFT_DIR = r'C:\Users\admin\Devel\clFFT-Full-2.12.2-Windows-x64'
    CLFFT_LIB_DIRS = [os.path.join(CLFFT_DIR, 'lib64\import')]
    CLFFT_INCL_DIRS = [os.path.join(CLFFT_DIR, 'include'), ]
    CL_DIR = os.getenv('AMDAPPSDKROOT')
    CL_INCL_DIRS = [os.path.join(CL_DIR, 'include')]
    EXTRA_COMPILE_ARGS = []
    EXTRA_LINK_ARGS = []
    
# macOS
elif 'Darwin' in system:
    CLFFT_DIR = r'/Users/gregor/Devel/clFFT'
    CLFFT_LIB_DIRS = [r'/Users/gregor/Devel/clFFT/src/library']
    CLFFT_INCL_DIRS = [os.path.join(CLFFT_DIR, 'src', 'include'), ]
    CL_INCL_DIRS = []
    EXTRA_COMPILE_ARGS = ['-stdlib=libc++']
    EXTRA_LINK_ARGS = ['-stdlib=libc++']

import Cython.Compiler.Options
Cython.Compiler.Options.generate_cleanup_code = 2

extensions = [
    Extension("gpyfft.gpyfftlib",
              [os.path.join('gpyfft', 'gpyfftlib.pyx')],
              include_dirs= CLFFT_INCL_DIRS + CL_INCL_DIRS,
              extra_compile_args=EXTRA_COMPILE_ARGS,
              extra_link_args=EXTRA_LINK_ARGS,
              libraries=['clFFT'],
              library_dirs = CLFFT_LIB_DIRS,
              language='c++',
              )
    ]

def copy_clfftdll_to_package():
    import shutil
    shutil.copy(
        os.path.join(CLFFT_DIR, 'bin', 'clFFT.dll'),
        'gpyfft')

    shutil.copy(
        os.path.join(CLFFT_DIR, 'bin', 'StatTimer.dll'),
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
    with open(os.path.join(dirname, "README.md"), "r") as fp:
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
