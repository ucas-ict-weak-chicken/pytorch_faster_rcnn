from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

numpy_include = np.get_include()

ext_modules = [
    Extension('utils.cython_bbox',
              ['utils/bbox.pyx'],
              extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
              include_dirs=[numpy_include]
              )
]
setup(
    name='faster_rcnn',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
