from setuptools import setup
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from Cython.Distutils import build_ext
import numpy as np

numpy_include = np.get_include()

ext_modules = [
    Extension('utils.cython_bbox',
              ['utils/bbox.pyx'],
              extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
              include_dirs=[numpy_include]
              ),
    CUDAExtension('nms.cuda_nms', [
            'nms/src/nms_cuda.cpp',
            'nms/src/nms_cuda_kernel.cu',
        ])
]
setup(
    name='faster_rcnn',
    cmdclass={'build_ext': BuildExtension},
    ext_modules=ext_modules
)
