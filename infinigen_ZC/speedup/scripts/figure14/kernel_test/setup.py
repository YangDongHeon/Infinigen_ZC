from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gather_copy',  # This is the module name that will be produced.
    ext_modules=[
        CUDAExtension(
            name='gather_copy',  # The extension module will be called "gather_copy".
            sources=['gather_copy.cu'],  # Your source file containing the code.
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O3', '-arch=sm_89']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
