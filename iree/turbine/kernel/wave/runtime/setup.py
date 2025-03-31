from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys
from pathlib import Path


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):

        # Ensure CMake is available
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        # Create build directory
        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_dir, exist_ok=True)

        # Get extension directory
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Configure CMake
        cmake_args = [
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
        ]
        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=build_dir)

        # Build CMake project
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)


setup(
    name="wave_runtime",
    version="0.1.0",
    ext_modules=[CMakeExtension("wave_runtime")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
