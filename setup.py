import os
import sys
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py


class Build(build_py):
    def run(self):
        protoc_command = ["make", "all"]
        if subprocess.call(protoc_command) != 0:
            sys.exit(-1)
        super().run()


setup(
    name='chamfer_distance',
    version='1.0',
    description='Python Distribution Utilities',
    packages=['chamfer_distance'],
    package_dir={'': 'src'},
    cmdclass={
        'build_py': Build,
    }
)
