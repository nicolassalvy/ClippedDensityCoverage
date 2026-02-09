from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys


class CustomInstall(install):
    def run(self):
        # Install dgm-eval in editable mode, otherwise it won't work
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",  # Too strict (exact versions)
                "-e",
                "git+https://github.com/layer6ai-labs/dgm-eval.git#egg=dgm_eval",
            ]
        )
        super().run()


setup(
    name="ClippedDensityCoverage",
    version="0.1",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires="==3.10",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "hydra-core",
        "tqdm",
        "top-pr",
        # for dgm-eval
        "pandas",
        "opencv-python",
        "open_clip_torch",
        "pillow",
        "scipy",
        "timm",
        "transformers",
    ],
    packages=find_packages(where="."),
    include_package_data=True,
    cmdclass={
        "install": CustomInstall,
    },
)
