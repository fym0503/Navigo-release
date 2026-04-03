import os
from pathlib import Path

from setuptools import find_packages, setup

BUILD_ID = os.environ.get("BUILD_BUILDID", "0")
ROOT = Path(__file__).parent


def read_requirements(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


with open(ROOT / "README.md", "r", encoding="utf-8") as handle:
    long_description = handle.read()


setup(
    name="navigo",
    version="0.1." + BUILD_ID,
    author="Yimin Fan",
    author_email="fanyimin@link.cuhk.edu.hk",
    license="BSD-2-Clause",
    license_files=["LICENSE"],
    description="Navigo package with tutorials and documentation in one repository.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["navigo", "navigo.*"]),
    install_requires=read_requirements(ROOT / "requirements.txt"),
    extras_require={"docs": read_requirements(ROOT / "docs" / "requirements.txt")},
)
