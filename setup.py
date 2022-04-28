from setuptools import setup, find_packages
from pathlib import Path

version_file = Path(__file__).parent.joinpath("ocr", "VERSION.txt")
version = version_file.read_text(encoding="UTF-8").strip()

install_requires = []

with open("requirements.txt") as f:
    lines = f.readlines()
    install_requires.extend(lines)

setup(
    name="ocr",
    version=version,
    packages=find_packages(include=["ocr", "ocr.*"], exclude=["*/datasets/*"]),
    install_requires=install_requires,
    python_requires=">=3.9",
    include_package_data=True,
    zip_safe=True,
)
