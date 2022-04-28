from setuptools import setup, find_packages, find_namespace_packages
from pathlib import Path

version_file = Path(__file__).cwd().joinpath("ocr_model", "VERSION.txt")
version = version_file.read_text(encoding="UTF-8").strip()

install_requires = []

with open("requirements.txt") as f:
    lines = f.readlines()
    install_requires.extend(lines)

setup(
    name="ocr_model",
    version=version,
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.9",
    include_package_data=True,
    entry_points={"console_scripts": ["ocr_model=ocr_model.__main__:main"]},
    zip_safe=True,
)
