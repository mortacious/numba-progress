from setuptools import setup, find_packages


# standalone import of a module (https://stackoverflow.com/a/58423785)
def import_module_from_path(path):
    """Import a module from the given path without executing any code above it
    """
    import importlib
    import pathlib
    import sys

    module_path = pathlib.Path(path).resolve()
    module_name = module_path.stem  # 'path/x.py' -> 'x'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    if module not in sys.modules:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules
    return module


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = import_module_from_path('numba_progress/_version.py').__version__

setup(
    name="numba-progress",
    version=version,
    author="Felix Igelbrink",
    author_email="felix.igelbrink@uni-osnabrueck.de",
    description="A progress bar implementation for numba functions using tqdm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mortacious/numba-progress",
    project_urls={
        "Bug Tracker": "https://github.com/mortacious/numba-progress/issues",
    },
    packages=find_packages(exclude=['examples']),
    install_requires=[
        'numpy',
        'numba>=0.52',
        'tqdm'
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    python_requires=">=3.7",
    zip_safe=True
)
