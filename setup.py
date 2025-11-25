from setuptools import setup, find_packages


version = {}
with open("scripts/_version.py", "r") as f:
    exec(f.read(), version)


setup(
    name="tennis-counterfactual",
    version=version["__version__"],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "xgboost",
        "scikit-learn",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "tennisctl=scripts.cli:main",
        ],
    },
    author="Mattia Campana",
    description="Tennis match-winning prediction and counterfactual analysis",
    url="https://github.com/campaneros/tennis-counterfactual",
)
