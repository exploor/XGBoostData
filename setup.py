from setuptools import setup, find_packages

setup(
    name='mrd',
    version='0.1',
    packages=find_packages(where='mrd'),  # Specify 'mrd' as the package root
    package_dir={'': 'mrd'},  # Tells setuptools that packages are in 'mrd' directory
    install_requires=[
        'pyyaml',
        'pandas',
        'numpy',
        'sqlalchemy',
        'streamlit',
        'scipy',
        'matplotlib',
        'plotly'
    ],
    include_package_data=True,
    package_data={
        'mrd.core': ['config.yaml'],
    },
)