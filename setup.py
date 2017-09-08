from setuptools import setup, find_packages

setup(
    name='runlmc',
    version='0.0',
    description=(
        'Structurally efficient multi-output linearly coregionalized'
        ' Gaussian Processes: it\'s tricky, tricky, tricky, tricky, tricky.'),
    packages=find_packages(),
    author='Vladimir Feinberg',
    install_requires=[
        'paramz',
        'scipy',
        'numpy',
        'climin',
        'numdifftools',
        'contexttimer',
        'pandas',
    ],
)
