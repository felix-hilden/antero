from setuptools import setup

setup(
    name='antero',
    version='0.1',
    description='Pre-processing utilities',
    url='https://github.com/felix-hilden/antero',
    author='Felix Hildén',
    license='MIT',
    packages=['antero'],
    zip_safe=True,
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'tqdm',
        'scipy',
        'matplotlib',
        'seaborn',
        'sklearn',
    ]
)
