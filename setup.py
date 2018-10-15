from setuptools import setup

setup(
    name='pputils',
    version='0.1',
    description='Pre-processing utilities',
    url='https://github.com/felix-hilden/pputils',
    author='Felix Hildén',
    license='MIT',
    packages=['pputils'],
    zip_safe=True,
    install_requires=[
        'numpy',
        'pandas'
    ]
)
