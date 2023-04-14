from setuptools import setup, find_packages

setup(
    name='cloth-funnels',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'filelock',
        'h5py',
        'numpy-quaternion',
        'opencv-python',
        'openexr',
        'potpourri3d==0.0.4',
        'pyquaternion',
        'ray[default]',
        'scipy',
        'tensorboardx',
        'transformations',
        'transforms3d',
        'trimesh',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    author='Alper',
    author_email='ac4983@columbia.edu',
    description='A Python package for cloth-funnels',
    url='https://github.com/columbia-ai-robotics/cloth-funnels',
    python_requires='>=3.9',
)
