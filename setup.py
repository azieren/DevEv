import setuptools

setuptools.setup(
    name="DevEv",
    version="0.0.24",
    url="https://github.com/azieren/DevEv",
    author="Nicolas Aziere",
    author_email="nicolas.aziere@gmail.com",
    description="3D visualization tool",
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    install_requires=['numpy',
        'matplotlib',
        'scipy',
        'PyOpenGL'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    include_package_data=True,
)