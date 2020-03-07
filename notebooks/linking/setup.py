from setuptools import setup

setup(
   name="HelioLinC2",
   license="BSD 3-Clause License",
   author="Siegfried Eggl",
   author_email="eggl@uw.edu",
   long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
   url="https://github.com/lsst_dm/lsst_dm_ssp",
   packages=["heliolinc2"],
   package_dir={"heliolinc2": "heliolinc2"},
   package_data={"heliolinc2": ["tests/*.txt"]},
   use_scm_version=True,
   setup_requires=["pytest-runner", "setuptools_scm"],
   tests_require=["pytest"],
)
