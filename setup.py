from distutils.core import setup

setup(
    name="TBML",
    version="0.1",
    packages=["Driver", "TBNN", "TBMix", "TKE_NN"],
    install_requires=["numpy>=1.22.0", "torch>=1.13.0", "pandas>=1.4.0",
                      "scikit-learn>=1.2.0", "matplotlib>=3.7.0", "seaborn>=0.12.0",
                      "scipy>=1.10.0"]
)