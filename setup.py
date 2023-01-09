import setuptools


setuptools.setup(
    name='regression_plots', version='1.0', scripts=['regression_plots/regression_plots.py'],
    author='Kevin Garcia', author_email='d.kevin0402@gmail.com',
    description='package that works as a tool in linear regression models and shows diagnostic graphs of the model.',
    url='https://github.com/kevillo/regression_plots', packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "License :: MIT License",
                 "Operating System :: OS Independent", ])
