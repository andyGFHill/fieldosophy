
# Welcome to Fieldosophy

<p align="center">
<center><img src="https://drive.google.com/uc?export=view&id=17fSqlCPBd06zf0jM2ghjKJdPrpoQUyqN" ></center>
</p>

Documentation available [here](https://andygfhill.github.io/fieldosophy).



## About

The Fieldosophy package supplies tools for working with Gaussian random fields. It is mainly focused on the SPDE approach and computations based on a finite element approximation of the Gaussian random field, please see papers: 

Lindgren, F., Rue, H., Lindström, J., 2011. An explicit link between Gaussian fields and Gaussian Markov random fields:
the stochastic partial differential equation approach. J. R. Stat. Soc. 73 (4), 423–498.

Bolin, D., Kirchner, K., 2020. The rational SPDE approach for Gaussian random fields with general smoothness. J. Comput.
Graph. Stat. (29), 274-285.

Hildeman, A., Bolin, D., Rychlik, I., 2020. Deformed SPDE models with an application to spatial modeling of significant wave height. Spatial Statistics. <https://doi.org/10.1016/j.spasta.2020.100449>

## License

Fieldosophy is licensed under BSD-3 (3-clause license). 
This basically means that it can be used and modified in open-source and proprietary projects. 
If derivative work is distributed for commercial or non-profit usage, it does not require distribution of source code. However, it do require that the original work is publicly recognized. Please read the "LICENSE"-file for details.
 



## Installation 

### Compiling the low-level library

Although Fieldosophy is a Python package, much of its functionality is based on lower level functions written in C and C++. 
These are packaged into a dynamically linked library that needs to be compiled.

Before compiling the library the dependencies have to be installed. These are:

* Eigen 
    Can be downloaded from: https://eigen.tuxfamily.org.
    Inside the "Makefile"-file in the main directory of Fieldosophy, point the variable "EIGENPATH" to the location of the "Eigen" directory of the Eigen library.
* GNU Scientific library
    Can be downloaded from: https://www.gnu.org/software/gsl. 
    Make sure to install it so that the libraries and headers are available in the default search paths of the compiler.

When the dependencies are installed and set up, compile the Fieldosophy library by typing "make" in the main directory of the Fieldosophy package. 
The library-file should now reside in the "./fieldosophy/libraries" directory.

### Installing the Python package

If Fieldosophy's low-level library has been compiled successfully and resides in the "./fieldosophy/libraries" directory, the Python package can be installed by typing: "pip install .".
Fieldosophy is now installed and can be imported in python using "import fieldosophy".

If you want to make sure that the installation is working properly, and you have the python package "unittest" installed, you can type "python -m unittest". This will execute the suite of unit tests to confirm the installation. 








