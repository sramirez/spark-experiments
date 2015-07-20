Information Theoretic Feature Selection Framework (Spark)
===========================================================

The present framework implements Feature Selection (FS) on Spark for its application on Big Data problems. The framework is interoperable with the MLlib Spark library and will be part of it in the close future.
This package contains a generic implementation of greedy Information Theoretic Feature Selection methods. The implementation is based on the common theoretic framework presented in [1]. Implementations of mRMR, InfoGain, JMI and other commonly used FS filters are provided. In addition, the framework can be extended with other criteria provided by the user as long as the process complies with the framework proposed in [1].

-- Improvements on FS:
* Support for sparse data (in progress).
* Pool optimization for high-dimensional datasets.
* Better efficiency: 150 seconds by selected feature for a 65M dataset with 631 attributes.
* Some important bugs fixed.

Fayyad's Discretizer and Experimental Utilities (Spark)
===========================================================

This package also contains an implementation of Fayyad's discretizer [2] based on Minimum Description Length Principle (MDLP) in order to treat non discrete datasets as well as two artificial dataset generators to test current and new feature selection criteria.

-- Improvements on discretizer:
* Support for sparse data.
* Multi-attribute processing. The whole process is carried out in a single step when the number of boundary points per attribute fits well in one partition (<= 100K boundary points per attribute). 
* Support for attributes with a huge number of boundary points (> 100K boundary points per attribute). Extremely rare situation. 
* Some bugs fixed. 

This software has been proved with two large real-world datasets such as:
* A dataset selected for the GECCO-2014 in Vancouver, July 13th, 2014 competition, which comes from the Protein Structure Prediction field (http://cruncher.ncl.ac.uk/bdcomp/). The dataset has 32 million instances, 631 attributes, 2 classes, 98% of negative examples and occupies, when uncompressed, about 56GB of disk space.
* Epsilon dataset: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon. 400K instances and 2K attributes.

## Contributors

- Sergio Ramírez-Gallego (sramirez@decsai.ugr.es) (main contributor and maintainer).
- Héctor Mouriño-Talín (h.mtalin@udc.es)
- David Martínez-Rego (dmartinez@udc.es)

## References
```
[1] Brown, G., Pocock, A., Zhao, M. J., & Luján, M. (2012). 
"Conditional likelihood maximisation: a unifying framework for information theoretic feature selection." 
The Journal of Machine Learning Research, 13(1), 27-66.
```
```
[2] Fayyad, U., & Irani, K. (1993).
"Multi-interval discretization of continuous-valued attributes for classification learning."


```
