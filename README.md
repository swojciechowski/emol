# Evolutionary Multi-Objective Learning
---

Package with multi-objective learning algorithms based on Evolutionary Multi-objective optimization. Currently, supports methods dedicated to imbalanced-data and preference-aware scenarios. 

**Project Status:** WIP

> Current version is fully functional but it is possible to find some bugs. Please feel free to describe them in **issues** section or provided fixes via **pull requests**.

## README

Implemented methods are strongly related to `pymoo` package (https://pymoo.org). 
This allows experimenting with different configurations of solvers (as provided by `pymoo` API), while maintaining basic configuration of "Machine Learning as Optimization" tasks presented in library. This approach is focusing on implementation of operators over solvers which are parameterized.

All implementations are compatible with `scikit-learn` package, where provided code extends it to multi-objective scenarios.

Code provided in "helpers" and "misc" directories is for additional tools, not related to main purpose of this library.

Most of the math operators are and should be used as `numpy` operators (especially operators requiring broadcasting). This approach is faster and preferred over standard python operators. This note especially refers to scenarios where new objective functions will be implemented. Note, that those will be evaluated many times in optimization loop.

## Installation

Please install this package using following command:

```
pip install git+https://github.com/swojciechowski/emol.git
```

or clone repository and install it in editable mode for developement.
