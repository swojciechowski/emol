# Evolutionary Multi-Objective Learning

Package with learning algorithms based on Evolutionary Multi-objective Approach.

**Current Package Status:** WIP

---

Implemented methods are strongly related to `pymoo` package (https://pymoo.org). 
This allows experimenting with different configurations of solvers (as provided by `pymoo` API), while maintaining basic configuration of "Machine Learning as Optimization" tasks presented in library. This approach is focusing on implementation of operators over solvers which are parameterized.

All implementations are compatible with `scikit-learn` package, where provided code extends it to multi-objective scenarios.

Tools exposed in "helpers" and "misc" directories are an additional code, not related to main purpose of this library.

Most of the math operators are and should be used as `numpy` operators (especially operators requiring broadcasting). This approach is faster and preferred over standard python operators. This note especially refers to scenarios where new objective functions will be implemented. Note, that those will be evaluated many times in optimization loop.
