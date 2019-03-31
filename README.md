# Antero

Assorted collection of data analysis and machine learning related things.

### Self-organising map
Minimal implementation of a self-organising map for both CPU (NumPy) and GPU (Tensorflow).

### Categorical

Three classes for one-hot encoding of various sorts.

* OneHotEncoder: simple one-hot encoder.
* NanHotEncoder: capable of transforming NaN values, inverting all-zero rows and handling unseen categories.
* CatHotEncoder: built around Pandas Categorical for ease of use. Similar to NanHotEncoder.
