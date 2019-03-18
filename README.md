# Antero

Assorted collection of data analysis and machine learning related things.

### Categorical

Three classes for one-hot encoding of various sorts.

* OneHotEncoder: simple one-hot encoder.
* NanHotEncoder: capable of transforming NaN values and inverting all-zero rows. Unseen categories are treated as NaNs.
* CatHotEncoder: built around Pandas Categorical for ease of use. Similar to NanHotEncoder.
