# pputils

Pre-processing utilities for data analytics and ML tasks.
So far only one-hot encoders for various tasks.

* OneHotEncoder: simple one-hot encoder.
* NanHotEncoder: encoder capable of transforming arrays with NaN values and inverting all-zero rows back to NaN values.
Unseen categories are treated as NaNs.
* CatHotEncoder: encoder built around the Pandas Categorical specification for ease of use. Similar to NanHotEncoder.
