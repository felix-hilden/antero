# Antero

Assorted collection of data analysis and machine learning related things.

### Self-organising map
Minimal implementation of a self-organising map for both CPU (NumPy) and GPU (Tensorflow).
Measures and visualisations are also included.
The map itself can span any number of dimensions, but visualisations are only supported
for the most common two-dimensional case.

```
from antero.som.cpu import SelfOrganisingMap
from antero.som.visual import heatmap, umatrix
from antero.som.measures import topographic_error

from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.preprocessing import RobustScaler

# Data
x, y = load_iris(True)
x, y = shuffle(x, y)
x = RobustScaler().fit_transform(x)

# Train
epochs = 1000
som = SelfOrganisingMap((20, 20), x.shape[-1], max_epochs=epochs)
som.train(x, epochs)

# Measure
umatrix(som.weights)
heatmap(som.project(x), som.shape, labels=y)
print('Topographic error:', topographic_error(x, som.weights))
```

### Categorical

Three classes for one-hot encoding of various sorts.

* OneHotEncoder: simple one-hot encoder.
* NanHotEncoder: capable of transforming NaN values, inverting all-zero rows and handling unseen categories.
* CatHotEncoder: built around Pandas Categorical for ease of use. Similar to NanHotEncoder.
