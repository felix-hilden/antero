# Antero

Assorted collection of data analysis and machine learning related things.

## Self-organising map
Minimal implementation of a self-organising map for both a CPU (NumPy) and a GPU (Tensorflow).
The implementations are completely interchangeable, only requiring a change in the import statement.
To use this SOM, no knowledge of Tensorflow is required. All data is consumed using NumPy arrays.

### Key features
* exponentially decreasing **learning rate** and **Gaussian neighbourhood** functions are constructed
    based on map dimensions and estimated maximum number of training epochs
* **mini-batch training** - optionally aggregate updates for a number of training steps before changing weights
* **visualisations and measures** - easy ways to view and measure results
    * U-matrix - visualise or measure map topology by average distance from a node to its neighbours
    * heatmap - visualise where input data is projected on the map, if labels are passed creates a figure for each class
    * topological error - percentage of best-matching units for which the second-best-matching unit is not a neighbour
* **n-dimensional** - any number of map dimensions can be specified,
though visualisations are only supported for two-dimensional maps.
* **built-in serialisation** - just call `save` and `load`
* **CPU and GPU** - fully interchangeable implementations even saving with one and loading with the other

### Complete example
```
from antero.som.cpu import SelfOrganisingMap
from antero.som.visual import heatmap, umatrix, class_pies
from antero.som.measures import topographic_error, embedding_accuracy

from sklearn.datasets import load_iris
from sklearn.preprocessing import RobustScaler

x, y = load_iris(True)
x = RobustScaler().fit_transform(x)

epochs = 1000
som = SelfOrganisingMap((7, 7), x.shape[-1], max_epochs=epochs)
som.train(x, epochs, shuffle=True)

umatrix(som)
heatmap(som, x)
class_pies(som, x, y)
print('Topographic error:', topographic_error(som, x))
print('Embedding accuracy:', embedding_accuracy(som, x))
```

## Categorical

Three classes for one-hot encoding of various sorts.

* `OneHotEncoder`: simple one-hot encoder.
* `NanHotEncoder`: capable of transforming `NaN` values, inverting all-zero rows and handling unseen categories.
* `CatHotEncoder`: built around `pandas.Categorical` for ease of use. Similar to `NanHotEncoder`.
