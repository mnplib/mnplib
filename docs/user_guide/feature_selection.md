# Feature Selection

Given a dataset X = {x<sub>1</sub>, ..., x<sub>p</sub>} composed by p features, and a target variable y, the _miscoding_ of the feature x<sub>j</sub> measures how difficult is to reconstruct y given x<sub>j</sub>, and the other way around. We are not only interested in to identify how much information x<sub>j</sub> contains about y, but also if x<sub>j</sub> contains additional information that is not related to y (which is a bad thing).

The `nescience.Miscoding` class allow us to compute the relevance of features, the quality of a dataset, and select the optimal subset of features to include in a study.

## Feature Relevance

Let's generate a synthetic dataset composed by 1000 random points belonging to 10 Gaussian blobs. The samples of the dataset are described by 20 features, from which only 4 are informative.

```python
from sklearn.datasets.samples_generator import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=4, n_redundant=0, n_repeated=0, n_classes=10, n_clusters_per_class=1, weights=None, flip_y=0, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=1)
```

Next figure shows the blobs projected into the two dimensional space defined by the features x<sub>8</sub> and x<sub>10</sub>.

![Ten Gaussian Blobs](https://github.com/rleiva/fastautoml/blob/master/images/10gaussianblobs.png)

Let's compute the miscoding of each feature with respect to the target variable.

```python
from nescience.miscoding import Miscoding
miscoding = Miscoding()
miscoding.fit(X, y)
msd = miscoding.miscoding_features(type='adjusted')
```

If we plot the values of the `msd` array, we will get something like the following figure:

![Feature Relevance](https://github.com/rleiva/fastautoml/blob/master/images/MiscodingGaussianBlobs.png)

As it was expected, the `Miscoding` class was able to identify the four relevant features of the dataset.

For more information about how to identify the relevance of features using the `Miscoding` class see the following blog entries:

* Miscoding of Random Distributions (TBD)
* [Correlation vs Miscoding](https://github.com/rleiva/fastautoml/wiki/Correlation-vs-Miscoding)

## Optimal Feature Selection

The class `Miscoding` allow us to select the optimal subset of features from our dataset to use to train a model. That subset will contain only those features that are relevant for the problem at hand. That means faster training time, and reduced risk of overfitting. For this purpose, we use the `partial` version of the concept of miscoding. `partial` miscoding is simmilar to the `adjusted` miscoding we have seen in the previous section, but with the difference that non relevant features have a negative value.

Let's generate a synthetic dataset composed by 1000 random points belonging to 10 Gaussian blobs. The samples of the dataset are described by 20 features, from which 14 are informative.

```python
from sklearn.datasets.samples_generator import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=14, n_redundant=0, n_repeated=0, n_classes=10, n_clusters_per_class=1, weights=None, flip_y=0, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=1)
```

And let's compute the `partial` miscoding of these features

```python
from nescience.miscoding import Miscoding
miscoding = Miscoding()
miscoding.fit(X, y)
mscd = miscoding.miscoding_features(type='partial')
```

Next figure shows the result of classifying the features:

![Feature Relevance](https://github.com/rleiva/fastautoml/blob/master/images/PartialMiscodingGaussianBlobs.png)

As we can see, non-relevant features have now a negative value. In our study, we should use only features with positive partial miscoding.

For more information about how to select the optimal subset of features see the following blog entries:

* Optimal Feature Selection (TBD)
* Miscoding of Models (TBD)

## Mathematical Formulation

Let's **X** = {**x<sub>1</sub>**, ..., **x<sub>p</sub>**} be a dataset composed by p features, **y** the target variable , and **x<sub>j</sub>** the _j−th_ feature.

We define the _feature miscoding_ of **x<sub>j</sub>** as a representation of **y** as:

![Feature Miscoding](https://github.com/rleiva/fastautoml/blob/master/images/math_featuremiscoding.png)

If **x** is a qualitative vector (either a feature or the target variable) taking values from a set labels **G** = {g<sub>1</sub>, ..., g<sub>l</sub>}, the Kolmogorov complexity of **x** can be approximated by:

![Kolmogorov Compression](https://github.com/rleiva/fastautoml/blob/master/images/math_kolmogorovcompression.png)

If **x** is a quantitative vector, it has to be discretized first.

We define the _adjusted miscoding_ as the normalized version of the complemens of the feature miscodings:

![Adjusted Miscoding](https://github.com/rleiva/fastautoml/blob/master/images/math_adjustedmiscoding.png)

We define the _partial miscoding_ of **x<sub>j</sub>** as a representation of **y**, as:

![Partial Miscoding](https://github.com/rleiva/fastautoml/blob/master/images/math_partialmiscoding.png)

Let's **Z** = {**z<sub>1</sub>**, ..., **z<sub>k</sub>**} be a subset of features of **X**. We define the miscoding of **Z** as a representation of **y** as:

![Miscoding](https://github.com/rleiva/fastautoml/blob/master/images/math_miscodingsubset.png)
