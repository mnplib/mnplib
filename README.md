# fastautoml 

``fastautoml`` is powerful and computationally efficient Python library for automated machine learning, intended for data scientists and with the goal of maximize their productivity.

## Prerequisites

``fastautoml`` requires:

 * scikit-learn (>= 0.22)

## User Installation

If you already have a working installation of ``scikit-learn`` and ``pandas``, the easiest way to install ``fastautoml`` is using ``pip``:

```
pip install fastautoml
```

## Running

The following example shows how to compute an optimal model for the MNIST dataset included with ``scikit-learn``.

```
from fastautoml.fastautoml import AutoClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = AutoClassifier()
model.fit(X_train, y_train)
print("Score:", model.score(X_test, y_test))
```

## Help

 * [User Guide](https://github.com/rleiva/fastautoml/wiki)
 * Reference API (TBD)
 * [Examples of usage](examples)

## Authors

[R. Leiva](https://github.com/rleiva) and [contributors](Contributors.md). If you want to contribute to this project, please contact with the main author.

## License

This project is licensed under the 3-Clause BSD license - see the [LICENSE.md](LICENSE.md) file for details.

## Funding

This project has received funding from the [IMDEA Networks Institute](https://www.networks.imdea.org/) and from the European Union's Horizon 2020 research and innovation programme under grant agreement No 732667 [RECAP](https://recap-project.eu/).

