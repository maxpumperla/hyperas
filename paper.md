---
title: 'Hyperas: Simple Hyperparameter Tuning for Keras Models'
tags:
  - Python
  - Hyperparameter Tuning
  - Deep Learning
  - Keras
  - Hyperopt
authors:
  - name: Max Pumperla
    affiliation: "1, 2"
affiliations:
  - name: IU Internationale Hochschule
    index: 1
  - name: Pathmind Inc.
    index: 2 
date: 19 November 2021
bibliography: paper.bib
    
---

# Summary

Hyperas is an extension of [Keras](https://keras.io/) [@chollet2015keras], which allows you to run hyperparameter optimization of your models using [Hyperopt](http://hyperopt.github.io/hyperopt/) [@bergstra2012hyperopt].
It was built to enable fast experimentation cycles for researchers and software developers.
With hyperas, you can set up your Keras models as you're used to and specify your hyperparameter search spaces in a convenient way, following the design principles suggested by the [Jinja project](https://jinja.palletsprojects.com/en/3.0.x/) [@jinja2008].

With hyperas, researchers can use the full power of hyperopt without sacrificing experimentation speed. 
Its documentation is hosted on [GitHub](https://github.com/maxpumperla/hyperas) and comes with suite of [examples]https://github.com/maxpumperla/hyperas/tree/master/examples) to get users started.


# Statement of need

Hyperas is in active use in the Python community and still has [thousands of weekly downloads](https://pypistats.org/packages/hyperas), which shows a clear need for this experimentation library.
Over the years, hyperas has been used and cited by [research papers](https://scholar.google.com/scholar?cluster=1375058734373368171&hl=en&oi=scholarr), mostly by [referring to Github](https://scholar.google.com/scholar?hl=de&as_sdt=0%2C5&q=hyperas+keras&btnG=).
Researchers that want to focus on their deep learning model definitions don't get bogged down by maintaining separate hyperparameter search spaces and configurations and can leverage hyperas to speed up their experiments.
After hyperas has been published, tools like Optuna [@akiba2019optuna] have adopted a similar approach to hyperparameter tuning.
KerasTuner [@omalley2019kerastuner] is officially supported by Keras itself, but does not have the same variety of hyperparameter search algorithms as hyperas.

# Design and API

Hyperas uses a Jinja-style template language to define search spaces implicitly in Keras model specifications.
Essentially, regular configuration values in a Keras layer, such as `Dropout(0.2)` get replaced by a [suitable distribution](https://github.com/maxpumperla/hyperas/blob/master/hyperas/distributions.py) like `Dropout({{uniform(0, 1)}})`.
To define a hyperas model, you proceed in two steps.
First, you set up a function that returns the data you want to train on, which could include features and labels for training, validation and test sets.
Schematically this would look as follows:

```python
def data():
    # Load your data here
    return x_train, y_train, x_test, y_test
```

Next, you have to specify a function that takes your data as input arguments, defines a Keras model with hyperas template handles (`{{}}`), fits the model to your data and returns a dictionary that has to at least contain a `loss` value to be minimized by hyperopt, e.g. validation loss or the negative of test accuracy, and the hyperopt `status` of the experiment.

```python
from hyperas.distributions import uniform
from hyperopt import STATUS_OK


def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    # ... add more layers
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # fit model
    model.fit(x_train, y_train, ...)

    # evaluate model and return loss
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = score[1]
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}
```

Lastly, you simply prompt the `optim` module of hyperas to `minimize` your model loss defined in `create_function`, using `data`, with a hyperparameter optimization algorithm like TPE or any other algorithm supported by hyperopt [@pmlr-v28-bergstra13].

```python
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe

best_run = optim.minimize(model=create_model,
                          data=data,
                          algo=tpe.suggest,
                          max_evals=10,
                          trials=Trials())
```

Furthermore, note that hyperas can run [hyperparameter tuning in parrallel](https://github.com/maxpumperla/hyperas#running-hyperas-in-parallel), using hyperopt's distributed MongoDB backend.

# Acknowledgements

We would like to thank all the open-source contributors that helped making `hyperas` what it is today.
It's a great honor to see your software continually used by the [community](https://github.com/maxpumperla/hyperas/network/dependents?package_id=UGFja2FnZS01MjIwODQ4OA%3D%3D).

# References
