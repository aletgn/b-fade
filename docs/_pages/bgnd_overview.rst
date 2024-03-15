Overview
========

Summary of the API
------------------

As the API reports, B-FADE collates the following sub-modules.

- :py:mod:`bfade.abstract` contains abstract classes which are use to opportunely define concrete classes for curves, viewers, Bayesian infrastructure.

- :py:mod:`bfade.dataset` is a collection of classes defining the structure of the datasets to be processed by B-FADE. The module include a class to generate datasets (grids for instance).

- :py:mod:`bfade.statistics` contains wrappers of `scipy` proability distributions and a class for Monte Carlo simulations.

- :py:mod:`bfade.viewers` includes graphical output utilities (concrete classes) allowing the user to inspect both data and results.

- :py:mod:`bfade.elhaddad` is a collection of concrete classes defining the representation of the El Haddad curve, managing the datasets and perform MAP over El Haddad parameters.

- :py:mod:`bfade.util` is a set of functions recalled in the above-mentioned modules to support their implementation.

Notation
--------

The documentation uses the following notation to describe the functionalities of B-FADE:

- :math:`P` denotes a proability distribution.

- :math:`D` is a dataset.

- :math:`\theta` is the aggregated vector of the El Haddad parameters.
