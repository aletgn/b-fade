Initialisation
==============

Required Modules
----------------

Regardless of the option chosen to run B-FADE, before starting to code the main body of a script, make sure to include in its header the following:

.. code-block:: python
    
    import numpy as np
    import pandas as pd
    import scipy

The included modules/functions shall be utilised throughout the execution. Specifically, ``scipy`` is imported to recall probability distributions for shaping priors. ``numpy`` is required to preprocess information to define the prior parameters, in most of the envisaged cases. Whilst ``pandas`` shall be exploited to recall the file readers.

Logging
-------

Since B-FADE implement basic logging functionalities across sub-modules, the user can easily (and optionally) customise the logging level via :py:mod:`bfade.util.logger_manager`. To do so, include the following line of code:

.. code-block:: python

    logger_manager(level="WARNING") # The default is "DEBUG"

where ``level`` can be selected amongst these:

- ``DEBUG``
- ``INFO``
- ``WARNING``
- ``CRITICAL``
- ``ERROR``

sorted by verbosity.

Graphical Output
----------------

The user can decide on the look-and-feel appearance of the plot that B-FADE outputs. This is done via :py:mod:`bfade.util.config_matplotlib`. This function modifies the settings of ``matplotlib`` according to the user's inputs, for instance:


.. code-block:: python

    config_matplotlib(font_size=12, font_family="serif", use_latex=True)
    
