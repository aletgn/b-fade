Running B-FADE
==============

Options
-------

B-FADE can be executed via:

- ``.py`` files whose configuration is directly coded inside the script
    
    .. code-block::

        python script_file.py

- ``.py`` files whose configuration is *externally* coded inside a ``.yaml`` file which is parsed at run-time. By this option the script must be run as:

    .. code-block::

        python script_file.py --config config_file.yaml

- ``.ipynb`` (Jupyter) notebooks. In this case, it shall not be possible write an external configuration file.

Remarks on yaml Files
---------------------

``.yaml`` files are typically used to arrange software configurations. Such files are coded this way:

    .. code-block:: yaml
        
        section_1:
         attribute_1: aString
         attribute_2: aNumber
         # ...
          subsection_1:
           attribute_3: [aList]
           # ...
        # ..

Sections and subsections are nested by *spaces* -- any number of spaces are allowed as long as they are consistently kept across nested statements. Although tabulations are permitted they are considered bad practice. The scientific notation is allowed as long as leading numbers are float, such as:

    .. code-block:: yaml
        
        good_float_1: 1.0e5
        good_float_2: 2.5e-5

Conversely, numbers such as:

    .. code-block:: yaml
        
        bad_float_1: 1e6

are treated as strings. Comments begin with ``#``.

Python's ``.yaml`` parser captures Lists, but not Tuples. The scientific notation is not allowed in ``.yaml`` files. Additionally, ``.yaml`` files are acquired as (nested) Dictionaries.