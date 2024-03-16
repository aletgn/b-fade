Save/Load
=========

Provided that the sub-modules of B-FADE are correctly imported *any* istantiated object can be saved and load through :py:mod:`bfade.util.save` and :py:mod:`bfade.util.load`, respectively.

Save
----
Suppose one wish to save the results of the MAP and both propagation regions, thus:

    .. code-block:: python

        save(bay, folder=aStringFolder)

Before ``folder`` any number of object can be written. ``save`` will keep the names that ``bay`` and ``mc`` were given in the resulting file. Note that the files are given with ``.bfd`` extension.


Load
----

To load previosly save objects, use:

    .. code-block:: python

        load(folder = aStringFolder, extension = ".bfd")

which returns *all* the objects ending with ``.bfd``. This is particularly suitable for loading multiple objects. Instead, to load an objects that matches a specific name, do:

    .. code-block:: python

        load(folder = aStringFolder, filename = aStringFilename)

These options can be combined as well:

    .. code-block:: python

        load(folder = aStringFolder, extension = ".bfd", filename = aStringFilename)