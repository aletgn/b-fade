Virtual Environment
===================

It is recommended working within a virtual environment, so please make sure that, for instance, ``virtualenv`` is properly configured. If you do not wish to configure a virtual environment, please skip to the next section. If ``pip`` is working, it is possible to install ``virtualenv`` by:

.. code-block::

	pip install virtualenv
	
Once this step is accomplished, run:
		
.. code-block::

	virtualenv bfade

to create the desired virtual environment. Then, activate it

.. code-block::

	. path_to_virtual_environment/bfade/bin/activate
	
on Linux, whilst:

.. code-block::

	path_to_virtual_environment/bfade/Script/activate
	
on Windows.

If you use ``conda`` to manage virtual environments, run:

.. code-block::

	conda create --name bfade

Activate the environment:

.. code-block::

	conda activate bfade

and finally install B-FADE:

.. code-block::

	conda install bfade

