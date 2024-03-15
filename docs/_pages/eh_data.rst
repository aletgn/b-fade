Dataset
=======

Units of Measurement
--------------------
Data must be provided complying with the following units of measurement:

- :math:`[\sqrt{\text{area}}] = \mu\text{m}`

- :math:`[\Delta\sigma] = \text{MPa}`

- :math:`[\Delta K] = \text{MPa}\sqrt{\text{m}}`

- :math:`[\Delta\sigma_w] = \text{MPa}`

- :math:`[\Delta K_{th,lc}] = \text{MPa}\sqrt{\text{m}}`

Structure
---------

B-FADE requires fatigue data & defect characterisation to accomplish the identification of the EH parameters. As concerns fatigue data, we assume that :math:`\mathsf{N}` experimental results have been collected, for given applied stress ranges :math:`\Delta\sigma` and for a fixed stress ratio :math:`R`, i.e. :math:`N_i,\Delta\sigma_i\ \forall\, i=1,2,\dots,\mathsf{N}`. Upon setting a runout threshold :math:`N_w`, the experimental points are distinguished as *Failed* and *Runout*, according to:

.. math::
	y_i = \begin{align}\begin{cases} \text{Runout}\ &\text{if}\ N_i \le N_w\\ \text{Failed}\ &\text{if}\ N_i > N_w \end{cases}\end{align}

which is translated, from a computational perspective as:

.. math::
	y_i = \begin{align}\begin{cases} 0\ &\text{if}\ N_i \le N_w\\ 1\ &\text{if}\ N_i > N_w \end{cases}\end{align}

B-FADE does not implement the above conditional, so data must be provided already labelled by :math:`y_i` (0 or 1).

We also assume that :math:`\sqrt{\text{area}}` of the *killer* defect that triggered fatigue failure has been measured using post-mortem investigation. Therefore, we can define the following dataset:

.. math::
	D = \{((\sqrt{\text{area}}_i, \Delta\sigma_i, Y_i), y_i) \mid i=1,2,\dots,N\}

Format
------

B-FADE accepts both ``.csv`` and spreadsheets as input files. Spreadsheets formats include ``.ods`` (LibreOffice, OpenOffice), and ``.xls`` & ``.xlsx`` (Microsoft Excel). The input file can be *headered* or *headerless*. Accordingly, the data can be loaded in two differente ways, though the former is more straightforward.

- Headered spreadsheets files should be arranged as:

		+-----------+-------------+----+--------+-----+
		| sqrt_area | delta_sigma | Y  | failed | test|
		+===========+=============+====+========+=====+
		|           |             |    |        |     |
		+-----------+-------------+----+--------+-----+

	
- Headered ``.csv`` files should be arranged as:

	.. code-block::

		sqrt_area,delta_sigma,y,failed,test

In both cases, the first two columns are obviously :math:`\sqrt{\text{area}}`, :math:`\Delta\sigma`, :math:`Y` is the geometric factor for defects. Regarding ``failed``, this column is binary, i.e. 0 and 1 means runout and failed samples, respectively. The last column serves the purpose of testing. This column, in fact, allows for performing a custom train/test split of the data. If you do not wish to do so fill out this column with 0. For headerless files the same columns must be provided without writing the column names.

Acquisition
-----------

No conditional statements regulates the acquisition of the different format files. To ensure more versatility the user can invoke a specific ``reader`` from ``pandas`` when istantiating :py:mod:`bfade.elhaddad.ElHaddadDataset` which subclass'es :py:mod:`bfade.dataset.Dataset`

- To read a spreadsheet, do:

	.. code-block::

			dat = ElHaddadDataset(reader=pd.read_excel, path="aString", sheet_name = "aSheet")

- To read a ``.csv``, do:

		.. code-block::

			dat = ElHaddadDataset(reader= pd.read_csv, path = "aString")

Regarding headerless files, more arguments must be passed, though.

- To read a spreadsheet, do:

	.. code-block::

			dat = ElHaddadDataset(reader=pd.read_excel, path="aString",
			                      sheet_name = "aSheet", usecols = cid, names = cname)

- To read a ``.csv``, do:

		.. code-block::

			dat = ElHaddadDataset(reader=pd.read_excel, path="aString",
							      usecols = cid, names = cname)

In both cases, ``cid`` and ``cname`` are List:

		.. code-block:: python

			cid = [0, 1, 2, 3, 4]
			cname = ["sqrt_area", "delta_sigma", "Y", "failed", "test"]

which must have the same length. The index in ``cid`` are the position of the column (starting from 0!), whereas ``cname`` is name the user gives to the column, eventually. The set of keys in ``cname`` is *mandatory* as B-FADE expects exactly them.

Importantly, the El Haddad curve has to be referred to a unique :math:`Y`. To do so, we invoke :py:mod:`bfade.elhaddad.ElHaddadDataset.pre_process`. If the `Y` is known to be unique beforehand, no arguments has to passed. Otherwise, we provide `Y_ref` as an argument. From the code perspective:

    .. code-block:: python

            dat.pre_process() # unique Y

In contrast:

    .. code-block:: python

            dat.pre_process(Y_ref=aFloat) # non-unique Y

In the latter case, B-FADE rescales the input values of :math:`\Delta K` by SIF equivalence:

	.. math::
		\Delta K_{ref} = \Delta K_{i}
		
hence:

	.. math::
		\Delta\sigma\, Y_{ref} \sqrt{\pi \sqrt{\text{area}}_{ref}} = \Delta\sigma\, Y_{i} \sqrt{\pi \sqrt{\text{area}}_{i}}
		
finally:

	.. math::
		\sqrt{\text{area}}_{ref}=\sqrt{\text{area}}_{i}\,\bigg({{Y_{i}} \over {Y_{ref}}}\bigg)^2

The user shall find the dataset stored in ``dat.data``.


Train/Test Split
----------------

**Forego this section if splitting is not required**. There are two way you can perform train/test split, and both are inherited from the superclass :py:mod:`bfade.elhaddad.dataset.Dataset`. The split is performed invoking :py:mod:`bfade.elhaddad.ElHaddadDataset.partition`:

- Random train/test split.

	.. code-block:: python

		dat.partition("random", test_size=0.2)

	which wraps ``train_test_split`` from ``sklearn.model_selection``. In this case, 80% samples are reserved for training the El Haddad parameters and 20% are treated as test samples.

- User-defined train/test split

	.. code-block:: python

		dat.partition("user") 

	this option requires users to indicate 0 or 1 in  ``test`` column of the input files, thus marking specimens as test (1), or train (0).

The invoked method returns two new instances of :py:mod:`bfade.elhaddad.ElHaddadDataset`, the training and test dataset, respectively.

.. Generation
.. ----------

.. B-FADE also offers a subclass of :py:mod:`bfade.elhaddad.Dataset` to create synthetic datasets (grids or tubes) for test/evaluation purposes, i.e. :py:mod:`bfade.datset.SyntheticDataset`. To this end, B-FADE defines :py:mod:`bfade.datagen.ElHaddadGrid`, inheriting from :py:mod:`bfade.elhaddad.ElHaddad`.

.. Initially, we can make up two reference values for :math:`\Delta K_{th,lc}` and :math:`\Delta\sigma_w` which function as the "reference values" whereby the dataset is generated. Accordingly, we invoke the constructor of :py:mod:`bfade.datagen.ElHaddadGrid`:

.. .. code-block:: python

.. 	grd = ElHaddadGrid(y=0.65, delta_sigma_w=aFloat, delta_k_th_lc=aFloat, name="aName") # a given name

.. Next, we generate a regular grid of points, which makes use of :py:mod:`bfade.util.grid_factory`:

.. .. code-block:: python

.. 	grd.make_grid(x_bounds=aList, y_bounds=aList, x_res=aInt, y_res=aInt)

.. Following, we perturb the grid by Gaussian random noise:

.. .. code-block:: python

.. 	grd.perturb_grid(std_sa=aFloat, std_ds=aFloat)

.. Finally, the dataset is exported by:

.. .. code-block:: python

.. 	grd.make_dataset(writer="aPandasWriter",
.. 					 extension="anExtension",
.. 					 folder="aFolder")