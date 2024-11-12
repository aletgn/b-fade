This page gathers the Jupyter Notebook, ``.py`` script, and ``.yaml`` configuration files shown in the previous sections. Just hover over the desired file and:

.. code-block::
    
    Mouse right click > Save Link As...

Notebooks
---------

All the Jupyter notebooks provided below are full compatible with `Google Colab <https://colab.research.google.com>`_, so they can be imported and run therein. If those are run Colab, please make sure B-FADE is install in you working environment. Hence, make a cell with this line:

.. code-block:: bash

    !pip install b-fade

To enable processing dataset files, these must be uploaded to Google Drive. Therefore the users' google drive must be mounted in Colab workspace. To do so, make a cell that imports the required package and mounts the remote folder:

.. code-block:: python

    from google.colab import drive
    drive.mount('/content/drive')

By default, Google Colab notebooks are located in subfolder ``/content/drive/My Drive/Colab Notebooks/``. Assuming the user uploaded their dataset file ``MyDataset.csv`` in such folder, B-FADE methods and function can get the dataset at: ``/content/drive/My Drive/Colab Notebooks/MyDataset.csv``.

Before running `Parametrised El Haddad`, please run `Fictitious El Haddad Datasets` to generate the dataset for test purpose. If notebooks are run in Google Colab upload the dataset files to Google Drive and access them as indicate above (as you would do with real datasets).

Available notebooks:

- :download:`Fictitious El Haddad Datasets <../_examples/eh_dataset.ipynb>`

- :download:`Parametrised El Haddad <../_examples/eh_notebook.ipynb>`

- :download:`Custom Classes <../_examples/custom_classes.ipynb>`

Python Scripts
--------------

- :download:`El Haddad Script<../_examples/eh_shell.py>`

Yaml Config Files
-----------------

- :download:`El Haddad Yaml<../_examples/eh_shell.yaml>`
