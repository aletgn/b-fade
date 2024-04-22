Curve
=====

Units of Measurement
--------------------
The quantities that the El Haddad curve necessitates must be provided complying with the following units of measurement:

- :math:`[\sqrt{\text{area}}] = \mu\text{m}`

- :math:`[\Delta\sigma] = \text{MPa}`

- :math:`[\Delta K] = \text{MPa}\sqrt{\text{m}}`

- :math:`[\Delta\sigma_w] = \text{MPa}`

- :math:`[\Delta K_{th,lc}] = \text{MPa}\sqrt{\text{m}}`

Implemented Variables
---------------------
As concerns the practical implementation of MAP for EH curves, please note that:

- :math:`\Delta K_{th,lc}` is represented as ``dk_th``

- :math:`\Delta\sigma_w` is represented as ``ds_w``

- :math:`Y` is represented, obviously, as ``Y``.

Having shorter names, yet sufficiently explanatory, helps contract the code.

Instantiation
-------------

The implementation of the El Haddad curve is given by subclassing from :py:mod:`bfade.abstract.AbstractCurve` and defining a concrete method for :py:mod:`bfade.abstract.AbstractCurve.equation`, i.e. :py:mod:`bfade.abstract.ElHaddadCurve.equation`, according to the prior section.

To instantiate an El Haddad curve, we do:

	.. code-block:: python

		eh = ElHaddadCurve(metrics=aCallable, dk_th=aFloat, ds_w=aFloat, Y=aFloat, name=aStringName)

So, we instantiate the curve via :math:`\Delta K_{th, lc}`, and :math:`\Delta\sigma_w`, whilst :math:`\sqrt{\text{area}_0}` is computed accordingly. The choice of ``metrics`` shall be discussed later. 

Preliminary Inspection
----------------------

We can graphically inspect the curve with:

	.. code-block:: python

			eh.inspect(np.linspace(aFloatStart, aFloatEnd, aIntStep), scale="log")