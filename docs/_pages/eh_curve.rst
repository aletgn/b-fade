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

Istantiation
------------

The implementation of the El Haddad curve is given by subclassing from :py:mod:`bfade.abstract.AbstractCurve` and defining a concrete method for :py:mod:`bfade.abstract.AbstractCurve.equation`, i.e. :py:mod:`bfade.abstract.ElHaddadCurve.equation`, according to the prior section.

To istantiate an El Haddad curve, we do:

	.. code-block:: python

		eh = ElHaddadCurve(metrics=aCallable, dk_th=aFloat, ds_w=aFloat, Y=0.65, name=aStringName)

So, we istantiate the curve via :math:`\Delta k_{th, lc}`, and :math:`\Delta\sigma_w`, whilst :math:`\sqrt{\text{area}_0}` is computed accordingly. The choice of ``metrics`` shall be discussed later. 

Preliminary Inspection
----------------------

We can graphically inspect the curve with:

	.. code-block:: python

			eh.inspect(np.linspace(aFloatStart, aFloatEnd, aIntStep), scale="log")