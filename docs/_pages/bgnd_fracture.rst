Fracture Mechanics
==================

Before proceeding it is worth recalling a few concepts of Fracture Mechanics, which shall turn out to be useful later on.

Stress Intensity Factor Range
-----------------------------

It is well-established that the severity of fatigue loads in the neighbouring region of a crack can be assess via the stress intensity factor range (SIF) :math:`\Delta K`:

.. math::
	\Delta K = Y\, \Delta\sigma \sqrt{\pi\, a}
	
where :math:`\Delta\sigma` is the applied stress range, :math:`Y` is the geometric factor of the crack, and :math:`a` is the characteristic length of the crack.

El-Haddad Curve
---------------

Assuming :math:`\Delta K` of the prior section as the fatigue crack driving force, it is possible to outline the fatigue endurance limit of flawed or cracked metals exploiting the El-Haddad (EH) curve, which is a semi-empirical model based on Linear Fracture Mechanics.

For the sake of illustrating the package's functionalities, we consider a metallic alloy containing defects. Furthermore, we restate :math:`\Delta K`, by associating  a characteristic crack length to the defects, in agreement with Murakami :cite:`murakami_ii_2019`. Hence:

.. math::
	a\mapsto \sqrt{\text{area}}

where :math:`\sqrt{\text{area}}` is the projected area of the defect onto the plane orthogonal to the direction of the applied fatigue load. Accordingly, the SIF range turns out to be:

.. math::
	\Delta K = Y\, \Delta\sigma \sqrt{\pi\, \sqrt{\text{area}}}

where, in this case, :math:`Y` is determined upon the distance between the defect's centroid and the free surface of the fatigue specimen :cite:`salvati_defect-based_2022`.

Given the above, the EH curve is analytically defined as :cite:`zerbst_damage_2021`:

.. math::
	\Delta\sigma = \Delta\sigma_w \sqrt{{{\sqrt{\text{area}}}\over{\sqrt{\text{area}_0}+\sqrt{\text{area}}}}}
	
where :math:`\Delta\sigma_w` is the fatigue endurance limit of the defect-free specimen, and :math:`\sqrt{\text{area}_0}` is the EH critical length, defined using the inverse of the SIF range for defects:

.. math::
	\sqrt{\text{area}_0} = \frac{1}{\pi} \bigg(\frac{\Delta K_{th,lc}}{Y\,\Delta\sigma_w}\bigg)^2
	
in which :math:`\Delta K_{th,lc}` is the SIF range threshold for long cracks. It is therefore clear that both :math:`\Delta K_{th,lc}` and :math:`\Delta\sigma_w` must be determined if one wishes to estimate the fatigue endurance limit of a given material. To do so, B-FADE implements the Maximum a Posteriori Estimation (MAP) approach showcased in :cite:`tognan_probabilistic_2023`. Once the posterior distribution of the parameters is known, it is opportunely post-processed the outline the probabilistic propagating crack region via both frequentist and Bayesian approaches.
	
References
----------
.. bibliography:: ../references.bib
   :style: unsrt