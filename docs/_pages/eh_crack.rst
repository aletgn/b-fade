Propagating Crack Region
========================

Frequentist (Monte Carlo)
-------------------------
The posterior can be sampled to determine the El Haddad curve at different failure probabilities, thereby obtaining the estimation of the probabilistic propagating crack region.

Firstly, let us instance an object of class :py:mod:`bfade.statistics.MonteCarlo`:

    .. code-block:: python

        mc = MonteCarlo(ElHaddadCurve)

In ``samples`` input how many samples must be drawn from the posterior. Now, we can sample:

- the *joint* posterior distribution

    .. code-block:: python

        mc.sample(aIntNSamples, "joint", bay)

- the *marginal* posterior distribution

    .. code-block:: python

        mc.sample(aIntNSamples, "marginals", bay)

That's it, the Monte Carlo simulation is accomplished. If we wish to computed the prediction interval, i.e. the frequentist propagating crack region, we invoke:

    .. code-block:: python

        mean, pred, x = mc.prediction_interval(aListXEgdes, aIntSamples, "log", aFloatConfidence)

which compute the :math:`\sqrt{\text{area}}`-wise prediction interval:

    .. math::

        P[\overline{\mathcal{E}^{(\mathsf{M})}} - \mathcal{P}^{(\mathsf{M})} \le \mathcal{E}^{({\mathsf{M}}+1)} \le \overline{\mathcal{E}^{(\mathsf{M})}} +\mathcal{P}^{(\mathsf{M})}] = \beta

where :math:`\overline{\mathcal{E}^{(\mathsf{M})}}` is the :math:`\sqrt{\text{area}}`-wise expected value of the set of El Haddad curves and :math:`\mathcal{P}^{(\mathsf{M})}` is the semi-amplitude of the prediction interval:

    .. math::
	    \mathcal{P}^{(\sf M)} = T_{\beta}\ S^{(\sf M)}  \sqrt{1 + 1/{\mathsf{M}}}

in which :math:`T_\beta` is the :math:`1-\beta/2` percentile of :math:`T`-Student distribution, and :math:`S^{(\sf M)}` is the :math:`\sqrt{\text{area}}`-wise standard deviation of the history of the :math:`\mathsf{M}` EH curves.

Bayesian
--------
An alternative approach to outlining the crack is offered in :py:mod:`bfade.ElHaddadBayes.predictive_posterior`. The method implements the numerical computation of the predictive posterior:

.. math::
    P[\mathbf{x}^* | D] = \int_\theta P[\mathbf{x}^* | \theta] P[\theta | D]\, d\theta

where :math:`\mathbf{x}^*` is an unseen datum, and :math:`P[\mathbf{x}^* | \theta]` is given by Logistic Regression. Since the integral is analytically intractable in most of the cases, the numerical approximation is implemented:

.. math::
    P[\mathbf{x}^* | D] = \sum_{i=1}^{\mathsf{P}} P[\mathbf{x}^* | \theta^{(i)}]

where :math:`\theta^{(i)}\, i = 1,2,\dots, \mathsf{P}` are samples drawn from the joint posterior.

Let us prepare a grid of points over which the posterior is evaluated. We user the utility :py:mod:`bfade.util.grid_factory`:

    .. code-block:: python

        X1, X2 = grid_factory("log",
                              aListX1Edges,
                              aListX2Edges,
                              aIntN1,
                              aIntN1)

which basically creates a regular rectangular grid (:math:`\sqrt{\text{area}}\times \Delta\sigma_w`) whose point are ``log``-spaced, spanning ``aListX1Edges`` and ``aListX2Edges``, with a number of ``aIntN1`` and ``aIntN1``.

The evaluation of the predictive posterior is done by:

    .. code-block:: python

        pred = bay.predictive_posterior(aIntSamples, dataset, aFunction)

where ``aIntSamples`` is the number of samples drawn from the posterior, and ``dataset`` is an instance of `bfade.dataset.Dataset`. Importantly, ``aFunction`` can be used to load numpy functions to process the predictions. For instance, we can compute the mean and the standard deviation by passing ``np.mean`` and ``np.std`` , respectively.