Bayesian Inference
==================

Initialisation
--------------

The dataset is ready to undergo Maximum a Posteriori Estimation (MAP). It is now time to build the Bayesian infrastructure. To this end, we invoke the constructor of :py:mod:`bfade.elhaddad.ElHaddadBayes`:

    .. code-block:: python

        bay = ElHaddadBayes("dk_th", "ds_w", Y=aFloat, name=dat.name) # The default name is 'Untitled'

where ``name`` can be any string, though we load ``name = dat.name`` (the name of the dataset) for later referencing. Using a consistent naming shall help organise exported files and figure. Please note that ``dk_th`` and ``ds_w`` are the probabilistic trainable parameters, whereas `Y` is regarded as deterministic. Specifically, we infer the vector :math:`\theta=\left[\Delta K_{th, lc}\quad \Delta\sigma_w\right]`, using the input feature vector :math:`\mathbf{x} = \left[\sqrt{\text{area}} \quad \Delta\sigma \right]`.

:py:mod:`bfade.abstract.AbstractBayes` comes with an intuitive interface to progressively build the elements of Bayes' theorem, which :py:mod:`bfade.elhaddad.ElHaddadBayes` inherits. 

Prior
-----

We initially define the prior over each parameter. B-FADE enables the prior to be instantiated from any ``scipy.stats``'s probability distribution. Alongside this, a custom Uniform distribution is provided in :py:mod:`bfade.statistics.uniform` (the motivation is highlighted in the function's documentation). In this example, we hypothesise a Gaussian and a Uniform distribution for :math:`\Delta K_{th,lc}` and :math:`\Delta\sigma_w`, respectively:

    .. math::
        \Delta K_{th,lc} \sim \mathcal{N(\mu, \sigma)}

    .. math::
        \Delta\sigma_w \sim \mathcal{U}

Please note that the standard deviation :math:`\sigma` was used in the definition of the Gaussian prior, instead of the variance :math:`\sigma^2`. This reflects the implementation.

- :math:`\Delta K_{th,lc}` is initialised by:

    .. code-block:: python

        # provide a list of values
        dk_th_list = aList
        
        mean = np.array(dk_th_list).mean() 
        # mean  = aNumber is allowed as well
        std = np.array(dk_th_list).std() 
        # std  = aNumber is allowed as well

        # initialisation
        from scipy.stats import norm
        bay.load_prior("dk_th", norm, loc=mean, scale = std)     

- :math:`\Delta \sigma_w` is initialised by:

    .. code-block:: python

        from bfade.statistics import uniform
        bay.load_prior("ds_w", uniform, unif_value = 1)

The log-prior is automatically (so, no invocation is needed) assembled by :py:mod:`bfade.abstract.AbstractBayes.log_prior` as the logarithm of the following expression:

    .. math::
        P[\theta] = P[\Delta K_{th,lc}]\, P[\Delta\sigma_w]

that is, the priors are composed as independent distributions.


Likelihood
----------
Since B-FADE handles a binary classification problem, users are expected to compute the **logarithm** of Bernoulli likelihood. The likelihood is given by:

    .. math::

         P[D|\theta] = \prod_{i=1}^{\mathsf{N}} P[\mathbf{x}_i | \theta]^{y_i}\,(1- P[\mathbf{x}_i | \theta])^{1-y_i}

where :math:`y_i` and :math:`P[\mathbf{x}_i | \theta]` and the ground-truth, and predicted label of the data, respectively (0 for runout samples, 1 for failed samples). The predicted values are computed via :py:mod:`bfade.ElHaddadBayes.predictor` (concrete method of :py:mod:`bfade.AbstractBayes.predictor`). This function, performs the logistic regression:

    .. math::
        P[\mathbf{x}_i | \theta] = \frac{1}{1 + \exp[-\mathcal{H}(\mathbf{x}_i, \theta)]}

where :math:`\mathcal{H}(\mathbf{x}_i, \theta)` is the signed distance between the sample :math:`\mathbf{x}_i` and the El Haddad curve with parameters :math:`\theta`. The signed distance is implemented in :py:mod:`bfade.elhaddad.ElHaddadCurve:signed_distance_to_dataset` and wrapped into :py:mod:`bfade.AbstractBayes.predictor`.

The set up the log-likelihood we only need to call the method:

    .. code-block:: python

        from sklearn.metrics import log_loss
        bay.load_log_likelihood(log_loss, normalize=aBool)

In this respect, we must pay attention to the keyword ``normalize``. Precisely, if we use uniform priors on both parameters (default when invoking ``ElHaddadBayes``) we are actually performing Maximum Likelihood Estimation, and the log-likelihood need not to be 'normalised':

    .. code-block:: python

        bay.load_log_likelihood(log_loss, normalize=False)

By contrast, if we use other priors, we are regularising the likelihood. Therefore, this requires:

    .. code-block:: python

        bay.load_log_likelihood(log_loss, normalize=True)

Now ``bay`` can compute the log-likelihood of the dataset, i.e. :math:`\log P[D|\theta]`.


Before proceeding, we ought to spend a few words on computing the signed distance that the log-likelihood entails. :py:mod:`bfade.ElHaddadBayes.predictor` computes the signed distance by finding the mimimum-distance point of each datum of :math:`D` over the log-log plane. The method, in fact, invokes:

    .. code-block:: python

        eh = ElHaddadCurve(metrics=np.log10, **all_pars)

using ``np.log10`` as ``metrics``. Then, the distances between the given and the mimimum-distance points are computed using the common Euclidean metrics over the lin-lin plane. If, for a certain reason, the user needs to compute the minimum-distance points over the lin-lin plane, just invoke the curve with ``metrics=identity``, where ``identity`` is :py:mod:`bfade.util.identity`.

Posterior
---------

The last element of Bayes' theorem is the posterior:

    .. math::
        P[\theta | D] = \frac{P[D | \theta]\, P[\theta]}{P[D]}

It is well-known that MAP involves maximising the log-posterior:

    .. math::
        \log P[\theta | D] = \log P[D | \theta] + \log P[\theta]

neglecting the log-evidence :math:`-\log P[D]`, as it is a constant. In this case, nothing else has to be coded. The method :py:mod:`bfade.elhaddad.ElHaddadBayes.log_posterior` takes care of arranging the log-posterior.

Maximum a Posteriori Estimation
-------------------------------

The last step entails executing MAP:

    .. math::
        \hat{\theta} = -\underset{\theta}{\text{argmin}}\ \log P[\theta | D]

B-FADE exploit the monotonicity of the logarithm to perform the computations, thus allowing the evidence's term (:math:`-P[D]`, constant) to be neglected in the optimisation.

MAP runs by:
    
    .. code-block:: python
        
        bay.MAP(dat, x0=[aFloadDkGuess, aFloadDsGuess])

where ``aFloadDkGuess`` and ``aFloadDsGuess`` are the initial guesses for the optimiser.

If MAP succeedes the optimal values of the parameters :math:`\hat{\theta} =  \begin{bmatrix} \Delta K_{th,lc}\quad \Delta\sigma_w \end{bmatrix}` are stored in ``bay.theta_hat``. Furthermore, the next stage required the inverse Hessian matrix computed at ``theta_hat``. If MAP succeedes, such matrix is stored in ``bay.ihess``.

Posterior Approximation
-----------------------

Once MAP succeedes the posterior distribution is approximated using Variational Inference. In this case the approximator is a multivariate Gaussian distribution (i.e. Laplace's approximation) such that:

    .. math::
        P[\theta | D ] = \mathcal{N}(\hat{\theta}, H^{-1})

where :math:`H^{-1}` is the inverse Hessian matrix of :math:`-\log P[\theta | D]` evaluated at :math:`\hat{\theta}`. This is the joint posterior distribution. The approximation automatically loads in case of MAP succeedes, which not only loads the *joint* posterior distribution (the last equation), but also extracts the *marginal* posterior distributions:

    .. math::
        \Delta K_{th,lc} \sim \mathcal{N}(\hat{\Delta K_{th,lc}}, H^{-1}_{11})

    .. math::
        \Delta\sigma_w \sim \mathcal{N}(\hat{\Delta\sigma_w}, H^{-1}_{22})