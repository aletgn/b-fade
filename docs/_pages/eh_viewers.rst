Viewers
=======

B-FADE is provided with three *viewers* to display the input data as well as results:

- :py:mod:`bfade.viewers.BayesViewer` displays:

    - the contour of the log-prior;
    - the contour of the log-likelihood;
    - the contour of the log-posterior.

- :py:mod:`bfade.viewers.LaplacePosteriorViewer` displays:

    - the contour of the joint posterior;
    - the marginal posterior distributions.

- :py:mod:`bfade.viewers.PreProViewer` arranges a common canvas to inspect:

    - datasets;
    - curves;
    - prediction intervals;
    - predictive posterior distribution.

Each viewer has a ``config`` method to configure the options to export the pictures. Both ``BayesViewer`` and ``LaplacePosteriorViewer`` are subclassed from :py:mod:`bfade.abstract.AbstractMAPViewer`, so their retains similar interfaces.

Bayes Viewer
------------

Initially, we instantiate a viewer:

    .. code-block:: python

        v = BayesViewer("dk_th", aListEdgesX1, aIntN1,
                        "ds_w", aListEdgesX2, aIntN2, name=aStringName)

where ``dk_th`` and ``ds_w`` selects the variables whereby the elements of the Bayes theorem are plotted. Next we the elements theorem in their logarithmic form via:

    .. code-block:: python

            v.contour("log_prior", bay)
            v.contour("log_likelihood", bay, dat)
            v.contour("log_posterior", bay, dat)

Laplace Posterior Viewer
------------------------

Similarly, we instantiate a viewer:

    .. code-block:: python

            l = LaplacePosteriorViewer("dk_th", aInt1, aIntN1,
                                        "ds_w", aInt2, aIntN2,
                                        name=aStringName)

where ``aInt*`` is the coverage factor. The we run:

    .. code-block:: python

        l.contour(bay)
        l.marginals("dk_th", bay)
        l.marginals("ds_w", bay)

to display the joint (``contour``) and the marginal posterior distributions. If one wishes to override the limits of the x- and y-axis, they would call ``config_contour``.

Pre- and Post-Processing Viewer 
-------------------------------

Before displaying anything we need to instantiate a viewer:

    .. code-block:: python
        
        pp = PreProViewer(aListEdgesX1, aListEdgesX2, aIntResCurves, scale="log")

Everything is set up, hence let us make a few examples. Bear in mind that the viewer progressively compose the figure based upon the inputs. Therefore, it allows many glyph to be overlaid. For convenience we assume that the dataset has been acquired and then split, so we have:

    - the training dataset in ``dat_tr``
    - the test dataset in ``dat_ts``

- Plot only training dataset:

    .. code-block:: python
        
        pp.view(train_data = dat_tr)

- Plot only test dataset:

    .. code-block:: python
            
            pp.view(test_data = dat_ts)

- Plot both datasets:

    .. code-block:: python
            
            pp.view(train_data = dat_tr, test_data = dat_ts)

- Plot an El Haddad curve ``eh`` along with the datasets:

    .. code-block:: python
            
            pp.view(train_data = dat_tr,
                    test_data = dat_ts,
                    curve = [bay.el_haddad_hat]) #list!

the argument is a list as the method allows for plotting multiple curves. Consider one instantiated via :py:mod:`bfade.elhaddad.ElHaddadCurve`, i.e. ``eh1``. Hence:

    .. code-block:: python
            
            pp.view(train_data = dat_tr,
                    test_data = dat_ts,
                    curve = [eh, eh1])

- Plot the frequentist propagating crack region along with the training dataset:

    .. code-block:: python

            pp.view(train_data = dat_tr, prediction_interval = mc, mc_samples=aInt,
                    mc_distribution=aString, mc_bayes=bay, confidence=aInt)

where ``aString`` can take either ``joint`` or ``marginals`` to determine how the posterior must be sampled.

- Plot the Bayesian *and* frequentist propagating crack region along with the training dataset and. An additional argument must be passed to indicate which statistical indicator has to be plot. For instance, the ``mean`` and  standard deviation ``std``:
    
    .. code-block:: python
        
        pp.view(train_data=dat_tr,
                 prediction_interval = mc, mc_samples=aInt,
                 mc_distribution=aString, mc_bayes=bay, confidence=aInt
                 predictive_posterior = bay, post_samples, post_data=dataset,
                 post_op=aFunction)

Importantly, ``post_op`` can be used to load ``numpy`` functions to process the predictions. For instance, we can compute the mean and the standard deviation by passing ``np.mean`` and ``np.std``, respectively.