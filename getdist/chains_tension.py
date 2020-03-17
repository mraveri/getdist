"""
This file contains the functions and utilities to compute agreement and
disagreement between two different chains either using a Gaussian approximation
for the posterior or not.

For more details on the method implemented see
`arxiv 1806.04649 <https://arxiv.org/pdf/1806.04649.pdf>`_
and `arxiv 1912.04880 <https://arxiv.org/pdf/1912.04880.pdf>`_.
If you find these methods and papers useful consider citing them
in your publications.
"""

import os
import time
import gc
import scipy
import numpy as np
from getdist import MCSamples, WeightedSamples
from getdist.gaussian_mixtures import GaussianND
from scipy.linalg import sqrtm

# imports for parallel calculations:
import multiprocessing
import joblib
# number of threads available:
if 'OMP_NUM_THREADS' in os.environ.keys():
    n_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    n_threads = multiprocessing.cpu_count()


###############################################################################
# Utilities:
###############################################################################


def from_confidence_to_sigma(P):
    """
    Transforms a probability to effective number of sigmas.
    This matches the input probability with the number of standard deviations
    that an event with the same probability would have had in a Gaussian
    distribution as in Eq. (G1) of
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_).

    .. math::
        n_{\\sigma}^{\\rm eff}(P) \\equiv \\sqrt{2} {\\rm Erf}^{-1}(P)

    :param P: the input probability.
    :return: the effective number of standard deviations.
    """
    if (np.all(P < 0.) or np.all(P > 1.)):
        raise ValueError('Input probability has to be between zero and one.\n',
                         'Input value is ', P)
    return np.sqrt(2.)*scipy.special.erfinv(P)


def from_sigma_to_confidence(nsigma):
    """
    Gives the probability of an event at a given number of standard deviations
    in a Gaussian distribution.

    :param nsigma: the input number of standard deviations.
    :return: the probability to exceed the number of standard deviations.
    """
    if (np.all(nsigma < 0.)):
        raise ValueError('Input nsigma has to be positive.\n',
                         'Input value is ', nsigma)
    return scipy.special.erf(nsigma/np.sqrt(2.))


def get_prior_covariance(chain, param_names=None):
    """
    Utility to estimate the prior covariance from the ranges of a chain.
    The flat range prior covariance
    (`link <https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_)
    is given by:

    .. math:: C_{ij} = \\delta_{ij} \\frac{( max(p_i) - min(p_i) )^2}{12}

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param param_names: optional choice of parameter names to
        restrict the calculation.
    :return: the estimated covariance of the prior.
    """
    # get the parameter names to use:
    if param_names is None:
        param_names = chain.getParamNames().getRunningNames()
    # get the ranges:
    _prior_min = []
    _prior_max = []
    for name in param_names:
        # lower bound:
        if name in chain.ranges.lower.keys():
            _prior_min.append(chain.ranges.lower[name])
        else:
            _prior_min.append(-1.e30)
            # upper bound:
        if name in chain.ranges.upper.keys():
            _prior_max.append(chain.ranges.upper[name])
        else:
            _prior_max.append(1.e30)
    _prior_min = np.array(_prior_min)
    _prior_max = np.array(_prior_max)
    #
    return np.diag((_prior_max-_prior_min)**2/12.)


def get_Neff(chain, prior_chain=None, param_names=None, prior_factor=1.0):
    """
    Function to compute the number of effective parameters constrained by a
    chain over the prior.
    The number of effective parameters is defined as in Eq. (29) of
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) as:

    .. math:: N_{\\rm eff} \\equiv
        N -{\\rm tr}[ \\mathcal{C}_\\Pi^{-1}\\mathcal{C}_p ]

    where :math:`N` is the total number of nominal parameters of the chain,
    :math:`\\mathcal{C}_\\Pi` is the covariance of the prior and
    :math:`\\mathcal{C}_p` is the posterior covariance.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param prior_chain: (optional) the prior chain.
        If the prior is not well approximated by
        a ranged prior and is informative it is better to explicitly
        use a prior only chain.
        If this is not given the algorithm will assume ranged priors with the
        ranges computed from the input chain.
    :param param_names: (optional) parameter names to restrict the
        calculation of :math:`N_{\\rm eff}`.
        If none is given the default assumes that all running parameters
        should be used.
    :param prior_factor: (optional) factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :return: the number of effective parameters.
    """
    # initialize param names:
    if param_names is None:
        param_names = chain.getParamNames().getRunningNames()
    else:
        chain_params = chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # initialize prior covariance:
    if prior_chain is not None:
        # check:
        prior_params = prior_chain.getParamNames().list()
        if not np.all([name in prior_params for name in param_names]):
            raise ValueError('Input parameter is not in the prior chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', prior_params)
        # get the prior covariance:
        C_Pi = prior_chain.cov(pars=param_names)
    else:
        C_Pi = get_prior_covariance(chain, param_names=param_names)
    # multiply by prior factor:
    C_Pi = prior_factor*C_Pi
    # get the posterior covariance:
    C_p = chain.cov(pars=param_names)
    # compute the number of effective parameters
    _temp = np.dot(np.linalg.inv(C_Pi), C_p)
    # compute Neff from the regularized spectrum of the eigenvalues:
    _eigv, _eigvec = np.linalg.eig(_temp)
    _eigv[_eigv > 1.] = 1.
    _eigv[_eigv < 0.] = 0.
    #
    _Ntot = len(_eigv)
    _Neff = _Ntot - np.sum(_eigv)
    #
    return _Neff


def KL_decomposition(matrix_a, matrix_b):
    """
    Computes the Karhunen–Loeve (KL) decomposition of the matrix A and B. \n
    Notice that B has to be real, symmetric and positive. \n
    The algorithm is taken from
    `this link <http://fourier.eng.hmc.edu/e161/lectures/algebra/node7.html>`_.

    :param matrix_a: the first matrix.
    :param matrix_b: the second matrix.
    :return: the KL eigenvalues and the KL eigenvectors.
    """
    # compute the eigenvalues of b, lambda_b:
    _lambda_b, _phi_b = np.linalg.eigh(matrix_b)
    # check that this is positive:
    if np.any(_lambda_b < 0.):
        raise ValueError('B is not positive definite\n',
                         'KL eigenvalues are ', _lambda_b)
    _sqrt_lambda_b = np.diag(1./np.sqrt(_lambda_b))
    _phib_prime = np.dot(_phi_b, _sqrt_lambda_b)
    _a_prime = np.dot(np.dot(_phib_prime.T, matrix_a), _phib_prime)
    _lambda, _phi_a = np.linalg.eigh(_a_prime)
    _phi = np.dot(np.dot(_phi_b, _sqrt_lambda_b), _phi_a)
    return _lambda, _phi


def QR_inverse(matrix):
    """
    Invert a matrix with the QR decomposition.
    This is much slower than standard inversion but has better accuracy
    for matrices with higher condition number.

    :param matrix: the input matrix.
    :return: the inverse of the matrix.
    """
    _Q, _R = np.linalg.qr(matrix)
    return np.dot(_Q, np.linalg.inv(_R.T))


def Silverman_ROT(num_params, num_samples):
    """
    Compute Silverman's rule of thumb bandwidth covariance scaling.
    This is the default scaling that is used to compute the KDE estimate of
    parameter shifts.

    :param num_params: the number of parameters in the chain.
    :param num_samples: the number of samples in the chain.
    :return: Silverman's scaling.
    """
    silverman_rot = ((4./(float(num_params)+2.))**(1./(float(num_params)+4.))
                     / float(num_samples)**(1./(float(num_params)+4.)))**2
    return silverman_rot


def gaussian_approximation(chain, param_names=None):
    """
    Function that computes the Gaussian approximation of a given chain.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param param_names: (optional) parameter names to restrict the
        Gaussian approximation.
        If none is given the default assumes that all parameters
        should be used.
    :return: :class:`~getdist.gaussian_mixtures.GaussianND` object with the
        Gaussian approximation of the chain.
    """
    # test the type of the chain:
    if not isinstance(chain, MCSamples):
        raise TypeError('Input chain is not of MCSamples type.')
    # get parameter names:
    if param_names is None:
        param_names = chain.getParamNames().list()
    else:
        chain_params = chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # get the mean:
    mean = chain.getMeans(pars=[chain.index[name]
                          for name in param_names])
    # get the covariance:
    cov = chain.cov(pars=param_names)
    # get the labels:
    param_labels = [_n.label for _n
                    in chain.getParamNames().parsWithNames(param_names)]
    # get label:
    if chain.label is not None:
        label = 'Gaussian_'+chain.label
    elif chain.name_tag is not None:
        label = 'Gaussian_'+chain.name_tag
    else:
        label = None
    # initialize the Gaussian distribution:
    gaussian_approx = GaussianND(mean, cov,
                                 names=param_names,
                                 labels=param_labels,
                                 label=label)
    #
    return gaussian_approx


def get_separate_mcsamples(chain):
    """
    Function that returns separate :class:`~getdist.mcsamples.MCSamples`
    for each sampler chain.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :return: list of :class:`~getdist.mcsamples.MCSamples` with the separate
        chains.
    """
    # get separate chains:
    _chains = chain.getSeparateChains()
    # copy the param names and ranges:
    _mc_samples = []
    for ch in _chains:
        temp = MCSamples()
        temp.paramNames = chain.getParamNames()
        temp.setSamples(ch.samples, weights=ch.weights, loglikes=ch.loglikes)
        temp.sampler = chain.sampler
        temp.ranges = chain.ranges
        temp.updateBaseStatistics()
        _mc_samples.append(temp.copy())
    #
    return _mc_samples


def clopper_pearson_binomial_trial(k, n, alpha=0.32):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected
    successes on n trials.

    :param k: number of success.
    :param n: total number of trials.
    :param alpha: (optional) confidence level.
    :return: lower and upper bound.
    """
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi


###############################################################################
# Parameter based estimators:
###############################################################################


def pareameter_diff_weighted_samples(samples_1, samples_2, boost=1,
                                     indexes_1=None, indexes_2=None):
    """
    Compute the parameter differences of two input weighted samples.
    The parameters of the difference samples are related to the
    parameters of the input samples, :math:`\\theta_1` and
    :math:`\\theta_2` by:

    .. math:: \\Delta \\theta \\equiv \\theta_1 - \\theta_2

    This function does not assume Gaussianity of the chain.
    This functions does assume that the parameter determinations from the two
    chains (i.e. the underlying data sets) are uncorrelated.
    Do not use this function for chains that are correlated.

    :param samples_1: :class:`~getdist.chains.WeightedSamples`
        first input weighted samples with :math:`n_1` samples.
    :param samples_2: :class:`~getdist.chains.WeightedSamples`
        second input weighted samples with :math:`n_2` samples.
    :param boost: (optional) boost the number of samples in the
        difference. By default the length of the difference samples
        will be the length of the longest one.
        Given two samples the full difference samples can contain
        :math:`n_1\\times n_2` samples but this is usually prohibitive
        for realistic chains.
        The boost parameters wil increase the number of samples to be
        :math:`{\\rm boost}\\times {\\rm max}(n_1,n_2)`.
        Default boost parameter is one.
        If boost is None the full difference chain is going to be computed
        (and will likely require a lot of memory and time).
    :param indexes_1: (optional) array with the indexes of the parameters to
        use for the first samples. By default this tries to use all
        parameters.
    :param indexes_2: (optional) array with the indexes of the parameters to
        use for the second samples. By default this tries to use all
        parameters.
    :return: :class:`~getdist.chains.WeightedSamples` the instance with the
        parameter difference samples.
    """
    # test for type, this function assumes that we are working with MCSamples:
    if not isinstance(samples_1, WeightedSamples):
        raise TypeError('Input samples_1 is not of WeightedSamples type.')
    if not isinstance(samples_2, WeightedSamples):
        raise TypeError('Input samples_2 is not of WeightedSamples type.')
    # get indexes:
    if indexes_1 is None:
        indexes_1 = np.arange(samples_1.samples.shape[1])
    if indexes_2 is None:
        indexes_2 = np.arange(samples_2.samples.shape[1])
    # check:
    if not len(indexes_1) == len(indexes_2):
        raise ValueError('The samples do not containt the same number',
                         'of parameters.')
    num_params = len(indexes_1)
    # order the chains so that the second chain is always with less points:
    if (len(samples_1.weights) >= len(samples_2.weights)):
        ch1, ch2 = samples_1, samples_2
        sign = +1.
        ind1, ind2 = indexes_1, indexes_2
    else:
        ch1, ch2 = samples_2, samples_1
        sign = -1.
        ind1, ind2 = indexes_2, indexes_1
    # get number of samples:
    num_samps_1 = len(ch1.weights)
    num_samps_2 = len(ch2.weights)
    if boost is None:
        sample_boost = num_samps_2
    else:
        sample_boost = min(boost, num_samps_2)
    # create the arrays (these might be big depending on boost level...):
    weights = np.empty((num_samps_1*sample_boost))
    difference_samples = np.empty((num_samps_1*sample_boost, num_params))
    if ch1.loglikes is not None and ch2.loglikes is not None:
        loglikes = np.empty((num_samps_1*sample_boost))
    else:
        loglikes = None
    # compute the samples:
    for ind in range(sample_boost):
        base_ind = int(float(ind)/float(sample_boost)*num_samps_2)
        _indexes = range(base_ind, base_ind+num_samps_1)
        # compute weights (as the product of the weights):
        weights[ind*num_samps_1:(ind+1)*num_samps_1] = \
            ch1.weights*np.take(ch2.weights, _indexes, mode='wrap')
        # compute the likelihood difference:
        if ch1.loglikes is not None and ch2.loglikes is not None:
            loglikes[ind*num_samps_1:(ind+1)*num_samps_1] = \
                ch1.loglikes-np.take(ch2.loglikes, _indexes, mode='wrap')
        # compute the difference samples:
        difference_samples[ind*num_samps_1:(ind+1)*num_samps_1, :] = \
            ch1.samples[:, ind1] \
            - np.take(ch2.samples[:, ind2], _indexes, axis=0, mode='wrap')
    # get additional informations:
    if samples_1.name_tag is not None and samples_2.name_tag is not None:
        name_tag = samples_1.name_tag+'_diff_'+samples_2.name_tag
    else:
        name_tag = None
    if samples_1.label is not None and samples_2.label is not None:
        label = samples_1.label+' diff '+samples_2.label
    else:
        label = None
    if samples_1.min_weight_ratio is not None and \
       samples_2.min_weight_ratio is not None:
        min_weight_ratio = min(samples_1.min_weight_ratio,
                               samples_2.min_weight_ratio)
    # initialize the weighted samples:
    diff_samples = WeightedSamples(ignore_rows=0,
                                   samples=sign*difference_samples,
                                   weights=weights, loglikes=loglikes,
                                   name_tag=name_tag, label=label,
                                   min_weight_ratio=min_weight_ratio)
    #
    return diff_samples


def parameter_diff_chain(chain_1, chain_2, boost=1):
    """
    Compute the chain of the parameter differences between the two input
    chains. The parameters of the difference chain are related to the
    parameters of the input chains, :math:`\\theta_1` and :math:`\\theta_2` by:

    .. math:: \\Delta \\theta \\equiv \\theta_1 - \\theta_2

    This function only returns the differences for the parameters that are
    common to both chains.
    This function preserves the chain separation (if any) so that the
    convergence of the difference chain can be tested.
    This function does not assume Gaussianity of the chain.
    This functions does assume that the parameter determinations from the two
    chains (i.e. the underlying data sets) are uncorrelated.
    Do not use this function for chains that are correlated.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        first input chain with :math:`n_1` samples
    :param chain_2: :class:`~getdist.mcsamples.MCSamples`
        second input chain with :math:`n_2` samples
    :param boost: (optional) boost the number of samples in the
        difference chain. By default the length of the difference chain
        will be the length of the longest chain.
        Given two chains the full difference chain can contain
        :math:`n_1\\times n_2` samples but this is usually prohibitive
        for realistic chains.
        The boost parameters wil increase the number of samples to be
        :math:`{\\rm boost}\\times {\\rm max}(n_1,n_2)`.
        Default boost parameter is one.
        If boost is None the full difference chain is going to be computed
        (and will likely require a lot of memory and time).
    :return: :class:`~getdist.mcsamples.MCSamples` the instance with the
        parameter difference chain.
    """
    # check input:
    if boost is not None:
        if boost < 1:
            raise ValueError('Minimum boost is 1\n Input value is ', boost)
    # test for type, this function assumes that we are working with MCSamples:
    if not isinstance(chain_1, MCSamples):
        raise TypeError('Input chain_1 is not of MCSamples type.')
    if not isinstance(chain_2, MCSamples):
        raise TypeError('Input chain_2 is not of MCSamples type.')
    # get the parameter names:
    param_names_1 = chain_1.getParamNames().list()
    param_names_2 = chain_2.getParamNames().list()
    # get the common names:
    param_names = [_p for _p in param_names_1 if _p in param_names_2]
    num_params = len(param_names)
    if num_params == 0:
        raise ValueError('There are no shared parameters to difference')
    # get the names and labels:
    diff_param_names = ['delta_'+name for name in param_names]
    diff_param_labels = ['\\Delta '+name.label for name in
                         chain_1.getParamNames().parsWithNames(param_names)]
    # get parameter indexes:
    indexes_1 = [chain_1.index[name] for name in param_names]
    indexes_2 = [chain_2.index[name] for name in param_names]
    # get separate chains:
    if not hasattr(chain_1, 'chain_offsets'):
        _chains_1 = [chain_1]
    else:
        _chains_1 = chain_1.getSeparateChains()
    if not hasattr(chain_2, 'chain_offsets'):
        _chains_2 = [chain_2]
    else:
        _chains_2 = chain_2.getSeparateChains()
    # set the boost:
    if chain_1.sampler == 'nested' \
       or chain_2.sampler == 'nested' or boost is None:
        chain_boost = max(len(_chains_1), len(_chains_2))
        sample_boost = None
    else:
        chain_boost = min(boost, max(len(_chains_1), len(_chains_2)))
        sample_boost = boost
    # get the combinations:
    if len(_chains_1) > len(_chains_2):
        temp_ind = np.indices((len(_chains_2), len(_chains_1)))
    else:
        temp_ind = np.indices((len(_chains_1), len(_chains_2)))
    ind1 = np.concatenate([np.diagonal(temp_ind, offset=i, axis1=1, axis2=2)[0]
                           for i in range(chain_boost)])
    ind2 = np.concatenate([np.diagonal(temp_ind, offset=i, axis1=1, axis2=2)[1]
                           for i in range(chain_boost)])
    chains_combinations = [[_chains_1[i], _chains_2[j]]
                           for i, j in zip(ind1, ind2)]
    # compute the parameter difference samples:
    diff_chain_samples = [pareameter_diff_weighted_samples(samp1,
                          samp2, boost=sample_boost, indexes_1=indexes_1,
                          indexes_2=indexes_2) for samp1, samp2
                          in chains_combinations]
    # create the samples:
    diff_samples = MCSamples(names=diff_param_names, labels=diff_param_labels)
    diff_samples.chains = diff_chain_samples
    diff_samples.makeSingle()
    # get the ranges:
    _ranges = {}
    for name, _min, _max in zip(diff_param_names,
                                np.amin(diff_samples.samples, axis=0),
                                np.amax(diff_samples.samples, axis=0)):
        _ranges[name] = [_min, _max]
    diff_samples.setRanges(_ranges)
    # initialize other things:
    if chain_1.name_tag is not None and chain_2.name_tag is not None:
        diff_samples.name_tag = chain_1.name_tag+'_diff_'+chain_2.name_tag
    # set distinction between base and derived parameters:
    _temp = diff_samples.getParamNames().list()
    _temp_paramnames = chain_1.getParamNames()
    for _nam in diff_samples.getParamNames().parsWithNames(_temp):
        _temp_name = _nam.name.replace('delta_', '')
        _nam.isDerived = _temp_paramnames.parWithName(_temp_name).isDerived
    # update and compute everything:
    diff_samples.updateBaseStatistics()
    diff_samples.deleteFixedParams()
    #
    return diff_samples


def _temp_kde_pdf(x, samples, weights):
    """
    Utility function to compute the KDE
    """
    X = x-samples
    return np.log(np.dot(weights, np.exp(-0.5*(X*X).sum(axis=1))))


def _temp_vec_kde_pdf(x, samples, weights):
    """
    Utility function to compute the KDE
    """
    X = np.subtract(x[np.newaxis, :, :], samples[:, np.newaxis, :])
    _temp = np.dot(weights, np.exp(-0.5*(X*X).sum(axis=2)))
    return np.log(_temp)


def _temp_kde_pdf_tree(x, samples, weights, data_tree, distance):
    # get KDE tree indexes:
    _index = data_tree.query_ball_point(x, r=distance)
    #
    return _temp_kde_pdf(x, samples[_index, :], weights[_index])


def exact_parameter_shift(diff_chain, param_names=None,
                          scale=None, method='brute_force',
                          feedback=1, **kwargs):
    """
    Compute the MCMC estimate of the probability of a parameter shift given
    an input parameter difference chain.
    This function uses a Kernel Density Estimate (KDE) algorithm discussed in
    (`Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_).
    If the difference chain contains :math:`n_{\\rm samples}` this algorithm
    scales as :math:`O(n_{\\rm samples}^2)` and might require long run times.
    For this reason the algorithm is parallelized with the
    joblib library.
    To compute the KDE density estimate several methods are implemented.

    :param diff_chain: :class:`~getdist.mcsamples.MCSamples`
        input parameter difference chain
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param scale: (optional) scale for the KDE smoothing.
        If none is provided the algorithm uses Silverman's
        rule of thumb scaling.
    :param method: (optional) a string containing the indication for the method
        to use in the KDE calculation. This can be very intensive so different
        techniques are provided.

        * method = brute_force is a parallelized brute force method. This
          method scales as :math:`O(n_{\\rm samples}^2)` and can be afforded
          only for small tensions. When suspecting a difference that is
          larger than 95% other methods are better.
        * method = nearest_elimination is a KD Tree based elimination method.
          For large tensions this scales as
          :math:`O(n_{\\rm samples}\\log(n_{\\rm samples}))`
          and in worse case scenarions, with small tensions, this can scale
          as :math:`O(n_{\\rm samples}^2)` but with significant overheads
          with respect to the brute force method.
          When expecting a statistically significant difference in parameters
          this is the recomended algorithm.

        Suggestion is to go with brute force for small problems, nearest
        elimination for big problems with signifcant tensions.
    :param feedback: (optional) print to screen the time taken
        for the calculation.
    :param kwargs: extra options to pass to the KDE algorithm.
        The nearest_elimination algorithm accepts the following optional
        arguments:

        * polish: (default True) after KD Tree elimination switch to brute
          force calculation on the remaining points.
        * stable_cycle: (default 2) number of elimination cycles that show
          no improvement in the result.
        * chunk_size: (default 40) chunk size for elimination cycles.
          For best perfornamces this parameter should be tuned to result
          in the greatest elimination rates.
        * smallest_improvement: (default 1.e-6) minimum percentage improvement
          rate before switching to brute force.
    :return: probability value and error estimate.
    """
    # import specific for this function:
    from tqdm import tqdm
    from scipy.spatial import cKDTree
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # some initial calculations:
    _samples_cov = diff_chain.cov(pars=param_names)
    _num_samples = np.sum(diff_chain.weights)
    _num_elements = len(diff_chain.weights)
    _num_params = len(ind)
    # scale for the kde:
    if scale is None:
        # number of effective samples:
        neff = np.sum(diff_chain.weights)**2/np.sum(diff_chain.weights**2)
        # get the scale:
        scale = Silverman_ROT(_num_params, neff)
        if feedback > 0:
            print('Neff samples:', neff)
    if feedback > 0:
        print('Smoothing scale:', scale)
    # define the Gaussian kernel smoothed pdf:
    _temp = np.diag([np.sqrt(scale) for _i in range(_num_params)])
    _temp = np.dot(np.dot(_temp, _samples_cov), _temp)
    _kernel_cov = np.linalg.inv(_temp)
    # whighten the samples:
    _temp = sqrtm(_kernel_cov)
    _white_samples = diff_chain.samples[:, ind].dot(_temp)
    # compute the KDE:
    t0 = time.time()
    if method == 'brute_force':
        # brute force parallelized method:
        with joblib.Parallel(n_jobs=n_threads) as parallel:
            if feedback > 1:
                _kde_eval_pdf = parallel(joblib.delayed(_temp_kde_pdf)
                                         (samp, _white_samples,
                                          diff_chain.weights)
                                         for samp in tqdm(_white_samples))
            else:
                _kde_eval_pdf = parallel(joblib.delayed(_temp_kde_pdf)
                                         (samp, _white_samples,
                                          diff_chain.weights)
                                         for samp in _white_samples)
        # probability of zero shift:
        _kde_prob_zero = _temp_kde_pdf(np.zeros(_num_params),
                                       _white_samples,
                                       diff_chain.weights)
        # filter for probability calculation:
        _filter = _kde_eval_pdf > _kde_prob_zero
    elif method == 'nearest_elimination':
        # method of nearest elimination:
        polish = kwargs.get('polish', True)
        stable_cycle = kwargs.get('stable_cycle', 2)
        chunk_size = kwargs.get('chunk_size', 40)
        smallest_improvement = kwargs.get('smallest_improvement', 1.e-6)
        # compute probability of zero:
        _kde_prob_zero = _temp_kde_pdf(np.zeros(_num_params),
                                       _white_samples,
                                       diff_chain.weights)
        _kde_prob_zero = np.exp(_kde_prob_zero)
        # build tree:
        if feedback > 0:
            print('Building KD-Tree')
        data_tree = cKDTree(_white_samples,
                            leafsize=40,
                            balanced_tree=True)
        # make sure that the weights are floats:
        _weights = diff_chain.weights.astype(np.float)
        # initial step uses weights to reduce the number of elements:
        _kde_eval_pdf = np.zeros(_num_elements)
        _filter = np.ones(_num_elements, dtype=bool)
        _last_n = 0
        _stable_cycle = 0
        # loop over the neighbours:
        if feedback > 0:
            print('Neighbours elimination')
        for i in range(_num_elements//chunk_size):
            ind_min = chunk_size*i
            ind_max = chunk_size*i+chunk_size
            _dist, _ind = data_tree.query(_white_samples[_filter],
                                          ind_max, n_jobs=-1)
            _kde_eval_pdf[_filter] += np.sum(
                _weights[_ind[:, ind_min:ind_max]]
                * np.exp(-0.5*np.square(_dist[:, ind_min:ind_max])), axis=1)
            _filter[_filter] = _kde_eval_pdf[_filter] < _kde_prob_zero
            _num_filtered = np.sum(_filter)
            if feedback > 1:
                print('nearest_elimination: chunk', i+1)
                print('    surviving elements', _num_filtered,
                      'of', _num_elements)
            # check if calculation has converged:
            _term_check = float(np.abs(_num_filtered-_last_n)) \
                / float(_num_elements) < smallest_improvement
            if _term_check and _num_filtered < _num_elements:
                _stable_cycle += 1
                if _stable_cycle >= stable_cycle:
                    break
            elif not _term_check and _stable_cycle > 0:
                _stable_cycle = 0
            elif _num_filtered == 0:
                break
            else:
                _last_n = _num_filtered
        # clean up memory:
        del(data_tree)
        # brute force the leftovers if wanted:
        if polish:
            if feedback > 0:
                print('nearest_elimination: polishing')
            with joblib.Parallel(n_jobs=n_threads) as parallel:
                if feedback > 1:
                    _kde_eval_pdf[_filter] = parallel(joblib.delayed(_temp_kde_pdf)
                        (samp, _white_samples, diff_chain.weights)
                        for samp in tqdm(_white_samples[_filter]))
                else:
                    _kde_eval_pdf[_filter] = parallel(joblib.delayed(_temp_kde_pdf)
                        (samp, _white_samples, diff_chain.weights)
                        for samp in _white_samples[_filter])
            _filter[_filter] = _kde_eval_pdf[_filter] < np.log(_kde_prob_zero)
            if feedback > 0:
                print('    surviving elements', np.sum(_filter),
                      'of', _num_elements)
    else:
        raise ValueError('Unknown method provided:', method)
    t1 = time.time()
    # clean up:
    gc.collect()
    # feedback:
    if feedback > 0:
        print('KDE method:', method)
        print('Time taken for KDE calculation:', round(t1-t0, 1), '(s)')
    # number of samples:
    if method == 'nearest_elimination':
        _num_filtered = _num_samples-np.sum(diff_chain.weights[_filter])
    else:
        _num_filtered = np.sum(diff_chain.weights[_filter])
    # probability and error estimate:
    _P = float(_num_filtered)/float(_num_samples)
    _low, _upper = clopper_pearson_binomial_trial(_num_filtered,
                                                  _num_samples, alpha=0.32)
    #
    return _P, _low, _upper


def Q_DM(chain_1, chain_2, prior_chain=None, param_names=None,
         cutoff=0.05, prior_factor=1.0):
    """
    Compute the value and degrees of freedom of the quadratic form giving the
    probability of a difference between the means of the two input chains,
    in the Gaussian approximation.

    This is defined as in
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) to be:

    .. math:: Q_{\\rm DM} \\equiv (\\theta_1-\\theta_2)
        (\\mathcal{C}_1+\\mathcal{C}_2
        -\\mathcal{C}_1\\mathcal{C}_\\Pi^{-1}\\mathcal{C}_2
        -\\mathcal{C}_2\\mathcal{C}_\\Pi^{-1}\\mathcal{C}_1)^{-1}
        (\\theta_1-\\theta_2)^T

    where :math:`\\theta_i` is the parameter mean of the i-th posterior,
    :math:`\\mathcal{C}` the posterior covariance and :math:`\\mathcal{C}_\\Pi`
    the prior covariance.
    :math:`Q_{\\rm DM}` is :math:`\\chi^2` distributed with number of degrees
    of freedom equal to the rank of the shift covariance.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_2: :class:`~getdist.mcsamples.MCSamples`
        the second input chain.
    :param prior_chain: (optional) the prior only chain.
        If the prior is not well approximated by a ranged prior and is
        informative it is better to explicitly use a prior only chain.
        If this is not given the algorithm will assume ranged priors
        with the ranges computed from the input chains.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param cutoff: (optional) the algorithms needs to detect prior
        constrained directions (that do not contribute to the test)
        from data constrained directions.
        This is achieved through a Karhunen–Loeve decomposition to avoid issues
        with physical dimensions of parameters and cutoff sets the minimum
        improvement with respect to the prior that is used.
        Default is five percent.
    :param prior_factor: (optional) factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :return: :math:`Q_{\\rm DM}` value and number of degrees of freedom.
        Since :math:`Q_{\\rm DM}` is :math:`\\chi^2` distributed the
        probability to exceed the test can be computed
        using :func:`scipy.stats.chi2.cdf`.
    """
    # initial checks:
    if cutoff < 0.0:
        raise ValueError('The KL cutoff has to be greater than zero.\n',
                         'Input value ', cutoff)
    # initialize param names:
    if param_names is None:
        param_names_1 = chain_1.getParamNames().getRunningNames()
        param_names_2 = chain_2.getParamNames().getRunningNames()
        # get common names:
        param_names = [name for name in param_names_1 if name in param_names_2]
        if len(param_names) == 0:
            raise ValueError('Chains do not have shared parameters.\n',
                             'Parameters for chain_1 ', param_names_1, '\n',
                             'Parameters for chain_2 ', param_names_2, '\n')
    else:
        param_names_1 = chain_1.getParamNames().list()
        param_names_2 = chain_2.getParamNames().list()
        _test_1 = np.all([name in param_names_1 for name in param_names])
        _test_2 = np.all([name in param_names_2 for name in param_names])
        if not _test_1 and not _test_2:
            raise ValueError('Input parameters not in the chains.\n',
                             'Input parameters ', param_names, '\n',
                             'Parameters for chain_1 ', param_names_1, '\n',
                             'Parameters for chain_2 ', param_names_2, '\n')
    # initialize prior covariance:
    if prior_chain is not None:
        # check:
        prior_params = prior_chain.getParamNames().list()
        if not np.all([name in prior_params for name in param_names]):
            raise ValueError('Input parameter is not in the prior chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', prior_params)
        # get the prior covariance:
        C_Pi = prior_chain.cov(pars=param_names)
    else:
        C_Pi1 = get_prior_covariance(chain_1, param_names=param_names)
        C_Pi2 = get_prior_covariance(chain_2, param_names=param_names)
        if not np.allclose(C_Pi1, C_Pi2):
            raise ValueError('The chains have different priors.')
        else:
            C_Pi = C_Pi1
    # scale prior covariance:
    C_Pi = prior_factor*C_Pi
    # get the posterior covariances:
    C_p1, C_p2 = chain_1.cov(pars=param_names), chain_2.cov(pars=param_names)
    # get the means:
    theta_1 = chain_1.getMeans(pars=[chain_1.index[name]
                               for name in param_names])
    theta_2 = chain_2.getMeans(pars=[chain_2.index[name]
                               for name in param_names])
    param_diff = theta_1-theta_2
    # do the calculation of Q:
    C_Pi_inv = QR_inverse(C_Pi)
    temp = np.dot(np.dot(C_p1, C_Pi_inv), C_p2)
    diff_covariance = C_p1 + C_p2 - temp - temp.T
    # take the directions that are best constrained over the prior:
    eig_1, eigv_1 = KL_decomposition(C_p1, C_Pi)
    eig_2, eigv_2 = KL_decomposition(C_p2, C_Pi)
    # get the smallest spectrum, if same use first:
    if np.sum(1./eig_1-1. > cutoff) <= np.sum(1./eig_2-1. > cutoff):
        eig, eigv = eig_1, eigv_1
    else:
        eig, eigv = eig_2, eigv_2
    # get projection matrix:
    proj_matrix = eigv[1./eig-1. > cutoff]
    # get dofs of Q:
    dofs = np.sum(1./eig-1. > cutoff)
    # project parameter difference:
    param_diff = np.dot(proj_matrix, param_diff)
    # project covariance:
    temp_cov = np.dot(np.dot(proj_matrix, diff_covariance), proj_matrix.T)
    # compute Q:
    Q_DM = np.dot(np.dot(param_diff, QR_inverse(temp_cov)), param_diff)
    #
    return Q_DM, dofs


def Q_UDM_KL_components(chain_1, chain_12, param_names=None):
    """
    Function that computes the Karhunen–Loeve (KL) decomposition of the
    covariance of a chain with the covariance of that chain joint with another
    one.
    This function is used for the parameter shift algorithm in
    update form.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples`
        the joint input chain.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :return: the KL eigenvalues, the KL eigenvectors and the parameter names
        that are used.
    """
    # initialize param names:
    if param_names is None:
        param_names_1 = chain_1.getParamNames().getRunningNames()
        param_names_12 = chain_12.getParamNames().getRunningNames()
        # get common names:
        param_names = [name for name in param_names_1
                       if name in param_names_12]
        if len(param_names) == 0:
            raise ValueError('Chains do not have shared parameters.\n',
                             'Parameters for chain_1 ', param_names_1, '\n',
                             'Parameters for chain_12 ', param_names_12, '\n')
    else:
        param_names_1 = chain_1.getParamNames().list()
        param_names_12 = chain_12.getParamNames().list()
        test_1 = np.all([name in param_names_1 for name in param_names])
        test_2 = np.all([name in param_names_12 for name in param_names])
        if not test_1 and not test_2:
            raise ValueError('Input parameters not in the chains.\n',
                             'Input parameters ', param_names, '\n',
                             'Parameters for chain_1 ', param_names_1, '\n',
                             'Parameters for chain_2 ', param_names_12, '\n')
    # get the posterior covariances:
    C_p1, C_p12 = chain_1.cov(pars=param_names), chain_12.cov(pars=param_names)
    # perform the KL decomposition:
    KL_eig, KL_eigv = KL_decomposition(C_p1, C_p12)
    #
    return KL_eig, KL_eigv, param_names


def Q_UDM_get_cutoff(chain_1, chain_2, chain_12,
                     prior_chain=None, param_names=None, prior_factor=1.0):
    """
    Function to estimate the optimal cutoff for the spectrum of parameter
    differences in update form.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_2: :class:`~getdist.mcsamples.MCSamples`
        the second chain that joined with the first one (modulo the prior)
        should give the joint chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples`
        the joint input chain.
    :param prior_chain: :class:`~getdist.mcsamples.MCSamples` (optional)
        If the prior is not well approximated by
        a ranged prior and is informative it is better to explicitly
        use a prior only chain.
        If this is not given the algorithm will assume ranged priors with the
        ranges computed from the input chain.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param prior_factor: (optional) factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :return: the optimal KL cutoff, KL eigenvalues, KL eigenvectors and the
        parameter names that are used.
    """
    # get all shared parameters:
    if param_names is None:
        param_names_1 = chain_1.getParamNames().getRunningNames()
        param_names_2 = chain_2.getParamNames().getRunningNames()
        # get common names:
        param_names = [name for name in param_names_1
                       if name in param_names_2]
        if len(param_names) == 0:
            raise ValueError('Chains do not have shared parameters.\n',
                             'Parameters for chain_1 ', param_names_1, '\n',
                             'Parameters for chain_2 ', param_names_2, '\n')
    # get the KL decomposition:
    KL_eig, KL_eigv, param_names = Q_UDM_KL_components(chain_1,
                                                       chain_12,
                                                       param_names=param_names)
    # get the cutoff that matches the dofs of Q_DMAP:
    N_1 = get_Neff(chain_1,
                   prior_chain=prior_chain,
                   param_names=param_names,
                   prior_factor=prior_factor)
    N_2 = get_Neff(chain_2,
                   prior_chain=prior_chain,
                   param_names=param_names,
                   prior_factor=prior_factor)
    N_12 = get_Neff(chain_12,
                    prior_chain=prior_chain,
                    param_names=param_names,
                    prior_factor=prior_factor)
    target_dofs = round(N_1 + N_2 - N_12)
    # compute the cutoff:

    def _helper(_c):
        return np.sum(KL_eig[KL_eig > 1.] > _c)-target_dofs
    # define the extrema:
    _a = 1.0
    _b = np.amax(KL_eig)
    # check bracketing:
    if _helper(_a)*_helper(_b) > 0:
        raise ValueError('Cannot find optimal cutoff.\n',
                         'This might be a problem with the prior.\n',
                         'You may try providing a prior chain.\n',
                         'KL spectrum:', KL_eig,
                         'Target dofs:', target_dofs)
    else:
        KL_cutoff = scipy.optimize.bisect(_helper, _a, _b)
    #
    return KL_cutoff, KL_eig, KL_eigv, param_names


def Q_UDM(chain_1, chain_12, cutoff=1.05, param_names=None):
    """
    Compute the value and degrees of freedom of the quadratic form giving the
    probability of a difference between the means of the two input chains,
    in update form with the Gaussian approximation.

    This is defined as in
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) to be:

    .. math:: Q_{\\rm UDM} \\equiv (\\theta_1-\\theta_{12})
        (\\mathcal{C}_1-\\mathcal{C}_{12})^{-1}
        (\\theta_1-\\theta_{12})^T

    where :math:`\\theta_1` is the parameter mean of the first posterior,
    :math:`\\theta_{12}` is the parameter mean of the joint posterior,
    :math:`\\mathcal{C}` the posterior covariance and :math:`\\mathcal{C}_\\Pi`
    the prior covariance.
    :math:`Q_{\\rm UDM}` is :math:`\\chi^2` distributed with number of degrees
    of freedom equal to the rank of the shift covariance.

    In case of uninformative priors the statistical significance of
    :math:`Q_{\\rm UDM}` is the same as the one reported by
    :math:`Q_{\\rm DM}` but offers likely mitigation against non-Gaussianities
    of the posterior distribution.
    In the case where both chains are Gaussian :math:`Q_{\\rm UDM}` is
    symmetric if the first input chain is swapped :math:`1\\leftrightarrow 2`.
    If the input distributions are not Gaussian it is better to use the most
    constraining chain as the base for the parameter update.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples`
        the joint input chain.
    :param cutoff: (optional) the algorithms needs to detect prior
        constrained directions (that do not contribute to the test)
        from data constrained directions.
        This is achieved through a Karhunen–Loeve decomposition to avoid issues
        with physical dimensions of parameters and cutoff sets the minimum
        improvement with respect to the prior that is used.
        Default is five percent.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :return: :math:`Q_{\\rm UDM}` value and number of degrees of freedom.
        Since :math:`Q_{\\rm UDM}` is :math:`\\chi^2` distributed the
        probability to exceed the test can be computed
        using :func:`scipy.stats.chi2.cdf`.
    """
    # get the cutoff and perform the KL decomposition:
    _temp = Q_UDM_KL_components(chain_1, chain_12, param_names=param_names)
    KL_eig, KL_eigv, param_names = _temp
    # get the parameter means:
    theta_1 = chain_1.getMeans(pars=[chain_1.index[name]
                               for name in param_names])
    theta_12 = chain_12.getMeans(pars=[chain_12.index[name]
                                 for name in param_names])
    shift = theta_1 - theta_12
    # do the Q_UDM calculation:
    _filter = np.logical_and(KL_eig > cutoff, KL_eig > 1.)
    Q_UDM = np.sum(np.dot(KL_eigv.T, shift)[_filter]**2./(KL_eig[_filter]-1.))
    dofs = np.sum(_filter)
    #
    return Q_UDM, dofs


###############################################################################
# Likelihood based estimators:
###############################################################################


def get_MAP_loglike(chain, feedback=True):
    """
    Utility function to obtain the data part of the maximum posterior for
    a given chain.
    The best possibility is that a separate file with the posterior
    explicit MAP is given. If this is not the case then the function will try
    to get the likelihood at MAP from the samples. This possibility is far more
    noisy in general.

    :param chain: :class:`~getdist.mcsamples.MCSamples`
        the input chain.
    :param feedback: logical flag to set whether the function should print
        a warning every time the explicit MAP file is not found.
        By default this is true.
    :return: the data log likelihood at maximum posterior.
    """
    # we first try to get the best fit from explicit maximization:
    try:
        # get the best fit from the explicit MAP:
        best_fit = chain.getBestFit(max_posterior=True)
        if len(best_fit.chiSquareds) == 0:
            _best_fit_data_like = best_fit.logLike
            if 'prior' in best_fit.getParamDict().keys():
                _best_fit_data_like -= best_fit.getParamDict()['prior']
        else:
            # get the total data likelihood:
            _best_fit_data_like = 0.0
            for _dat in best_fit.chiSquareds:
                _best_fit_data_like += _dat[1].chisq
    except Exception as ex:
        # we use the best fit from the chains.
        # This is noisy so we print a warning:
        if feedback:
            print(ex)
            print('WARNING: using MAP from samples. This can be noisy.')
        _best_fit_data_like = 0.0
        # get chi2 list:
        chi_list = [name for name in chain.getLikeStats().list()
                    if 'chi2_' in name]
        # assume that we have chi2_data and the chi_2 prior:
        if 'chi2_prior' in chi_list:
            chi_list = chi_list[:chi_list.index('chi2_prior')]
        # if empty we have to guess:
        if len(chi_list) == 0:
            _best_fit_data_like = chain.getLikeStats().logLike_sample
        else:
            for name in chi_list:
                _best_fit_data_like += \
                    chain.getLikeStats().parWithName(name).bestfit_sample
    # normalize:
    _best_fit_data_like = -0.5*_best_fit_data_like
    #
    return _best_fit_data_like


def Q_MAP(chain, num_data, prior_chain=None,
          normalization_factor=0.0, prior_factor=1.0, feedback=True):
    """
    Compute the value and degrees of freedom of the quadratic form giving
    the goodness of fit measure at maximum posterior (MAP), in
    Gaussian approximation.

    This is defined as in
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) to be:

    .. math:: Q_{\\rm MAP} \\equiv -2\\ln \\mathcal{L}(\\theta_{\\rm MAP})

    where :math:`\\mathcal{L}(\\theta_{\\rm MAP})` is the data likelihood
    evaluated at MAP.
    In Gaussian approximation this is distributed as:

    .. math:: Q_{\\rm MAP} \\sim \\chi^2(d-N_{\\rm eff})

    where :math:`d` is the number of data points and :math:`N_{\\rm eff}`
    is the number of effective parameters, as computed by the function
    :func:`getdist.chains_tension.get_Neff`.

    :param chain: :class:`~getdist.mcsamples.MCSamples`
        the input chain.
    :param num_data: number of data points.
    :param prior_chain: (optional) the prior chain.
        If the prior is not well approximated by
        a ranged prior and is informative it is better to explicitly
        use a prior only chain.
        If this is not given the algorithm will assume ranged priors with the
        ranges computed from the input chain.
    :param normalization_factor: (optional) likelihood normalization factor.
        This should make the likelihood a chi square.
    :param prior_factor: (optional) factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :param feedback: logical flag to set whether the function should print
        a warning every time the explicit MAP file is not found.
        By default this is true.
    :return: :math:`Q_{\\rm MAP}` value and number of degrees of freedom.
        Since :math:`Q_{\\rm MAP}` is :math:`\\chi^2` distributed the
        probability to exceed the test can be computed
        using :func:`scipy.stats.chi2.cdf`.
    """
    # get the best fit:
    best_fit_data_like = get_MAP_loglike(chain, feedback=feedback)
    # get the number of effective parameters:
    Neff = get_Neff(chain, prior_chain=prior_chain, prior_factor=prior_factor)
    # compute Q_MAP:
    Q_MAP = -2.*best_fit_data_like + normalization_factor
    # compute the number of degrees of freedom:
    dofs = float(num_data) - Neff
    #
    return Q_MAP, dofs


def Q_DMAP(chain_1, chain_2, chain_12, prior_chain=None,
           param_names=None, prior_factor=1.0, feedback=True):
    """
    Compute the value and degrees of freedom of the quadratic form giving
    the goodness of fit loss measure, in Gaussian approximation.

    This is defined as in
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) to be:

    .. math:: Q_{\\rm DMAP} \\equiv Q_{\\rm MAP}^{12} -Q_{\\rm MAP}^{1}
        -Q_{\\rm MAP}^{2}

    where :math:`Q_{\\rm MAP}^{12}` is the joint likelihood at maximum
    posterior (MAP) and :math:`Q_{\\rm MAP}^{i}` is the likelihood at MAP
    for the two single data sets.
    In Gaussian approximation this is distributed as:

    .. math:: Q_{\\rm DMAP} \\sim \\chi^2(N_{\\rm eff}^1 + N_{\\rm eff}^2 -
        N_{\\rm eff}^{12})

    where :math:`N_{\\rm eff}` is the number of effective parameters,
    as computed by the function :func:`getdist.chains_tension.get_Neff`
    for the joint and the two single data sets.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_2: :class:`~getdist.mcsamples.MCSamples`
        the second input chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples`
        the joint input chain.
    :param prior_chain: (optional) the prior chain.
        If the prior is not well approximated by
        a ranged prior and is informative it is better to explicitly
        use a prior only chain.
        If this is not given the algorithm will assume ranged priors with the
        ranges computed from the input chain.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param prior_factor: (optional) factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :param feedback: logical flag to set whether the function should print
        a warning every time the explicit MAP file is not found.
        By default this is true.
    :return: :math:`Q_{\\rm DMAP}` value and number of degrees of freedom.
        Since :math:`Q_{\\rm DMAP}` is :math:`\\chi^2` distributed the
        probability to exceed the test can be computed
        using :func:`scipy.stats.chi2.cdf`.
    """
    # check that all chains have the same running parameters:

    # get the data best fit for the chains:
    best_fit_data_like_1 = get_MAP_loglike(chain_1, feedback=feedback)
    best_fit_data_like_2 = get_MAP_loglike(chain_2, feedback=feedback)
    best_fit_data_like_12 = get_MAP_loglike(chain_12, feedback=feedback)
    # get the number of effective parameters:
    Neff_1 = get_Neff(chain_1,
                      prior_chain=prior_chain,
                      param_names=param_names,
                      prior_factor=prior_factor)
    Neff_2 = get_Neff(chain_2,
                      prior_chain=prior_chain,
                      param_names=param_names,
                      prior_factor=prior_factor)
    Neff_12 = get_Neff(chain_12,
                       prior_chain=prior_chain,
                       param_names=param_names,
                       prior_factor=prior_factor)
    # compute delta Neff:
    dofs = Neff_1 + Neff_2 - Neff_12
    # compute Q_DMAP:
    Q_DMAP = -2.*best_fit_data_like_12 \
        + 2.*best_fit_data_like_1 \
        + 2.*best_fit_data_like_2
    #
    return Q_DMAP, dofs
