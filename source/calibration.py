import numpy as np
from scipy.stats import chi2
from lifelines import KaplanMeierFitter


def d_calibration(event, predictions, bins=10):
    """
    Calculate the D-Calibration metric for survival predictions.

    Args:
        event (array-like): Array of event indicators (0 for censored, 1 for uncensored).
        predictions (array-like): Array of predicted survival probabilities.
        bins (int): Number of bins to divide the predictions into.

    Returns:
        dict: Dictionary containing the D-Calibration metric and related statistics.

    """
    event_indicators = np.array(event).astype(bool)
    
    # Calculate the bin index for each prediction
    bin_index = np.minimum(np.floor(predictions*bins), bins-1).astype(int)
    censored_bin_indexes = bin_index[~event_indicators]
    uncensored_bin_indexes = bin_index[event_indicators]
    
    # Calculate the contribution of censored predictions to each bin
    censored_predictions = predictions[~event_indicators]
    censored_contribution = 1 - (censored_bin_indexes/bins) * (1/censored_predictions)
    censored_following_contribution = 1 / (bins*censored_predictions)
    
    # Create a contribution pattern matrix for following contributions
    contribution_pattern = np.tril(np.ones([bins, bins]), k=-1).astype(bool)
    
    # Calculate the following contributions for censored predictions
    following_contributions = np.matmul(censored_following_contribution,
                                         contribution_pattern[censored_bin_indexes])
    
    # Calculate the single contributions for censored predictions
    single_contributions = np.matmul(censored_contribution,
                                         np.eye(bins)[censored_bin_indexes])

    # Calculate the contributions for uncensored predictions
    uncensored_contributions = np.sum(np.eye(bins)[uncensored_bin_indexes], axis=0)
    
    # Calculate the bin count by summing the contributions
    bin_count = (single_contributions + following_contributions + uncensored_contributions)
    
    # Calculate the chi-square statistic for D-Calibration
    chi2_statistic = np.sum(
        np.square(bin_count - len(predictions)/bins) / (len(predictions)/bins)
    )
    
    # Calculate the p-value using the chi-square distribution
    p_value = 1 - chi2.cdf(chi2_statistic, bins-1)
    
    return dict(
        p_value=p_value,
        bin_proportions=bin_count/len(predictions),
        censored_contributions=(single_contributions + following_contributions)/len(predictions),
        uncensored_contributions=uncensored_contributions / len(predictions),
        d_calibration=chi2_statistic,
    )


def xcal(points, is_alive, nbins=20, differentiable=False, gamma=1.0):
    new_is_alive = is_alive.copy()
    new_is_alive[points > 1.-1e-4] = 0
    
    points = points.reshape((-1,1))
    bin_width = 1.0 / nbins
    bin_indices = np.arange(nbins).reshape(1,-1).astype(float)
    bin_a = bin_indices * bin_width
    noise = 1e-6/nbins*np.random.rand(*bin_indices.shape)
    if not differentiable:
        noise = noise * 0.
    cum_noise = np.cumsum(noise)
    bin_width = np.array([bin_width]*nbins) + cum_noise
    bin_b = bin_a + bin_width
    
    bin_b_max = bin_b[:,-1]
    bin_b = bin_b/bin_b_max
    bin_a[:,1:] = bin_b[:,:-1]
    bin_width = bin_b - bin_a
    
    points_cens = points[new_is_alive==1]
    upper_diff_for_soft_cens = bin_b - points_cens
    bin_b[:,-1] = 2
    bin_a[:,0] = -1
    
    lower_diff_cens = points_cens - bin_a
    upper_diff_cens = bin_b - points_cens
    diff_product_cens = lower_diff_cens * upper_diff_cens
    
    def sigmoid(z):
        return 1 / (1+np.exp(-z))
    if differentiable:
        bin_index_ohe = sigmoid(gamma*diff_product_cens)
        exact_bins_next = sigmoid(-gamma*lower_diff_cens)
    else:
        bin_index_ohe = (lower_diff_cens >= 0).astype(float) * (upper_diff_cens > 0).astype(float)
        exact_bins_next = (lower_diff_cens <= 0).astype(float)
    
    EPS = 1e-13
    right_censored_interval_size = 1- points_cens + EPS
    upper_diff_within_bin = (upper_diff_for_soft_cens * bin_index_ohe)
    
    full_bin_assigned_weight = (exact_bins_next*bin_width.reshape(1,-1)/right_censored_interval_size.reshape(-1,1)).sum(0)
    partial_bin_assigned_weight = (upper_diff_within_bin/right_censored_interval_size).sum(0)
    
    points_uncens = points[new_is_alive==0]
    lower_diff = points_uncens - bin_a
    upper_diff = bin_b - points_uncens
    diff_product = lower_diff * upper_diff
    
    if differentiable:
        soft_membership = sigmoid(gamma*diff_product)
        fraction_in_bins = soft_membership.sum(0)
    else:
        exact_membership = (lower_diff >= 0).astype(float) * (upper_diff > 0).astype(float)
        fraction_in_bins = exact_membership.sum(0)
        
    frac_in_bins = (fraction_in_bins + full_bin_assigned_weight + partial_bin_assigned_weight) / points.shape[0]
    return np.square(frac_in_bins - bin_width).sum()


def calibration_curve(out, e, t, a, group, eval_time, typ='KM', ret_bins=False,
                      strat='quantile', n_bins=10):
    
    if typ == 'IPCW':
        return _calibration_curve_ipcw(out, e, t, a, group, eval_time,
                                       ret_bins=ret_bins, strat=strat,
                                       n_bins=n_bins)
    else:
        return _calibration_curve_km(out, e, t, a, group, eval_time,
                                       ret_bins=ret_bins, strat=strat,
                                       n_bins=n_bins)
    
    
def _calibration_curve_km(out, e, t, a, group, eval_time, ret_bins=True,
                          strat='uniform', n_bins=10):
    """
    Calculates the calibration curve for a survival model using Kaplan-Meier estimation.
    
    :param out: The variable `out` represents the predicted outcome or score for each observation
    :param e: The parameter `e` represents the event indicator variable. It is a binary variable that
    indicates whether an event (such as death or failure) has occurred for each observation
    :param t: The parameter `t` represents the time variable, which is the time at which an event occurs
    or is censored. It is a numpy array or pandas Series containing the time values
    :param a: The parameter "a" represents a grouping variable. It is used to group the data based on a
    specific variable
    :param group: The "group" parameter is used to specify a particular group within the data. It is
    used to filter the data based on a specific group
    :param eval_time: The `eval_time` parameter is the time at which the prediction is evaluated. It is
    used to calculate the predicted survival probability at that specific time point
    :param ret_bins: The parameter `ret_bins` is a boolean flag that determines whether the function
    should return the bin totals or not. If `ret_bins` is set to `True`, the function will return the
    bin totals along with the other outputs. If `ret_bins` is set to `False`, the function, defaults to
    True (optional)
    :param strat: The parameter "strat" in the function `_calibration_curve_km` determines the strategy
    used to create the bins for calibration. It can take two possible values:, defaults to uniform
    (optional)
    :param n_bins: The parameter `n_bins` represents the number of bins to divide the `out` values into
    for calibration. It determines the granularity of the calibration curve, defaults to 10 (optional)
    :return: different values depending on the value of the `ret_bins` parameter.
    """
    out_ = out.copy()
    if group is not None:
        mask = a == group
        e = e[mask]
        t = t[mask]
        out = out[mask]
    
    y = t > eval_time
    
    if strat == 'quantile':
        quantiles = [(1. / n_bins) * i for i in range(n_bins+1)]
        outbins = np.quantile(out, quantiles)
    
    if strat == 'uniform':
        binlen = (out.max() - out.min()) / n_bins
        outbins = [out.min() + i * binlen for i in range(n_bins+1)]
    
    prob_true = []
    prob_pred = []
    
    ece = 0
    
    bin_total = [0 for _ in range(n_bins)]
    
    for n_bin in range(n_bins):
        binmin = outbins[n_bin]
        binmax = outbins[n_bin+1]
        
        scorebin = (out >= binmin) & (out <= binmax)
        
        weight = float(scorebin.sum()) / len(out)
        
        bin_total[n_bin] += scorebin.sum()
        
        out_ = out[scorebin]
        y_ = y[scorebin]
        
        if out_.size == 0:
            out_mean = 0
        else:
            out_mean = out_.mean()
        
        if t[scorebin].size == 0 or e[scorebin].size == 0:
            pred = 0
        else:
            pred = KaplanMeierFitter().fit(t[scorebin], e[scorebin]).predict(eval_time)

        prob_true.append(pred)
        prob_pred.append(out_.mean())
        
        gap = abs(prob_pred[-1] - prob_true[-1])
        
        ece += weight * gap
        
    if ret_bins:
        return prob_true, prob_pred, bin_total, ece
    else:
        return prob_true, prob_pred, ece