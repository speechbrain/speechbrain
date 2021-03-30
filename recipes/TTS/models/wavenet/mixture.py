# coding: utf-8
# Code is adapted from:
# https://github.com/pclucas14/pixel-cnn-pp
# https://github.com/openai/pixel-cnn

from __future__ import with_statement, print_function, absolute_import

import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def discretized_mix_logistic_loss(y_hat, y, num_classes=256,
                                  log_scale_min=-7.0, reduce=True):
    """Discretized mixture of logistic distributions loss

    Note that it is assumed that input is scaled to [-1, 1].

    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.

    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0
    nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(y, log_scale_min=-7.0,
                                         clamp_log_scale=False):
    """
    Sample from discretized mixture of logistic distributions

    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value

    Returns:
        Tensor: sample in range of [-1, 1].
    """
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = to_one_hot(argmax, nr_mix)
    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    if clamp_log_scale:
        log_scales = torch.clamp(log_scales, min=log_scale_min)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

    return x


# we can easily define discretized version of the gaussian loss, however,
# use continuous version as same as the https://clarinet-demo.github.io/
def mix_gaussian_loss(y_hat, y, log_scale_min=-7.0, reduce=True):
    """Mixture of continuous gaussian distributions loss

    Note that it is assumed that input is scaled to [-1, 1].

    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.
    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    C = y_hat.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y_hat.size(1) % 3 == 0
        nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters.
    if C == 2:
        # special case for C == 2, just for compatibility
        logit_probs = None
        means = y_hat[:, :, 0:1]
        log_scales = torch.clamp(y_hat[:, :, 1:2], min=log_scale_min)
    else:
        #  (B, T, num_mixtures) x 3
        logit_probs = y_hat[:, :, :nr_mix]
        means = y_hat[:, :, nr_mix:2 * nr_mix]
        log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    dist = Normal(loc=0., scale=torch.exp(log_scales))
    # do we need to add a trick to avoid log(0)?
    log_probs = dist.log_prob(centered_y)

    if nr_mix > 1:
        log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        if nr_mix == 1:
            return -torch.sum(log_probs)
        else:
            return -torch.sum(log_sum_exp(log_probs))
    else:
        if nr_mix == 1:
            return -log_probs
        else:
            return -log_sum_exp(log_probs).unsqueeze(-1)


def sample_from_mix_gaussian(y, log_scale_min=-7.0):
    """
    Sample from (discretized) mixture of gaussian distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    C = y.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y.size(1) % 3 == 0
        nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)

    if C == 2:
        logit_probs = None
    else:
        logit_probs = y[:, :, :nr_mix]

    if nr_mix > 1:
        # sample mixture indicator from softmax
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_probs.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = to_one_hot(argmax, nr_mix)

        # Select means and log scales
        means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
        log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    else:
        if C == 2:
            means, log_scales = y[:, :, 0], y[:, :, 1]
        elif C == 3:
            means, log_scales = y[:, :, 1], y[:, :, 2]
        else:
            assert False, "shouldn't happen"

    scales = torch.exp(log_scales)
    dist = Normal(loc=means, scale=scales)
    x = dist.sample()

    x = torch.clamp(x, min=-1.0, max=1.0)
    return x
