"""
Multi-Microphone Signal Processing

Authors
-------
 * Francois Grondin 2020
 * William Aris 2020
"""

import torch
import speechbrain.processing.decomposition as dc


def gev(xs, rss, rns):
    """
    xs : (batch, time_step, n_fft, 2, n_channels)
    rss : (batch, time_step, n_fft, 2, n_mics + n_pairs)
    rns : (batch, time_step, n_fft, 2, n_mics + n_pairs)

    ys : (batch, time_step, n_fft, 2, 1)
    """

    # Extracting data
    n_channels = xs.shape[4]
    p = rss.shape[4]

    # Computing the eigenvectors
    rss_rns = torch.cat((rss, rns), dim=4)

    rss_rns_val, rss_rns_idx = torch.unique(rss_rns, return_inverse=True, dim=1)

    rss = rss_rns_val[..., range(0, p)]
    rns = rss_rns_val[..., range(p, 2 * p)]

    rns = dc.pos_def(rns)
    vs, _ = dc.gevd(rss, rns)

    # Beamforming
    f_re = vs[..., (n_channels - 1), 0]
    f_im = vs[..., (n_channels - 1), 1]

    ws_re = f_re[:, rss_rns_idx]
    ws_im = -1.0 * f_im[:, rss_rns_idx]

    xs_re = xs[..., 0, :]
    xs_im = xs[..., 1, :]

    ys_re = torch.sum((ws_re * xs_re - ws_im * xs_im), dim=3, keepdim=True)
    ys_im = torch.sum((ws_re * xs_im + ws_im * xs_re), dim=3, keepdim=True)

    # Assembling the output
    ys = torch.stack((ys_re, ys_im), 3)

    return ys
