import torch
import numpy as np

def traj_metrics(frame_embs, highfreq_cut=0.25):
    """
    frame_embs: [T+1, D] (includes CLS at 0). Returns dict of scalar metrics.
    """
    Z = frame_embs[1:]            # drop CLS -> [T, D]
    Z = torch.as_tensor(Z)
    T, D = Z.shape

    # 1st & 2nd differences
    d1 = Z[1:] - Z[:-1]           # [T-1, D]
    d2 = d1[1:] - d1[:-1]         # [T-2, D]
    eps = 1e-8

    speed = torch.norm(d1, dim=-1)              # [T-1]
    acc   = torch.norm(d2, dim=-1)              # [T-2]

    # curvature (angle between successive steps)
    num = (d1[1:] * d1[:-1]).sum(dim=-1)        # [T-2]
    den = (torch.norm(d1[1:], dim=-1)*torch.norm(d1[:-1], dim=-1) + eps)
    cosang = torch.clamp(num/den, -1.0, 1.0)
    curv = torch.acos(cosang)                   # [T-2]

    arc_len = speed.sum()
    straightness = torch.norm(Z[-1]-Z[0]) / (arc_len + eps)

    # frequency stats on time series of each dim (use numpy FFT for convenience)
    Zc = Z.detach().cpu().numpy()
    # remove mean per dim (DC)
    Zc = Zc - Zc.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(Zc, axis=0)               # [F, D]
    power = (fft.real**2 + fft.imag**2)         # [F, D]
    freq = np.fft.rfftfreq(Zc.shape[0], d=1.0)  # normalized freq [0..0.5]
    # high-frequency mask
    hf_mask = freq >= highfreq_cut
    hf_power = power[hf_mask].sum()
    tot_power = power.sum() + 1e-8
    hf_ratio = float(hf_power / tot_power)
    # spectral centroid
    spec_centroid = float((power * freq[:, None]).sum() / tot_power)

    # PCA dimension for 95% var
    # (use SVD on (T x D))
    U,S,Vh = np.linalg.svd(Zc, full_matrices=False)
    var = (S**2) / (S**2).sum()
    pca95 = int(np.searchsorted(np.cumsum(var), 0.95) + 1)

    return {
        "speed_mean": float(speed.mean()),
        "speed_std":  float(speed.std(unbiased=False)),
        "acc_energy": float((acc**2).mean().sqrt()),   # RMS accel (jerkiness)
        "curv_median": float(curv.median()),
        "arc_length": float(arc_len),
        "straightness": float(straightness),
        "hf_ratio": hf_ratio,
        "spectral_centroid": spec_centroid,
        "pca95_dims": pca95,
    }