#!/usr/bin/env python
import numpy as np

def _per_patient_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    pe = (y_pred[mask] - y_true[mask]) / y_true[mask] * 100.0
    mdpe = np.nanmedian(pe)
    mdape = np.nanmedian(np.abs(pe))
    wobble = np.nanmedian(np.abs(pe - mdpe))
    return mdpe, mdape, wobble

def varvel_population(y_true_list, y_pred_list):
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true_list and y_pred_list must have the same length")
    mdpe_list = []
    mdape_list = []
    wobble_list = []
    for yt, yp in zip(y_true_list, y_pred_list):
        mdpe, mdape, wobble = _per_patient_metrics(yt, yp)
        if not np.isnan(mdpe):
            mdpe_list.append(mdpe)
        if not np.isnan(mdape):
            mdape_list.append(mdape)
        if not np.isnan(wobble):
            wobble_list.append(wobble)
    mdpe_arr = np.asarray(mdpe_list, dtype=np.float64)
    mdape_arr = np.asarray(mdape_list, dtype=np.float64)
    wobble_arr = np.asarray(wobble_list, dtype=np.float64)
    if mdpe_arr.size == 0 or mdape_arr.size == 0 or wobble_arr.size == 0:
        return {
            "MDPE": {"median": np.nan, "IQR": (np.nan, np.nan)},
            "MDAPE": {"median": np.nan, "IQR": (np.nan, np.nan)},
            "Wobble": {"median": np.nan, "IQR": (np.nan, np.nan)},
        }
    mdpe_med = float(np.nanmedian(mdpe_arr))
    mdpe_q1 = float(np.nanpercentile(mdpe_arr, 25))
    mdpe_q3 = float(np.nanpercentile(mdpe_arr, 75))
    mdape_med = float(np.nanmedian(mdape_arr))
    mdape_q1 = float(np.nanpercentile(mdape_arr, 25))
    mdape_q3 = float(np.nanpercentile(mdape_arr, 75))
    wobble_med = float(np.nanmedian(wobble_arr))
    wobble_q1 = float(np.nanpercentile(wobble_arr, 25))
    wobble_q3 = float(np.nanpercentile(wobble_arr, 75))
    return {
        "MDPE": {"median": mdpe_med, "IQR": (mdpe_q1, mdpe_q3)},
        "MDAPE": {"median": mdape_med, "IQR": (mdape_q1, mdape_q3)},
        "Wobble": {"median": wobble_med, "IQR": (wobble_q1, wobble_q3)},
    }
