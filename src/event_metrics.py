"""
Event Detection and Timing Skill Metrics

This module implements flood event detection skill (POD, FAR, CSI) and 
peak timing metrics for evaluating probabilistic streamflow forecasts.

Based on:
- Event extraction via threshold exceedance
- 1-to-1 event matching by peak timing
- Detection metrics: Probability of Detection (POD), False Alarm Ratio (FAR), 
  Critical Success Index (CSI)
- Timing metrics: Peak timing bias, MAE, RMSE
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def _find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find contiguous runs of True values in boolean array.
    
    Parameters:
    -----------
    mask : np.ndarray
        Boolean mask array
    
    Returns:
    --------
    list of (start_idx, end_idx) tuples (inclusive)
    """
    mask = mask.astype(bool)
    if mask.size == 0:
        return []
    
    # Find changes in the mask
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0]
    
    # Handle edge cases
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_close_runs(runs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    """
    Merge runs separated by gaps <= max_gap timesteps.
    
    Parameters:
    -----------
    runs : list of (start, end) tuples
        Must be sorted by start time
    max_gap : int
        Maximum gap (in timesteps) to merge across
    
    Returns:
    --------
    list of merged (start, end) tuples
    """
    if not runs:
        return []
    
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        gap = s - prev_e - 1
        if gap <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    
    return [(s, e) for s, e in merged]


def extract_events(x: np.ndarray, 
                  threshold: float, 
                  min_duration: int = 1, 
                  merge_gap: int = 0) -> List[Dict]:
    """
    Extract threshold exceedance events from time series.
    
    Parameters:
    -----------
    x : np.ndarray
        1D time series array
    threshold : float
        Exceedance threshold (e.g., Q95 of training data)
    min_duration : int, optional
        Minimum event duration in timesteps (default: 1)
    merge_gap : int, optional
        Maximum gap to merge across (default: 0)
    
    Returns:
    --------
    list of event dictionaries with keys:
        - start: start index
        - end: end index (inclusive)
        - peak_idx: index of peak value
        - peak_val: peak value
        - duration: event duration in timesteps
    """
    runs = _find_runs(x >= threshold)
    runs = _merge_close_runs(runs, max_gap=merge_gap)
    
    events = []
    for s, e in runs:
        duration = e - s + 1
        if duration < min_duration:
            continue
        
        seg = x[s:e+1]
        peak_rel = int(np.argmax(seg))
        peak_idx = s + peak_rel
        
        events.append({
            "start": s,
            "end": e,
            "peak_idx": peak_idx,
            "peak_val": float(x[peak_idx]),
            "duration": duration
        })
    
    return events


def match_events_by_peak(obs_events: List[Dict], 
                        pred_events: List[Dict], 
                        tau: int) -> List[Tuple[int, int, int]]:
    """
    Match observed and predicted events using 1-to-1 greedy matching.
    
    Events match if their peak times are within ±tau timesteps.
    Greedy matching prioritizes pairs with smallest timing difference.
    
    Parameters:
    -----------
    obs_events : list of dict
        Observed events from extract_events()
    pred_events : list of dict
        Predicted events from extract_events()
    tau : int
        Maximum timing tolerance (in timesteps)
    
    Returns:
    --------
    list of (obs_idx, pred_idx, dt) tuples where:
        - obs_idx: index in obs_events
        - pred_idx: index in pred_events  
        - dt: timing difference (pred_peak - obs_peak)
    """
    if not obs_events or not pred_events:
        return []
    
    # Build all candidate pairs within timing tolerance
    candidates = []
    for i, oe in enumerate(obs_events):
        for j, pe in enumerate(pred_events):
            dt = pe["peak_idx"] - oe["peak_idx"]
            if abs(dt) <= tau:
                candidates.append((abs(dt), i, j, dt))
    
    # Greedy matching: smallest |dt| first
    candidates.sort(key=lambda t: t[0])
    
    matched_obs = set()
    matched_pred = set()
    matches = []
    
    for _, i, j, dt in candidates:
        if i in matched_obs or j in matched_pred:
            continue
        matched_obs.add(i)
        matched_pred.add(j)
        matches.append((i, j, dt))
    
    return matches


def detection_and_timing_metrics(q_obs: np.ndarray,
                                q_pred: np.ndarray,
                                threshold: float = None,
                                q: float = 0.95,
                                min_duration: int = 1,
                                merge_gap: int = 0,
                                tau: int = 1) -> Dict:
    """
    Compute event detection skill (POD/FAR/CSI) and peak timing metrics.
    
    Parameters:
    -----------
    q_obs : np.ndarray
        Observed streamflow time series (1D)
    q_pred : np.ndarray
        Predicted streamflow time series (1D)
    threshold : float, optional
        Absolute threshold for event detection. If None, computed as quantile q of q_obs
    q : float, optional
        Quantile level for threshold (default: 0.95 = 95th percentile)
    min_duration : int, optional
        Minimum event duration in timesteps (default: 1)
    merge_gap : int, optional
        Maximum gap to merge events across (default: 0)
    tau : int, optional
        Peak timing tolerance in timesteps (default: 1)
    
    Returns:
    --------
    dict with keys:
        Detection metrics:
        - threshold: threshold value used
        - n_obs_events: number of observed events
        - n_pred_events: number of predicted events
        - H: hits (matched events)
        - M: misses (unmatched observed events)
        - F: false alarms (unmatched predicted events)
        - POD: Probability of Detection = H / (H + M)
        - FAR: False Alarm Ratio = F / (H + F)
        - CSI: Critical Success Index = H / (H + M + F)
        
        Timing metrics (for matched events):
        - peak_dt_bias: mean timing error (days)
        - peak_dt_mae: mean absolute timing error (days)
        - peak_dt_rmse: root mean square timing error (days)
        - peak_dt_all: array of all timing errors
        
        Additional info:
        - matches: list of matched (obs_idx, pred_idx, dt) tuples
        - obs_events: list of observed event dicts
        - pred_events: list of predicted event dicts
    """
    q_obs = np.asarray(q_obs, dtype=float).flatten()
    q_pred = np.asarray(q_pred, dtype=float).flatten()
    
    if q_obs.shape != q_pred.shape:
        raise ValueError(f"Shape mismatch: q_obs {q_obs.shape} != q_pred {q_pred.shape}")
    
    # Determine threshold
    if threshold is None:
        threshold = float(np.quantile(q_obs, q))
    
    # Extract events
    obs_events = extract_events(q_obs, threshold, min_duration=min_duration, merge_gap=merge_gap)
    pred_events = extract_events(q_pred, threshold, min_duration=min_duration, merge_gap=merge_gap)
    
    # Match events
    matches = match_events_by_peak(obs_events, pred_events, tau=tau)
    
    # Compute detection metrics
    H = len(matches)  # Hits
    M = len(obs_events) - H  # Misses
    F = len(pred_events) - H  # False alarms
    
    POD = H / (H + M) if (H + M) > 0 else np.nan
    FAR = F / (H + F) if (H + F) > 0 else np.nan
    CSI = H / (H + M + F) if (H + M + F) > 0 else np.nan
    
    # Compute timing metrics on matched events
    if H > 0:
        dts = np.array([dt for (_, _, dt) in matches], dtype=float)
        timing_bias = float(np.mean(dts))
        timing_mae = float(np.mean(np.abs(dts)))
        timing_rmse = float(np.sqrt(np.mean(dts**2)))
    else:
        dts = np.array([], dtype=float)
        timing_bias = timing_mae = timing_rmse = np.nan
    
    return {
        # Threshold and event counts
        "threshold": threshold,
        "n_obs_events": len(obs_events),
        "n_pred_events": len(pred_events),
        
        # Detection metrics
        "H": H,
        "M": M,
        "F": F,
        "POD": POD,
        "FAR": FAR,
        "CSI": CSI,
        
        # Timing metrics
        "peak_dt_bias": timing_bias,
        "peak_dt_mae": timing_mae,
        "peak_dt_rmse": timing_rmse,
        "peak_dt_all": dts,
        
        # Additional info
        "matches": matches,
        "obs_events": obs_events,
        "pred_events": pred_events
    }


def compute_multi_quantile_event_metrics(q_obs: np.ndarray,
                                        q_pred_median: np.ndarray,
                                        q_pred_q05: np.ndarray,
                                        q_pred_q95: np.ndarray,
                                        threshold: float = None,
                                        q: float = 0.95,
                                        **kwargs) -> Dict:
    """
    Compute event detection and timing metrics for all three quantiles.
    
    Useful for quantile ensemble models to evaluate detection skill across
    different forecast quantiles.
    
    Parameters:
    -----------
    q_obs : np.ndarray
        Observed streamflow (1D)
    q_pred_median : np.ndarray
        Median forecast (1D)
    q_pred_q05 : np.ndarray
        5th percentile forecast (1D)
    q_pred_q95 : np.ndarray
        95th percentile forecast (1D)
    threshold : float, optional
        Event threshold (if None, computed from q_obs)
    q : float, optional
        Quantile for threshold (default: 0.95)
    **kwargs : additional arguments passed to detection_and_timing_metrics
        (min_duration, merge_gap, tau)
    
    Returns:
    --------
    dict with keys 'median', 'q05', 'q95', each containing full metrics dict
    """
    # Compute threshold once from observations if not provided
    if threshold is None:
        threshold = float(np.quantile(q_obs, q))
    
    return {
        'median': detection_and_timing_metrics(q_obs, q_pred_median, threshold=threshold, q=q, **kwargs),
        'q05': detection_and_timing_metrics(q_obs, q_pred_q05, threshold=threshold, q=q, **kwargs),
        'q95': detection_and_timing_metrics(q_obs, q_pred_q95, threshold=threshold, q=q, **kwargs)
    }
