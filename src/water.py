
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def semi_synthetic_analysis(
    file_path: str,
    treat_col: str = "treatment",
    outcome_col: str = "usage_end",
    bias_col: str = "income",
    K: int = 5,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Run the semi‑synthetic analysis on a Stata data set.

    Parameters
    ----------
    file_path : str
        Path to the Stata ``.dta`` file containing the data.  The file
        must contain at least the treatment, outcome and bias columns.
    treat_col : str, optional
        Name of the binary treatment indicator (1 for treated, 0 for
        control).  Defaults to ``"treatment"``.  The column is cast to
        integer if necessary.
    outcome_col : str, optional
        Name of the outcome variable measured after the intervention.
        Defaults to ``"usage_end"``.  It is expected to be numeric.
    bias_col : str, optional
        Name of the covariate used to induce sampling bias.  This
        covariate should be continuous or ordered.  Defaults to
        ``"income"``.
    K : int, optional
        Number of blocks to partition the data into.  Each block is
        further split into three sub‑blocks corresponding to the RCT,
        observational and negative control samples.  Defaults to 5.
    random_state : int, optional
        Seed used for random shuffling of the data.  Setting this
        produces reproducible splits.  Defaults to 0.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``ate_population``
            The difference in means between treated and control units
            computed over the entire data set.  This serves as the
            ground truth estimate for benchmarking.
        ``ate1_mean``
            The average of the block‑level RCT estimates (dataset #1).
        ``ate_obs_mean``
            The average of the bias‑corrected observational estimates
            ``ate2 - diff3`` across blocks.
        ``block_results``
            A list of dictionaries, one per block, containing
            ``ate1``, ``ate2``, ``diff3`` and ``ate_obs``.  ``ate1``
            corresponds to the RCT estimate from sub‑block 1; ``ate2``
            corresponds to the difference in means between biased
            treated units and random controls in sub‑block 2; ``diff3``
            corresponds to the difference between high‑income and
            low‑income controls in sub‑block 3; and ``ate_obs`` is the
            bias‑corrected estimate ``ate2 - diff3``.

    Notes
    -----
    For the observational and negative control analyses we impose bias
    using a simple median split of the ``bias_col``: units with values
    above the median are considered "high" and those below or equal are
    considered "low".  You may wish to customise this rule by
    replacing the median with another quantile or using a weighted
    sampling scheme.  The code will operate correctly as long as the
    ``bias_col`` contains numerical values.
    """
    # Load data from Stata file
    df = pd.read_stata(file_path)

    # Ensure treatment column is numeric (0/1)
    df[treat_col] = df[treat_col].astype(int)

    # Compute ground truth ATE from full data
    treated_all = df[df[treat_col] == 1]
    control_all = df[df[treat_col] == 0]
    ate_population = treated_all[outcome_col].mean() - control_all[outcome_col].mean()

    # Shuffle data before partitioning into blocks
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n = len(df_shuffled)
    block_size = n // K

    # Collect results across blocks
    block_results: List[Dict[str, float]] = []
    ate1_list: List[float] = []
    ate_obs_list: List[float] = []

    for k in range(K):
        start = k * block_size
        # Last block takes the remainder
        if k == K - 1:
            block = df_shuffled.iloc[start:]
        else:
            block = df_shuffled.iloc[start:start + block_size]

        # Split block into three roughly equal sub‑blocks
        m = len(block)
        s = m // 3
        sub1 = block.iloc[:s].copy()
        sub2 = block.iloc[s:2 * s].copy()
        sub3 = block.iloc[2 * s:].copy()

        # Dataset #1: unbiased RCT estimate from sub1
        treated1 = sub1[sub1[treat_col] == 1]
        control1 = sub1[sub1[treat_col] == 0]
        if len(treated1) > 0 and len(control1) > 0:
            ate1 = treated1[outcome_col].mean() - control1[outcome_col].mean()
        else:
            ate1 = np.nan

        # Dataset #2: biased treated vs. random controls from sub2
        treated2 = sub2[sub2[treat_col] == 1]
        control2 = sub2[sub2[treat_col] == 0]
        # Define high‑income treated group using median split
        if len(treated2) > 0:
            median_income_treat = treated2[bias_col].median()
            biased_treated = treated2[treated2[bias_col] > median_income_treat]
        else:
            biased_treated = pd.DataFrame(columns=sub2.columns)
        # All controls in sub2 form the random control group
        random_controls = control2.copy()
        if not biased_treated.empty and not random_controls.empty:
            ate2 = biased_treated[outcome_col].mean() - random_controls[outcome_col].mean()
        else:
            ate2 = np.nan

        # Dataset #3: negative control within treated units (but we use
        # control units only) from sub3
        control3 = sub3[sub3[treat_col] == 0]
        if len(control3) > 0:
            median_income_control = control3[bias_col].median()
            # Biased sample: high‑income controls
            biased_control = control3[control3[bias_col] > median_income_control]
            # Random controls: low‑income controls
            random_control3 = control3[control3[bias_col] <= median_income_control]
        else:
            biased_control = pd.DataFrame(columns=sub3.columns)
            random_control3 = pd.DataFrame(columns=sub3.columns)
        if not biased_control.empty and not random_control3.empty:
            diff3 = biased_control[outcome_col].mean() - random_control3[outcome_col].mean()
        else:
            diff3 = np.nan

        # Bias‑corrected observational estimate: E[Y(1)|X] - E[Y(0)] minus
        # the bias from the negative control (approximate b)
        if np.isnan(ate2) or np.isnan(diff3):
            ate_obs = np.nan
        else:
            ate_obs = ate2 - diff3

        # Store block‑level results
        block_results.append({
            'block': k,
            'ate1': ate1,
            'ate2': ate2,
            'diff3': diff3,
            'ate_obs': ate_obs,
        })
        ate1_list.append(ate1)
        ate_obs_list.append(ate_obs)

    # Compute averages across blocks ignoring NaNs
    ate1_mean = float(np.nanmean(ate1_list)) if len(ate1_list) > 0 else np.nan
    ate_obs_mean = float(np.nanmean(ate_obs_list)) if len(ate_obs_list) > 0 else np.nan

    return {
        'ate_population': float(ate_population),
        'ate1_mean': ate1_mean,
        'ate_obs_mean': ate_obs_mean,
        'block_results': block_results,
    }


def _print_results(res: Dict[str, Any]) -> None:
    """Helper for pretty printing results to the console."""
    print(f"ATE from full data (ground truth): {res['ate_population']:.4f}")
    print(f"Average RCT estimate across blocks (dataset #1): {res['ate1_mean']:.4f}")
    print(f"Average bias‑corrected observational estimate: {res['ate_obs_mean']:.4f}")
    print("\nPer‑block results:")
    for r in res['block_results']:
        print(
            f"  Block {r['block']:2d}: ate1 = {r['ate1']:.4f}, "
            f"ate2 = {r['ate2']:.4f}, diff3 = {r['diff3']:.4f}, "
            f"ate_obs = {r['ate_obs']:.4f}"
        )


if __name__ == '__main__':
    # Default parameters for the command line
    filepath = '../clean.dta'
    treat_column = 'treatment'
    outcome_column = 'usage_end'
    bias_column = 'income'
    K_blocks = 5

    # Run analysis
    try:
        results = semi_synthetic_analysis(
            filepath,
            treat_col=treat_column,
            outcome_col=outcome_column,
            bias_col=bias_column,
            K=K_blocks,
            random_state=0,
        )
        _print_results(results)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data file exists at {filepath}.")