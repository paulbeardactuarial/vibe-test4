import numpy as np
import pandas as pd
from pathlib import Path

_BASE_DIR = Path(__file__).parent
_df = pd.read_csv(_BASE_DIR / "data" / "pma92c20_pfa92c20.csv")
_qx_dict = _df.set_index(["sex", "age"])["qx"].to_dict()


def _survival_probs(gender, start_age, improvement_rate, max_age=120):
    """Return array of t_p_x for t = 0, 1, ..., max_age - start_age.

    Applies cohort projection: mortality at age x+j projected j years forward
    is q_{x+j} * (1 - improvement_rate)^j.
    """
    n = max_age - start_age + 1
    tpx = np.ones(n)
    for t in range(1, n):
        age = start_age + t - 1
        q_base = _qx_dict.get((gender, age), 1.0)
        q = min(q_base * (1 - improvement_rate) ** (t - 1), 1.0)
        tpx[t] = tpx[t - 1] * (1 - q)
    return tpx


def _pv_scalar(gender, age, discount_rate, improvement_rate,
               annuity_type, gender2, age_difference, joint_type):
    v = 1.0 / (1.0 + discount_rate)
    tpx = _survival_probs(gender, age, improvement_rate)
    n = len(tpx)
    disc = v ** np.arange(n)

    if annuity_type == "single":
        return float(np.dot(disc, tpx))

    age2 = age + age_difference
    tpy = _survival_probs(gender2, age2, improvement_rate)

    n = min(len(tpx), len(tpy))
    tpx = tpx[:n]
    tpy = tpy[:n]
    disc = disc[:n]

    if joint_type == "first_death":
        return float(np.dot(disc, tpx * tpy))
    else:  # second_death / last survivor
        return float(np.dot(disc, tpx + tpy + tpx * tpy))


def pv_annuity(gender, age, discount_rate, mortality_improvement_rate,
               annuity_type="single", gender2=None, age_difference=None,
               joint_type=None):
    """Calculate present value of a whole-of-life annuity-due.

    Parameters
    ----------
    gender : str or array-like of str
        'male' or 'female'
    age : int or array-like of int
        Starting age(s)
    discount_rate : float
        Annual discount rate, e.g. 0.05
    mortality_improvement_rate : float
        Annual cohort mortality improvement rate, e.g. 0.01
    annuity_type : {'single', 'joint'}
    gender2 : str, optional
        Second life gender for joint annuity
    age_difference : int, optional
        age2 = age + age_difference
    joint_type : {'first_death', 'second_death'}, optional
        first_death: pays while both alive; second_death: pays while at least one alive

    Returns
    -------
    float or np.ndarray
    """
    scalar_input = np.ndim(age) == 0

    age_arr = np.atleast_1d(np.asarray(age, dtype=int))
    if isinstance(gender, str):
        gender_arr = np.full(age_arr.shape, gender, dtype=object)
    else:
        gender_arr = np.asarray(gender, dtype=object)

    vec = np.vectorize(
        lambda g, a: _pv_scalar(
            g, int(a), discount_rate, mortality_improvement_rate,
            annuity_type, gender2, age_difference, joint_type
        )
    )
    result = vec(gender_arr, age_arr)

    return float(result.flat[0]) if scalar_input else result
