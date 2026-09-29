import numpy as np
import pandas as pd
import pytest
import scipy.stats as sts

import decoupler as dc


def _zscore_input(values):
    data = pd.DataFrame([values], index=["sample"], columns=[f"s{i}" for i in range(6)])
    net = pd.DataFrame(
        {
            "source": ["kinase"] * 3,
            "target": ["s0", "s1", "s2"],
            "weight": [1.0] * 3,
        }
    )
    return data, net


def _expected_score(values, *, center):
    values = np.asarray(values, dtype=float)
    target_mean = np.mean(values[:3])
    global_mean = np.mean(values) if center else 0.0
    return (target_mean - global_mean) * np.sqrt(3) / np.std(values, ddof=1)


@pytest.mark.parametrize("flavor,center", [("KSEA", True), ("RoKAI", False)])
def test_zscore_flavors_follow_their_published_formulas(flavor, center):
    values = [2.0, 3.0, 4.0, 3.0, 3.0, 3.0]
    data, net = _zscore_input(values)

    scores, pvals = dc.mt.zscore(data=data, net=net, tmin=3, flavor=flavor)

    expected_score = _expected_score(values, center=center)
    expected_pval = 2 * sts.norm.sf(abs(expected_score))
    assert scores.loc["sample", "kinase"] == pytest.approx(expected_score)
    assert pvals.loc["sample", "kinase"] == pytest.approx(expected_pval)


def test_zscore_ksea_is_translation_invariant_and_default_is_rokai():
    values = np.array([1.0, 2.0, 3.0, -2.0, -2.0, -2.0])
    shifted = values + 5.0
    data, net = _zscore_input(values)
    shifted_data, _ = _zscore_input(shifted)

    ksea, ksea_pvals = dc.mt.zscore(data=data, net=net, tmin=3, flavor="KSEA")
    shifted_ksea, shifted_ksea_pvals = dc.mt.zscore(data=shifted_data, net=net, tmin=3, flavor="KSEA")
    rokai, rokai_pvals = dc.mt.zscore(data=data, net=net, tmin=3, flavor="RoKAI")
    default, default_pvals = dc.mt.zscore(data=data, net=net, tmin=3)

    np.testing.assert_allclose(shifted_ksea, ksea)
    np.testing.assert_allclose(shifted_ksea_pvals, ksea_pvals)
    np.testing.assert_allclose(default, rokai)
    np.testing.assert_allclose(default_pvals, rokai_pvals)


def test_zscore_flavors_agree_when_input_is_globally_centered():
    data, net = _zscore_input([1.0, 2.0, 3.0, -2.0, -2.0, -2.0])

    ksea, ksea_pvals = dc.mt.zscore(data=data, net=net, tmin=3, flavor="KSEA")
    rokai, rokai_pvals = dc.mt.zscore(data=data, net=net, tmin=3, flavor="RoKAI")

    np.testing.assert_allclose(ksea, rokai)
    np.testing.assert_allclose(ksea_pvals, rokai_pvals)


@pytest.mark.parametrize("flavor", ["KSEA", "RoKAI"])
def test_func_zscore(
    mat,
    adjmat,
    flavor,
):
    X, obs, var = mat
    es, pv = dc.mt._zscore._func_zscore(mat=X, adj=adjmat, flavor=flavor)
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()
