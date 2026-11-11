"""Published-reference accounting must never hide unresolved simulated trials."""

import pytest

from examples.probability.published_reference import summarize_export


def export(outcomes):
    return {"config": {"kind": "coin", "horizon": 6}, "labels": ["Heads", "Tails"],
            "trials": [{"outcome": value} for value in outcomes],
            "counts": [outcomes.count(0), outcomes.count(1)], "unresolved": outcomes.count(-1)}


def test_conditional_heads_and_all_trial_unresolved_use_different_denominators():
    summary = summarize_export(export([0] * 82 + [1] * 76 + [-1] * 98))
    assert summary["total"] == 256
    assert summary["resolved"] == 158
    assert summary["heads_given_resolved"] == pytest.approx(82 / 158)
    assert summary["unresolved_fraction"] == pytest.approx(98 / 256)
    assert summary["heads_given_resolved_interval_95"] == pytest.approx([0.4415585434, 0.5955147754])


def test_all_unresolved_does_not_invent_a_fifty_percent_estimate():
    summary = summarize_export(export([-1, -1]))
    assert summary["heads_given_resolved"] is None
    assert summary["heads_given_resolved_interval_95"] is None
    assert summary["unresolved_fraction"] == 1


def test_stale_totals_are_rejected():
    data = export([0, 1, -1])
    data["counts"] = [2, 1]
    with pytest.raises(ValueError, match="disagree"):
        summarize_export(data)


@pytest.mark.parametrize("outcomes", [[2], [0.5], [True]])
def test_invalid_outcomes_are_rejected(outcomes):
    with pytest.raises(ValueError, match="integer"):
        summarize_export(export(outcomes))


def test_empty_and_non_coin_exports_are_rejected():
    with pytest.raises(ValueError, match="no completed"):
        summarize_export(export([]))
    data = export([0])
    data["config"]["kind"] = "die"
    with pytest.raises(ValueError, match="coin export"):
        summarize_export(data)
