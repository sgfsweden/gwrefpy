import json

import numpy as np
import pandas as pd
import pytest

from gwrefpy import Model, Well


@pytest.fixture()
def model() -> Model:
    """Model with two observation wells and three reference wells.

    With linear regression at offset 3.5D the RMSE ranking is known:
    obsA: refGood < refMid < refBad
    obsB: refBad < refGood < refMid
    """
    obs = pd.Series(
        [11.4, 11.7, 11.8],
        index=pd.to_datetime(["2023-01-07", "2023-02-01", "2023-02-25"]),
    )
    ref = pd.Series(
        [8.9, 9.2, 9.3, 9.3, 9.5],
        index=pd.to_datetime(
            ["2023-01-08", "2023-02-03", "2023-02-08", "2023-02-25", "2023-02-28"]
        ),
    )
    model = Model(name="collection")
    model.add_well(
        [
            Well("obsA", is_reference=False, timeseries=obs),
            Well("obsB", is_reference=False, timeseries=obs + [0.0, 0.3, -0.2]),
            Well("refGood", is_reference=True, timeseries=obs - 2 + [0, 1e-3, 0]),
            Well("refMid", is_reference=True, timeseries=ref),
            Well(
                "refBad",
                is_reference=True,
                timeseries=ref + np.array([0.3, -0.2, 0, 0.1, 0]),
            ),
        ]
    )
    return model


def fit(model: Model, obs: str, ref: str, **kwargs):
    kwargs.setdefault("offset", "3.5D")
    return model.fit(obs, ref, report=False, **kwargs)


def fit_all(model: Model) -> None:
    for obs in ["obsA", "obsB"]:
        for ref in ["refGood", "refMid", "refBad"]:
            fit(model, obs, ref)


def names(collection) -> list[str]:
    return [f.name for f in collection]


# --- naming -----------------------------------------------------------------


def test_default_name_is_readable(model) -> None:
    assert fit(model, "obsA", "refMid").name == "obsA~refMid:linearregression"
    assert (
        fit(model, "obsA", "refMid", method="npolyfit", degree=2).name
        == "obsA~refMid:npolyfit2"
    )
    assert (
        fit(model, "obsA", "refMid", method="chebyshev", degree=1).name
        == "obsA~refMid:chebyshev1"
    )


def test_refit_with_same_name_replaces_in_place(model) -> None:
    fit(model, "obsA", "refMid")
    fit(model, "obsA", "refGood")
    refit = fit(model, "obsA", "refMid", offset="5D")

    assert names(model.fits) == [
        "obsA~refMid:linearregression",
        "obsA~refGood:linearregression",
    ]
    assert model.fits[0] is refit


def test_explicit_name_collision_replaces(model) -> None:
    fit(model, "obsA", "refMid", name="mine")
    replacement = fit(model, "obsB", "refGood", name="mine")

    assert len(model.fits) == 1
    assert model.fits["mine"] is replacement


# --- filter -----------------------------------------------------------------


def test_filter_by_obs_and_ref_accepts_names_and_wells(model) -> None:
    fit_all(model)
    ref_good = model.get_wells("refGood")

    result = model.fits.filter(obs="obsA", ref=ref_good)

    assert names(result) == ["obsA~refGood:linearregression"]


def test_filter_lists_match_any_value(model) -> None:
    fit_all(model)

    result = model.fits.filter(obs="obsB", ref=["refGood", "refBad"])

    assert names(result) == [
        "obsB~refGood:linearregression",
        "obsB~refBad:linearregression",
    ]


def test_filter_by_method(model) -> None:
    fit(model, "obsA", "refMid")
    fit(model, "obsA", "refMid", method="npolyfit", degree=2)
    fit(model, "obsA", "refMid", method="chebyshev", degree=1)

    assert names(model.fits.filter(method="npolyfit")) == ["obsA~refMid:npolyfit2"]
    assert names(model.fits.filter(method=["linearregression", "chebyshev"])) == [
        "obsA~refMid:linearregression",
        "obsA~refMid:chebyshev1",
    ]


def test_filter_by_name_and_predicate(model) -> None:
    fit_all(model)

    by_name = model.fits.filter(name="obsA~refMid:linearregression")
    by_predicate = model.fits.filter(lambda f: f.ref_well.name == "refBad")

    assert names(by_name) == ["obsA~refMid:linearregression"]
    assert names(by_predicate) == [
        "obsA~refBad:linearregression",
        "obsB~refBad:linearregression",
    ]


def test_filter_unknown_method_raises(model) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        model.fits.filter(method="linreg")


def test_filter_without_matches_warns_and_is_empty(model, caplog) -> None:
    fit_all(model)

    result = model.fits.filter(obs="obsTypo")

    assert len(result) == 0
    assert "No fits matched" in caplog.text


# --- get / best / top -------------------------------------------------------


def test_get_returns_the_single_match(model) -> None:
    fit_all(model)

    result = model.fits.get(obs="obsB", ref="refMid")

    assert result.name == "obsB~refMid:linearregression"


def test_get_raises_unless_exactly_one_match(model) -> None:
    fit_all(model)

    with pytest.raises(ValueError, match="3 fits"):
        model.fits.get(obs="obsA")
    with pytest.raises(ValueError, match="No fit"):
        model.fits.get(obs="obsA", ref="nope")


def test_best_returns_lowest_rmse_fit(model) -> None:
    fit_all(model)

    assert model.fits.best().name == "obsA~refGood:linearregression"
    assert model.fits.filter(obs="obsB").best().name == "obsB~refBad:linearregression"


def test_best_of_empty_collection_raises(model) -> None:
    with pytest.raises(ValueError, match="empty"):
        model.fits.best()


def test_top_without_by_ranks_the_whole_collection(model) -> None:
    fit_all(model)

    result = model.fits.filter(obs="obsA").top(n=2)

    assert names(result) == [
        "obsA~refGood:linearregression",
        "obsA~refMid:linearregression",
    ]


def test_top_by_obs_gives_best_per_obs_well(model) -> None:
    fit_all(model)

    assert names(model.fits.top(by="obs")) == [
        "obsA~refGood:linearregression",
        "obsB~refBad:linearregression",
    ]
    assert names(model.fits.top(n=2, by="obs")) == [
        "obsA~refGood:linearregression",
        "obsA~refMid:linearregression",
        "obsB~refBad:linearregression",
        "obsB~refGood:linearregression",
    ]


def test_top_by_several_keys(model) -> None:
    fit(model, "obsA", "refMid")
    fit(model, "obsA", "refBad")
    fit(model, "obsA", "refMid", method="npolyfit", degree=1)
    fit(model, "obsA", "refGood", method="npolyfit", degree=1)

    result = model.fits.top(by=("obs", "method"))

    assert names(result) == [
        "obsA~refMid:linearregression",
        "obsA~refGood:npolyfit1",
    ]


def test_top_ties_go_to_the_earliest_fit(model) -> None:
    fit(model, "obsA", "refMid", name="first")
    fit(model, "obsA", "refMid", name="second")

    assert model.fits.best().name == "first"
    assert names(model.fits.top()) == ["first"]


def test_top_rejects_unknown_group_key(model) -> None:
    with pytest.raises(ValueError, match="Unknown grouping"):
        model.fits.top(by="well")


# --- remove / clear / keep_best ---------------------------------------------


def test_remove_by_fit_name_or_collection(model) -> None:
    fit_all(model)
    a_good = model.fits["obsA~refGood:linearregression"]

    model.fits.remove(a_good)
    model.fits.remove("obsA~refMid:linearregression")
    model.fits.remove(model.fits.filter(obs="obsB", ref=["refGood", "refMid"]))

    assert names(model.fits) == [
        "obsA~refBad:linearregression",
        "obsB~refBad:linearregression",
    ]


def test_remove_by_filter_returns_removed_fits(model) -> None:
    fit_all(model)
    fit(model, "obsA", "refMid", method="npolyfit", degree=1)

    removed = model.fits.remove(obs="obsA", method="linearregression")

    assert names(removed) == [
        "obsA~refGood:linearregression",
        "obsA~refMid:linearregression",
        "obsA~refBad:linearregression",
    ]
    assert names(model.fits.filter(obs="obsA")) == ["obsA~refMid:npolyfit1"]
    assert len(model.fits) == 4


def test_remove_dry_run_changes_nothing(model) -> None:
    fit_all(model)

    would_remove = model.fits.remove(ref="refBad", dry_run=True)

    assert names(would_remove) == [
        "obsA~refBad:linearregression",
        "obsB~refBad:linearregression",
    ]
    assert len(model.fits) == 6


def test_remove_rejects_ambiguous_or_empty_calls(model) -> None:
    fit_all(model)

    with pytest.raises(TypeError):
        model.fits.remove()
    with pytest.raises(TypeError):
        model.fits.remove("obsA~refMid:linearregression", obs="obsA")
    assert len(model.fits) == 6


def test_remove_missing_target_raises_key_error(model) -> None:
    fit_all(model)
    fit(model, "obsA", "refMid", name="elsewhere")
    detached = model.fits.remove("elsewhere")

    with pytest.raises(KeyError):
        model.fits.remove("nope")
    with pytest.raises(KeyError):
        model.fits.remove(detached[0])
    assert len(model.fits) == 6


def test_remove_by_filter_without_matches_is_a_no_op(model) -> None:
    fit_all(model)

    removed = model.fits.remove(obs="obsTypo")

    assert len(removed) == 0
    assert len(model.fits) == 6


def test_clear_removes_everything(model) -> None:
    fit_all(model)

    model.fits.clear()

    assert len(model.fits) == 0


def test_keep_best_keeps_n_best_per_group_in_original_order(model) -> None:
    fit_all(model)

    removed = model.fits.keep_best(2, by="obs")

    assert names(model.fits) == [
        "obsA~refGood:linearregression",
        "obsA~refMid:linearregression",
        "obsB~refGood:linearregression",
        "obsB~refBad:linearregression",
    ]
    assert names(removed) == [
        "obsA~refBad:linearregression",
        "obsB~refMid:linearregression",
    ]


def test_keep_best_leaves_small_groups_alone(model) -> None:
    fit(model, "obsA", "refMid")

    removed = model.fits.keep_best(3, by="obs")

    assert len(removed) == 0
    assert len(model.fits) == 1


def test_keep_best_dry_run_changes_nothing(model) -> None:
    fit_all(model)

    would_remove = model.fits.keep_best(1, by="obs", dry_run=True)

    assert len(would_remove) == 4
    assert len(model.fits) == 6


def test_keep_best_requires_by(model) -> None:
    with pytest.raises(TypeError):
        model.fits.keep_best(3)


def test_detached_collections_cannot_be_modified(model) -> None:
    fit_all(model)
    detached = model.fits.filter(obs="obsA")

    with pytest.raises(TypeError, match="detached"):
        detached.remove(obs="obsA")
    with pytest.raises(TypeError, match="detached"):
        detached.keep_best(1, by="obs")
    with pytest.raises(TypeError, match="detached"):
        detached.clear()
    assert len(model.fits) == 6
    assert len(detached) == 3


# --- to_dataframe / display -------------------------------------------------


def test_to_dataframe_matches_fits_summary_columns(model) -> None:
    fit_all(model)

    df = model.fits.filter(obs="obsA").to_dataframe()

    assert len(df) == 3
    assert df["ref_well_name"].tolist() == ["refGood", "refMid", "refBad"]
    assert set(df["obs_well_name"]) == {"obsA"}
    assert set(df["method"]) == {"LinRegResult"}
    assert {"rmse", "n_points", "time_offset", "linreg_slope"} <= set(df.columns)


def test_to_dataframe_of_empty_collection_is_empty(model) -> None:
    assert model.fits.to_dataframe().empty


def test_repr_summarises_the_collection(model) -> None:
    fit_all(model)

    assert repr(model.fits) == "FitCollection(6 fits, 2 obs wells, 3 ref wells)"
    assert "<table" in model.fits._repr_html_()


# --- integration with Model -------------------------------------------------


def test_fit_with_lists_returns_a_detached_collection(model) -> None:
    result = model.fit(
        ["obsA", "obsB"], ["refGood", "refMid"], offset="3.5D", report=False
    )

    assert names(result) == [
        "obsA~refGood:linearregression",
        "obsB~refMid:linearregression",
    ]
    with pytest.raises(TypeError, match="detached"):
        result.clear()
    assert len(model.fits) == 2


# --- deprecated API ---------------------------------------------------------


def test_get_fits_is_deprecated_but_keeps_its_shape(model) -> None:
    fit(model, "obsA", "refMid")
    fit(model, "obsA", "refGood")

    with pytest.deprecated_call():
        many = model.get_fits("obsA")
    with pytest.deprecated_call():
        one = model.get_fits("refGood")
    with pytest.deprecated_call():
        none = model.get_fits("obsB")

    assert isinstance(many, list) and len(many) == 2
    assert one.name == "obsA~refGood:linearregression"
    assert none is None


def test_fits_summary_is_deprecated(model) -> None:
    fit_all(model)

    with pytest.deprecated_call():
        summary = model.fits_summary()

    pd.testing.assert_frame_equal(summary, model.fits.to_dataframe())


def test_remove_fits_by_n_is_deprecated(model) -> None:
    fit_all(model)

    with pytest.deprecated_call():
        model.remove_fits_by_n("obsA", 1)

    assert names(model.fits.filter(obs="obsA")) == ["obsA~refGood:linearregression"]
    assert len(model.fits) == 4


# --- persistence ------------------------------------------------------------


def test_save_and_load_keeps_fits_and_names(model, tmp_path) -> None:
    fit_all(model)
    model.save_project(str(tmp_path / "project"))

    loaded = Model(name=str(tmp_path / "project.gwref"))

    assert names(loaded.fits) == names(model.fits)


def test_loading_duplicate_names_suffixes_instead_of_replacing(
    model, tmp_path, caplog
) -> None:
    fit(model, "obsA", "refMid", name="dup")
    fit(model, "obsA", "refGood", name="other")
    fit(model, "obsB", "refBad", name="legacy-uuid-1234")
    path = tmp_path / "project.gwref"
    model.save_project(str(path))
    data = json.loads(path.read_text())
    data["fits"][1]["name"] = "dup"
    path.write_text(json.dumps(data))

    loaded = Model(name=str(path))

    assert names(loaded.fits) == ["dup", "dup#2", "legacy-uuid-1234"]
    assert loaded.fits["dup#2"].ref_well.name == "refGood"
    assert "dup#2" in caplog.text


def test_fit_with_lists_rejects_a_single_shared_name(model) -> None:
    with pytest.raises(ValueError, match="list of names"):
        model.fit(
            ["obsA", "obsB"],
            ["refGood", "refMid"],
            offset="3.5D",
            name="run1",
            report=False,
        )
    assert len(model.fits) == 0


def test_get_fits_still_ignores_unknown_methods(model) -> None:
    fit(model, "obsA", "refMid")
    fit(model, "obsA", "refGood")

    with pytest.deprecated_call():
        result = model.get_fits("obsA", method="bogus")

    assert len(result) == 2
