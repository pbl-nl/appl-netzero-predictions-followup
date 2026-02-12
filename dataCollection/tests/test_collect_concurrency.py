from __future__ import annotations

from unittest.mock import patch

from dataCollection.collect import run_all_collectors


@patch("dataCollection.collect._run_primap")
@patch("dataCollection.collect._run_worldbank")
@patch("dataCollection.collect._run_vdem")
@patch("dataCollection.collect._run_rsf")
@patch("dataCollection.collect._run_inform")
@patch("dataCollection.collect._run_iea")
@patch("dataCollection.collect._prepare_iea_execution", return_value=(False, "skip"))
def test_iea_skip_still_runs_all_workers(
    _prepare,
    run_iea,
    run_inform,
    run_rsf,
    run_vdem,
    run_worldbank,
    run_primap,
):
    run_iea.side_effect = lambda should_run, reason: type("R", (), {"collector": "IEA", "status": "skipped", "exit_code": None, "message": reason, "duration_sec": 0.0})()
    run_inform.side_effect = lambda: type("R", (), {"collector": "Inform", "status": "success", "exit_code": 0, "message": None, "duration_sec": 0.1})()
    run_rsf.side_effect = lambda: type("R", (), {"collector": "RSF", "status": "success", "exit_code": 0, "message": None, "duration_sec": 0.1})()
    run_vdem.side_effect = lambda: type("R", (), {"collector": "V-Dem", "status": "success", "exit_code": 0, "message": None, "duration_sec": 0.1})()
    run_worldbank.side_effect = lambda: type("R", (), {"collector": "WorldBank", "status": "success", "exit_code": 0, "message": None, "duration_sec": 0.1})()
    run_primap.side_effect = lambda: type("R", (), {"collector": "PRIMAP", "status": "success", "exit_code": 0, "message": None, "duration_sec": 0.1})()

    run_all_collectors()

    run_iea.assert_called_once()
    run_inform.assert_called_once()
    run_rsf.assert_called_once()
    run_vdem.assert_called_once()
    run_worldbank.assert_called_once()
    run_primap.assert_called_once()


@patch("dataCollection.collect._prepare_iea_execution", return_value=(True, "ok"))
def test_iea_preflight_true_passed_to_worker(_prepare):
    with patch("dataCollection.collect._run_iea") as run_iea, patch(
        "dataCollection.collect._run_inform"
    ) as run_inform, patch("dataCollection.collect._run_rsf") as run_rsf, patch(
        "dataCollection.collect._run_vdem"
    ) as run_vdem, patch("dataCollection.collect._run_worldbank") as run_worldbank, patch(
        "dataCollection.collect._run_primap"
    ) as run_primap:
        run_iea.side_effect = lambda should_run, reason: type("R", (), {"collector": "IEA", "status": "success", "exit_code": 0, "message": None, "duration_sec": 0.1})()
        for fn, name in [
            (run_inform, "Inform"),
            (run_rsf, "RSF"),
            (run_vdem, "V-Dem"),
            (run_worldbank, "WorldBank"),
            (run_primap, "PRIMAP"),
        ]:
            fn.side_effect = lambda n=name: type("R", (), {"collector": n, "status": "success", "exit_code": 0, "message": None, "duration_sec": 0.1})()

        run_all_collectors()

        args, _ = run_iea.call_args
        assert args[0] is True
        assert args[1] == "ok"
