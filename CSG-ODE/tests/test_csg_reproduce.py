from __future__ import annotations

import csv
import json

from reproduce_csg_ode import AGGREGATE_FIELDS, SweepRun, aggregate


def test_aggregate_writes_csv_when_all_results_are_excluded(tmp_path) -> None:
    result_path = tmp_path / "variant-result.json"
    result_path.write_text(json.dumps({"status": "dataset_variant_complete"}), encoding="utf-8")
    run = SweepRun(
        dataset="pems08",
        task="interp",
        obs_ratio=0.4,
        seed=1991,
        variant="full",
        run_name="variant",
        command=["python", "run_csg_ode.py"],
        result_path=str(result_path),
    )

    assert aggregate([run], tmp_path) == []
    assert json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8")) == []
    with (tmp_path / "aggregate.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [AGGREGATE_FIELDS]
