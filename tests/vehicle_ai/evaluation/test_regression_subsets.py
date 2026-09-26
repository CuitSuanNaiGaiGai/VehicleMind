from modules.vehicle_ai.evaluation import regression_report


def summarize(items):
    assert hasattr(regression_report, "summarize_subsets")
    return regression_report.summarize_subsets(items)


def item(case_id, trial_index=1, semantic="pass", task=True):
    return dict(
        case_id=case_id,
        trial_index=trial_index,
        semantic_verdict=semantic,
        task_pass=task,
    )


def test_subsets_keep_failures_and_use_actual_intersection():
    result = summarize(
        [
            item("C04"),
            item("C04", 2, "fail", False),
            item("R01", 1, "pass", False),
            item("A01"),
        ]
    )
    assert result["context_grounding"] == {
        "passed": 2,
        "total": 3,
        "case_ids": ["C04", "R01"],
    }
    assert result["stale_unknown"] == {"passed": 1, "total": 2, "case_ids": ["C04"]}
    assert result["per_trial_task_success"] == {
        "1": {"passed": 2, "total": 3, "case_ids": ["A01", "C04", "R01"]},
        "2": {"passed": 0, "total": 1, "case_ids": ["C04"]},
    }


def test_full_subsets_have_frozen_denominators():
    ids = (
        [f"C{i:02}" for i in range(1, 7)]
        + [f"R{i:02}" for i in range(1, 7)]
        + [f"X{i:02}" for i in range(1, 11)]
    )
    result = summarize([item(case, trial) for case in ids for trial in (1, 2, 3)])
    assert result["context_grounding"]["total"] == 66
    assert result["stale_unknown"]["total"] == 18
    assert result["stale_unknown"]["case_ids"] == [
        "C04",
        "C05",
        "R06",
        "X06",
        "X07",
        "X08",
    ]
    assert [value["passed"] for value in result["per_trial_task_success"].values()] == [
        22,
        22,
        22,
    ]


def test_empty_subset_keeps_zero_denominator():
    result = summarize([item("A01")])
    assert result["context_grounding"] == {"passed": 0, "total": 0, "case_ids": []}
    assert result["stale_unknown"] == {"passed": 0, "total": 0, "case_ids": []}


def test_subset_tables_show_both_models_before_after_and_na():
    from modules.vehicle_ai.evaluation import regression_subsets

    data = summarize([item("C04"), item("C04", 2, "fail", False), item("C04", 3)])
    comparison = {
        "providers": {
            provider: {"before": {}, "after": {"descriptive_subsets": data}}
            for provider in ("qwen", "glm")
        }
    }
    for output in (
        regression_subsets.render_subset_markdown(comparison),
        regression_subsets.render_subset_html(comparison),
    ):
        for expected in (
            "QWEN",
            "GLM",
            "优化前",
            "A5 后",
            "N/A",
            "2/3",
            "C04",
            "事后描述性子集",
            "不是逐事实或人工准确率",
            "不能相加",
            "第 1 次：1/1",
            "第 2 次：0/1",
            "第 3 次：1/1",
        ):
            assert expected in output


def test_validated_batch_and_report_include_subset_results(tmp_path, monkeypatch):
    # Reuse the existing complete hash-bound fixture and report scaffold.
    import test_regression_report as existing

    aggregate = existing.aggregate_batch
    captured = []

    def capture_batch(*args, **kwargs):
        result = aggregate(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(existing, "aggregate_batch", capture_batch)
    existing.test_aggregate_batch_validates_hashes_and_keeps_unavailable_usage_null(
        tmp_path
    )
    subsets = captured[0]["descriptive_subsets"]
    assert subsets["context_grounding"] == {
        "passed": 1,
        "total": 1,
        "case_ids": ["C01"],
    }
    assert subsets["stale_unknown"]["total"] == 0

    def with_subset(renderer):
        def render(comparison):
            for pair in comparison["providers"].values():
                for key in ("before", "after"):
                    pair[key]["descriptive_subsets"] = subsets
            output = renderer(comparison)
            assert "事后描述性子集" in output
            assert "1/1 (100.0%)；case IDs：C01" in output
            assert "N/A；case IDs：无" in output
            return output

        return render

    monkeypatch.setattr(existing, "render_html", with_subset(existing.render_html))
    monkeypatch.setattr(
        existing, "render_markdown", with_subset(existing.render_markdown)
    )
    existing.test_html_uses_chinese_and_does_not_merge_stage_denominators()
