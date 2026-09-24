from __future__ import annotations

from modules.perception_audit.report import build_summary, render_html, render_markdown


def fixture_results():
    catalog = {
        "attempted": {"cabin": 2, "road": 1},
        "items": [
            {"id": "cabin/a.mp4", "domain": "cabin", "source_status": "unknown"},
            {"id": "cabin/b.mp4", "domain": "cabin", "source_status": "unknown"},
            {"id": "road/c.mp4", "domain": "road", "source_status": "unknown"},
        ],
    }
    results = [
        {
            "id": "cabin/a.mp4",
            "domain": "cabin",
            "status": "success",
            "processed_frames": 10,
            "valid_output_frames": 8,
            "state_counts": {"NORMAL": 7, "UNKNOWN": 1},
            "state_duration_seconds": {"NORMAL": 0.7, "UNKNOWN": 0.1},
            "face_visible_frames": 7,
            "eye_closed_frames": 2,
            "yawn_output_frames": 1,
        },
        {
            "id": "cabin/b.mp4",
            "domain": "cabin",
            "status": "failed",
            "error": "视频无法打开",
            "processed_frames": 0,
            "valid_output_frames": 0,
        },
        {
            "id": "road/c.mp4",
            "domain": "road",
            "status": "success",
            "processed_frames": 5,
            "valid_output_frames": 5,
            "object_counts": {"car": 9},
            "lane_detected_frames": 3,
            "drivable_detected_frames": 4,
            "lane_output_flips": 1,
            "drivable_output_flips": 2,
        },
    ]
    return catalog, results


def test_summary_keeps_video_and_frame_denominators_separate():
    catalog, results = fixture_results()
    summary = build_summary(catalog, results)
    assert summary["domains"]["cabin"]["videos"] == {"successful": 1, "attempted": 2}
    assert summary["domains"]["cabin"]["frames"] == {"valid_output": 8, "processed": 10}
    assert summary["domains"]["road"]["videos"] == {"successful": 1, "attempted": 1}
    assert summary["domains"]["road"]["output"]["object_counts"] == {"car": 9}
    assert summary["domains"]["cabin"]["output"]["state_duration_seconds"] == {
        "NORMAL": 0.7,
        "UNKNOWN": 0.1,
    }
    assert summary["accuracy"] == {
        "status": "not_evaluated",
        "reason": "无对齐真值标签",
    }
    assert summary["domains"]["cabin"]["failures"] == {"视频无法打开": 1}


def test_chinese_reports_only_show_aggregate_and_escape_html():
    catalog, results = fixture_results()
    results[1]["error"] = "<script>/Users/private/key</script>"
    summary = build_summary(catalog, results)
    markdown = render_markdown(summary)
    html = render_html(summary)
    assert "舱内" in markdown and "舱外" in html
    assert "1 / 2 条视频" in markdown
    assert "8 / 10 帧" in markdown
    assert "not_evaluated" in markdown and "not_evaluated" in html
    assert '<main class="layout">' in html
    assert '<section class="metric-grid">' in html
    assert "<table" in html
    assert "<pre>" not in html
    for report in (markdown, html):
        assert "a.mp4" not in report
        assert "/Users/private" not in report
        assert "<script>" not in report
        assert "Precision" not in report


def test_local_html_can_show_each_video_without_absolute_paths():
    catalog, results = fixture_results()
    html = render_html(build_summary(catalog, results), results=results)
    assert "a.mp4" in html and "b.mp4" in html and "c.mp4" in html
    assert "逐视频核对" in html
    assert "视频无法打开" in html
    assert "/Users/" not in html


def test_zero_frame_denominator_has_no_percentage():
    catalog, results = fixture_results()
    results[0]["processed_frames"] = 0
    results[0]["valid_output_frames"] = 0
    html = render_html(build_summary(catalog, results))
    assert "无可计算比例" in html


def test_local_sample_timestamp_links_to_relative_video():
    catalog, results = fixture_results()
    results[0]["samples"] = [
        {"frame_index": 12, "timestamp_ms": 1200, "output": {"driver_state": "NORMAL"}}
    ]
    html = render_html(
        build_summary(catalog, results),
        results=results,
        video_links={"cabin/a.mp4": "../../../data/perception/cabin/a.mp4"},
    )
    assert "../../../data/perception/cabin/a.mp4#t=1.2" in html
    assert "1.2 秒" in html
