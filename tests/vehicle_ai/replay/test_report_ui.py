import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3] / "apps/vehicle_ai_demo/replay_ui/report.js"
)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js 不可用")
def test_report_ui_renders_failed_assertion_and_quality_in_chinese() -> None:
    harness = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function node() {
  return {
    textContent: '', children: [], src: '',
    classList: {add() {}},
    append(...children) { this.children.push(...children); },
  };
}
const roots = {};
const data = {
  description: '失败场景', media: {cabin: 'c.gif', road: 'r.gif'},
  observations: {}, limitations: ['仅用于演示'],
  provenance: {git_commit: 'abc', created_at_utc: 'now'}, config_sha256: '123',
  summary: {
    passed: false, scenario_id: 'failure', semantic_sha256: 'abcdef123456',
    unauthorized_sensitive_executions: 0, metrics: {total_ms: 1},
    final_context: {driver: {state: 'STALE', risk: 'INVALID'}},
    assertions: [{name: 'vehicle.navigation_state', passed: false,
      expected: 'ACTIVE', actual: 'IDLE'}],
  },
  trace: [{at_ms: 1, kind: 'confirmation', data: {
    tool_name: 'start_navigation', success: false, error: 'CONFIRMATION_EXPIRED',
  }}],
};
const document = {
  getElementById(id) {
    if (id === 'vehiclemind-data') return {textContent: JSON.stringify(data)};
    return roots[id] ??= node();
  },
  createElement() { return node(); },
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {document});
function allText(item) {
  return [item.textContent, ...item.children.flatMap(allText)].join(' ');
}
assert.equal(roots['status-badge'].textContent, '失败');
assert.match(allText(roots['context-grid']), /已过期/);
assert.match(allText(roots['context-grid']), /无效/);
assert.match(allText(roots['assertions']), /预期：进行中/);
assert.match(allText(roots['assertions']), /实际：未启动/);
assert.match(allText(roots['timeline']), /错误码：CONFIRMATION_EXPIRED/);
"""
    result = subprocess.run(
        ["node", "-e", harness, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
