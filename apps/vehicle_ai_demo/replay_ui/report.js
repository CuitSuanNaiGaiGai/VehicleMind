"use strict";

const data = JSON.parse(document.getElementById("vehiclemind-data").textContent);
const summary = data.summary;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function valueText(value) {
  if (value === null) return "UNKNOWN";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

document.getElementById("scenario-description").textContent = data.description;
const status = document.getElementById("status-badge");
status.textContent = summary.passed ? "PASS" : "FAIL";
status.classList.add(summary.passed ? "pass" : "fail");
document.getElementById("cabin-media").src = data.media.cabin;
document.getElementById("road-media").src = data.media.road;

const metrics = [
  ["Scenario", summary.scenario_id],
  ["Semantic trace", summary.semantic_sha256.slice(0, 12)],
  ["Safety violations", summary.unauthorized_sensitive_executions],
  ["Runtime", `${summary.metrics.total_ms.toFixed(2)} ms`],
];
const metricRoot = document.getElementById("headline-metrics");
for (const [label, value] of metrics) {
  const card = element("article", "metric");
  card.append(element("span", "label", label), element("strong", "", value));
  metricRoot.append(card);
}

const contextRoot = document.getElementById("context-grid");
for (const [domain, values] of Object.entries(summary.final_context)) {
  const card = element("article", "context-card");
  card.append(element("span", "label", "Resolved context"));
  card.append(element("h2", "", domain));
  const observation = data.observations[domain] || data.observations[domain === "driver" ? "cabin" : domain];
  if (observation) {
    for (const [key, value] of Object.entries(observation)) {
      const row = element("div", "context-row observation-row");
      row.append(element("span", "", key), element("code", "", valueText(value)));
      card.append(row);
    }
  }
  for (const [key, value] of Object.entries(values)) {
    const row = element("div", "context-row");
    row.append(element("span", "", key), element("code", "", valueText(value)));
    card.append(row);
  }
  contextRoot.append(card);
}

const visibleKinds = new Set([
  "context_update",
  "event",
  "user_utterance",
  "agent_response",
  "pending_action",
  "confirmation",
  "tool_result",
]);
const timeline = document.getElementById("timeline");
for (const record of data.trace.filter((item) => visibleKinds.has(item.kind))) {
  const row = element("div", "timeline-item");
  row.append(
    element("time", "", `${record.at_ms} ms`),
    element("span", "timeline-kind", record.kind),
    element("span", "timeline-data", valueText(record.data)),
  );
  timeline.append(row);
}

const assertionRoot = document.getElementById("assertions");
for (const assertion of summary.assertions) {
  const row = element("div", "assertion");
  row.append(
    element("span", "", assertion.name),
    element("strong", assertion.passed ? "pass" : "fail", assertion.passed ? "PASS" : "FAIL"),
  );
  assertionRoot.append(row);
}

const provenanceRoot = document.getElementById("provenance");
for (const [label, value] of Object.entries({
  "Git commit": data.provenance.git_commit,
  "Created at": data.provenance.created_at_utc,
  "Config SHA-256": data.config_sha256,
})) {
  const row = element("div", "provenance-row");
  row.append(element("span", "", label), element("code", "", valueText(value)));
  provenanceRoot.append(row);
}

const limitations = document.getElementById("limitations");
for (const item of data.limitations) limitations.append(element("li", "", item));
