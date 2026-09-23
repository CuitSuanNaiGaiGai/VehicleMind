"use strict";

const data = JSON.parse(document.getElementById("vehiclemind-data").textContent);
const summary = data.summary;

const labels = {
  driver: "驾驶员", road: "道路", vehicle: "车辆", cabin: "舱内",
  source: "来源", confidence: "置信度", valid: "有效", at_ms: "观测时间",
  freshness: "新鲜度", presence: "驾驶员存在", state: "驾驶状态",
  risk: "风险等级", eye_closed: "闭眼", eye_closure_seconds: "持续闭眼秒数",
  perclos: "闭眼比例", recent_yawns: "近期哈欠次数", blink_count: "眨眼次数",
  vehicle_count: "车辆数", pedestrian_count: "行人数", rider_count: "骑行者数",
  traffic_light_count: "交通灯数", traffic_sign_count: "交通标志数",
  lane_detected: "检测到车道", drivable_area_detected: "检测到可行驶区域",
  traffic_level: "交通密度", total_objects: "目标总数", speed_kmh: "车速",
  gear: "挡位", navigation_state: "导航状态", navigation_destination: "导航目的地",
  navigation_destination_id: "目的地标识", ac_enabled: "空调开启",
  cabin_temperature_c: "舱内温度", target_temperature_c: "目标温度",
  driver_window_open: "驾驶员车窗开启", media_playing: "媒体播放中",
  media_title: "媒体标题", volume: "音量", processing_ms: "处理耗时",
  old_risk: "原风险等级", new_risk: "新风险等级",
  driver_state: "驾驶状态", vehicle_speed_kmh: "车速", poi_id: "地点标识",
};
const values = {
  PRESENT: "在位", ABSENT: "不在位", NORMAL: "正常", DROWSY: "疲劳",
  DISTRACTED: "分心", UNKNOWN: "未知", LOW: "低", MEDIUM: "中",
  HIGH: "高", CRITICAL: "严重", MODERATE: "中等", IDLE: "未启动",
  ACTIVE: "进行中", "recorded-replay": "录制观测回放",
  recorded_cabin_perception: "录制的舱内观测",
  recorded_driving_perception: "录制的道路观测",
  recorded_vehicle_state: "录制的车辆状态",
  cabin_perception: "舱内感知", driving_perception: "道路感知",
};
const events = {
  DRIVER_PRESENT: "驾驶员在位", DRIVER_STATE_CHANGED: "驾驶状态变化",
  DRIVER_RISK_CHANGED: "驾驶风险变化", HIGH_RISK_DETECTED: "检测到高风险",
  TRAFFIC_LEVEL_CHANGED: "交通密度变化", LANE_LOST: "车道丢失",
  DRIVABLE_AREA_LOST: "可行驶区域丢失",
};
const tools = {
  search_nearby_rest_area: "查找附近服务区",
  start_navigation: "启动导航", cancel_navigation: "取消导航",
  set_driver_window: "设置驾驶员车窗",
};
const kinds = {
  context_update: "上下文更新", event: "风险事件", user_utterance: "用户输入",
  agent_response: "Agent 回复", pending_action: "待确认动作",
  confirmation: "用户确认", tool_result: "工具结果",
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function displayLabel(key) {
  return labels[key] ?? `未翻译字段：${key}`;
}

function valueText(value, key = "") {
  if (value === null) return key === "confidence" ? "未提供" : "缺失";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "string") return values[value] ?? value;
  if (Array.isArray(value)) return value.map((item) => valueText(item)).join("、");
  if (typeof value === "object") return detailText(value);
  return String(value);
}

function detailText(object) {
  return Object.entries(object)
    .map(([key, value]) => `${displayLabel(key)}：${valueText(value, key)}`)
    .join("；");
}

function toolText(name) {
  return tools[name] ?? `未翻译工具：${name}`;
}

function recordText(record) {
  const item = record.data;
  switch (record.kind) {
    case "context_update":
      return `${labels[item.domain] ?? item.domain}；${detailText(item.values ?? {})}；${item.valid ? "有效" : "无效"}`;
    case "event":
      return `${events[item.type] ?? `未翻译事件：${item.type}`}；${detailText(item.data ?? {})}`;
    case "user_utterance": return item.text;
    case "agent_response": return item.summary;
    case "pending_action":
      return `${toolText(item.tool_name)}待确认；${detailText(item.arguments ?? {})}`;
    case "confirmation":
      return `${toolText(item.tool_name)}：${item.success ? "确认成功" : "确认失败"}`;
    case "tool_result":
      return `${toolText(item.name)}：${item.success ? "执行成功" : "执行失败"}${item.error ? `；错误码：${item.error}` : ""}`;
    default: return "详情见原始追踪文件";
  }
}

function assertionText(name) {
  if (name.startsWith("event:")) return `事件：${events[name.slice(6)] ?? name.slice(6)}`;
  if (name.startsWith("tool:")) return `工具：${toolText(name.slice(5))}`;
  if (name.startsWith("vehicle.")) return `车辆：${displayLabel(name.slice(8))}`;
  const known = {
    unauthorized_sensitive_executions: "未经确认的敏感动作执行次数",
    scripted_responses_consumed: "脚本化回复全部消耗",
  };
  return known[name] ?? displayLabel(name);
}

document.getElementById("scenario-description").textContent = data.description;
const status = document.getElementById("status-badge");
status.textContent = summary.passed ? "通过" : "失败";
status.classList.add(summary.passed ? "pass" : "fail");
document.getElementById("cabin-media").src = data.media.cabin;
document.getElementById("road-media").src = data.media.road;

const metrics = [
  ["场景标识", summary.scenario_id],
  ["语义追踪摘要", summary.semantic_sha256.slice(0, 12)],
  ["安全违规次数", summary.unauthorized_sensitive_executions],
  ["运行耗时", `${summary.metrics.total_ms.toFixed(2)} 毫秒`],
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
  card.append(element("span", "label", "统一上下文"));
  card.append(element("h2", "", labels[domain] ?? displayLabel(domain)));
  const observation = data.observations[domain] || data.observations[domain === "driver" ? "cabin" : domain];
  if (observation) {
    for (const [key, value] of Object.entries(observation)) {
      const row = element("div", "context-row observation-row");
      row.append(element("span", "", displayLabel(key)), element("code", "", valueText(value, key)));
      card.append(row);
    }
  }
  for (const [key, value] of Object.entries(values)) {
    const row = element("div", "context-row");
    row.append(element("span", "", displayLabel(key)), element("code", "", valueText(value, key)));
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
    element("time", "", `${record.at_ms} 毫秒`),
    element("span", "timeline-kind", kinds[record.kind]),
    element("span", "timeline-data", recordText(record)),
  );
  timeline.append(row);
}

const assertionRoot = document.getElementById("assertions");
for (const assertion of summary.assertions) {
  const row = element("div", "assertion");
  row.append(
    element("span", "", assertionText(assertion.name)),
    element("strong", assertion.passed ? "pass" : "fail", assertion.passed ? "通过" : "失败"),
  );
  assertionRoot.append(row);
}

const provenanceRoot = document.getElementById("provenance");
for (const [label, value] of Object.entries({
  "Git 提交": data.provenance.git_commit,
  "创建时间": data.provenance.created_at_utc,
  "配置 SHA-256": data.config_sha256,
})) {
  const row = element("div", "provenance-row");
  row.append(element("span", "", label), element("code", "", valueText(value)));
  provenanceRoot.append(row);
}

const limitations = document.getElementById("limitations");
for (const item of data.limitations) limitations.append(element("li", "", item));
