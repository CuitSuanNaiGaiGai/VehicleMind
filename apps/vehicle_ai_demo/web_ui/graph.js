"use strict";

// 拓扑来自编译后的 Graph；这里只定义节点摆放和中文职责标签。
const NODE_INFO = {
  __start__: [40, 23, "入口", "接入本轮请求"],
  __end__: [700, 23, "结束", "本轮执行结束"],
  receive_request: [40, 102, "请求接入", "恢复任务并选择业务路径"],
  build_context: [260, 102, "构造 Context", "按现有规则构造模型输入"],
  decide: [480, 102, "模型决策", "调用当前模型，解析工具调用"],
  respond: [700, 102, "生成响应", "记录回复，路由到确认或结束"],
  validate_tool: [40, 232, "参数校验", "核对原始 JSON、参数与 Schema"],
  prepare_tool: [260, 232, "工具准备", "预算校验、计划推进和敏感动作准备"],
  execute_tool: [480, 232, "工具执行", "经 Tool Registry 和安全策略执行"],
  observe_tool: [700, 232, "观察结果", "将工具结果写回状态并选择下一步"],
  dialogue_interpret: [40, 362, "对话解释", "解释可选业务对话中的意图"],
  dialogue_reduce: [260, 362, "状态归约", "更新业务约束、候选与任务状态"],
  dialogue_execute: [480, 362, "业务对话执行", "处理候选查询或选择操作"],
  retry_tool: [700, 362, "只读重试", "按既有预算执行获准的只读重试"],
  await_confirmation: [40, 482, "等待用户确认", "interrupt 暂停，等待显式恢复输入"],
  confirm_action: [260, 482, "处理确认", "按动作 ID 恢复原始动作并校验授权"],
  invalid_resume: [480, 482, "拒绝无效恢复", "无效恢复输入不会执行敏感操作"],
  reconcile_write: [700, 482, "未知结果回读", "回读车机状态，禁止自动重放写操作"],
};
const MAIN_EDGES = new Set([
  "__start__:receive_request", "receive_request:build_context", "build_context:decide",
  "decide:validate_tool", "validate_tool:prepare_tool", "prepare_tool:execute_tool",
  "execute_tool:observe_tool", "observe_tool:decide", "observe_tool:retry_tool",
  "observe_tool:reconcile_write", "observe_tool:respond", "decide:respond",
  "respond:await_confirmation", "await_confirmation:confirm_action", "confirm_action:__end__",
  "respond:__end__", "retry_tool:observe_tool", "reconcile_write:respond",
  "receive_request:dialogue_interpret", "dialogue_interpret:dialogue_reduce",
  "dialogue_reduce:dialogue_execute", "dialogue_execute:respond", "await_confirmation:receive_request",
]);

function svgElement(tag, attributes = {}, text = null) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  if (text !== null) element.textContent = text;
  return element;
}

class WorkflowDiagram {
  constructor(svg, detail) {
    this.svg = svg;
    this.detail = detail;
    this.visits = new Map();
    this.failures = new Set();
    this.edges = [];
    this.visitedEdges = new Set();
    this.nodes = new Map();
    this.active = null;
    this.waiting = false;
    this.allEdges = false;
  }

  mount(topology) {
    this.svg.replaceChildren();
    this.nodes.clear();
    this.edges = [];
    const defs = svgElement("defs");
    const marker = svgElement("marker", {id: "edge-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse"});
    marker.append(svgElement("path", {d: "M 0 0 L 10 5 L 0 10 z", fill: "#91a38d"}));
    defs.append(marker);
    this.svg.append(defs);
    [ [40, 90, "决策与响应"], [40, 218, "工具调用"], [40, 348, "业务对话 / 恢复"], [40, 468, "人工确认 / 安全执行"] ].forEach(([x, y, text]) => {
      this.svg.append(svgElement("text", {x, y, class: "graph-group-label"}, text));
    });
    topology.edges.forEach(edge => {
      const source = NODE_INFO[edge.source];
      const target = NODE_INFO[edge.target];
      if (!source || !target) return;
      const key = `${edge.source}:${edge.target}`;
      const sx = source[0] + 170, sy = source[1] + 27;
      const tx = target[0], ty = target[1] + 27;
      let path;
      if (source[1] === target[1] && sx < tx) {
        path = `M${sx},${sy} L${tx},${ty}`;
      } else if (source[0] === target[0]) {
        const bottom = source[1] < target[1];
        const x = source[0] + 85;
        path = `M${x},${source[1] + (bottom ? 55 : 0)} L${x},${target[1] + (bottom ? 0 : 55)}`;
      } else {
        const offset = sx < tx ? 45 : 65;
        path = `M${sx},${sy} C${sx + offset},${sy} ${tx - offset},${ty} ${tx},${ty}`;
      }
      const element = svgElement("path", {d: path, class: `graph-edge ${edge.conditional ? "conditional" : ""}`, "marker-end": "url(#edge-arrow)"});
      element.append(svgElement("title", {}, `${edge.source} → ${edge.target}`));
      this.svg.append(element);
      this.edges.push({key, element});
    });
    topology.nodes.forEach(name => {
      const info = NODE_INFO[name];
      if (!info) return;
      const [x, y, label, description] = info;
      const group = svgElement("g", {class: "graph-node", transform: `translate(${x},${y})`, tabindex: "0", role: "button", "aria-label": `${label}：${name}`});
      group.append(svgElement("rect", {width: 170, height: 55}));
      group.append(svgElement("text", {x: 12, y: 23, class: "label"}, label));
      group.append(svgElement("text", {x: 12, y: 42, class: "name"}, name));
      const counter = svgElement("text", {x: 158, y: 23, class: "count", "text-anchor": "end"});
      group.append(counter);
      group.append(svgElement("title", {}, description));
      const select = () => { this.detail.textContent = `${label}：${description}`; };
      group.addEventListener("click", select);
      group.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); }
      });
      this.svg.append(group);
      this.nodes.set(name, {group, counter});
    });
    this.paint();
  }

  reset(trace = []) {
    this.visits.clear();
    this.failures.clear();
    this.visitedEdges.clear();
    this.active = null;
    this.waiting = false;
    trace.forEach(event => {
      this.visits.set(event.node, (this.visits.get(event.node) || 0) + 1);
      if (event.route) this.visitedEdges.add(`${event.node}:${event.route === "end" ? "__end__" : event.route}`);
    });
    this.paint();
  }

  event(event) {
    if (event.phase === "start") {
      this.active = event.name;
      this.waiting = false;
    } else if (event.phase === "waiting") {
      this.active = event.name;
      this.waiting = true;
    } else {
      this.visits.set(event.name, (this.visits.get(event.name) || 0) + 1);
      this.active = null;
      if (event.phase === "error") this.failures.add(event.name);
    }
    this.paint();
  }

  updateRoute(trace) {
    const last = trace?.at(-1);
    if (last?.route) this.visitedEdges.add(`${last.node}:${last.route === "end" ? "__end__" : last.route}`);
    this.paint();
  }

  paint() {
    this.nodes.forEach(({group, counter}, name) => {
      group.classList.toggle("visited", this.visits.has(name));
      group.classList.toggle("failed", this.failures.has(name));
      group.classList.toggle("running", this.active === name && !this.waiting);
      group.classList.toggle("waiting", this.active === name && this.waiting);
      counter.textContent = this.visits.has(name) ? `×${this.visits.get(name)}` : "";
    });
    this.edges.forEach(({key, element}) => {
      element.classList.toggle("current-path", this.visitedEdges.has(key));
      element.style.display = this.allEdges || MAIN_EDGES.has(key) || this.visitedEdges.has(key) ? "" : "none";
    });
  }
}
