"use strict";

const byId = id => document.getElementById(id);
const diagram = new WorkflowDiagram(byId("graph"), byId("node-detail"));
let view = null;
let token = null;
let stream = null;
let selectedTab = "timeline";
let nodeEvents = [];
let localSubmitting = false;
let connected = false;
let seenEvent = 0;
let runCount = 0;

function showError(message) {
  byId("error").textContent = message || "";
  byId("error").hidden = !message;
}

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {cache: "no-store"} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-VehicleMind-Token": token},
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 403 && payload !== undefined && data.error?.includes("会话令牌")) {
      const refreshed = await fetch("/api/bootstrap", {cache: "no-store"});
      const snapshot = await refreshed.json();
      if (refreshed.ok) {
        connect(snapshot);
        throw new Error("会话已变化，页面已同步；请核对当前待确认动作后重新操作。");
      }
    }
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function message(role, text) {
  if (!text) return;
  byId("conversation").querySelector(".empty-chat")?.remove();
  const item = document.createElement("div");
  item.className = `message ${role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "你" : "VEHICLEMIND";
  const body = document.createElement("div");
  body.className = "message-text";
  body.textContent = text;
  item.append(label, body);
  byId("conversation").append(item);
  byId("conversation").scrollTop = byId("conversation").scrollHeight;
}

function hydrateConversation(snapshot) {
  byId("conversation").replaceChildren();
  snapshot.conversation.forEach(item => {
    if (item.role === "user" || item.role === "assistant") message(item.role, item.content);
  });
  if (!byId("conversation").children.length) {
    const empty = document.createElement("div");
    empty.className = "empty-chat";
    const heading = document.createElement("h3");
    heading.textContent = "从一个真实请求开始";
    const help = document.createElement("p");
    help.textContent = "选一个示例，或输入自己的请求。";
    empty.append(heading, help);
    byId("conversation").append(empty);
  }
}

function updateControls() {
  const busy = !!view?.busy || localSubmitting;
  ["send", "new-session", "provider", "recover"].forEach(id => { byId(id).disabled = busy || !connected; });
  const pending = view?.pending_action;
  const expired = pending && Date.now() / 1000 >= pending.expires_at;
  byId("confirm").disabled = busy || !pending || expired || !connected;
  byId("reject").disabled = busy || !pending || expired || !connected;
  if (pending) {
    const seconds = Math.max(0, Math.ceil(pending.expires_at - Date.now() / 1000));
    byId("expiry").textContent = seconds ? `${seconds} 秒后过期` : "已过期，请重新请求";
  }
}

function render(next) {
  view = next;
  const state = view.state;
  const pending = view.pending_action;
  byId("provider").value = view.provider;
  byId("model-badge").textContent = view.provider === "scripted" ? "离线脚本" : view.provider.toUpperCase();
  byId("model-note").textContent = view.provider === "scripted" ? `规则生成 Tool Call · ${view.demo_delay_seconds * 1000} ms 演示模型延迟 · 不调用在线 API` : "在线模型决策 · 使用服务器配置的 API Key";
  byId("thread-id").textContent = `thread_id: ${view.thread_id}`;
  const labels = {COMPLETED: "已完成", AWAITING_CONFIRMATION: "等待确认", AWAITING_INPUT: "等待输入", RUNNING: "执行中", FAILED: "失败", CANCELLED: "已取消", IDLE: "空闲"};
  byId("task-status").textContent = labels[state.task?.status] || state.task?.status || "尚无任务";
  byId("execution-label").textContent = view.busy ? "正在执行" : view.next.includes("await_confirmation") ? "等待人工确认" : view.next.length ? "执行已中断" : "准备就绪";
  byId("active-node").textContent = view.active_node || (view.busy ? "接入请求中" : "等待请求");
  byId("node-count").textContent = `${runCount} 次节点执行`;
  byId("pending").hidden = !pending;
  if (pending) {
    byId("pending-title").textContent = pending.display_text;
    byId("pending-tool").textContent = pending.tool_name;
    byId("pending-arguments").textContent = JSON.stringify({action_id: pending.action_id, arguments: pending.arguments}, null, 2);
  }
  byId("recover").hidden = !view.next.length || view.next.includes("await_confirmation");
  if (view.last_error) showError(view.last_error);
  diagram.updateRoute(state.graph_trace);
  updateControls();
  renderEvidence();
}

function renderEvidence() {
  const area = byId("evidence");
  if (!view) return;
  area.replaceChildren();
  if (selectedTab === "timeline") {
    if (!nodeEvents.length) {
      const empty = document.createElement("p");
      empty.className = "empty-evidence";
      empty.textContent = "执行事件将在这里实时出现。";
      area.append(empty);
    }
    nodeEvents.slice(-100).forEach(event => {
      const row = document.createElement("div");
      row.className = "event-row";
      const time = document.createElement("span");
      time.className = "event-time";
      time.textContent = event.elapsed !== undefined ? `${event.elapsed.toFixed(0)} ms` : new Date(event.timestamp * 1000).toLocaleTimeString("zh-CN", {hour12: false});
      const name = document.createElement("span");
      name.className = "event-name";
      name.textContent = event.name;
      const phase = document.createElement("span");
      phase.className = `event-phase ${event.phase}`;
      phase.textContent = {start: "开始", finish: "完成", waiting: "等待确认", error: "异常", restored: "已恢复"}[event.phase];
      row.append(time, name, phase);
      area.append(row);
    });
    area.scrollTop = area.scrollHeight;
  } else {
    const state = view.state;
    const data = selectedTab === "tools" ? {calls: state.tool_calls, arguments: state.arguments, result: state.tool_result, confirmation: state.confirmation_result, task_tool_results: state.task?.tool_results} : selectedTab === "context" ? {context: state.context, messages: state.messages} : selectedTab === "vehicle" ? state.simulated_vehicle || {message: "尚无执行中的车辆快照"} : {task: state.task, pending_action: view.pending_action, budget: state.budget, model_rounds: state.model_rounds, stop_reason: state.stop_reason, next: view.next};
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(data, null, 2);
    area.append(pre);
  }
}

function consume(event) {
  if (event.id <= seenEvent) return;
  seenEvent = event.id;
  const data = event.payload;
  if (event.kind === "operation_start") {
    showError(null);
    diagram.reset();
    runCount = 0;
    nodeEvents = [];
    view.busy = true;
    if (data.text) message("user", data.text);
    else message("user", {confirm: "确认执行当前动作", reject: "拒绝当前动作", recover: "恢复中断执行"}[data.operation]);
    render(view);
  } else if (event.kind === "node") {
    diagram.event(data);
    if (data.phase === "start") runCount += 1;
    nodeEvents.push({...data, timestamp: event.timestamp});
    view.active_node = data.phase === "start" || data.phase === "waiting" ? data.name : null;
    render(view);
  } else if (event.kind === "state" || event.kind === "operation_end") {
    if (event.kind === "operation_end") {
      diagram.active = data.active_node;
      diagram.waiting = data.next.includes("await_confirmation");
      diagram.paint();
    }
    render(data);
  } else if (event.kind === "response") {
    message("assistant", typeof data.result === "string" ? data.result : data.result.message);
  } else if (event.kind === "operation_error") {
    showError(data.message);
  }
}

function connect(data) {
  if (stream) stream.close();
  token = data.token;
  seenEvent = data.last_event_id || 0;
  runCount = (data.state.graph_trace || []).length;
  nodeEvents = (data.state.graph_trace || []).map(item => ({name: item.node, phase: "restored", elapsed: item.elapsed_ms}));
  diagram.reset(data.state.graph_trace || []);
  diagram.mount(data.graph);
  diagram.active = data.active_node;
  diagram.waiting = data.next.includes("await_confirmation");
  diagram.paint();
  hydrateConversation(data);
  render(data);
  stream = new EventSource(`/api/events?cursor=${seenEvent}`);
  stream.addEventListener("open", () => {
    connected = true;
    byId("connection").textContent = "事件流已连接";
    byId("connection-dot").classList.add("connected");
    updateControls();
  });
  stream.addEventListener("snapshot", event => {
    const snapshot = JSON.parse(event.data);
    if (snapshot.stream_id !== view.stream_id || snapshot.resync) {
      stream.close();
      api("/api/bootstrap").then(connect).catch(error => showError(error.message));
    } else render(snapshot);
  });
  stream.addEventListener("update", event => { consume(JSON.parse(event.data)); });
  stream.addEventListener("error", () => {
    connected = false;
    byId("connection").textContent = "连接中断，正在重连";
    byId("connection-dot").classList.remove("connected");
    updateControls();
  });
}

async function submit(kind, payload) {
  if (localSubmitting || view?.busy || !connected) return false;
  localSubmitting = true;
  updateControls();
  try { await api(`/api/${kind}`, payload); return true; }
  catch (error) { showError(error.message); return false; }
  finally { localSubmitting = false; updateControls(); }
}

byId("chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const text = byId("request").value.trim();
  if (!text) return;
  if (await submit("chat", {text})) byId("request").value = "";
});
byId("request").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); byId("chat-form").requestSubmit(); }
});
document.querySelectorAll("[data-example]").forEach(button => button.addEventListener("click", () => { byId("request").value = button.dataset.example; byId("request").focus(); }));
byId("confirm").addEventListener("click", () => { if (view?.pending_action) submit("confirm", {action_id: view.pending_action.action_id}); });
byId("reject").addEventListener("click", () => { if (view?.pending_action) submit("reject", {action_id: view.pending_action.action_id}); });
byId("recover").addEventListener("click", () => submit("recover", {}));
byId("new-session").addEventListener("click", async () => {
  localSubmitting = true; updateControls();
  try { const data = await api("/api/session", {provider: byId("provider").value}); showError(null); connect(data); }
  catch (error) { showError(error.message); }
  finally { localSubmitting = false; updateControls(); }
});
byId("all-edges").addEventListener("change", event => { diagram.allEdges = event.target.checked; diagram.paint(); });
document.querySelectorAll("[data-tab]").forEach(button => button.addEventListener("click", () => {
  selectedTab = button.dataset.tab;
  document.querySelectorAll("[data-tab]").forEach(item => item.setAttribute("aria-selected", String(item === button)));
  byId("evidence").setAttribute("aria-labelledby", button.id);
  renderEvidence();
}));
setInterval(updateControls, 1000);
api("/api/bootstrap").then(connect).catch(error => showError(error.message));
