# A3 LightRAG Knowledge Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an opt-in, reproducible A3 knowledge retrieval path in which LightRAG supplies scoped evidence and the existing VehicleMind Agent decides how to answer.

**Architecture:** Run two isolated LightRAG REST sidecars with fixed ports, data directories and indexes (`vehicle_common` and `vehiclemind_demo`). VehicleMind maps a validated profile to one allowlisted endpoint, calls `/query/data` in `mix` and context-only mode through an adapter, and exposes that retrieval as an on-demand read-only Agent tool. Existing Agent remains responsible for final Chinese responses, safety policy and vehicle actions; the offline replay path remains independent of LightRAG.

**Tech Stack:** Python 3.13 main application, existing ToolRegistry/VehicleAgent/Qwen/GLM clients, YAML configuration, pytest, Ruff and source-size checks; isolated pinned LightRAG REST sidecar (`lightrag-hku==1.5.7` subject to API contract smoke test), DashScope-compatible embedding endpoint/model configured from environment, JSON/YAML/Markdown knowledge catalog, deterministic Python evaluation, JSON/JSONL traces and Chinese HTML report.

## Global Constraints

- Keep main application Python `>=3.13,<3.14`; do not add LightRAG dependencies to its dependency set.
- Run fixed separate REST services for `vehicle_common` and `vehiclemind_demo`; do not rely on per-request workspace headers or user-provided URLs.
- Unknown or absent profile can query only the common endpoint; no demo-only evidence may be retrieved or cited.
- Use LightRAG context-only retrieval; never use its generated answer as VehicleMind's final answer.
- Knowledge retrieval is read-only and cannot create `PendingAction`, authorize tools, or override A2 safety policy.
- Only send relevant context fields whose quality is `KNOWN`; keep live perception, user statements, tool results and static knowledge distinguishable.
- Keep the existing no-key offline replay runnable without LightRAG, model downloads, or service processes.
- Keep each new Python module/script below 500 lines and the demo entry point below 300 lines; split by responsibility.
- Freeze 30 internal questions: 20 answerable, 5 no-answer, 5 profile-mismatch. They are AI-assisted internal cases, not independent human gold labels.
- Report measured calls/latency and provider-reported VehicleMind Agent tokens when available. LightRAG internal token usage is `N/A` unless its service/provider response explicitly exposes it; do not infer token/cost totals.
- Preserve failed trials and source manifests; do not tune the question set or metric denominators after seeing results.

---

## File Map

| Path | Responsibility |
|---|---|
| `modules/vehicle_ai/knowledge/models.py` | Small typed results: source reference, retrieved chunk, retrieval result and typed retrieval error. |
| `modules/vehicle_ai/knowledge/catalog.py` | Load/validate source manifest, content hashes, profile labels and source IDs. |
| `modules/vehicle_ai/knowledge/profile_router.py` | Resolve only configured profiles to fixed local endpoints. |
| `modules/vehicle_ai/knowledge/lightrag_client.py` | HTTP contract, context-only query, response validation and reference mapping. |
| `modules/vehicle_ai/knowledge/index_builder.py` | Upload catalog documents to the correct service, poll status, produce manifest. |
| `modules/vehicle_ai/knowledge/tool.py` | Adapt the client to one read-only `ToolDefinition`; compact evidence for Agent. |
| `modules/vehicle_ai/knowledge/trace.py` | Persist retrieval request, returned candidates, citations and measured timings. |
| `modules/vehicle_ai/evaluation/rag_cases.py` | Frozen 30-case test set and expected evidence/profile. |
| `modules/vehicle_ai/evaluation/rag_metrics.py` | Recall@5, Citation Support review fields, abstention and profile leakage metrics. |
| `modules/vehicle_ai/evaluation/rag_report.py` | Chinese HTML report rendering and evidence/source drilldown. |
| `modules/config/knowledge.yaml` | Fixed endpoints, timeouts, top-k and corpus/index paths; no secrets. |
| `modules/config/agent_policy.yaml` | Read-only risk declaration for `search_vehicle_knowledge`. |
| `config/knowledge/` | Source manifest plus public Markdown texts, stable source IDs and licensing notes. |
| `tools/lightrag/pyproject.toml`, `tools/lightrag/uv.lock` | Isolated pinned LightRAG service environment. |
| `tools/lightrag/config/`, `scripts/lightrag_*.py` | Sidecar configuration, lifecycle, health checks and index commands. |
| `tests/vehicle_ai/knowledge/` | Unit/contract/integration tests with fake HTTP and optional live smoke markers. |
| `tests/vehicle_ai/evaluation/test_rag_*.py` | Frozen case, metric and report tests. |
| `docs/`, `README.md`, `todolist.md` | Local setup, opt-in commands, limitations, verified progress. |

## Shared Interfaces

```python
@dataclass(frozen=True)
class KnowledgeReference:
    source_id: str
    title: str
    section: str
    source_uri: str
    profile: str

@dataclass(frozen=True)
class RetrievedChunk:
    source_id: str
    text: str
    rank: int
    reference: KnowledgeReference

@dataclass(frozen=True)
class RetrievalResult:
    profile: str
    query: str
    chunks: tuple[RetrievedChunk, ...]
    latency_ms: float
    request_id: str
    error_code: str | None = None
    error_message: str | None = None

class ProfileRouter:
    def resolve(self, profile: str | None) -> tuple[str, str]: ...
    # Returns (effective_profile, fixed_base_url), never a URL from caller input.

class LightRAGClient:
    def query(self, profile: str | None, query: str, *, top_k: int = 5) -> RetrievalResult: ...

class KnowledgeCatalog:
    def load(self, manifest_path: Path) -> tuple[KnowledgeSource, ...]: ...
    def validate(self, sources: Sequence[KnowledgeSource]) -> None: ...
```

`ToolDefinition` arguments expose only `{ "query": string }`; profile and KNOWN context are injected by VehicleMind. Tool result returns concise context plus stable `[Kxx]` source IDs and a machine-readable evidence list. It never accepts `profile`, `workspace`, `base_url`, file path or action arguments from the model.

---

### Task 1: Define Retrieval Models, Catalog and Fixed Profile Routing

**Files:**
- Create: `modules/vehicle_ai/knowledge/__init__.py`
- Create: `modules/vehicle_ai/knowledge/models.py`
- Create: `modules/vehicle_ai/knowledge/catalog.py`
- Create: `modules/vehicle_ai/knowledge/profile_router.py`
- Create: `modules/config/knowledge.yaml`
- Test: `tests/vehicle_ai/knowledge/test_knowledge_catalog.py`
- Test: `tests/vehicle_ai/knowledge/test_profile_router.py`

**Interfaces:** Provide the `KnowledgeReference`, `RetrievedChunk`, `RetrievalResult`, `KnowledgeSource`, `KnowledgeCatalog.load/validate`, and `ProfileRouter.resolve` interfaces defined above. `resolve(None)` and any unrecognized profile return `("vehicle_common", common_url)`.

- [x] **Step 1: Write catalog and routing tests** covering duplicate IDs, missing fields, incorrect SHA-256, missing source text, profile labels outside the allowlist, unknown profile fallback and attempts to inject URLs.
- [x] **Step 2: Run focused tests and confirm failure** with `python -m pytest tests/vehicle_ai/knowledge/test_knowledge_catalog.py tests/vehicle_ai/knowledge/test_profile_router.py -q`; observed: collection failed because the knowledge package did not exist.
- [x] **Step 3: Implement immutable models, strict catalog parsing and allowlisted routing.** Use `yaml.safe_load`, resolve source text paths under `config/knowledge/` only, recompute SHA-256 from UTF-8 bytes, reject traversal and reject duplicate IDs. Configuration defines exactly `vehicle_common` and `vehiclemind_demo` base URLs and a positive timeout; secrets are not stored there.
- [x] **Step 4: Run focused tests and lint**; observed: 26 focused tests and 493 full tests passed; Ruff, mypy and source-size checks passed.
- [x] **Step 5: Commit** as `feat(rag): add scoped knowledge catalog and profile router`.

### Task 2: Pin and Smoke-Test the Isolated LightRAG Sidecars

**Files:**
- Create: `tools/lightrag/pyproject.toml`
- Create: `tools/lightrag/uv.lock`
- Create: `tools/lightrag/config/common.env.example`
- Create: `tools/lightrag/config/demo.env.example`
- Create: `scripts/lightrag_services.py`
- Create: `tests/vehicle_ai/knowledge/test_lightrag_service_config.py`
- Create: `tests/vehicle_ai/knowledge/test_lightrag_contract_live.py`

**Interfaces:** `lightrag_services.py` exposes `validate_sidecar_config`, `start_services`, `stop_services`, and `healthcheck_services`; commands accept no arbitrary remote host and use only the two fixed loopback ports configured in `modules/config/knowledge.yaml`.

- [x] **Step 1: Write config and service contract tests** requiring two distinct ports, two distinct work directories, fixed profile mapping, env-only API keys, and explicit refusal of a service using a mismatched index directory.
- [x] **Step 2: Run focused tests**; observed: collection failed because the sidecar manager module did not exist.
- [x] **Step 3: Add a separate uv project pinned to `lightrag-hku==1.5.7`.** Generated `uv.lock` and installed the isolated environment; OpenAI-compatible Qwen and embedding bindings come from existing environment variables.
- [x] **Step 4: Implement lifecycle/health checks** with fixed process scope, startup timeout and environment-only credentials. Ordinary offline commands do not start services.
- [x] **Step 5: Run optional live API contract smoke test** against both loopback services. Observed: both services became healthy; `vehiclemind_demo` upload/status/`mix` context-only query passed; `vehicle_common` upload and indexing passed, an initial 45-second client query timeout occurred, and a subsequent query with a 180-second timeout returned one chunk and one reference. The temporary smoke indexes were moved out of the project runtime directory.
- [x] **Step 6: Run unit tests and commit** as `build(rag): isolate and pin LightRAG sidecars`.

### Task 3: Build a Traceable 20-Source Knowledge Corpus

**Files:**
- Create: `config/knowledge/source_catalog.yaml`
- Create: `config/knowledge/common/` public source Markdown files
- Create: `config/knowledge/demo/` VehicleMind demo-specific source Markdown files
- Create: `scripts/validate_knowledge_catalog.py`
- Test: `tests/vehicle_ai/knowledge/test_public_knowledge_catalog.py`

**Interfaces:** Each `KnowledgeSource` serializes `source_id`, `title`, `source_uri`, `section`, `version`, `published_at`, `profile`, `topic`, `text_path`, `sha256`, `license_note`, and `status`. The catalog contains at least 20 sources; common sources are copied byte-for-byte into both service manifests with matching IDs/hashes.

- [x] **Step 1: Write catalog tests** for minimum count, required provenance, license note, hash integrity and disjoint common/demo source IDs. Common-source duplication into both service indexes is verified in Task 4.
- [x] **Step 2: Run the test**; observed: both tests failed because the catalog was absent.
- [x] **Step 3: Create the corpus** from locatable NHTSA/CDC guidance and a fixed VehicleMind commit; label simulated capabilities as project design, not a real vehicle manual. Source metadata and Markdown body are separate; IDs are `K001`–`K020`.
- [x] **Step 4: Run `python scripts/validate_knowledge_catalog.py`**; observed: 20 valid sources (10 common, 10 demo), manifest SHA-256 printed, no missing section/version/license metadata.
- [x] **Step 5: Review every public text for attribution and commit** as `docs(rag): add sourced vehicle knowledge catalog`.

### Task 4: Index Management with Atomic Publication

**Files:**
- Create: `modules/vehicle_ai/knowledge/index_builder.py`
- Create: `scripts/build_knowledge_indexes.py`
- Create: `tests/vehicle_ai/knowledge/test_index_builder.py`
- Modify: `.gitignore`

**Interfaces:** `IndexBuilder.build(profile: str, sources: Sequence[KnowledgeSource]) -> IndexBuildResult`; only the two configured profiles are accepted. `IndexBuildResult` contains profile, source count, manifest SHA-256, indexed timestamp, LightRAG version and success status. Sidecar databases, graph/vector storage and process logs are ignored by Git.

- [x] **Step 1: Write fake-HTTP tests** for profile corpus, upload, status, timeout, rejected upload, unknown profile and preservation of old published index.
- [x] **Step 2: Run tests**; observed: collection failed because the index builder module was absent.
- [x] **Step 3: Implement build flow** in a staging LightRAG service; publish the entire completed storage/input generation only after every source is processed, retaining a recoverable prior generation.
- [x] **Step 4: Run tests and size check**; observed: 8 focused tests passed, Ruff/mypy/source-size passed. Real common index processed 10/10; real demo index processed 20/20. Both formal services started and returned catalog-mapped evidence (`vehicle_common`: K004/K008/K007/K001/K005; `vehiclemind_demo`: K019/K011/K013/K015/K020).
- [x] **Step 5: Commit** as `feat(rag): add reproducible atomic index builds`.

### Task 5: Implement Context-Only HTTP Adapter and Retrieval Trace

**Files:**
- Create: `modules/vehicle_ai/knowledge/lightrag_client.py`
- Create: `modules/vehicle_ai/knowledge/trace.py`
- Test: `tests/vehicle_ai/knowledge/test_lightrag_client.py`
- Test: `tests/vehicle_ai/knowledge/test_retrieval_trace.py`

**Interfaces:** `LightRAGClient.query(profile, query, *, top_k=5)` returns `RetrievalResult`. The request body fixes `mode="mix"`, `only_need_context=true`, `include_references=true`; limit candidates to configured maximum 5. Trace records effective profile, request ID, query, returned source IDs/ranks/text, LightRAG version, elapsed milliseconds and error code.

- [x] **Step 1: Write fake HTTP contract tests** for context-only request, fixed endpoint, catalog mapping, top-k bound, foreign sources, invalid responses, timeout and exclusion of LightRAG answer text.
- [x] **Step 2: Run focused tests**; observed: collection failed because client and trace modules did not exist.
- [x] **Step 3: Implement adapter** with a 120-second bounded query timeout, response validation, stable source lookup and typed errors. The request contains only query text and fixed retrieval options.
- [x] **Step 4: Implement append-only per-run JSONL trace** of request, evidence, error and elapsed time; credentials are absent from the trace schema.
- [x] **Step 5: Run tests and commit** as `feat(rag): query LightRAG evidence with traceable citations`. Observed: 9 focused adapter/trace tests pass; real common-index query returned five catalog-mapped sources, starting with `K004`, in 13,982 ms.

### Task 6: Expose Retrieval as an On-Demand Read-Only Agent Tool

**Files:**
- Create: `modules/vehicle_ai/knowledge/tool.py`
- Modify: `modules/vehicle_ai/tools/__init__.py`
- Modify: `modules/config/agent_policy.yaml`
- Modify: `modules/vehicle_ai/agent/prompts.py`
- Modify: `modules/vehicle_ai/runtime.py`
- Test: `tests/vehicle_ai/knowledge/test_knowledge_tool.py`
- Test: `tests/vehicle_ai/test_knowledge_agent_integration.py`

**Interfaces:** `build_knowledge_tool(client, context_manager, catalog) -> ToolDefinition` defines one read-only function `search_vehicle_knowledge(query: str)`. Its handler injects current validated profile and only relevant context values whose `field_quality` is `KNOWN`; it returns at most five evidence chunks and stable citations. It never receives profile/URL from LLM arguments.

- [x] **Step 1: Write tests** for ordinary greeting, opt-in knowledge tool call, profile/URL argument rejection, stale context omission, read-only execution, retrieval error and no PendingAction.
- [x] **Step 2: Run tests**; observed: integration tests failed on missing `knowledge_profile` runtime parameter and tool unit tests failed on missing module.
- [x] **Step 3: Implement tool factory and policy entry** with query-only schema, read-only risk and no confirmation side effects.
- [x] **Step 4: Update Agent instructions** to require on-demand evidence retrieval, Chinese responses, source citations and separation of live state from static knowledge.
- [x] **Step 5: Run integration and full relevant tests**; observed: 525 passed, 2 optional live tests skipped; Ruff, mypy and source-size gate passed.
- [x] **Step 6: Commit** as `feat(agent): add on-demand read-only knowledge retrieval`.

### Task 7: Freeze Evaluation Set and Compute Evidence Metrics

**Files:**
- Create: `modules/vehicle_ai/evaluation/rag_cases.py`
- Create: `modules/vehicle_ai/evaluation/rag_metrics.py`
- Create: `modules/vehicle_ai/evaluation/rag_runner.py`
- Create: `tests/vehicle_ai/evaluation/test_rag_cases.py`
- Create: `tests/vehicle_ai/evaluation/test_rag_metrics.py`
- Create: `tests/vehicle_ai/evaluation/test_rag_runner.py`
- Create: `config/knowledge/eval_cases.yaml`

**Interfaces:** `RagCase` contains `case_id`, `query`, `profile`, `answerable`, `expected_source_ids`, `expected_abstention_reason`; `compute_recall_at_k(cases, results, k=5) -> Metric`; `compute_abstention_metrics(...)`; `compute_profile_leaks(...) -> int`. Citation Support records facts, citation IDs, support decision, reviewer type and review note; AI assistance is explicitly tagged.

- [x] **Step 1: Write deterministic metric tests** for answerable-only Recall@5, separate abstention groups, scope leakage, fact-level citation support and unavailable token usage.
- [x] **Step 2: Run metric tests**; observed: collection failed because the RAG metric modules did not exist.
- [x] **Step 3: Freeze 30 cases** in YAML: 20 answerable, 5 no-answer, 5 profile mismatch; case file SHA-256 is recorded by the loader.
- [x] **Step 4: Implement deterministic runner and metrics** retaining raw retrieval and Agent outcomes, failures, exact metric counts and provider-reported Agent usage when supplied. Unreviewed citation support and LightRAG internal tokens remain null.
- [x] **Step 5: Run offline fixture tests**; observed: 530 full tests passed, 2 optional online tests skipped; Ruff, mypy and source-size checks passed.
- [x] **Step 6: Commit** as `test(rag): freeze scoped retrieval evaluation and metrics`.

### Task 8: Produce Chinese Evidence Report and Opt-In Demo Workflow

**Files:**
- Create: `modules/vehicle_ai/evaluation/rag_report.py`
- Create: `scripts/run_knowledge_eval.py`
- Create: `docs/guide/knowledge-rag-demo.md`
- Modify: `README.md`
- Modify: `todolist.md`
- Test: `tests/vehicle_ai/evaluation/test_rag_report.py`
- Test: `tests/vehicle_ai/knowledge/test_offline_independence.py`

**Interfaces:** CLI supports `--mode online-rag --provider qwen|glm --profile vehicle_common|vehiclemind_demo`; a distinct `--build-indexes` flag controls index builds. Successful run writes `runs/rag_eval/<run_id>/report.html`, `trial.jsonl`, `summary.json`, source/case hashes and runtime manifest. Report display is Chinese; metric names are English followed by Chinese meanings.

- [x] **Step 1: Write report/offline tests** for answer, sources, quote, tool use, live-state provenance, profile, errors, latency/tokens, 30-case output and default offline independence.
- [x] **Step 2: Run focused tests**; observed: report tests failed collection because the report module was absent.
- [x] **Step 3: Implement Chinese HTML report** with readable answer/evidence cards, source links, collapsed machine details and bilingual metric labels. Show N/A for unavailable tokens and unreviewed metrics.
- [x] **Step 4: Implement opt-in command flow** with fixed services, optional index build, online Agent run, AI-assisted citation and abstention reviews, run snapshots, HTML report and Chinese README/guide instructions.
- [x] **Step 5: Verify end to end**; observed: 538 tests passed, 2 optional live tests skipped; Ruff, mypy, source-size and diff checks passed. Offline replay printed `通过: drowsy-rest-stop`. Both real indexes queried successfully; full 30-case Qwen run produced a report and a separate 5-case scope follow-up produced a second report.
- [x] **Step 6: Update A3 checklist and prepare commit** as `feat(rag): deliver opt-in knowledge-grounded demo and report`.

## Plan Self-Review

- **Spec coverage:** profile isolation, fixed sidecars, source provenance, LightRAG-only evidence, context quality, failure behavior, atomic indexing, on-demand read-only Agent tool, 30-case evaluation, all requested metrics, Chinese report, offline independence and explicit launch flow are assigned to Tasks 1–8.
- **Limitations represented:** 20 sources and 30 AI-assisted cases remain small internal evidence; no independent human gold, no ablation, no production-scale claim, and no fabricated LightRAG token usage.
- **Type consistency:** shared models and `ProfileRouter.resolve` / `LightRAGClient.query` are declared once; index, Agent and evaluation tasks consume those same interfaces.
- **File-size discipline:** responsibilities split across catalog/router/client/index/tool/trace and report/metrics/runner modules; source-size gate is required before completion.
- **Scope:** no new perception models, custom retrieval framework, autonomous driving action authority, or changes to the offline demo path are included.
