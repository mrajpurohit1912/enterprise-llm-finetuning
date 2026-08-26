# Phase 6: Decompose
# Dependency Graph, Critical Path & Milestone Schedule

**Document Version:** 1.0.0  
**Status:** Approved  

---

## 1. Task Dependency Graph

The following Mermaid graph displays the exact prerequisites and dependencies between developer tasks:

```mermaid
flowchart TD
    %% Milestone 1
    subgraph M1["Milestone 1: Core Foundation (PR #1)"]
        CORE101["CORE-101: Pydantic Schemas"]
        CORE102["CORE-102: Domain Interfaces"]
        CORE101 --> CORE103["CORE-103: YamlConfigLoader"]
        CORE102 --> CORE103
    end

    %% Milestone 2
    subgraph M2["Milestone 2: Parallel Track Development (PR #2 - #5)"]
        CORE102 --> DATA201["DATA-201: HF Dataset Loader"]
        CORE102 --> DATA202["DATA-202: Chat Formatter"]
        DATA201 --> DATA203["DATA-203: DatasetProcessor"]
        DATA202 --> DATA203

        CORE102 --> MODEL301["MODEL-301: 4-bit ModelLoader"]
        CORE101 --> MODEL302["MODEL-302: PEFT LoRA Config"]
        
        CORE102 --> OBS401["OBS-401: Prometheus Callback"]
        CORE101 --> OBS402["OBS-402: W&B Telemetry"]
        CORE102 --> REG403["REG-403: Safetensors Registry"]

        CORE101 --> OPS503["OPS-503: GitHub Actions CI"]
    end

    %% Milestone 3
    subgraph M3["Milestone 3: Pipeline Integration (PR #6)"]
        DATA203 --> TRAIN303["TRAIN-303: SFTTrainer & Pipeline Usecase"]
        MODEL301 --> TRAIN303
        MODEL302 --> TRAIN303
        OBS401 --> TRAIN303
        OBS402 --> TRAIN303
        REG403 --> TRAIN303
    end

    %% Milestone 4
    subgraph M4["Milestone 4: Serving & Release (PR #7 - #8)"]
        TRAIN303 --> SERV501["SERV-501: Async vLLM Service"]
        TRAIN303 --> CLI502["CLI-502: Presentation CLI"]
        SERV501 --> RELEASE["Platform v1.0.0 Release"]
        CLI502 --> RELEASE
    end
```

---

## 2. Critical Path Analysis

The **Critical Path** (the longest sequence of dependent activities determining the minimum project duration) is:

$$\text{CORE-101/102} \longrightarrow \text{DATA-201/202} \longrightarrow \text{DATA-203} \longrightarrow \text{TRAIN-303} \longrightarrow \text{SERV-501} \longrightarrow \text{RELEASE}$$

* **Total Estimated Velocity:** 2 Sprints (4 weeks total for a 4-engineer team).
* **Risk Mitigation:** Developers on Track 4 (Observability) and Track 5 (CI/CD) run 100% in parallel without blocking the critical path.
