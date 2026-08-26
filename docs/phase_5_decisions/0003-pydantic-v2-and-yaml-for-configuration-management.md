# ADR-0003: Pydantic v2 and YAML for Configuration Engine

* **Status:** Accepted
* **Deciders:** Lead ML Architect, Backend Lead
* **Date:** 2026-08-25

## Context and Problem Statement
Fine-tuning experiments involve over 30 interrelated hyperparameters across data, tokenizer, precision, hardware, and optimizer. Passing arguments through long CLI flags (`--lr 0.0002 --r 16 --alpha 32 ...`) is error-prone, untracked, and non-reproducible. We need a declarative, validated, version-controllable configuration mechanism.

## Decision Drivers
* **Pre-flight Validation:** Fail instantly with clear diagnostic messages if a user supplies invalid types or missing required fields before allocating expensive GPU resources.
* **Immutability:** Configurations must remain frozen (`frozen=True`) during execution to prevent accidental runtime state corruption.
* **Readability:** Human-readable configuration format easily versioned in Git.

## Considered Options
1. **Plain Python Config Files (`config.py`):** Dynamic but allows runtime mutation and lacks schema validation.
2. **Hydra / OmegaConf:** Powerful for research, but introduces complex hierarchical syntax and runtime magic.
3. **Pydantic v2 + YAML:** High-performance Rust-backed schema validation with declarative YAML files.

## Decision Outcome
Chosen Option: **Pydantic v2 with YAML files (`YamlConfigLoader`)**.
* `src/domain/schemas/config_schema.py` defines immutable Pydantic models.
* `src/infrastructure/config/yaml_loader.py` reads YAML into strongly-typed `ExperimentConfig`.

### Positive Consequences
* Catches configuration errors in $< 5\text{ms}$ before GPU initialization.
* Auto-generates JSON Schema for IDE autocomplete and web UI form builders.
