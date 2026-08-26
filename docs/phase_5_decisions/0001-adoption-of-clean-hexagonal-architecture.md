# ADR-0001: Adoption of Clean / Hexagonal Architecture

* **Status:** Accepted
* **Deciders:** Lead ML Architect, Principal Engineer, Platform Team
* **Date:** 2026-08-25

## Context and Problem Statement
Machine learning fine-tuning repositories frequently degrade into monolithic, untestable scripts where Hugging Face API calls, data downloads, training loops, and hardware setup are entangled. When external libraries update their APIs or when new data sources (S3, Snowflake) are introduced, monolithic scripts require full rewrites and break existing workflows.

## Decision Drivers
* **Maintainability:** Clear separation between business rules and third-party ML libraries.
* **Testability:** Ability to unit test data loaders and pipeline use cases on CPU runners in CI/CD using lightweight mocks without downloading gigabytes of weights.
* **Team Velocity:** Allow multiple engineers to work in parallel on data ingestion, model strategies, and telemetry without merge conflicts.

## Considered Options
1. **Procedural Scripting (Notebooks / Flat Scripts):** Fast for PoCs, disastrous for enterprise platforms.
2. **Layered 3-Tier Architecture:** Controller $\to$ Service $\to$ Data Access (Leads to database/library coupling).
3. **Clean / Hexagonal Architecture (Ports & Adapters):** Domain $\to$ Application $\to$ Infrastructure $\to$ Presentation.

## Decision Outcome
Chosen Option: **Clean / Hexagonal Architecture (Ports & Adapters)**.
* Core domain rules and schemas live in `src/domain/` with zero third-party framework dependencies.
* Business workflows live in `src/application/usecases/` and `src/application/services/`.
* External libraries (`transformers`, `boto3`, `trl`) are strictly encapsulated in `src/infrastructure/` behind domain interfaces.

### Positive Consequences
* Adapters (e.g. `S3DatasetLoader`, `HuggingFaceDatasetLoader`) can be swapped seamlessly via configuration.
* Unit test coverage can reach $\ge 80\%$ without needing live GPU clusters.

### Negative Consequences
* Slightly higher initial boilerplate (interfaces, factories, dependency injection wiring).
