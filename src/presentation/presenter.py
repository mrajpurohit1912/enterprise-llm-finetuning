"""
src/presentation/presenter.py
Presentation layer responsible for console rendering, banners, and scorecards.
Decouples terminal output formatting from business orchestration.
"""

import sys

from src.domain.schemas.compute_schema import RemoteJobResult, RemoteJobStatus
from src.domain.schemas.pipeline_schema import TrainPipelineResult


class ConsolePresenter:
    """Encapsulates all terminal ASCII banners, scorecard rendering, and summaries."""

    @staticmethod
    def render_remote_submission(res: RemoteJobResult, provider_name: str) -> None:
        """Render remote job submission receipt."""
        print("\n=======================================================")
        print("        REMOTE COMPUTE JOB SUBMISSION SUMMARY          ")
        print("=======================================================")
        print(f"Experiment Name   : {res.job_id}")
        print(f"Compute Backend   : {provider_name}")
        print(f"Remote Job ID     : {res.remote_job_id}")
        print(f"Status            : {res.status.value.upper()}")
        print(f"Receipt Location  : outputs/.remote_jobs/{res.job_id}.json")
        print("-------------------------------------------------------")
        print("Run 'python -m src.main --remote-status' to check progress.")
        print("Run 'python -m src.main --remote-watch' to stream and auto-download.")
        print("=======================================================\n")

    @staticmethod
    def render_remote_status(res: RemoteJobResult, provider_name: str) -> None:
        """Render remote job execution status."""
        print("\n=======================================================")
        print("             REMOTE COMPUTE STATUS SUMMARY             ")
        print("=======================================================")
        print(f"Remote Job ID     : {res.remote_job_id}")
        print(f"Compute Backend   : {provider_name}")
        print(f"Status            : {res.status.value.upper()}")
        print("=======================================================\n")

    @staticmethod
    def render_remote_pull(res: RemoteJobResult) -> None:
        """Render remote artifact download confirmation."""
        print("\n=======================================================")
        print("            REMOTE ARTIFACT RETRIEVAL SUMMARY          ")
        print("=======================================================")
        print(f"Remote Job ID     : {res.remote_job_id}")
        print(f"Status            : {res.status.value.upper()}")
        print(f"Downloaded Path   : {res.artifact_path}")
        print("=======================================================\n")

    @staticmethod
    def render_remote_watch_header() -> None:
        """Render remote watch stream header."""
        print("\n=======================================================")
        print("            WATCHING REMOTE COMPUTE EXECUTION          ")
        print("=======================================================")

    @staticmethod
    def render_remote_watch_tick(
        timestamp: str, remote_job_id: str | None, status: str
    ) -> None:
        """Render a single polling tick during watch mode."""
        print(f"[{timestamp}] Job '{remote_job_id}' Status: {status.upper()}")

    @staticmethod
    def render_remote_watch_completion(res: RemoteJobResult) -> None:
        """Render completion status after watch poll loop exits."""
        if res.status == RemoteJobStatus.COMPLETED:
            print(f"\n✔ Job completed! Artifacts downloaded to: {res.artifact_path}")
        else:
            print("\n✖ Job reported failure. Check remote Kaggle logs for details.")
        print("=======================================================\n")

    @staticmethod
    def render_pipeline_summary(result: TrainPipelineResult) -> None:
        """Render comprehensive fine-tuning pipeline scorecard."""
        print("\n=======================================================")
        print("          FINE-TUNING PIPELINE RUN SUMMARY             ")
        print("=======================================================")
        print(f"Experiment Name   : {result.experiment_name}")
        print(f"Execution Status  : {result.status}")
        print(f"Elapsed Time      : {result.duration_seconds}s")
        print(f"Artifacts Output  : {result.output_dir}")
        if result.registered_model_uri:
            print(f"Registered Model  : {result.registered_model_uri}")
        if result.train_loss is not None:
            print(f"Final Train Loss  : {result.train_loss:.4f}")
        if result.dataset_size:
            print(f"Dataset Partitions: {result.dataset_size}")
        if result.evaluation is not None:
            print(f"Task Type         : {result.evaluation.task_type.value}")
            print(f"Evaluation F1     : {result.evaluation.metrics.f1_score:.4f}")
            print(f"Exact Match (EM)  : {result.evaluation.metrics.exact_match:.4f}")
            print(f"ROUGE-L           : {result.evaluation.metrics.rouge_l or 0.0:.4f}")
            if result.evaluation.metrics.tolerance_accuracy_5pct is not None:
                print(
                    f"Tolerance Acc (<=5%): {result.evaluation.metrics.tolerance_accuracy_5pct * 100:.2f}%"
                )
            if result.evaluation.metrics.tolerance_accuracy_1pct is not None:
                print(
                    f"Strict Acc (<=1%)  : {result.evaluation.metrics.tolerance_accuracy_1pct * 100:.2f}%"
                )
            if result.evaluation.metrics.sign_accuracy is not None:
                print(
                    f"Sign Accuracy     : {result.evaluation.metrics.sign_accuracy * 100:.2f}%"
                )
            if result.evaluation.metrics.parseable_number_rate is not None:
                print(
                    f"Parseable Rate    : {result.evaluation.metrics.parseable_number_rate * 100:.2f}%"
                )
            if result.evaluation.metrics.median_relative_error is not None:
                print(
                    f"Median Rel Error  : {result.evaluation.metrics.median_relative_error * 100:.2f}%"
                )
            if result.evaluation.delta_metrics:
                delta_f1 = result.evaluation.delta_metrics.get("delta_f1_score", 0.0)
                delta_em = result.evaluation.delta_metrics.get("delta_exact_match", 0.0)
                print(f"Delta Uplift (ΔF1): {delta_f1:+.4f}")
                print(f"Delta Uplift (ΔEM): {delta_em:+.4f}")
            if result.evaluation.tpot_ms_p50 is not None:
                print(
                    f"TPOT (P50 Decode) : {result.evaluation.tpot_ms_p50:.2f} ms/token"
                )
            if result.evaluation.ttft_ms_p50 is not None:
                print(f"TTFT (P50 Prefill): {result.evaluation.ttft_ms_p50:.2f} ms")
            if result.evaluation.peak_memory_mb is not None:
                print(f"Peak GPU VRAM     : {result.evaluation.peak_memory_mb:.2f} MB")
            gate = "PASSED" if result.evaluation.passed_quality_gate else "FAILED"
            print(f"Quality Gate      : {gate}")
        print("=======================================================\n")

    @staticmethod 
    def render_registry_summary(model_name: str, published_uri: str, provider_name: str) -> None:
        """Render model registry publication confirmation."""
        print("\n=======================================================")
        print("          MODEL REGISTRY PUBLICATION SUMMARY           ")
        print("=======================================================") 
        print(f"Model Identifier  : {model_name}") 
        print(f"Registry Backend  : {provider_name}")
        print(f"Published URI     : {published_uri}")
        print("=======================================================\n")

    @staticmethod
    def render_error(title: str, message: str) -> None:
        """Render domain or fatal error message to stderr."""
        print(f"[{title}] {message}", file=sys.stderr)
