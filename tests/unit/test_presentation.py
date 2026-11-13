"""
tests/unit/test_presentation.py
Unit tests for the presentation layer: CLI parsing, composition root, handlers, and presenter.
"""

import argparse
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.domain.exceptions import DomainError
from src.domain.schemas.compute_schema import RemoteJobResult, RemoteJobStatus
from src.domain.schemas.config_schema import (
    DatasetConfig,
    DatasetSourceType,
    ExperimentConfig,
    ExperimentInfo,
    LLMModelConfig,
    MonitoringConfig,
    TrainingArgs,
)
from src.domain.schemas.evaluation_schema import (
    EvaluationConfig,
    EvaluationResult,
    MetricScores,
    TaskType,
)
from src.domain.schemas.pipeline_schema import TrainPipelineResult
from src.presentation.cli import main, parse_args, run_cli
from src.presentation.composition_root import (
    build_merge_usecase,
    build_remote_usecases,
    build_train_pipeline,
)
from src.presentation.handlers import (
    handle_merge_action,
    handle_pipeline_action,
    handle_remote_action,
    is_remote_action,
)
from src.presentation.presenter import ConsolePresenter


class TestPresentationLayer(unittest.TestCase):
    """Unit tests for the presentation layer components."""

    def setUp(self) -> None:
        self.sample_config = ExperimentConfig(
            experiment=ExperimentInfo(name="unit-test-exp"),
            dataset=DatasetConfig(
                source=DatasetSourceType.HUGGINGFACE, dataset_name="test/dataset"
            ),
            llm_model=LLMModelConfig(llm_model_id="test/model"),
            training_args=TrainingArgs(),
            monitoring=MonitoringConfig(),
            evaluation=EvaluationConfig(evaluator_type="deterministic"),
        )

    # -------------------------------------------------------------------------
    # Argument Parsing Tests
    # -------------------------------------------------------------------------
    def test_parse_args_defaults(self) -> None:
        """Default arguments should match configuration conventions."""
        args = parse_args([])
        self.assertEqual(args.config, "src/finetuning_config.yaml")
        self.assertEqual(args.remote_provider, "kaggle")
        self.assertFalse(args.eval_only)
        self.assertFalse(args.merging_only)
        self.assertFalse(args.remote_submit)
        self.assertFalse(args.remote_status)
        self.assertFalse(args.remote_pull)
        self.assertFalse(args.remote_watch)

    def test_parse_args_custom_flags(self) -> None:
        """Custom flags should properly populate the namespace."""
        args = parse_args(
            ["--config", "custom.yaml", "--eval-only", "--remote-job-id", "job123"]
        )
        self.assertEqual(args.config, "custom.yaml")
        self.assertTrue(args.eval_only)
        self.assertEqual(args.remote_job_id, "job123")

    # -------------------------------------------------------------------------
    # ConsolePresenter Tests
    # -------------------------------------------------------------------------
    def test_render_remote_submission(self) -> None:
        """Render remote submission should print key fields."""
        res = RemoteJobResult(
            job_id="exp-1", remote_job_id="user/exp-1", status=RemoteJobStatus.QUEUED
        )
        with patch("builtins.print") as mock_print:
            ConsolePresenter.render_remote_submission(res, "kaggle")
            self.assertTrue(mock_print.called)

    def test_render_pipeline_summary(self) -> None:
        """Render pipeline summary should print execution metrics and evaluation scorecard."""
        result = TrainPipelineResult(
            experiment_name="test-exp",
            status="SUCCESS",
            output_dir=Path("outputs/test"),
            duration_seconds=12.5,
            train_loss=0.4567,
            dataset_size={"train": 100},
            evaluation=EvaluationResult(
                experiment_name="test-exp",
                model_id="test/model",
                eval_split="test",
                total_samples=10,
                latency_per_sample_ms=50.0,
                throughput_tokens_per_sec=20.0,
                task_type=TaskType.TEXT_GENERATION,
                metrics=MetricScores(f1_score=0.95, exact_match=0.90, rouge_l=0.92),
                passed_quality_gate=True,
            ),
        )
        with patch("builtins.print") as mock_print:
            ConsolePresenter.render_pipeline_summary(result)
            self.assertTrue(mock_print.called)

    def test_render_error(self) -> None:
        """Error rendering should output to stderr."""
        with patch("sys.stderr.write") as mock_stderr:
            ConsolePresenter.render_error("Domain Error", "Test error message")
            # print(..., file=sys.stderr) calls write on stderr
            self.assertTrue(mock_stderr.called)

    # -------------------------------------------------------------------------
    # Composition Root Tests
    # -------------------------------------------------------------------------
    @patch("src.presentation.composition_root.TransformerLlmModelLoader")
    @patch("src.presentation.composition_root.HuggingFaceTokenizer")
    def test_build_merge_usecase(
        self, mock_tok: MagicMock, mock_llm: MagicMock
    ) -> None:
        """Composition root should construct MergeModelUseCase with loaders."""
        usecase = build_merge_usecase(self.sample_config)
        self.assertIsNotNone(usecase)

    @patch("src.presentation.composition_root.ComputeProviderFactory.get_provider")
    def test_build_remote_usecases(self, mock_get_provider: MagicMock) -> None:
        """Composition root should instantiate submit and sync use cases."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        submit_uc, sync_uc, provider = build_remote_usecases("kaggle")
        self.assertEqual(provider, mock_provider)
        self.assertEqual(submit_uc._provider, mock_provider)
        self.assertEqual(sync_uc._provider, mock_provider)

    @patch("src.presentation.composition_root.DatasetLoaderFactory")
    @patch("src.presentation.composition_root.HuggingFaceTokenizer")
    @patch("src.presentation.composition_root.TrainerFactory.get_trainer")
    @patch("src.presentation.composition_root.EvaluatorFactory.get_evaluator")
    @patch("src.presentation.composition_root.TransformerLlmModelLoader")
    def test_build_train_pipeline(
        self,
        mock_llm: MagicMock,
        mock_eval_fac: MagicMock,
        mock_trainer_fac: MagicMock,
        mock_tok: MagicMock,
        mock_ds: MagicMock,
    ) -> None:
        """Composition root should wire all adapters into TrainPipelineUsecase."""
        mock_loader = MagicMock()
        pipeline = build_train_pipeline(self.sample_config, mock_loader)
        self.assertIsNotNone(pipeline)

    # -------------------------------------------------------------------------
    # Action Handlers Tests
    # -------------------------------------------------------------------------
    def test_is_remote_action(self) -> None:
        """is_remote_action returns True only if a remote flag is present."""
        args_no = argparse.Namespace(
            remote_submit=False,
            remote_status=False,
            remote_pull=False,
            remote_watch=False,
        )
        self.assertFalse(is_remote_action(args_no))

        args_yes = argparse.Namespace(
            remote_submit=True,
            remote_status=False,
            remote_pull=False,
            remote_watch=False,
        )
        self.assertTrue(is_remote_action(args_yes))

    @patch("src.presentation.handlers.build_merge_usecase")
    def test_handle_merge_action(self, mock_build: MagicMock) -> None:
        """handle_merge_action delegates to MergeModelUseCase."""
        mock_uc = MagicMock()
        mock_build.return_value = mock_uc

        code = handle_merge_action(self.sample_config)
        self.assertEqual(code, 0)
        mock_uc.merge_and_unload_model.assert_called_once()

    @patch("src.presentation.handlers.build_train_pipeline")
    def test_handle_pipeline_action(self, mock_build: MagicMock) -> None:
        """handle_pipeline_action runs pipeline and renders scorecard."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock()
        mock_build.return_value = mock_pipeline
        mock_presenter = MagicMock()

        code = handle_pipeline_action(
            config=self.sample_config,
            config_loader=MagicMock(),
            eval_only=False,
            presenter=mock_presenter,
        )
        self.assertEqual(code, 0)
        mock_pipeline.run.assert_called_once_with(eval_only=False)
        mock_presenter.render_pipeline_summary.assert_called_once()

    @patch("src.presentation.handlers.build_remote_usecases")
    def test_handle_remote_submit(self, mock_build_remote: MagicMock) -> None:
        """handle_remote_action submit workflow."""
        mock_submit = MagicMock()
        mock_submit.execute.return_value = RemoteJobResult(
            job_id="test", remote_job_id="test/1", status=RemoteJobStatus.QUEUED
        )
        mock_sync = MagicMock()
        mock_provider = MagicMock(provider_name="kaggle")
        mock_build_remote.return_value = (mock_submit, mock_sync, mock_provider)

        args = argparse.Namespace(
            remote_submit=True,
            remote_status=False,
            remote_pull=False,
            remote_watch=False,
            remote_provider="kaggle",
        )

        with patch("src.presentation.handlers.YamlConfigLoader") as mock_yaml:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_config.return_value = self.sample_config
            mock_yaml.return_value = mock_loader_instance

            mock_presenter = MagicMock()
            code = handle_remote_action(
                args, Path("dummy.yaml"), presenter=mock_presenter
            )
            self.assertEqual(code, 0)
            mock_submit.execute.assert_called_once()
            mock_presenter.render_remote_submission.assert_called_once()

    # -------------------------------------------------------------------------
    # CLI Controller & Exit Codes Tests
    # -------------------------------------------------------------------------
    @patch("src.presentation.cli.handle_pipeline_action")
    @patch("src.presentation.cli.YamlConfigLoader")
    def test_run_cli_success(
        self, mock_loader_cls: MagicMock, mock_handle: MagicMock
    ) -> None:
        """run_cli should return 0 on successful pipeline run."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_config.return_value = self.sample_config
        mock_loader_cls.return_value = mock_loader_instance
        mock_handle.return_value = 0

        exit_code = run_cli(["--config", "src/finetuning_config.yaml"])
        self.assertEqual(exit_code, 0)

    @patch("src.presentation.cli.YamlConfigLoader")
    def test_run_cli_domain_error(self, mock_loader_cls: MagicMock) -> None:
        """run_cli should catch DomainError, render message, and return 1."""
        mock_loader_cls.side_effect = DomainError("Configuration file missing")
        mock_presenter = MagicMock()

        exit_code = run_cli(["--config", "nonexistent.yaml"], presenter=mock_presenter)
        self.assertEqual(exit_code, 1)
        mock_presenter.render_error.assert_called_once_with(
            "Domain Error", "Configuration file missing"
        )

    @patch("src.presentation.cli.YamlConfigLoader")
    def test_run_cli_keyboard_interrupt(self, mock_loader_cls: MagicMock) -> None:
        """run_cli should catch KeyboardInterrupt and return 130."""
        mock_loader_cls.side_effect = KeyboardInterrupt()
        mock_presenter = MagicMock()

        exit_code = run_cli(["--config", "dummy.yaml"], presenter=mock_presenter)
        self.assertEqual(exit_code, 130)
        mock_presenter.render_error.assert_called_once_with(
            "Aborted", "Execution cancelled by user."
        )

    @patch("src.presentation.cli.run_cli")
    def test_main_calls_sys_exit_on_failure(self, mock_run: MagicMock) -> None:
        """main() should invoke sys.exit if run_cli returns non-zero."""
        mock_run.return_value = 1
        with self.assertRaises(SystemExit) as cm:
            main(["--invalid"])
        self.assertEqual(cm.exception.code, 1)

    @patch("src.presentation.cli.run_cli")
    def test_main_returns_cleanly_on_zero(self, mock_run: MagicMock) -> None:
        """main() should return cleanly without sys.exit if run_cli returns 0."""
        mock_run.return_value = 0
        # Should not raise SystemExit
        main([])
