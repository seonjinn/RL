from types import SimpleNamespace
from unittest.mock import Mock

from nemo_rl.models.generation.vllm.stat_logging import (
    flush_cudagraph_metrics,
)


def test_flush_cudagraph_metrics_is_opt_in() -> None:
    logger_manager = SimpleNamespace(log=Mock())
    llm = SimpleNamespace(logger_manager=logger_manager)

    assert flush_cudagraph_metrics(llm, {}) is False
    logger_manager.log.assert_not_called()


def test_flush_cudagraph_metrics_logs_accumulated_stats() -> None:
    logger_manager = SimpleNamespace(log=Mock())
    llm = SimpleNamespace(logger_manager=logger_manager)

    assert flush_cudagraph_metrics(llm, {"cudagraph_metrics": True}) is True
    logger_manager.log.assert_called_once_with()


def test_flush_cudagraph_metrics_handles_missing_logger_manager() -> None:
    llm = SimpleNamespace(logger_manager=None)

    assert flush_cudagraph_metrics(llm, {"cudagraph_metrics": True}) is False
