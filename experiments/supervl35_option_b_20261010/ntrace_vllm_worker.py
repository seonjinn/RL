"""Bounded ntrace capture for the standalone vLLM 0.29 decode experiment."""

from __future__ import annotations

import os
from typing import Any

from ntrace import NemoRLRolloutTraceController
from vllm.v1.worker.gpu_worker import Worker


class NtraceWorker(Worker):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace_controller: NemoRLRolloutTraceController | None = None
        self._decode_delay = int(os.environ.get("NTRACE_DECODE_DELAY_ITERATIONS", "8192"))
        self._decode_count = int(os.environ.get("NTRACE_DECODE_NUM_ITERS", "32"))
        if self._decode_delay < 0 or self._decode_count <= 0:
            raise ValueError("ntrace decode delay must be nonnegative and count positive")
        self._profile_requested = False
        self._profile_iteration = 0

    def init_device(self) -> None:
        super().init_device()
        self.trace_controller = NemoRLRolloutTraceController(rank=self.rank)

    def compile_or_warm_up_model(self) -> Any:
        controller = self._controller()
        original_capture = self.model_runner.capture_model

        def capture_with_provenance() -> Any:
            token = controller.begin_engine_initialization(name="vllm_cudagraph_capture")
            try:
                return original_capture()
            finally:
                controller.end_engine_initialization(token)

        self.model_runner.capture_model = capture_with_provenance
        try:
            return super().compile_or_warm_up_model()
        finally:
            self.model_runner.capture_model = original_capture

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        controller = self._controller()
        if is_start:
            if self._profile_requested:
                raise RuntimeError("ntrace profile is already running")
            self._profile_iteration = 0
            self._profile_requested = True
        else:
            self._profile_requested = False
            controller.assert_trace_completed()

    def execute_model(self, scheduler_output: Any) -> Any:
        if not self._profile_requested:
            return super().execute_model(scheduler_output)

        iteration = self._profile_iteration
        self._profile_iteration += 1
        if not self._decode_delay <= iteration < self._decode_delay + self._decode_count:
            return super().execute_model(scheduler_output)

        controller = self._controller()
        controller.begin_rollout(step_id=iteration)
        try:
            result = super().execute_model(scheduler_output)
        except BaseException:
            controller.abort_rollout(reason="execute_model_failed")
            raise
        controller.finish_rollout()
        return result

    def shutdown(self) -> None:
        try:
            if self.trace_controller is not None:
                self.trace_controller.close()
        finally:
            super().shutdown()

    def _controller(self) -> NemoRLRolloutTraceController:
        if self.trace_controller is None:
            raise RuntimeError("ntrace controller was not initialized")
        return self.trace_controller
