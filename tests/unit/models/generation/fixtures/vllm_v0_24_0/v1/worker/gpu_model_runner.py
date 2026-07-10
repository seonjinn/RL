# ruff: noqa: F821
# Exact executable excerpt from vllm/v1/worker/gpu_model_runner.py at
# ee0da84ab9e04ac7610e28580af62c365e898389.
class GPUModelRunner:
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if self.routed_experts_initialized:
            self.routed_experts_capturer.clear_buffer()

        return None
