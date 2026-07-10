# Exact excerpt from vllm/v1/core/sched/output.py at ee0da84ab9e04ac7610e28580af62c365e898389.
class SchedulerOutput:
    # Dynamic speculative decoding: optimal K chosen by scheduler.
    # Number of spec tokens to schedule for the next step.
    num_spec_tokens_to_schedule: int = 0

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        raise NotImplementedError
