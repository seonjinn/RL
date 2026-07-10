# Exact excerpt from vllm/config/speculative.py at ee0da84ab9e04ac7610e28580af62c365e898389.
class SpeculativeConfig:
    # dynamic speculative decoding control
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None
    """Batch-size schedule used to dynamically choose speculative-token count.

    Each entry is ``(range_start, range_end, num_speculative_tokens)`` with an
    inclusive batch-size range.
    """

    # params generated in the post-init stage
