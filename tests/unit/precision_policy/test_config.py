import pytest

from nemo_rl.precision_policy.config import PrecisionPolicyConfig


def test_minimal_routed_scope_defaults_training_to_bf16() -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "routed-middle",
                    "role": "moe.routed_expert",
                    "layers": {"exclude_first": 2, "exclude_last": 1},
                    "rollout": "mxfp8",
                }
            ]
        }
    )
    assert policy.schema_version == 1
    assert policy.default == "bf16"
    assert policy.scopes[0].training is None
    assert policy.scopes[0].layers.index_space == "global_decoder"


@pytest.mark.parametrize(
    "bad",
    [
        {"default": "mxfp8", "scopes": []},
        {"scopes": [{"id": "x", "role": "moe.routed_expert"}]},
        {
            "scopes": [
                {
                    "id": "x",
                    "role": "moe.routed_expert",
                    "advanced_match": {},
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "advanced_match": {"graph": []},
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "role": "moe.routed_expert",
                    "layers": {"exclude_first": -1},
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "role_typo": "moe.routed_expert",
                    "rollout": "mxfp8",
                }
            ]
        },
    ],
)
def test_invalid_or_ambiguous_policy_is_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PrecisionPolicyConfig.model_validate(bad)
