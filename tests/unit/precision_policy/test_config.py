import math

import pytest
from pydantic import ValidationError

from nemo_rl.precision_policy.config import (
    PrecisionPolicyConfig,
    parse_precision_policy,
)


def test_minimal_routed_scope_defaults_training_to_bf16() -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "routed-middle",
                    "roles": ["moe.routed_expert"],
                    "layers": {"exclude_first": 2, "exclude_last": 1},
                    "rollout": "mxfp8",
                }
            ]
        }
    )
    assert policy.schema_version == 1
    assert policy.default == "bf16"
    assert policy.scopes[0].training is None
    assert policy.scopes[0].layers is not None
    assert policy.scopes[0].layers.index_space == "global_decoder"


def test_scope_roles_are_non_empty_unique_and_canonically_sorted() -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "multi-role",
                    "roles": ["moe.routed_expert", "attention.qkvo"],
                    "rollout": "mxfp8",
                }
            ]
        }
    )

    assert policy.scopes[0].roles == ["attention.qkvo", "moe.routed_expert"]

    for roles in ([], [""], [" "], ["moe.routed_expert", "moe.routed_expert"]):
        with pytest.raises(ValidationError):
            PrecisionPolicyConfig.model_validate(
                {"scopes": [{"id": "invalid", "roles": roles, "rollout": "mxfp8"}]}
            )


def test_direct_model_validation_rejects_singular_role() -> None:
    with pytest.raises(ValidationError, match="Undocumented precision policy field"):
        PrecisionPolicyConfig.model_validate(
            {
                "scopes": [
                    {
                        "id": "legacy",
                        "role": "moe.routed_expert",
                        "rollout": "mxfp8",
                    }
                ]
            }
        )


def test_parse_precision_policy_translates_legacy_singular_role_once() -> None:
    raw = {
        "scopes": [
            {
                "id": "legacy",
                "role": "moe.routed_expert",
                "rollout": "mxfp8",
            }
        ]
    }

    policy = parse_precision_policy(raw)

    assert policy is not None
    assert policy.scopes[0].roles == ["moe.routed_expert"]
    assert raw["scopes"][0] == {
        "id": "legacy",
        "role": "moe.routed_expert",
        "rollout": "mxfp8",
    }
    dumped_scope = policy.model_dump()["scopes"][0]
    assert dumped_scope["roles"] == ["moe.routed_expert"]
    assert "role" not in dumped_scope


def test_parse_precision_policy_rejects_role_and_roles_together() -> None:
    with pytest.raises(ValueError, match="role.*roles"):
        parse_precision_policy(
            {
                "scopes": [
                    {
                        "id": "ambiguous",
                        "role": "moe.routed_expert",
                        "roles": ["attention.qkvo"],
                        "rollout": "mxfp8",
                    }
                ]
            }
        )


def test_advanced_match_preserves_two_graph_identity_predicates() -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "decoder-projections",
                    "advanced_match": {
                        "graph_instance_id": "main",
                        "semantic_graph_path": ["text.decoder", "text.embedding"],
                        "module_kind": "linear",
                    },
                    "rollout": "mxfp8",
                }
            ]
        }
    )

    advanced_match = policy.scopes[0].advanced_match
    assert advanced_match is not None
    assert advanced_match.graph_instance_id == "main"
    assert advanced_match.semantic_graph_path == ["text.decoder", "text.embedding"]


@pytest.mark.parametrize(
    ("field_name", "predicate"),
    [
        ("graph_instance_id", [""]),
        ("graph_instance_id", [" "]),
        ("graph_instance_id", ["main", ""]),
        ("semantic_graph_path", [""]),
        ("semantic_graph_path", [" "]),
        ("semantic_graph_path", ["main", ""]),
        ("model_part", [""]),
        ("model_part", [" "]),
        ("model_part", ["main", ""]),
        ("module_kind", [""]),
        ("module_kind", [" "]),
        ("module_kind", ["main", ""]),
        ("parameter_role", [""]),
        ("parameter_role", [" "]),
        ("parameter_role", ["main", ""]),
    ],
)
def test_advanced_match_rejects_blank_string_predicate_members(
    field_name: str, predicate: list[str]
) -> None:
    with pytest.raises(ValidationError):
        PrecisionPolicyConfig.model_validate(
            {
                "scopes": [
                    {
                        "id": "x",
                        "advanced_match": {field_name: predicate},
                        "rollout": "mxfp8",
                    }
                ]
            }
        )


@pytest.mark.parametrize("predicate", [[""], [" "], ["expert", ""]])
def test_advanced_match_rejects_blank_attribute_predicate_members(
    predicate: list[str],
) -> None:
    with pytest.raises(ValidationError):
        PrecisionPolicyConfig.model_validate(
            {
                "scopes": [
                    {
                        "id": "x",
                        "advanced_match": {"attributes": {"expert": predicate}},
                        "rollout": "mxfp8",
                    }
                ]
            }
        )


def test_advanced_match_preserves_zero_and_false_attribute_predicate_members() -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "expert-attributes",
                    "advanced_match": {"attributes": {"expert_index": [0, False]}},
                    "rollout": "mxfp8",
                }
            ]
        }
    )

    advanced_match = policy.scopes[0].advanced_match
    assert advanced_match is not None
    assert advanced_match.attributes["expert_index"] == [0, False]


def test_addresses_allow_same_semantic_id_on_distinct_graph_instances() -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "explicit-projections",
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.proj",
                        },
                        {
                            "graph_instance_id": "mtp.0",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.proj",
                        },
                    ],
                    "rollout": "mxfp8",
                }
            ]
        }
    )

    addresses = policy.scopes[0].addresses
    assert addresses is not None
    assert [address.graph_instance_id for address in addresses] == ["main", "mtp.0"]


def test_scope_atomic_conflict_omission_is_preserved_after_round_trip() -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "atomic_conflict": "expand",
            "scopes": [
                {
                    "id": "routed-middle",
                    "roles": ["moe.routed_expert"],
                    "rollout": "mxfp8",
                }
            ],
        }
    )

    assert policy.atomic_conflict == "expand"
    assert policy.scopes[0].atomic_conflict is None
    reparsed = PrecisionPolicyConfig.model_validate(policy.model_dump())
    assert reparsed.scopes[0].atomic_conflict is None


@pytest.mark.parametrize(
    ("policy_mode", "scope_mode"),
    [("expand", "error"), ("error", "expand")],
)
def test_scope_atomic_conflict_local_override_is_preserved(
    policy_mode: str, scope_mode: str
) -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "atomic_conflict": policy_mode,
            "scopes": [
                {
                    "id": "routed-middle",
                    "roles": ["moe.routed_expert"],
                    "rollout": "mxfp8",
                    "atomic_conflict": scope_mode,
                }
            ],
        }
    )

    assert policy.scopes[0].atomic_conflict == scope_mode
    reparsed = PrecisionPolicyConfig.model_validate(policy.model_dump())
    assert reparsed.scopes[0].atomic_conflict == scope_mode


def test_omitted_layers_remain_none_after_round_trip() -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "routed-middle",
                    "roles": ["moe.routed_expert"],
                    "rollout": "mxfp8",
                }
            ]
        }
    )

    assert policy.scopes[0].layers is None
    reparsed = PrecisionPolicyConfig.model_validate(policy.model_dump())
    assert reparsed.scopes[0].layers is None


@pytest.mark.parametrize(
    "layers",
    [{}, {"exclude_first": 0, "exclude_last": 0}],
)
def test_explicit_layers_remain_structural_selector_after_round_trip(
    layers: dict[str, int],
) -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "routed-middle",
                    "roles": ["moe.routed_expert"],
                    "layers": layers,
                    "rollout": "mxfp8",
                }
            ]
        }
    )

    selector = policy.scopes[0].layers
    assert selector is not None
    assert selector.index_space == "global_decoder"
    assert selector.exclude_first == 0
    assert selector.exclude_last == 0
    reparsed_selector = (
        PrecisionPolicyConfig.model_validate(policy.model_dump()).scopes[0].layers
    )
    assert reparsed_selector is not None
    assert reparsed_selector == selector


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("as_list", [False, True])
def test_nonfinite_attribute_predicates_are_rejected(
    nonfinite: float, as_list: bool
) -> None:
    predicate: object = [nonfinite] if as_list else nonfinite
    with pytest.raises(ValidationError):
        PrecisionPolicyConfig.model_validate(
            {
                "scopes": [
                    {
                        "id": "expert-attributes",
                        "advanced_match": {
                            "attributes": {"capacity_factor": predicate}
                        },
                        "rollout": "mxfp8",
                    }
                ]
            }
        )


def test_attribute_scalar_types_survive_round_trip() -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "typed-attributes",
                    "advanced_match": {
                        "attributes": {
                            "float_zero": 0.0,
                            "integer_zero": 0,
                            "boolean_false": False,
                            "negative_zero": -0.0,
                            "typed_list": [0.0, 0, False, -0.0],
                        }
                    },
                    "rollout": "mxfp8",
                }
            ]
        }
    )

    for candidate in (
        policy,
        PrecisionPolicyConfig.model_validate(policy.model_dump()),
    ):
        advanced_match = candidate.scopes[0].advanced_match
        assert advanced_match is not None
        attributes = advanced_match.attributes
        assert type(attributes["float_zero"]) is float
        assert type(attributes["integer_zero"]) is int
        assert type(attributes["boolean_false"]) is bool
        assert type(attributes["negative_zero"]) is float
        assert math.copysign(1.0, attributes["negative_zero"]) == -1.0
        typed_list = attributes["typed_list"]
        assert isinstance(typed_list, list)
        assert [type(item) for item in typed_list] == [float, int, bool, float]
        negative_zero = typed_list[-1]
        assert type(negative_zero) is float
        assert math.copysign(1.0, negative_zero) == -1.0


@pytest.mark.parametrize("field_name", ["exclude_first", "exclude_last"])
@pytest.mark.parametrize("value", [True, False, 0.0, 1.0, "0", "1"])
def test_layer_exclusions_reject_coercive_values(
    field_name: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        PrecisionPolicyConfig.model_validate(
            {
                "scopes": [
                    {
                        "id": "routed-middle",
                        "roles": ["moe.routed_expert"],
                        "layers": {field_name: value},
                        "rollout": "mxfp8",
                    }
                ]
            }
        )


@pytest.mark.parametrize("field_name", ["exclude_first", "exclude_last"])
@pytest.mark.parametrize("value", [0, 1])
def test_layer_exclusions_accept_exact_nonnegative_ints(
    field_name: str, value: int
) -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {
            "scopes": [
                {
                    "id": "routed-middle",
                    "roles": ["moe.routed_expert"],
                    "layers": {field_name: value},
                    "rollout": "mxfp8",
                }
            ]
        }
    )

    selector = policy.scopes[0].layers
    assert selector is not None
    assert type(getattr(selector, field_name)) is int
    assert getattr(selector, field_name) == value


@pytest.mark.parametrize("field_name", ["exclude_first", "exclude_last"])
def test_layer_exclusions_reject_negative_integers(field_name: str) -> None:
    with pytest.raises(ValidationError):
        PrecisionPolicyConfig.model_validate(
            {
                "scopes": [
                    {
                        "id": "routed-middle",
                        "roles": ["moe.routed_expert"],
                        "layers": {field_name: -1},
                        "rollout": "mxfp8",
                    }
                ]
            }
        )


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_require_match_rejects_coercive_boolean_values(value: object) -> None:
    with pytest.raises(ValidationError):
        PrecisionPolicyConfig.model_validate({"require_match": value, "scopes": []})


@pytest.mark.parametrize("value", [False, True])
def test_require_match_accepts_exact_booleans(value: bool) -> None:
    policy = PrecisionPolicyConfig.model_validate(
        {"require_match": value, "scopes": []}
    )
    assert type(policy.require_match) is bool
    assert policy.require_match is value


@pytest.mark.parametrize("value", [True, False, 1.0, "1", 2])
def test_schema_version_requires_an_exact_integer(value: object) -> None:
    with pytest.raises(ValidationError):
        PrecisionPolicyConfig.model_validate({"schema_version": value, "scopes": []})


def test_schema_version_accepts_exact_integer_one() -> None:
    policy = PrecisionPolicyConfig.model_validate({"schema_version": 1, "scopes": []})
    assert type(policy.schema_version) is int
    assert policy.schema_version == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"default": "mxfp8", "scopes": []},
        {"scopes": [{"id": "x", "roles": ["moe.routed_expert"]}]},
        {
            "scopes": [
                {
                    "id": "x",
                    "roles": ["moe.routed_expert"],
                    "advanced_match": {},
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "roles": ["moe.routed_expert"],
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
        {
            "scopes": [
                {
                    "id": "x",
                    "advanced_match": {"graph_instance_id": []},
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "advanced_match": {"semantic_graph_path": ""},
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "advanced_match": {"graph": "text.decoder"},
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "semantic_addresses": [],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [{"semantic_ids": ["text.decoder.layer.0.proj"]}],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.proj",
                            "unknown": "value",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [
                        {
                            "graph_instance_id": " main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.proj",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": " text.decoder",
                            "semantic_id": "text.decoder.layer.0.proj",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": " text.decoder.layer.0.proj",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.embedding.layer.0.proj",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoderish.layer.0.proj",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.proj",
                        },
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.proj",
                        },
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "roles": ["moe.routed_expert"],
                    "addresses": [
                        {
                            "graph_instance_id": "main",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.proj",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [
                        {
                            "graph_instance_id": "decoder.0",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.proj",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
        {
            "scopes": [
                {
                    "id": "x",
                    "addresses": [
                        {
                            "graph_instance_id": "mtp.",
                            "semantic_graph_path": "text.decoder",
                            "semantic_id": "text.decoder.layer.0.proj",
                        }
                    ],
                    "rollout": "mxfp8",
                }
            ]
        },
    ],
)
def test_invalid_or_ambiguous_policy_is_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PrecisionPolicyConfig.model_validate(bad)
