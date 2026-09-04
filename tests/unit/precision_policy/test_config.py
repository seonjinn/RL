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
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
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
                    "role": "moe.routed_expert",
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
