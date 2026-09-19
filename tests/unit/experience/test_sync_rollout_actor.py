from unittest.mock import MagicMock

import pytest

from nemo_rl.experience.sync_rollout_actor import SyncRolloutActor
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
    GenerationNextPhase,
)


@pytest.mark.parametrize("next_phase", list(GenerationNextPhase))
def test_next_phase_helper_forwards_intent_to_generation(
    next_phase: GenerationNextPhase,
) -> None:
    actor_class = SyncRolloutActor.__ray_metadata__.modified_class
    actor = actor_class.__new__(actor_class)
    actor.policy_generation = MagicMock(spec=GenerationInterface)
    actor.policy_generation.finish_generation_for_next_phase.return_value = True

    finish_for_next_phase = getattr(actor, "_finish_generation_for_next_phase", None)
    assert finish_for_next_phase is not None
    assert finish_for_next_phase(next_phase) is True
    actor.policy_generation.finish_generation_for_next_phase.assert_called_once_with(
        next_phase
    )
