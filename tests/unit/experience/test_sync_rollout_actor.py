from unittest.mock import MagicMock

from nemo_rl.experience.sync_rollout_actor import SyncRolloutActor
from nemo_rl.models.generation.interfaces import GenerationInterface


def test_actor_finish_cannot_mark_driver_owned_discard_state() -> None:
    actor_class = SyncRolloutActor.__ray_metadata__.modified_class
    actor = actor_class.__new__(actor_class)
    actor.policy_generation = MagicMock(spec=GenerationInterface)
    actor.policy_generation.finish_generation.return_value = True

    finish_generation = getattr(actor, "_finish_generation", None)
    assert finish_generation is not None
    assert finish_generation() is True
    actor.policy_generation.finish_generation.assert_called_once_with()
    actor.policy_generation.finish_generation_for_next_phase.assert_not_called()
