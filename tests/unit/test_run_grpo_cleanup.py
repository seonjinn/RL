from unittest.mock import Mock

from examples.run_grpo import _shutdown_sync_resources


def test_shutdown_sync_resources_releases_ray_users_before_teardown() -> None:
    policy = Mock()
    generation = Mock()
    policy_cluster = Mock()
    generation_cluster = Mock()
    logger = Mock()
    monitor = logger.gpu_monitor

    _shutdown_sync_resources(
        policy, generation, (policy_cluster, generation_cluster), logger
    )

    monitor.stop.assert_called_once_with()
    logger.finish.assert_called_once_with()
    generation.shutdown.assert_called_once_with()
    policy.shutdown.assert_called_once_with()
    policy_cluster.shutdown.assert_called_once_with()
    generation_cluster.shutdown.assert_called_once_with()
    assert logger.gpu_monitor is None


def test_shutdown_sync_resources_does_not_double_shutdown_colocated_policy() -> None:
    policy = Mock()
    cluster = Mock()
    logger = Mock()
    logger.gpu_monitor = None

    _shutdown_sync_resources(policy, policy, (cluster, cluster), logger)

    policy.shutdown.assert_called_once_with()
    cluster.shutdown.assert_called_once_with()
