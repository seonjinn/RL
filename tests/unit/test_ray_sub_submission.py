from pathlib import Path


def test_ray_sub_does_not_add_a_singleton_slurm_dependency():
    ray_sub_path = Path(__file__).parents[2] / "ray.sub"
    assert "#SBATCH --dependency=singleton" not in ray_sub_path.read_text()
