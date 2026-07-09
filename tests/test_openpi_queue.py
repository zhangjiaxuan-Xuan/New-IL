from __future__ import annotations

from new_il.libero.openpi_queue import (
    CLAIM_DONE,
    CLAIM_JOB,
    claim_job_deficit,
    make_openpi_queue,
    record_success,
)


def test_make_openpi_queue_and_claim_by_deficit(tmp_path) -> None:
    queue_dir = tmp_path / "queue"
    n_jobs = make_openpi_queue(
        queue_dir,
        task_suite_name="libero_spatial",
        task_ids=[0, 1],
        attempts_per_task=2,
        seed=7,
    )
    assert n_jobs == 4

    status, job, claimed = claim_job_deficit(queue_dir, worker_id=0, per_task_target=1, total_target=2)
    assert status == CLAIM_JOB
    assert job is not None
    assert claimed is not None
    assert job["task_id"] == 0
    record_success(queue_dir, task_id=0, worker_id=0)
    claimed.unlink()

    status, job, claimed = claim_job_deficit(queue_dir, worker_id=1, per_task_target=1, total_target=2)
    assert status == CLAIM_JOB
    assert job is not None
    assert claimed is not None
    assert job["task_id"] == 1


def test_claim_done_after_total_target(tmp_path) -> None:
    queue_dir = tmp_path / "queue"
    make_openpi_queue(queue_dir, "libero_spatial", [0, 1], attempts_per_task=2, seed=7)
    record_success(queue_dir, task_id=0, worker_id=0)
    record_success(queue_dir, task_id=1, worker_id=0)

    status, job, claimed = claim_job_deficit(queue_dir, worker_id=0, per_task_target=1, total_target=2)
    assert status == CLAIM_DONE
    assert job is None
    assert claimed is None
