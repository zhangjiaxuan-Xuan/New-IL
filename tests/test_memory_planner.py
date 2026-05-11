import pytest

from new_il.training.memory import GpuInfo, max_batch_from_free_memory, plan_memory


def test_max_batch_rounds_down_to_multiple_of_four() -> None:
    assert max_batch_from_free_memory(
        40.0,
        memory_fraction=0.9,
        reserve_gb=4.0,
        gb_per_sample=3.0,
        batch_multiple=4,
    ) == 8


def test_plan_memory_uses_two_gpus_and_effective_batch() -> None:
    plan = plan_memory(
        [
            GpuInfo(index=0, total_mb=80 * 1024, free_mb=70 * 1024),
            GpuInfo(index=1, total_mb=80 * 1024, free_mb=60 * 1024),
            GpuInfo(index=2, total_mb=80 * 1024, free_mb=10 * 1024),
        ],
        min_free_gb=20,
        max_gpus=2,
        target_global_batch=128,
        memory_fraction=0.9,
        reserve_gb=4,
        gb_per_sample=4,
        batch_multiple=4,
    )

    assert plan.selected_gpus == [0, 1]
    assert plan.per_device_batch_size == 12
    assert plan.per_device_batch_size % 4 == 0
    assert plan.grad_accumulation_steps == 6
    assert plan.effective_batch_size == 144


def test_manual_batch_rejected_when_oom_risk_without_override() -> None:
    with pytest.raises(ValueError, match="exceeds safe estimate"):
        plan_memory(
            [GpuInfo(index=0, total_mb=24 * 1024, free_mb=24 * 1024)],
            min_free_gb=10,
            max_gpus=1,
            target_global_batch=64,
            memory_fraction=0.9,
            reserve_gb=4,
            gb_per_sample=4,
            batch_multiple=4,
            per_device_batch_size=16,
        )


def test_manual_batch_allowed_with_oom_risk_override() -> None:
    plan = plan_memory(
        [GpuInfo(index=0, total_mb=24 * 1024, free_mb=24 * 1024)],
        min_free_gb=10,
        max_gpus=1,
        target_global_batch=64,
        memory_fraction=0.9,
        reserve_gb=4,
        gb_per_sample=4,
        batch_multiple=4,
        per_device_batch_size=16,
        allow_oom_risk=True,
    )

    assert plan.per_device_batch_size == 16
    assert plan.oom_risk


def test_manual_batch_must_be_multiple() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        plan_memory(
            [GpuInfo(index=0, total_mb=80 * 1024, free_mb=80 * 1024)],
            per_device_batch_size=10,
            batch_multiple=4,
        )
