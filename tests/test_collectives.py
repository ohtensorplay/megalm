import pytest

from megalm.runtime.collectives import (
    NOOP_COLLECTIVE,
    CollectiveError,
    DistributedCollectiveGroup,
    RecordingCollectiveGroup,
    TPGroupConfig,
    bind_all_reduce,
)


class _Tensor:
    """Minimal stand-in with the one method the reduction path uses."""

    def __init__(self, value):
        self.value = value

    def copy_(self, other):
        self.value = other.value
        return self


def test_tp1_noop_group_is_inert_and_free():
    assert NOOP_COLLECTIVE.enabled is False
    assert NOOP_COLLECTIVE.world_size == 1
    tensor = _Tensor(7)
    NOOP_COLLECTIVE.all_reduce(tensor)
    assert tensor.value == 7
    assert NOOP_COLLECTIVE.broadcast_object({"x": 1}) == {"x": 1}
    NOOP_COLLECTIVE.barrier()
    NOOP_COLLECTIVE.check_health()
    NOOP_COLLECTIVE.close()
    assert bind_all_reduce(NOOP_COLLECTIVE) is bind_all_reduce(NOOP_COLLECTIVE)


def test_bind_all_reduce_returns_direct_callable():
    fake = RecordingCollectiveGroup(rank=1, world_size=2)
    fn = bind_all_reduce(fake)
    tensor = _Tensor(3)
    fn(tensor)
    assert fake.all_reduce_calls == [1]


def test_recording_group_can_simulate_reduction():
    fake = RecordingCollectiveGroup(
        rank=0,
        world_size=2,
        reduce=lambda tensor, rank: _Tensor(tensor.value + rank + 1),
    )
    tensor = _Tensor(5)
    fake.all_reduce(tensor)
    assert tensor.value == 6
    assert fake.broadcasts == []


def test_recording_group_health_and_lifecycle():
    fake = RecordingCollectiveGroup()
    fake.check_health()
    fake.fail_health = True
    with pytest.raises(CollectiveError):
        fake.check_health()
    fake.barrier()
    fake.broadcast_object("plan")
    fake.close()
    assert fake.barriers == 1
    assert fake.broadcasts == ["plan"]
    assert fake.closed is True


def test_group_config_validation():
    with pytest.raises(CollectiveError):
        TPGroupConfig(rank=0, world_size=0)
    with pytest.raises(CollectiveError):
        TPGroupConfig(rank=2, world_size=2)
    with pytest.raises(CollectiveError):
        TPGroupConfig(rank=0, world_size=2, init_method="")
    with pytest.raises(CollectiveError):
        TPGroupConfig(rank=0, world_size=1, timeout_seconds=0)


def test_world_size_one_creates_no_process_group():
    group = DistributedCollectiveGroup(TPGroupConfig(rank=0, world_size=1))
    assert group.enabled is False
    tensor = _Tensor(9)
    group.all_reduce(tensor)
    assert tensor.value == 9
    assert group.broadcast_object([1, 2]) == [1, 2]
    group.check_health()
    group.close()
    group.close()
