from types import SimpleNamespace

import pytest
import tensorplay

from megalm.kernels.decode_relay import relay_decode_batch
from megalm.kernels.speculative_relay import relay_speculative_chain
from megalm.runtime.graphs import DecodeGraphRunner


pytestmark = pytest.mark.skipif(
    not tensorplay.cuda.is_available(),
    reason="decode relay requires CUDA",
)


def test_speculative_relay_fuses_metadata_and_zero_stride_mrope() -> None:
    device = tensorplay.device("cuda")
    length = 4
    source_positions = tensorplay.arange(17, 21, device=device)
    source_mrope = source_positions.unsqueeze(0).expand(3, -1)
    target_input_ids = tensorplay.full((length,), -1, device=device)
    target_positions = tensorplay.full((length,), -1, device=device)
    target_slots = tensorplay.full((length,), -1, device=device)
    target_blocks = tensorplay.full(
        (1, 3),
        -1,
        dtype=tensorplay.int32,
        device=device,
    )
    target_lengths = tensorplay.full(
        (length,),
        -1,
        dtype=tensorplay.int32,
        device=device,
    )
    target_mrope = tensorplay.full(
        (3, length),
        -1,
        device=device,
    )

    relay_speculative_chain(
        tensorplay.tensor((11, 12, 13, 14), device=device),
        source_positions,
        tensorplay.tensor((31, 32, 33, 34), device=device),
        tensorplay.tensor(((2, 3, -1),), dtype=tensorplay.int32, device=device),
        target_input_ids,
        target_positions,
        target_slots,
        target_blocks,
        target_lengths,
        source_mrope=source_mrope,
        target_mrope=target_mrope,
    )
    tensorplay.cuda.synchronize()

    assert target_input_ids.tolist() == [11, 12, 13, 14]
    assert target_positions.tolist() == [17, 18, 19, 20]
    assert target_slots.tolist() == [31, 32, 33, 34]
    assert target_blocks.tolist() == [[2, 3, -1]]
    assert target_lengths.tolist() == [18, 19, 20, 21]
    tensorplay.testing.assert_close(
        target_mrope,
        source_positions.unsqueeze(0).repeat(3, 1),
    )


@pytest.mark.parametrize(
    ("has_gdn_state", "has_mrope"),
    (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ),
)
def test_decode_relay_keeps_optional_metadata_independent(
    has_gdn_state: bool,
    has_mrope: bool,
) -> None:
    device = tensorplay.device("cuda")
    active_rows = 2
    bucket_size = 4
    max_blocks = 3
    source_input_ids = tensorplay.tensor((11, 12), device=device)
    source_positions = tensorplay.tensor((7, 9), device=device)
    source_slots = tensorplay.tensor((21, 22), device=device)
    source_block_tables = tensorplay.tensor(
        ((1, 2, -1), (3, -1, -1)),
        dtype=tensorplay.int32,
        device=device,
    )
    source_sequence_lengths = tensorplay.tensor(
        (8, 10),
        dtype=tensorplay.int32,
        device=device,
    )
    source_active = tensorplay.ones(
        active_rows,
        dtype=tensorplay.bool,
        device=device,
    )
    source_gdn = (
        tensorplay.tensor(
            ((31, 32), (33, 34)),
            dtype=tensorplay.int32,
            device=device,
        )
        if has_gdn_state
        else None
    )
    source_mrope = (
        tensorplay.tensor(
            ((1, 2), (3, 4), (5, 6)),
            device=device,
        )
        if has_mrope
        else None
    )
    target_input_ids = tensorplay.full((bucket_size,), -1, device=device)
    target_positions = tensorplay.full((bucket_size,), -1, device=device)
    target_slots = tensorplay.full((bucket_size,), -1, device=device)
    target_block_tables = tensorplay.full(
        (bucket_size, max_blocks),
        -9,
        dtype=tensorplay.int32,
        device=device,
    )
    target_sequence_lengths = tensorplay.full(
        (bucket_size,),
        -1,
        dtype=tensorplay.int32,
        device=device,
    )
    target_active = tensorplay.ones(
        bucket_size,
        dtype=tensorplay.bool,
        device=device,
    )
    target_gdn = (
        tensorplay.full(
            (bucket_size, 2),
            -1,
            dtype=tensorplay.int32,
            device=device,
        )
        if has_gdn_state
        else None
    )
    target_mrope = (
        tensorplay.full(
            (3, bucket_size),
            -1,
            device=device,
        )
        if has_mrope
        else None
    )

    relay_decode_batch(
        source_input_ids,
        source_positions,
        source_slots,
        source_block_tables,
        source_sequence_lengths,
        source_active,
        target_input_ids,
        target_positions,
        target_slots,
        target_block_tables,
        target_sequence_lengths,
        target_active,
        source_gdn,
        source_mrope,
        target_gdn,
        target_mrope,
    )
    tensorplay.cuda.synchronize()

    tensorplay.testing.assert_close(
        target_input_ids,
        tensorplay.tensor((11, 12, 0, 0), device=device),
    )
    tensorplay.testing.assert_close(
        target_positions,
        tensorplay.tensor((7, 9, 0, 0), device=device),
    )
    tensorplay.testing.assert_close(
        target_slots,
        tensorplay.tensor((21, 22, 0, 0), device=device),
    )
    tensorplay.testing.assert_close(
        target_sequence_lengths,
        tensorplay.tensor((8, 10, 1, 1), dtype=tensorplay.int32, device=device),
    )
    tensorplay.testing.assert_close(
        target_active,
        tensorplay.tensor((True, True, False, False), device=device),
    )
    tensorplay.testing.assert_close(
        target_block_tables,
        tensorplay.tensor(
            (
                (1, 2, -1),
                (3, -1, -1),
                (0, 0, 0),
                (0, 0, 0),
            ),
            dtype=tensorplay.int32,
            device=device,
        ),
    )
    if has_gdn_state:
        assert source_gdn is not None and target_gdn is not None
        tensorplay.testing.assert_close(target_gdn[:active_rows], source_gdn)
        tensorplay.testing.assert_close(
            target_gdn[active_rows:],
            tensorplay.zeros((2, 2), dtype=tensorplay.int32, device=device),
        )
    if has_mrope:
        assert source_mrope is not None and target_mrope is not None
        tensorplay.testing.assert_close(
            target_mrope[:, :active_rows],
            source_mrope,
        )
        tensorplay.testing.assert_close(
            target_mrope[:, active_rows:],
            tensorplay.zeros((3, 2), dtype=tensorplay.int64, device=device),
        )


def test_decode_relay_respects_zero_stride_mrope_view() -> None:
    device = tensorplay.device("cuda")
    active_rows = 4
    bucket_size = 8
    source_positions = tensorplay.arange(
        26,
        26 + active_rows,
        device=device,
    )
    source_mrope = source_positions.unsqueeze(0).expand(3, -1)
    assert source_mrope.stride() == (0, 1)
    target_mrope = tensorplay.full(
        (3, bucket_size),
        -1,
        device=device,
    )

    relay_decode_batch(
        tensorplay.arange(active_rows, device=device),
        source_positions,
        tensorplay.arange(active_rows, device=device),
        tensorplay.zeros(
            (active_rows, 2),
            dtype=tensorplay.int32,
            device=device,
        ),
        tensorplay.ones(active_rows, dtype=tensorplay.int32, device=device),
        tensorplay.ones(active_rows, dtype=tensorplay.bool, device=device),
        tensorplay.zeros(bucket_size, dtype=tensorplay.long, device=device),
        tensorplay.zeros(bucket_size, dtype=tensorplay.long, device=device),
        tensorplay.zeros(bucket_size, dtype=tensorplay.long, device=device),
        tensorplay.zeros(
            (bucket_size, 2),
            dtype=tensorplay.int32,
            device=device,
        ),
        tensorplay.zeros(bucket_size, dtype=tensorplay.int32, device=device),
        tensorplay.zeros(bucket_size, dtype=tensorplay.bool, device=device),
        None,
        source_mrope,
        None,
        target_mrope,
    )
    tensorplay.cuda.synchronize()

    tensorplay.testing.assert_close(
        target_mrope[:, :active_rows],
        source_mrope,
    )
    tensorplay.testing.assert_close(
        target_mrope[:, active_rows:],
        tensorplay.zeros(
            (3, bucket_size - active_rows),
            dtype=tensorplay.int64,
            device=device,
        ),
    )


@pytest.mark.parametrize(
    ("hybrid_state", "uses_mrope", "expects_gdn", "expects_mrope"),
    (
        (False, False, False, False),
        (False, True, False, True),
        (True, True, True, True),
    ),
)
def test_graph_bucket_uses_captured_model_capabilities(
    hybrid_state: bool,
    uses_mrope: bool,
    expects_gdn: bool,
    expects_mrope: bool,
) -> None:
    runner = DecodeGraphRunner.__new__(DecodeGraphRunner)
    runner.model = SimpleNamespace(device=tensorplay.device("cuda"))
    runner.max_blocks_per_sequence = 2
    runner.page_size = 16
    runner.hybrid_state = hybrid_state
    runner.uses_mrope = uses_mrope

    batch = runner._static_batch(2)

    assert (batch.gdn_state_indices is not None) is expects_gdn
    assert (batch.mrope_positions is not None) is expects_mrope
