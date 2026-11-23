import tensorplay

from megalm.models.qwen2_5_omni.mrope import (
    build_omni_mrope_positions,
)


def test_video_temporal_positions_follow_seconds_per_grid() -> None:
    token_types = tensorplay.tensor(
        (0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0),
        dtype=tensorplay.long,
    )

    positions, delta = build_omni_mrope_positions(
        token_types,
        tensorplay.empty((0, 3), dtype=tensorplay.long),
        video_grid_thw=tensorplay.tensor(((2, 4, 4),)),
        video_seconds_per_grid=((0.5, 0.5),),
        spatial_merge_size=2,
        position_ids_per_second=25,
    )

    expected = tensorplay.tensor(
        (
            (0, 1, 2, 2, 2, 2, 14, 14, 14, 14, 15),
            (0, 1, 2, 2, 3, 3, 2, 2, 3, 3, 15),
            (0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 15),
        ),
        dtype=tensorplay.long,
    )
    tensorplay.testing.assert_close(positions, expected)
    assert delta == 5


def test_audio_positions_are_one_dimensional_and_count_checked() -> None:
    token_types = tensorplay.tensor(
        (0, 0, 3, 3, 3, 0),
        dtype=tensorplay.long,
    )

    positions, delta = build_omni_mrope_positions(
        token_types,
        tensorplay.empty((0, 3), dtype=tensorplay.long),
        audio_feature_counts=(3,),
        spatial_merge_size=2,
        position_ids_per_second=25,
    )

    expected = tensorplay.arange(6).view(1, -1).expand(3, -1)
    tensorplay.testing.assert_close(positions, expected)
    assert delta == 0
