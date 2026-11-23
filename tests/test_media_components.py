import tensorplay
import tensorplay.nn.functional as F

from megalm.kernels.vision_attention import (
    build_grouped_varlen_attention_layout,
    grouped_varlen_attention,
)
from megalm.models.components.media_patches import (
    pack_image_patches,
    pack_video_patches,
)


def test_two_identical_frames_match_image_temporal_packing() -> None:
    image = tensorplay.arange(
        3 * 28 * 28,
        dtype=tensorplay.float32,
    ).view(3, 28, 28)

    image_patches = pack_image_patches(
        image,
        in_channels=3,
        temporal_patch_size=2,
        patch_size=14,
        spatial_merge_size=2,
    )
    video_patches, grid = pack_video_patches(
        tensorplay.stack((image, image)),
        in_channels=3,
        temporal_patch_size=2,
        patch_size=14,
        spatial_merge_size=2,
    )

    assert grid == (1, 2, 2)
    tensorplay.testing.assert_close(video_patches, image_patches)


def test_grouped_varlen_attention_reuses_uniform_direct_layout() -> None:
    tensorplay.manual_seed(7)
    query = tensorplay.randn(12, 2, 4)
    key = tensorplay.randn_like(query)
    value = tensorplay.randn_like(query)
    layout = build_grouped_varlen_attention_layout(
        tensorplay.tensor((0, 4, 8, 12), dtype=tensorplay.int32)
    )

    assert len(layout.groups) == 1
    assert layout.groups[0].flat_indices is None
    actual = grouped_varlen_attention(
        query,
        key,
        value,
        layout,
        scale=0.5,
    )
    expected = F.scaled_dot_product_attention(
        query.view(3, 4, 2, 4).transpose(1, 2),
        key.view(3, 4, 2, 4).transpose(1, 2),
        value.view(3, 4, 2, 4).transpose(1, 2),
        dropout_p=0.0,
        scale=0.5,
    ).transpose(1, 2).reshape_as(query)
    tensorplay.testing.assert_close(actual, expected)


def test_grouped_varlen_attention_matches_segment_reference() -> None:
    tensorplay.manual_seed(11)
    query = tensorplay.randn(9, 2, 4)
    key = tensorplay.randn_like(query)
    value = tensorplay.randn_like(query)
    boundaries = (0, 2, 5, 7, 9)
    layout = build_grouped_varlen_attention_layout(
        tensorplay.tensor(boundaries, dtype=tensorplay.int32)
    )

    assert len(layout.groups) == 2
    actual = grouped_varlen_attention(
        query,
        key,
        value,
        layout,
        scale=0.5,
    )
    expected = tensorplay.empty_like(actual)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        expected[start:end] = F.scaled_dot_product_attention(
            query[start:end].transpose(0, 1)[None, :],
            key[start:end].transpose(0, 1)[None, :],
            value[start:end].transpose(0, 1)[None, :],
            dropout_p=0.0,
            scale=0.5,
        )[0].transpose(0, 1)
    tensorplay.testing.assert_close(actual, expected)
