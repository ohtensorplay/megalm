import hashlib

import tensorplay

from megalm.runtime.vision_cache import (
    GPUVisionCache,
    VisionCacheKey,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_image_cache_key_separates_detail_levels() -> None:
    low = VisionCacheKey(
        model_digest=_digest("model"),
        processor_digest=_digest("processor"),
        content_digest=_digest("image"),
        detail="low",
    )
    high = VisionCacheKey(
        model_digest=_digest("model"),
        processor_digest=_digest("processor"),
        content_digest=_digest("image"),
        detail="high",
    )
    assert low != high


def test_video_cache_preserves_temporal_metadata_across_hits() -> None:
    cache = GPUVisionCache(
        capacity_bytes=1_024,
        width=4,
        dtype=tensorplay.float32,
        device="cpu",
    )
    key = VisionCacheKey(
        model_digest=_digest("model"),
        processor_digest=_digest("processor"),
        content_digest=_digest("video"),
        modality="video",
    )
    value = cache.insert_and_acquire(
        key,
        tensorplay.arange(32, dtype=tensorplay.float32).view(8, 4),
        (2, 4, 4),
        "first",
        frame_feature_counts=(4, 4),
        timestamps=(0.25, 0.75),
        seconds_per_grid=(0.5, 0.5),
    )
    cache.release_request("first")

    hit = cache.acquire(key, "second")

    assert hit is not None
    assert hit.embedding.data_ptr() == value.embedding.data_ptr()
    assert hit.frame_feature_counts == (4, 4)
    assert hit.timestamps == (0.25, 0.75)
    assert hit.seconds_per_grid == (0.5, 0.5)
    cache.release_request("second")
    cache.assert_integrity()
