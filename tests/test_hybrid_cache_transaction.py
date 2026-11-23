import tensorplay

from megalm.runtime.cache import KVCacheManager
from megalm.runtime.hybrid_cache import GDNStateCache


def test_gdn_shadow_state_swaps_without_reassigning_request() -> None:
    manager = KVCacheManager(
        num_pages=4,
        page_size=4,
        max_sequences=2,
        max_model_len=8,
        device="cpu",
        enable_prefix_cache=False,
        gdn_state_slots=4,
        gdn_states_per_request=2,
    )
    handle = manager.admit("request")
    active = manager.gdn_state_indices("request")
    shadow = manager.gdn_state_indices("request", shadow=True)

    assert active != shadow
    manager.swap_gdn_state("request")
    assert manager.handle("request") == handle
    assert manager.gdn_state_indices("request") == shadow
    assert manager.gdn_state_indices("request", shadow=True) == active

    manager.assert_integrity()
    manager.release("request")
    manager.assert_integrity()


def test_gdn_state_copy_copies_conv_and_recurrent_arenas() -> None:
    cache = GDNStateCache(
        layer_indices=(1, 3),
        num_slots=2,
        conv_width=6,
        conv_kernel_dim=3,
        num_value_heads=2,
        key_head_dim=2,
        value_head_dim=4,
        dtype=tensorplay.float16,
        device="cpu",
    )
    cache.conv[:, 0].fill_(3)
    cache.recurrent[:, 0].fill_(7)

    cache.copy_state((0, 0), (1, 1))

    assert tensorplay.equal(cache.conv[:, 1], cache.conv[:, 0])
    assert tensorplay.equal(cache.recurrent[:, 1], cache.recurrent[:, 0])
