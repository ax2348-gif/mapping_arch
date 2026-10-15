"""The Z^2 contract is frozen: every batch an old-contract model receives
is byte-identical to what commit 4dc94de produced."""
from tests.golden import cases

GOLDEN = {
    'train d0 s0': '2a4d7a96cbb70d55',
    'train d0 s1': '0aa7aded9976c7e6',
    'eval d0': '4bdb9d9da85bd159',
    'train d1 s0': '2cab6544d1c55062',
    'train d1 s1': '056100e7fb4c60c1',
    'eval d1': 'add7d679090ba91c',
}


def test_batches_unchanged():
    assert cases() == GOLDEN


def test_z2_batch_has_no_se2_keys():
    import numpy as np
    from mapping.data import Config, load_bank, make_batch
    bank, _ = load_bank('data/mnist.npz', 32, 'train', False)
    b = make_batch(np.random.default_rng(0), Config(), bank, 4)
    assert set(b) == {'observations', 'directions', 'positions',
                      'query_offset', 'target', 'episodes'}
