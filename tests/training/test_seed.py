from capstone.training.seed import seed_everything


def test_seed_everything_returns_the_requested_seed():
    assert seed_everything(42) == 42

