import contextlib
from types import SimpleNamespace

import pytest

from capstone.training.trainer import _autocast_factory


def fake_torch(bf16_supported: bool = True):
    calls: list[dict] = []

    def autocast(**kwargs):
        calls.append(kwargs)
        return contextlib.nullcontext()

    return SimpleNamespace(
        cuda=SimpleNamespace(is_bf16_supported=lambda: bf16_supported),
        autocast=autocast,
        bfloat16="bfloat16",
    ), calls


def test_default_precision_is_fp32_and_does_not_autocast():
    torch, calls = fake_torch()

    precision, autocast = _autocast_factory({}, torch, SimpleNamespace(type="cuda"))

    assert precision == "fp32"
    assert autocast is contextlib.nullcontext
    with autocast():
        pass
    assert calls == []


def test_bf16_autocasts_on_a_capable_cuda_device():
    torch, calls = fake_torch()

    precision, autocast = _autocast_factory({"precision": "BF16"}, torch, SimpleNamespace(type="cuda"))

    assert precision == "bf16"
    with autocast():
        pass
    assert calls == [{"device_type": "cuda", "dtype": "bfloat16"}]


@pytest.mark.parametrize(
    ("training", "device", "bf16_supported", "message"),
    [
        ({"precision": "fp16"}, SimpleNamespace(type="cuda"), True, "must be 'fp32' or 'bf16'"),
        ({"precision": "bf16"}, SimpleNamespace(type="cpu"), True, "requires a CUDA device"),
        ({"precision": "bf16"}, SimpleNamespace(type="cuda"), False, "requires a CUDA device"),
    ],
)
def test_unsupported_precision_is_refused_instead_of_falling_back(training, device, bf16_supported, message):
    torch, _ = fake_torch(bf16_supported)

    with pytest.raises(ValueError, match=message):
        _autocast_factory(training, torch, device)
