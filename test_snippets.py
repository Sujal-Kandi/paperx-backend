"""
Run from the folder that holds schemas.py:  python -m pytest test_snippets.py -q
Needs: pip install pytest torch

Executes the EXACT code strings shown on the website (in section order, sharing
one namespace), so the page can never show a snippet that hasn't been tested.
"""
import json
import math
import os
import pathlib

import torch

from schemas import Paper

HERE = pathlib.Path(__file__).resolve().parent
CANDIDATES = [
    os.environ.get("PAPERX_CONTENT", ""),
    HERE / "attention.json",                                        # flat layout
    HERE.parent / "attention.json",
    HERE.parents[1] / "frontend" / "content" / "attention.json",   # backend/tests layout
]
CONTENT = next(pathlib.Path(c) for c in CANDIDATES if c and pathlib.Path(c).exists())


def load():
    return json.loads(CONTENT.read_text(encoding="utf-8"))


def namespace():
    """Exec every section's code in order, like a reader following the page."""
    ns = {}
    for s in load()["sections"]:
        if s.get("code"):
            exec(s["code"]["pytorch"], ns)
    return ns


def test_content_matches_schema():
    Paper.model_validate(load())


def test_scaled_dot_product_attention_snippet():
    attn = namespace()["scaled_dot_product_attention"]
    B, T, dk, dv = 2, 5, 64, 32
    q, k, v = torch.randn(B, T, dk), torch.randn(B, T, dk), torch.randn(B, T, dv)
    out, w = attn(q, k, v)
    assert out.shape == (B, T, dv)
    assert w.shape == (B, T, T)
    assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-5)
    _, wm = attn(q, k, v, torch.tril(torch.ones(T, T)))
    assert torch.all(torch.triu(wm, diagonal=1) == 0)


def test_multi_head_attention_snippet():
    torch.manual_seed(0)
    mha = namespace()["MultiHeadAttention"]().eval()
    B, T = 2, 6
    x = torch.randn(B, T, 512)
    assert mha(x, x, x).shape == (B, T, 512)
    # causal mask: changing future tokens must not change earlier outputs
    mask = torch.tril(torch.ones(T, T))
    x2 = x.clone()
    x2[:, 3:] += 1.0
    o1, o2 = mha(x, x, x, mask), mha(x2, x2, x2, mask)
    assert torch.allclose(o1[:, :3], o2[:, :3], atol=1e-5)


def test_sinusoidal_positions_snippet():
    pe = namespace()["sinusoidal_positions"](50, 512)
    assert pe.shape == (50, 512)
    assert pe.abs().max() <= 1.0
    assert torch.all(pe[0, 0::2] == 0) and torch.all(pe[0, 1::2] == 1)
    # matches the paper's formula at pos=5, i=4 (dimension 2i = 8)
    assert math.isclose(pe[5, 8].item(), math.sin(5 / 10000 ** (8 / 512)), rel_tol=1e-4)
    assert math.isclose(pe[5, 9].item(), math.cos(5 / 10000 ** (8 / 512)), rel_tol=1e-4)


def test_encoder_layer_snippet():
    torch.manual_seed(0)
    layer = namespace()["EncoderLayer"]().eval()
    x = torch.randn(2, 7, 512)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.allclose(y.mean(-1), torch.zeros(2, 7), atol=1e-4)  # LayerNorm output