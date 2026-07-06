import torch

from rodiloco.model import (
    ModelConfig,
    Transformer,
    apply_rope,
    build_model,
    build_rope_cache,
    reference_attention,
)


def test_forward_shapes_and_loss():
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, max_seq_len=32)
    model = build_model(cfg)
    idx = torch.randint(0, 256, (3, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 16, 256)
    assert loss.ndim == 0 and loss.item() > 0


def test_attention_matches_reference():
    torch.manual_seed(0)
    B, H, T, d = 2, 4, 10, 16
    q, k, v = (torch.randn(B, H, T, d) for _ in range(3))
    ref = reference_attention(q, k, v)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ref, sdpa, atol=1e-4)


def test_causality_future_tokens_do_not_leak():
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = build_model(cfg).eval()
    idx = torch.randint(0, 64, (1, 12))
    logits_full, _ = model(idx)
    # changing the LAST token must not change earlier positions' logits
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % 64
    logits_mod, _ = model(idx2)
    assert torch.allclose(logits_full[:, :-1], logits_mod[:, :-1], atol=1e-5)


def test_rope_rotation_norm_preserved():
    cos, sin = build_rope_cache(8, 16, 10000.0, torch.device("cpu"), torch.float32)
    x = torch.randn(1, 2, 8, 16)
    y = apply_rope(x, cos, sin)
    # a rotation preserves per-vector norm
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_num_params_excludes_tied_embedding():
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4)
    model = Transformer(cfg)
    assert model.num_params(non_embedding=True) < model.num_params(non_embedding=False)


def test_model_overfits_tiny_batch():
    # a real convergence smoke test: the loop must be able to drive loss down
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, d_model=64, n_layers=2, n_heads=4, max_seq_len=16)
    model = build_model(cfg)
    from rodiloco.optim import AdamW

    opt = AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 32, (4, 16))
    y = torch.randint(0, 32, (4, 16))
    first = None
    for _ in range(60):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5
