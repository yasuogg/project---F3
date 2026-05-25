"""Smoke test: PPO checkpoint round-trip with auto-resume payload."""
import tempfile
import torch
from pathlib import Path


def test_checkpoint_roundtrip():
    # tiny model & optimizer; just verify the save() payload shape
    m = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    payload = {"policy": m.state_dict(), "optimizer": opt.state_dict(),
               "global_step": 1234, "cfg": {"foo": "bar"}}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ppo_last.pt"
        torch.save(payload, p)
        sd = torch.load(p, map_location="cpu")
        assert sd["global_step"] == 1234
        assert "policy" in sd and "optimizer" in sd
        m2 = torch.nn.Linear(4, 2)
        m2.load_state_dict(sd["policy"])
