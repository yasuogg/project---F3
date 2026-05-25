"""BC pretrain the refiner policy from collected success traces."""
from omegaconf import OmegaConf
import tyro

from rlwa.rl.bc import train_bc


def main(
    agent_cfg: str = "configs/agent/rl_refiner.yaml",
    train_cfg: str = "configs/train/ppo.yaml",
    device: str = "cuda",
):
    ac = OmegaConf.load(agent_cfg)
    tc = OmegaConf.load(train_cfg)
    cfg = OmegaConf.merge(ac, tc)
    train_bc(cfg, device=device)


if __name__ == "__main__":
    tyro.cli(main)
