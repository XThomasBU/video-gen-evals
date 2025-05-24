import warnings
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger

warnings.filterwarnings('ignore')
log = get_pylogger(__name__)


class TokenHMRPredictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        from .tokenhmr.lib.models import load_tokenhmr

        model, _ = load_tokenhmr(
            checkpoint_path=cfg.checkpoint,
            model_cfg=cfg.model_config,
            is_train_state=False,
            is_demo=True
        )
        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:, :3, :, :],
            'mask': (x[:, 3, :, :]).clip(0, 1),
        }
        model_out = self.model(batch)

        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out


class PHALP_Prime_TokenHMR(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = TokenHMRPredictor(self.cfg)


@dataclass
class Human4DConfig(FullConfig):
    checkpoint: Optional[str] = None
    model_config: Optional[str] = None
    output_dir: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)


class TokenHMRTrackGenerator:
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_config: Optional[str] = None,
        overrides: Optional[dict] = None,
    ):
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.overrides = overrides or {}
        
        # Initialize Hydra
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path="tokenhmr/lib/configs_hydra")

    def run(self, video_path, output_dir = "outputs_DEL"):
        # Initialize hydra config with overrides
        cfg: DictConfig = hydra.compose(
            config_name="config",
            overrides=[
                f"video.source={video_path}",
                f"video.output_dir={output_dir}",
                *(f"{k}={v}" for k, v in self.overrides.items())
            ]
        )

        if self.checkpoint:
            cfg.checkpoint = self.checkpoint
        if self.model_config:
            cfg.model_config = self.model_config

        tracker = PHALP_Prime_TokenHMR(cfg)
        tracker.track()


# Usage example
if __name__ == "__main__":
    runner = TokenHMRTrackGenerator(
        checkpoint="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt",
        model_config="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml",
        overrides={"render.colors": "slahmr"}
    )
    runner.run("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/v_JumpingJack_g20_c01/v_JumpingJack_g20_c01_full.mp4")
