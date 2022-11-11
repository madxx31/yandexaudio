import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import os
from pytorch_lightning.callbacks import LearningRateMonitor
from data_module import DataModule
from model import Model
import yaml


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(17)
    os.environ["WANDB_API_KEY"] = "04cc10900943b2d11063303c05b967e980794041"
    wandb_logger = WandbLogger(project="yandexaudio")
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg))
    dm = DataModule(cfg)
    model = Model(cfg)
    # print(model)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = Trainer(**cfg.trainer, logger=wandb_logger, callbacks=[lr_monitor])
    trainer.fit(model, dm)
    if cfg.submit:
        trainer.test(model, datamodule=dm)

    # trainer.save_checkpoint("model.ckpt", weights_only=True)


if __name__ == "__main__":
    main()
