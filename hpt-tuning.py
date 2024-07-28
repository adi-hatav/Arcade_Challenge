import pytorch_lightning as pl
import torch
import gc

import wandb
from model import VesselSegmentationModel

if __name__ == "__main__":
    wandb.login()
    sweep_config = {
        "method": "random",
        "metric": {
            "name": "f1_score",
            "goal": "maximize",
        },
        "parameters": {
            "batch_size": {
                "values": [8],
            },
            "lr": {
                "distribution": "uniform",
                "min": 5e-4,
                "max": 1e-2,
            },
            "lambda_kl": {
                "distribution": "normal",
                "mu": 0.1,
                "sigma": 0.02,
            },
            "lambda_l2": {
                "distribution": "normal",
                "mu": 0.1,
                "sigma": 0.02,
            },
            "lambda_bce": {
                "values": [0.0, 0.5, 1.0],
            },
            "lambda_dice": {
                "values": [0.0, 0.5, 1.0],
            },
            "lambda_tversky": {
                "values": [0.0, 0.5, 1.0],
            },
            "lambda_label": {
                "values": [0.0, 0.5, 1e1],
            },
            "lambda_focal": {
                "values": [0.0, 0.1, 0.5, 1.0],
            },
            "drop_enc": {
                "values": [0.1, 0.3, 0.5],
            },
            "drop_label": {
                "values": [0.1, 0.3, 0.5],
            },
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="DMI-2024")
    wandb_logger = pl.loggers.WandbLogger()

    def clear_memory():
        gc.collect()
        torch.cuda.empty_cache()

    def train():
        # clear_memory()
        torch.set_float32_matmul_precision('medium')
        wandb.init()
        config = wandb.config
        run = wandb.run
        run_id = run.id
        model = VesselSegmentationModel(config)
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor="val_dice_loss",
                dirpath=f"./models/{run_id}",
                save_top_k=3,
                mode="min",
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]
        trainer = pl.Trainer(
            max_epochs=5,
            accelerator="auto",
            callbacks=callbacks,
            log_every_n_steps=1,
            enable_progress_bar=True,
            logger=wandb_logger,
        )
        trainer.fit(model)
        trainer.test(model)

    wandb.agent(sweep_id, train, count=20)