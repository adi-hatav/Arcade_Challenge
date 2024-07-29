import pytorch_lightning as pl
import torch
import os

import wandb
from model import VesselSegmentationModel

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    wandb.login()
    sweep_config = {
        "method": "random",
        "metric": {
            "name": "val_f1",
            "goal": "maximize",
        },
        "parameters": {
            "batch_size": {
                "values": [8],
            },
            "lr": {
                "distribution": "uniform",
                "min": 1e-4,
                "max": 5e-4,
            },
            "lambda_kl": {
                "distribution": "normal",
                "mu": 0.1,
                "sigma": 0.02,
            },
            "lambda_l2": {
                'values': [0.1, 0.01],
            },
            "lambda_bce": {
                "values": [0.0, 0.5, 1.0],
            },
            "lambda_dice": {
                "values": [1.0, 2.0],
            },
            "lambda_tversky": {
                "values": [0.0, 1.0],
            },
            "lambda_label": {
                "values": [0.25, 0.5],
            },
            "drop_enc": {
                "values": [0.3, 0.5],
            },
            "drop_label": {
                "values": [0.5, 0.9],
            },
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="DMI-2024")
    wandb_logger = pl.loggers.WandbLogger()

    def train():
        torch.set_float32_matmul_precision('medium')
        wandb.init()
        config = wandb.config
        run = wandb.run
        run_id = run.id
        model = VesselSegmentationModel(config)
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor="val_f1",
                dirpath=f"./models/wandb_sweep/{run_id}",
                save_top_k=3,
                mode="min",
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="auto",
            callbacks=callbacks,
            log_every_n_steps=1,
            enable_progress_bar=True,
            logger=wandb_logger,
        )
        trainer.fit(model)
        trainer.test(model)
    wandb.agent(sweep_id, train, count=3)