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
                "distribution": "q_log_uniform_values",
                "q": 4,
                "min": 8,
                "max": 64,
            },
            "lr": {
                "distribution": "uniform",
                "min": 1e-4,
                "max": 7e-3,
            },
            "lambda_kl": {
                "distribution": "uniform",
                "min": 0.05,
                "max": 0.2,
            },
            "lambda_l2": {
                "values": [1e-1, 5e-2, 1e-4],
            },
            "lambda_bce": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.5,
            },
            "lambda_dice": {
                "distribution": "normal",
                "mu": 1.0,
                "sigma": 0.4,
            },
            "lambda_tversky": {
                "values": [0.0, 0.1, 0.5, 1.0],
            },
            "lambda_label": {
                "values": [0.0, 0.5, 1e1],
            },
            "lambda_focal": {
                "values": [0.0, 0.1, 0.5, 1.0],
            },
            "drop_enc": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.4,
            },
            "drop_label": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.5,
            },
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="DMI-2024")
    wandb_logger = pl.loggers.WandbLogger()

    def clear_memory():
        gc.collect()
        torch.cuda.empty_cache()

    def train():
        clear_memory()
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
            max_epochs=1,
            accelerator="auto",
            callbacks=callbacks,
            log_every_n_steps=1,
            enable_progress_bar=True,
            logger=wandb_logger,
        )
        trainer.fit(model)
        trainer.test(model)

    wandb.agent(sweep_id, train, count=20)