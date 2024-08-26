import os
import pytorch_lightning as pl
import torch
import wandb
from model import VesselSegmentationModel

if __name__ == "__main__":
    experiment_name = "initial_train"
    if not os.path.exists("models"):
        os.makedirs("models")

    # Initialize wandb
    wandb.login()
    wandb.init(name=experiment_name)

    # Clear the cuda cache
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Create the model
    model = VesselSegmentationModel(
        config={ # Just an example, you can change these values
            "in_channels": 3,
            "out_shape": (25, 512, 512),
            "gn_groups": 8,
            "n_init_features": 32,
            "drop_enc": 0.2,
            "drop_label": 0.4,
            "lr": 5e-4,
            "lambda_kl": 0.1,
            "lambda_l2": 0.1,
            "lambda_bce": 0.0,
            "lambda_dice": 1.0,
            "lambda_tversky": 0.0,
            "lambda_label": 0.0,
            "alpha_tversky": 0.4,
            "beta_tversky": 0.6,
            "mse_root": False,
        }
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_f1",
            dirpath=f"./models/{experiment_name}",
            save_top_k=5,
            mode="max",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    # Continue training
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=True,
        # fast_dev_run=True,
        logger=pl.loggers.WandbLogger(),
    )

    # Train the model
    trainer.fit(model)
    trainer.test(model)
