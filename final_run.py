import pytorch_lightning as pl
import torch

import wandb
from model import VesselSegmentationModel

def train(config, epochs=250):
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    wandb.init()
    run = wandb.run
    run_id = run.id
    model = VesselSegmentationModel(config)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_f1",
            dirpath=f"./models/final_model_mse_fix/{run_id}",
            save_top_k=3,
            mode="max",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    wandb_logger = pl.loggers.WandbLogger()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=True,
        logger=wandb_logger,
    )
    trainer.fit(model)
    trainer.test(model)
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    wandb.login()

    api = wandb.Api()
    sweep = api.sweep("nlp-course-zvi-and-tomer/DMI-2024/sweeps/nqyqzk32")

    # Get best run parameters
    best_run = sweep.best_run(order='val_f1')
    best_parameters = best_run.config
    # best_parameters['in_channels'] = 1
    print(f'Best perfoming run: {best_run.id}\n\nWith parameters: {best_parameters}')
    print('Starting training with best parameters for 250 epochs...')
    train(config=best_parameters, epochs=250)
