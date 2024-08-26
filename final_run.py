import pytorch_lightning as pl
import torch
import os
import wandb
from model import VesselSegmentationModel

if not os.path.exists('models'):
    os.makedirs('models')

def train(config, epochs=250):
    """
    Train the VesselSegmentationModel using the provided configuration.

    Args:
        config (dict): Configuration parameters for the model.
        epochs (int): Number of epochs to train for. Defaults to 250.
    """
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    
    # Initialize wandb run
    wandb.init()
    run = wandb.run
    run_id = run.id
    
    # Create model instance
    model = VesselSegmentationModel(config)
    
    # Define callbacks for model checkpointing and learning rate monitoring
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_f1",
            dirpath=f"./models/final_model_mse_fix/{run_id}",
            save_top_k=3,
            mode="max",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    
    # Set up WandB logger
    wandb_logger = pl.loggers.WandbLogger()
    
    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=True,
        logger=wandb_logger,
    )
    
    # Train the model
    trainer.fit(model)
    
    # Test the model
    trainer.test(model)
    
    # Finish the wandb run
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    # Login to wandb
    wandb.login()

    # Access the wandb API
    api = wandb.Api()
    
    # Fetch the sweep information
    sweep = api.sweep("nlp-course-zvi-and-tomer/DMI-2024/sweeps/nqyqzk32")

    # Get the best run parameters from the sweep
    best_run = sweep.best_run(order='val_f1')
    best_parameters = best_run.config
    
    print(f'Best performing run: {best_run.id}\n\nWith parameters: {best_parameters}')
    print('Starting training with best parameters for 250 epochs...')
    
    # Train the model using the best parameters
    train(config=best_parameters, epochs=250)
