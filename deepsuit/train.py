import os
import torch
import torch.optim as optim
from .config import save_yaml
from loguru import logger


# Setup optimizer and learning rate scheduler
def get_optim(config, model):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"]["value"],
        weight_decay=config["optim"]["adamw"]["weight_decay"]
    )
    return optimizer


def get_scheduler(config, optimizer):
    # ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler"]["reducelr"]["mode"],
        factor=config["scheduler"]["reducelr"]["factor"],
        patience=config["scheduler"]["reducelr"]["patience"],
        threshold=config["scheduler"]["reducelr"]["threshold"],
        min_lr=config["scheduler"]["reducelr"]["min_lr"],
        verbose=True
    )
    # add cosine...
    
    # add warm up
    
    return scheduler


import signal
class EarlyStopping:
    def __init__(self, patience=5, mode='max'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stopped_epoch = 0
        self.early_stop = False
        self.mode = mode
        self.best_model_state = None  # Store best model state
        self.config = None
        self.model = None

        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.handle_signal)

    def save_checkpoint(self):
        """Save the best model and update config."""
        if self.config is None or self.model is None:
            logger.error("Config or model is not set; cannot save checkpoint.")
            return

        model_dir = os.path.join(self.config["exp"]["dir"], "model")
        os.makedirs(model_dir, exist_ok=True)
        best_model_path = os.path.join(model_dir, "best_model.pth")
        torch.save(self.best_model_state, best_model_path)
        self.config['exp']['best_model_path'] = best_model_path
        save_yaml(self.config)
        logger.info(f"Best model saved to {best_model_path}")

    def handle_signal(self, signum, frame):
        """Handle Ctrl+C signal to save the best model."""
        logger.info(f"Signal {signum} received. Saving the best model before exiting...")
        self.save_checkpoint()
        exit(0)

    def __call__(self, val_loss, epoch, config, model):
        """Check for early stopping conditions."""
        self.config = config  # Set config
        self.model = model  # Set model

        if self.mode == "max":
            improve_condition = self.best_score is None or val_loss >= self.best_score
        elif self.mode == "min":
            improve_condition = self.best_score is None or val_loss <= self.best_score
        else:
            logger.error(f"Invalid mode: {self.mode}")
            return

        if improve_condition:
            self.best_score = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()  # Update best model state
            self.save_checkpoint()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                self.early_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}.")

        logger.info(f"Early stopping counter: {self.counter}, Best score: {self.best_score:.4f}")