import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import hydra
import lightning as L
import omegaconf
import torch

from molink.data import get_dataloader, get_tokenizer
from molink.model import MolinkRWKV
from molink.utils.training import get_last_checkpoint

# Enable Tensor Cores for faster training on Ada/Ampere GPUs
torch.set_float32_matmul_precision('medium')

# Allow loading checkpoints with OmegaConf configs (PyTorch 2.6+)
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Safe resolver for arithmetic operations only
def safe_multiply(x, y):
    """Safely multiply two numbers."""
    return int(x) * int(y)

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("mul", safe_multiply)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (int(x) + int(y) - 1) // int(y))


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def train(config):
    logger.info("="*80)
    logger.info("Starting Molink Training")
    logger.info("="*80)

    # Log system information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    logger.info(f"Working directory: {os.getcwd()}")

    # Log configuration
    logger.info("\n" + "="*80)
    logger.info("Configuration:")
    logger.info("="*80)
    logger.info(f"Data source: {config.data.source}")
    if config.data.source == "hf":
        logger.info(f"  Dataset: {config.data.dataset_name}")
        logger.info(f"  Split: {config.data.split}")
        logger.info(f"  Streaming: {config.data.streaming}")
    else:
        logger.info(f"  File path: {config.data.file_path}")
    logger.info(f"Tokenizer: {config.data.tokenizer_name}")
    logger.info(f"Global batch size: {config.loader.global_batch_size}")
    logger.info(f"Number of workers: {config.loader.num_workers}")
    logger.info(f"Max training steps: {config.trainer.max_steps}")
    logger.info(f"Precision: {config.trainer.precision}")
    logger.info(f"Gradient clipping: {config.trainer.gradient_clip_val}")

    # Initialize wandb logger
    wandb_logger = None
    if hasattr(config, 'wandb') and config.wandb.get('name') is not None:
        logger.info(f"Initializing Weights & Biases logging (project: {config.wandb.project}, run: {config.wandb.name})")
        try:
            wandb_logger = L.pytorch.loggers.WandbLogger(
                config=omegaconf.OmegaConf.to_object(config),
                **config.wandb,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb logger: {e}")
            logger.warning("Continuing without wandb logging")
    else:
        logger.info("Wandb logging disabled (no run name specified)")

    # Load tokenizer
    logger.info("\n" + "="*80)
    logger.info("Loading tokenizer...")
    logger.info("="*80)
    try:
        tokenizer = get_tokenizer(config)
        logger.info(f"Tokenizer loaded successfully")
        logger.info(f"  Vocab size: {tokenizer.vocab_size}")
        logger.info(f"  BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
        logger.info(f"  EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        logger.info(f"  PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    # Initialize model
    logger.info("\n" + "="*80)
    logger.info("Initializing model...")
    logger.info("="*80)
    try:
        model = MolinkRWKV(config, tokenizer=tokenizer)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    # Load data
    logger.info("\n" + "="*80)
    logger.info("Loading training data...")
    logger.info("="*80)
    try:
        train_dataloader = get_dataloader(config, tokenizer=tokenizer)
        if hasattr(train_dataloader.dataset, '__len__'):
            logger.info(f"Dataset size: {len(train_dataloader.dataset):,} examples")
        logger.info(f"Number of batches per epoch: {len(train_dataloader):,}")
        logger.info(f"Batch size per device: {config.loader.batch_size}")
        logger.info("Data loader created successfully")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Check for checkpoint
    logger.info("\n" + "="*80)
    logger.info("Checking for checkpoints...")
    logger.info("="*80)
    ckpt_path = get_last_checkpoint(config.callback.dirpath)
    if ckpt_path:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    else:
        logger.info("No checkpoint found, starting from scratch")

    # Initialize trainer
    logger.info("\n" + "="*80)
    logger.info("Initializing trainer...")
    logger.info("="*80)
    try:
        # Select strategy based on number of devices
        num_devices = config.trainer.devices if isinstance(config.trainer.devices, int) else torch.cuda.device_count()
        if num_devices > 1:
            strategy = hydra.utils.instantiate(
                {"_target_": "lightning.pytorch.strategies.DDPStrategy", "find_unused_parameters": False}
            )
            logger.info(f"Using DDP strategy for {num_devices} GPUs")
        else:
            strategy = "auto"
            logger.info("Using single device strategy")

        trainer = hydra.utils.instantiate(
            config.trainer,
            default_root_dir=os.getcwd(),
            callbacks=[
                hydra.utils.instantiate(config.callback),
                L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
            ],
            strategy=strategy,
            logger=wandb_logger,
            enable_progress_bar=True,
        )
        logger.info("Trainer initialized successfully")
        logger.info(f"Checkpoint directory: {config.callback.dirpath}")
        logger.info(f"Saving checkpoints every {config.callback.every_n_train_steps} steps")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        raise

    # Start training
    logger.info("\n" + "="*80)
    logger.info("Starting training loop...")
    logger.info("="*80)
    try:
        trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
        logger.info("\n" + "="*80)
        logger.info("Training completed successfully!")
        logger.info("="*80)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    train()