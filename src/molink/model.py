import logging
import lightning as L
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from molink.data import get_tokenizer
from molink.rwkv7 import RWKV7Config, RWKV7ForCausalLM

logger = logging.getLogger(__name__)


class MolinkRWKV(L.LightningModule):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.config = config
        self.tokenizer = tokenizer or get_tokenizer(config)
        self.config.model.vocab_size = self.tokenizer.vocab_size

        rwkv_config = RWKV7Config(
            vocab_size=self.config.model.vocab_size,
            n_layer=self.config.model.n_layer,
            n_embd=self.config.model.n_embd,
            head_size=self.config.model.head_size,
            max_seq_len=self.config.model.max_seq_len,
            dropout=self.config.model.dropout,
            layer_norm_eps=self.config.model.layer_norm_eps,
            use_cuda_wkv=self.config.model.get("use_cuda_wkv", False),
        )
        self.model = RWKV7ForCausalLM(rwkv_config)

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        # Calculate and log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
        logger.info(f"Model config: layers={rwkv_config.n_layer}, embd={rwkv_config.n_embd}, "
                   f"head_size={rwkv_config.head_size}, vocab_size={rwkv_config.vocab_size}")
        logger.info(f"Max sequence length: {rwkv_config.max_seq_len}, dropout: {rwkv_config.dropout}")
        logger.info(f"Using CUDA WKV kernel: {rwkv_config.use_cuda_wkv}")

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)

        # Log batch information periodically
        if batch_idx % 100 == 0:
            logger.info(f"Step {self.global_step}: Processing batch {batch_idx} with shape {input_ids.shape}")
            # Calculate actual sequence lengths
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1)
                logger.info(f"  Sequence length stats - min: {seq_lengths.min().item()}, "
                           f"max: {seq_lengths.max().item()}, mean: {seq_lengths.float().mean().item():.1f}")

        # Forward pass
        logits = self.model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Calculate loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Calculate perplexity
        perplexity = torch.exp(loss)

        # Calculate token accuracy (for valid tokens only)
        with torch.no_grad():
            predictions = shift_logits.argmax(dim=-1)
            valid_mask = shift_labels != -100
            if valid_mask.any():
                correct = (predictions == shift_labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float()
            else:
                accuracy = torch.tensor(0.0, device=loss.device)

        # Log comprehensive metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log("train/perplexity", perplexity, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)
        self.log("train/batch_size", float(input_ids.size(0)), on_step=False, on_epoch=True, sync_dist=True, logger=True)

        # Log detailed info every N steps
        if self.global_step % 100 == 0:
            logger.info(f"Step {self.global_step}: loss={loss.item():.4f}, "
                       f"perplexity={perplexity.item():.2f}, accuracy={accuracy.item():.4f}")

        return loss

    def on_train_epoch_start(self):
        logger.info(f"Starting epoch {self.current_epoch}")

    def on_train_epoch_end(self):
        logger.info(f"Completed epoch {self.current_epoch}")

    def on_before_optimizer_step(self, optimizer):
        # Log gradient norms
        if self.global_step % 100 == 0:
            grad_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            self.log("train/grad_norm", grad_norm, on_step=True, prog_bar=False, logger=True)
            logger.info(f"Step {self.global_step}: gradient_norm={grad_norm:.4f}")

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith("bias") or "ln" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        decay_params = sum(p.numel() for p in decay)
        no_decay_params = sum(p.numel() for p in no_decay)
        logger.info(f"Optimizer parameter groups: {decay_params:,} with decay, {no_decay_params:,} without decay")

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.optim.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
        )

        logger.info(f"Optimizer: AdamW(lr={self.config.optim.lr}, "
                   f"betas=({self.config.optim.beta1}, {self.config.optim.beta2}), "
                   f"weight_decay={self.config.optim.weight_decay}, eps={self.config.optim.eps})")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.optim.warmup_steps,
            num_training_steps=self.config.trainer.max_steps,
        )

        logger.info(f"Learning rate scheduler: Linear warmup for {self.config.optim.warmup_steps} steps, "
                   f"then linear decay until {self.config.trainer.max_steps} steps")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "lr",
            },
        }


