"""
Model implementations.
Notation: [B, r, c]   = [batch size, sequence length, features]
"""

import math

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import SpearmanCorrCoef
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
from contextlib import nullcontext
import os

try:
    import torchsort
except ImportError:
    torchsort = None

from core.config import (
    ScalingConfig,
    Pipeline,
    ModelClass,
    SchedulerLR,
    TrainingBehavior,
    training_behavior,
)


def unpack_batch(batch):
    if len(batch) == 4:
        expr, gene_ids, mask, cell_type = batch
        return expr, gene_ids, mask, cell_type, None, None, None
    if len(batch) == 6:
        expr, gene_ids, mask, cell_type, y_dataset, leaf_id = batch
        return expr, gene_ids, mask, cell_type, y_dataset, leaf_id, None
    if len(batch) == 7:
        expr, gene_ids, mask, cell_type, y_dataset, leaf_id, position_ids = batch
        return expr, gene_ids, mask, cell_type, y_dataset, leaf_id, position_ids
    raise ValueError(f"Unsupported batch format with {len(batch)} items.")


def print_decoded_examples(model, tokenizer, batch):
    exprs, genes, mask, _, _, _, position_ids = unpack_batch(batch)
    genes_masked = genes.clone()
    genes_masked[mask] = model.mask_token
    print("")
    print("------------------EXAMPLES--------------------")
    print(tokenizer.batch_decode(genes_masked[:4]))
    print("----------------------------------------------")
    output = model.forward(genes_masked, exprs, position_ids=position_ids)
    logits = output["gene_logits"]
    preds = logits.argmax(dim=-1)
    print(tokenizer.batch_decode(preds[:4]))
    print("---------------ONLY MASKED--------------------")
    print(tokenizer.decode(genes[mask]))
    print("----------------------------------------------")
    print(tokenizer.decode(preds[mask]))


def compute_spearman_expressions(expr_preds, expr, mask, batch_size):
    # Spearman's correlation between ranks
    spearman = SpearmanCorrCoef(num_outputs=batch_size)
    full_spear = spearman(expr_preds.T, expr.T).mean()

    mask_spear = 0.0
    for i in range(batch_size):
        row_preds = expr_preds[i][mask[i]]
        row_expr = expr[i][mask[i]]
        # if no masked values for that cell skip
        if row_preds.shape[0] == 0:
            continue
        spearman = SpearmanCorrCoef(num_outputs=1)
        mask_spear += spearman(row_preds.reshape(-1, 1), row_expr.reshape(-1, 1))
    mask_spear /= batch_size
    return full_spear, mask_spear


def get_spearman_from_ranked_lists(pred_row, true_row):
    spearman = SpearmanCorrCoef(num_outputs=1)
    # Create a combined list of unique IDs from both tensors
    combined_ids = torch.unique(torch.cat((pred_row, true_row)))

    # Create tensors for ranks in both input tensors
    pred_ranks = torch.full((len(combined_ids),), len(pred_row), dtype=torch.long)
    true_ranks = torch.full((len(combined_ids),), len(true_row), dtype=torch.long)

    # Assign ranks based on positions in the input tensors
    for idx, id in enumerate(combined_ids):
        if id in pred_row:
            indices = (pred_row == id).nonzero(as_tuple=True)[0]
            pred_ranks[idx] = indices[0].item()

        if id in true_row:
            indices = (true_row == id).nonzero(as_tuple=True)[0]
            true_ranks[idx] = indices[0].item()

    # Compute Spearman's rank correlation for the current batch
    batch_spearman = spearman(
        pred_ranks.reshape(-1, 1).float(), true_ranks.reshape(-1, 1).float()
    )
    return batch_spearman


def compute_spearman_ranked_ids(ranked_pred, ranked_true, mask, batch_size):
    # Spearman's correlation between ranks

    full_spearman = 0.0
    mask_spearman = 0.0

    for i in range(batch_size):
        # Extract the rows for the current batch
        pred_row = ranked_pred[i]
        true_row = ranked_true[i]
        mask_row = mask[i]

        batch_spearman = get_spearman_from_ranked_lists(pred_row, true_row)
        full_spearman += batch_spearman

        if pred_row[mask_row].shape[0] == 0:
            continue
        masked_batch_spearman = get_spearman_from_ranked_lists(
            pred_row[mask_row], true_row[mask_row]
        )
        mask_spearman += masked_batch_spearman

    # Average Spearman correlation across the batch
    full_spearman /= batch_size
    mask_spearman /= batch_size

    return full_spearman, mask_spearman


def masked_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits[mask]
    masked_targets = targets[mask].to(dtype=masked_logits.dtype)
    if masked_logits.numel() == 0:
        return logits.sum() * 0.0
    return torch.nn.functional.binary_cross_entropy_with_logits(
        masked_logits,
        masked_targets,
    )


def masked_binary_cross_entropy_with_logits_old(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Legacy masked BCE used before empty masks retained a gradient path."""
    masked_logits = logits[mask]
    masked_targets = targets[mask].to(dtype=masked_logits.dtype)
    if masked_logits.numel() == 0:
        return logits.new_tensor(0.0)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        masked_logits,
        masked_targets,
    )


def masked_active_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active_mask = mask & (targets > 0)
    if not torch.any(active_mask):
        return predictions.sum() * 0.0
    return torch.nn.functional.mse_loss(
        predictions[active_mask],
        targets[active_mask],
    )


def masked_active_mse_loss_old(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Legacy active MSE used before empty masks retained a gradient path."""
    active_mask = mask & (targets > 0)
    if not torch.any(active_mask):
        return predictions.new_tensor(0.0)
    return torch.nn.functional.mse_loss(
        predictions[active_mask],
        targets[active_mask],
    )


def legacy_binned_losses(predictions, active_logits, targets, mask):
    """Original binned loss formulas, including historical empty-mask NaNs."""
    mse_mask = (mask & (active_logits.sigmoid() > 0.5)).to(float)
    mse = (
        F.mse_loss(predictions * mse_mask, targets * mse_mask, reduction="sum")
        / mse_mask.sum()
    )
    bce_mask = mask.to(float)
    bce = (
        F.binary_cross_entropy_with_logits(
            active_logits * bce_mask, (targets > 0) * bce_mask, reduction="sum"
        )
        / bce_mask.sum()
    )
    return mse, bce


# Base Model
class BaseModel(pl.LightningModule):
    """
    Implements the basic embed-encode-decode logic:

        1. Embed a pair of (gene_ids, gene_expression) into a single vector of d_model dimensions

        2. Process this vector according to a given encoder (default is nn.Identity)

        3. Decodes the encoded representation to generate:
            - gene expression prediction (float)
            - probability of active gene (float in [0,1])
    """

    def __init__(self, config: ScalingConfig):
        super().__init__()

        # Set a priori
        self.training_behavior = training_behavior(config)
        self.mask_value = -1

        # What is required by the model
        self.pipeline = config.metadata.pipeline
        self.lr = config.model.lr
        self.dropout = config.model.dropout
        self.total_genes = config.model.total_genes
        self.n_steps = config.trainer.n_steps
        self.d_model = config.model.d_model
        self.lr_scheduler_class = config.trainer.lr_scheduler
        self.config = config

        # mask token is pipeline dependent
        if (
            config.metadata.pipeline == Pipeline.DEFAULT
            or config.metadata.pipeline == Pipeline.GENEFORMER
            or config.metadata.pipeline == Pipeline.GENECORPUS
        ):
            self.mask_token = self.total_genes + 1
        else:  # only tokenized text for now
            self.mask_token = 1

        # Embedder & Decoder are pipeline-dependent
        if self.pipeline == Pipeline.DEFAULT:
            # Embed gene tokens and expression and sum together (all visible)
            self.embedder = GeneExprEmbedding(
                n_tokens=config.model.total_genes,
                d_model=config.model.d_model,
                pad_id=config.model.total_genes,
            )

            # Decode into active prob and predicted expr
            self.decoder = ExprDecoder(
                d_model=config.model.d_model, dropout=self.dropout
            )

        elif (
            self.pipeline == Pipeline.GENEFORMER
            or self.pipeline == Pipeline.GENECORPUS
            or self.pipeline == Pipeline.TOKENIZED_TEXT
        ):
            # Embed only gene tokens (some are masked)
            self.embedder = nn.Embedding(
                num_embeddings=config.model.total_genes + 2,  # vocab + pad + mask
                embedding_dim=config.model.d_model,
                padding_idx=config.model.total_genes
                if self.pipeline != Pipeline.TOKENIZED_TEXT
                else 0,
            )

            self.positional = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout,
                max_len=config.model.context_length,
            )

            # Decode into probability over gene tokens
            self.decoder = GeneDecoder(
                d_model=config.model.d_model,
                vocab_size=config.model.total_genes + 2,
                dropout=self.dropout,
            )

        if self.pipeline == Pipeline.TOKENIZED_TEXT:
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=config.paths.path_to_tokenizer
            )
        else:
            self.tokenizer = None

        self.compute_spearman = config.trainer.compute_spearman
        # Main Encoder
        self.encoder = nn.Identity()

    @property
    def total_params(self) -> int:
        """
        Returns the total number of parameters in the model (computed on the fly).
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def total_params_h(self) -> str:
        """
        Returns total params in human readable format.
        """
        return f"{self.total_params:,}"

    def log_spearman(self, full_spear, mask_spear):
        # log spearman, differentiate bw train and val,
        # thus avoid writing multiple functions
        if self.training:
            mode = "train"
        else:
            mode = "val"
        self.log(f"{mode}_spearman_full", full_spear, prog_bar=True)
        self.log(f"{mode}_spearman_mask", mask_spear, prog_bar=True)

    def get_attention_mask(self, gene_ids: torch.Tensor) -> torch.Tensor:
        """Return True for real and MLM-mask tokens, and False for padding."""
        pad_token = 0 if self.pipeline == Pipeline.TOKENIZED_TEXT else self.total_genes
        return gene_ids != pad_token

    def get_encoder_output(self, g, x, position_ids=None):
        if self.training_behavior != TrainingBehavior.CORRECTED:
            return self.get_encoder_output_old(g, x, position_ids=position_ids)
        # Encode genes and expressions

        if self.pipeline == Pipeline.DEFAULT:
            total_embs = self.embedder(g, x)

        elif (
            self.pipeline == Pipeline.GENEFORMER
            or self.pipeline == Pipeline.GENECORPUS
            or self.pipeline == Pipeline.TOKENIZED_TEXT
        ):
            total_embs = self.embedder(g)
            total_embs = self.positional(total_embs, positions=position_ids)

        # MLM-mask tokens remain valid attention inputs; only padding is hidden.
        attention_mask = self.get_attention_mask(g)
        if isinstance(self.encoder, nn.Identity):
            encoder_output = self.encoder(total_embs)
        else:
            encoder_output = self.encoder(total_embs, attention_mask=attention_mask)

        return encoder_output

    def get_encoder_output_old(self, g, x, position_ids=None):
        """Legacy encoder path that did not pass a padding attention mask."""
        if self.pipeline == Pipeline.DEFAULT:
            total_embs = self.embedder(g, x)
        elif (
            self.pipeline == Pipeline.GENEFORMER
            or self.pipeline == Pipeline.GENECORPUS
            or self.pipeline == Pipeline.TOKENIZED_TEXT
        ):
            total_embs = self.embedder(g)
            total_embs = self.positional(total_embs, positions=position_ids)
        legacy_forward = getattr(self.encoder, "forward_old", None)
        if legacy_forward is not None:
            return legacy_forward(total_embs)
        return self.encoder(total_embs)

    def store_embeddings_and_types(
        self, expr, gene_ids, y_dataset, cell_type, leaf_id, position_ids=None
    ):
        # Set up automatic mixed precision context if applicable
        is_amp = self.trainer.precision in ["16-mixed", "bf16-mixed"]
        amp_dtype = (
            torch.bfloat16 if "bf16" in self.trainer.precision else torch.float16
        )
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (self.device.type == "cuda" and is_amp)
            else nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            encoder_output = self.get_encoder_output(
                gene_ids, expr, position_ids=position_ids
            )
            cell_embeddings = encoder_output.mean(dim=1)

            self.all_embeddings.append(cell_embeddings.detach().cpu())
            self.all_labels.append(cell_type.detach().cpu())
            self.all_dataset_ids.append(y_dataset.detach().cpu())
            self.all_leaf_ids.extend([str(x) for x in leaf_id])

    def store_embeddings_and_types_old(
        self, expr, gene_ids, y_dataset, cell_type, leaf_id, position_ids=None
    ):
        """Legacy embedding collection without padding-aware attention."""
        is_amp = self.trainer.precision in ["16-mixed", "bf16-mixed"]
        amp_dtype = (
            torch.bfloat16 if "bf16" in self.trainer.precision else torch.float16
        )
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (self.device.type == "cuda" and is_amp)
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            encoder_output = self.get_encoder_output_old(
                gene_ids, expr, position_ids=position_ids
            )
            cell_embeddings = encoder_output.mean(dim=1)
            self.all_embeddings.append(cell_embeddings.detach().cpu())
            self.all_labels.append(cell_type.detach().cpu())
            self.all_dataset_ids.append(y_dataset.detach().cpu())
            self.all_leaf_ids.extend([str(x) for x in leaf_id])

    def forward(
        self,
        g: torch.Tensor,  # [B, r]
        x: torch.Tensor,  # [B, r]
        position_ids: torch.Tensor | None = None,
    ) -> dict:
        # Check dtypes
        assert x.dtype == torch.float, f"Expected expr dtype torch.float, got {x.dtype}"
        assert g.dtype == torch.long, (
            f"Expected gene_ids dtype torch.long, got {g.dtype}"
        )

        # The encodings we want to learn
        encoder_output = self.get_encoder_output(g, x, position_ids=position_ids)

        # Decode
        output = self.decoder(encoder_output)

        return output

    def forward_old(
        self,
        g: torch.Tensor,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> dict:
        """Legacy forward path without padding-aware attention."""
        assert x.dtype == torch.float, f"Expected expr dtype torch.float, got {x.dtype}"
        assert g.dtype == torch.long, (
            f"Expected gene_ids dtype torch.long, got {g.dtype}"
        )
        encoder_output = self.get_encoder_output_old(g, x, position_ids=position_ids)
        return self.decoder(encoder_output)

    def step_default(self, batch, batch_idx, split):
        # Read in the batch, types are (float, int, bool, _)
        expr, gene_ids, mask, cell_type, y_dataset, leaf_id, _ = unpack_batch(batch)

        assert mask.dtype == torch.bool, (
            f"Expected mask dtype torch.bool, got {mask.dtype}"
        )

        # Apply mask
        expr_masked = expr.clone().detach()
        expr_masked[mask] = -1.0

        # Pass through the model
        output = self.forward(gene_ids, expr_masked)

        if split != "train" and self.compute_spearman:
            # Compute Spearman
            full_spearman, mask_spearman = compute_spearman_expressions(
                output["mlm"], expr, mask, batch_size=expr.shape[0]
            )
            self.log_spearman(full_spearman, mask_spearman)

        mse_loss = masked_active_mse_loss(
            output["mlm"],
            expr,
            mask,
        )

        bce_loss = masked_binary_cross_entropy_with_logits(
            output["active_logits"],
            expr > 0,
            mask,
        )

        # Also track accuracy
        with torch.no_grad():
            active_probs = torch.nn.functional.sigmoid(output["active_logits"])
            masked_correct = ((active_probs > 0.5) == expr.bool())[mask]
            accuracy = (
                masked_correct.float().mean()
                if masked_correct.numel()
                else active_probs.new_zeros(())
            )
            dispersion = output["mlm"].std(dim=-1).mean()

        # Store cell embeddings and types
        if split == "val":
            self.store_embeddings_and_types(
                expr, gene_ids, y_dataset, cell_type, leaf_id
            )

        return {
            "losses": {"bce": bce_loss, "mse": mse_loss},
            "metrics": {"accuracy": accuracy, "dispersion": dispersion},
        }

    def step_geneformer(self, batch, batch_idx, split="train"):
        # Read in the batch, types are (float, int, bool, _)
        expr, gene_ids, mask, cell_type, y_dataset, leaf_id, position_ids = (
            unpack_batch(batch)
        )

        # Apply mask
        gene_ids_masked = gene_ids.clone().detach()
        gene_ids_masked[mask] = self.mask_token  # depends on pipeline

        # Pass through the model
        # [B, r, vocab_size]
        output = self.forward(gene_ids_masked, expr, position_ids=position_ids)
        output = output["gene_logits"]

        # Compute BCE
        B, r, vocab_size = output.shape
        mask = mask.reshape(B * r)
        logits = output.reshape(B * r, vocab_size)[mask]
        targets = gene_ids.reshape(B * r)[mask]
        bce_loss = (
            torch.nn.functional.cross_entropy(logits, targets)
            if targets.numel()
            else output.sum() * 0.0
        )

        # Also track accuracy & dispersion (entropy)
        with torch.no_grad():
            if targets.numel():
                accuracy = (logits.argmax(dim=-1) == targets).to(float).mean()
                probs = torch.softmax(logits, dim=-1)
                dispersion = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                max_dispersion = torch.log2(torch.tensor(vocab_size))
                dispersion /= max_dispersion  # normalize to [0, 1]
            else:
                accuracy = output.new_zeros(())
                dispersion = output.new_zeros(())
            if (
                split != "train"
                and self.pipeline == Pipeline.TOKENIZED_TEXT
                and self.tokenizer is not None
                and batch_idx == 0
            ):
                print_decoded_examples(self, self.tokenizer, batch)

            if split != "train" and self.compute_spearman:
                # Compute Spearman
                full_spearman, mask_spearman = compute_spearman_ranked_ids(
                    output.argmax(dim=-1), gene_ids, mask, batch_size=expr.shape[0]
                )
                self.log_spearman(full_spearman, mask_spearman)

        # Store cell embeddings and types
        if split == "val":
            self.store_embeddings_and_types(
                expr,
                gene_ids,
                y_dataset,
                cell_type,
                leaf_id,
                position_ids=position_ids,
            )

        return {
            "losses": {"bce": bce_loss},
            "metrics": {"accuracy": accuracy, "dispersion": dispersion},
        }

    def step_default_old(self, batch, batch_idx, split):
        """Legacy default-pipeline step from before masking/attention fixes."""
        expr, gene_ids, mask, cell_type, y_dataset, leaf_id, _ = unpack_batch(batch)
        assert mask.dtype == torch.bool, (
            f"Expected mask dtype torch.bool, got {mask.dtype}"
        )

        expr_masked = expr.clone().detach()
        expr_masked[mask] = -1.0
        output = self.forward_old(gene_ids, expr_masked)

        if split != "train" and self.compute_spearman:
            full_spearman, mask_spearman = compute_spearman_expressions(
                output["mlm"], expr, mask, batch_size=expr.shape[0]
            )
            self.log_spearman(full_spearman, mask_spearman)

        if self.training_behavior == TrainingBehavior.LEGACY:
            mse_loss, bce_loss = legacy_binned_losses(
                output["mlm"], output["active_logits"], expr, mask
            )
        else:
            mse_loss = masked_active_mse_loss_old(output["mlm"], expr, mask)
            bce_loss = masked_binary_cross_entropy_with_logits_old(
                output["active_logits"], expr > 0, mask
            )

        with torch.no_grad():
            active_probs = torch.nn.functional.sigmoid(output["active_logits"])
            accuracy = ((active_probs > 0.5) == expr.bool())[mask].sum() / mask.sum()
            dispersion = output["mlm"].std(dim=-1).mean()

        if split == "val":
            self.store_embeddings_and_types_old(
                expr, gene_ids, y_dataset, cell_type, leaf_id
            )

        return {
            "losses": {"bce": bce_loss, "mse": mse_loss},
            "metrics": {"accuracy": accuracy, "dispersion": dispersion},
        }

    def step_geneformer_old(self, batch, batch_idx, split="train"):
        """Legacy token-pipeline step from before masking/attention fixes."""
        expr, gene_ids, mask, cell_type, y_dataset, leaf_id, position_ids = (
            unpack_batch(batch)
        )
        gene_ids_masked = gene_ids.clone().detach()
        gene_ids_masked[mask] = self.mask_token
        output = self.forward_old(gene_ids_masked, expr, position_ids=position_ids)[
            "gene_logits"
        ]

        batch_size, sequence_length, vocab_size = output.shape
        flat_mask = mask.reshape(batch_size * sequence_length)
        logits = output.reshape(batch_size * sequence_length, vocab_size)[flat_mask]
        targets = gene_ids.reshape(batch_size * sequence_length)[flat_mask]
        bce_loss = torch.nn.functional.cross_entropy(logits, targets)

        with torch.no_grad():
            accuracy = (logits.argmax(dim=-1) == targets).to(float).mean()
            probs = torch.softmax(logits, dim=-1)
            dispersion = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            max_dispersion = torch.log2(torch.tensor(vocab_size))
            dispersion /= max_dispersion
            if (
                split != "train"
                and self.pipeline == Pipeline.TOKENIZED_TEXT
                and self.tokenizer is not None
                and batch_idx == 0
            ):
                print_decoded_examples(self, self.tokenizer, batch)
            if split != "train" and self.compute_spearman:
                full_spearman, mask_spearman = compute_spearman_ranked_ids(
                    output.argmax(dim=-1),
                    gene_ids,
                    flat_mask,
                    batch_size=expr.shape[0],
                )
                self.log_spearman(full_spearman, mask_spearman)

        if split == "val":
            self.store_embeddings_and_types_old(
                expr,
                gene_ids,
                y_dataset,
                cell_type,
                leaf_id,
                position_ids=position_ids,
            )

        return {
            "losses": {"bce": bce_loss},
            "metrics": {"accuracy": accuracy, "dispersion": dispersion},
        }

    def step_old(self, batch, batch_idx, split="train"):
        """Dispatch to the legacy pipeline step."""
        if self.pipeline == Pipeline.DEFAULT:
            return self.step_default_old(batch, batch_idx, split)
        if (
            self.pipeline == Pipeline.GENEFORMER
            or self.pipeline == Pipeline.GENECORPUS
            or self.pipeline == Pipeline.TOKENIZED_TEXT
        ):
            return self.step_geneformer_old(batch, batch_idx, split)
        raise NotImplementedError(f"Pipeline {self.pipeline} not implemented")

    def step(self, batch, batch_idx, split="train"):
        if self.training_behavior != TrainingBehavior.CORRECTED:
            return self.step_old(batch, batch_idx, split)
        if self.pipeline == Pipeline.DEFAULT:
            return self.step_default(batch, batch_idx, split)
        elif (
            self.pipeline == Pipeline.GENEFORMER
            or self.pipeline == Pipeline.GENECORPUS
            or self.pipeline == Pipeline.TOKENIZED_TEXT
        ):
            return self.step_geneformer(batch, batch_idx, split)

    def training_step(self, batch, batch_idx):
        # for geneformer: dispersion = entropy of the predicted ids
        # for default: dispersion = std of predicted bins
        output = self.step(batch, batch_idx, split="train")
        self.log_stuff(output, split="train")
        return sum(output["losses"].values())

    def training_step_old(self, batch, batch_idx):
        """Legacy Lightning training entry point."""
        output = self.step_old(batch, batch_idx, split="train")
        self.log_stuff(output, split="train")
        return sum(output["losses"].values())

    def validation_step(self, batch, batch_idx):
        output = self.step(batch, batch_idx, split="val")
        self.log_stuff(output, split="val")
        return sum(output["losses"].values())

    def validation_step_old(self, batch, batch_idx):
        """Legacy Lightning validation entry point."""
        output = self.step_old(batch, batch_idx, split="val")
        self.log_stuff(output, split="val")
        return sum(output["losses"].values())

    def on_validation_start(self):
        self.all_embeddings = []
        self.all_labels = []
        self.all_dataset_ids = []
        self.all_leaf_ids = []

    def build_embedding_artifact(self):
        return {
            "global_step": int(self.global_step),
            "embeddings": torch.cat(self.all_embeddings, dim=0),
            "labels": torch.cat(self.all_labels, dim=0),
            "dataset_ids": (
                torch.cat(self.all_dataset_ids, dim=0) if self.all_dataset_ids else None
            ),
            "leaf_ids": self.all_leaf_ids,
        }

    def save_data(self, data, filename_prefix=None):
        # Built the path to ckpt dir
        ckpt_dir = os.path.join(
            self.config.paths.path_to_ckpt_dir,
            self.config.metadata.title,  # needed when we want to collect multiple runs where same params vary intra run and other inter run
            self.config.metadata.run_name,
        )

        if not os.path.exists(ckpt_dir):
            # Create dir
            os.makedirs(ckpt_dir)
            # Copy the config
            self.config.save(os.path.join(ckpt_dir, "config.yml"))

        # Save the data in a pytorch optimized format
        filename = (
            f"{self.config.metadata.title}_"
            f"{self.config.metadata.run_name}_"
            f"{self.global_step}"
        )
        if filename_prefix:
            filename = f"{filename_prefix}_{filename}"
        # first save to temp file
        torch.save(data, os.path.join(ckpt_dir, f"{filename}.pt.tmp"))
        # then move, so consumer will not read partial file
        # this work because file movement is atomic operation
        # whereas write is not
        os.replace(
            src=os.path.join(ckpt_dir, f"{filename}.pt.tmp"),
            dst=os.path.join(ckpt_dir, f"{filename}.pt"),
        )

    def on_validation_end(self):
        # create dict to be saved
        data = self.build_embedding_artifact()

        # save data in checkpoint dir
        self.save_data(data)

        if getattr(self, "_save_persistent_embedding_checkpoint", False):
            self.save_data(data, filename_prefix="checkpoint")

        # cleanup memory
        del self.all_embeddings
        del self.all_labels
        del self.all_dataset_ids
        del self.all_leaf_ids
        self._save_persistent_embedding_checkpoint = False

    def on_train_end(self):
        # Built the path to ckpt dir
        ckpt_dir = os.path.join(
            self.config.paths.path_to_ckpt_dir,
            self.config.metadata.title,  # needed when we want to collect multiple runs where same params vary intra run and other inter run
            self.config.metadata.run_name,
        )

        if not os.path.exists(ckpt_dir):
            # Create dir
            os.makedirs(ckpt_dir)
            # Copy the config
            self.config.save(os.path.join(ckpt_dir, "config.yml"))

        trainer = getattr(self, "trainer", None)
        max_steps = getattr(trainer, "max_steps", None)
        is_step_complete = (
            max_steps is None or max_steps < 0 or self.global_step >= max_steps
        )
        if is_step_complete:
            with open(os.path.join(ckpt_dir, "DONE"), "w") as f:
                f.write("done")

    def log_stuff(self, output, split):
        losses_to_log = {
            f"{split}_{loss}": output["losses"][loss]
            for loss in output["losses"].keys()
        }
        if "metrics" in output.keys():
            metrics_to_log = {
                f"{split}_{metric}": output["metrics"][metric]
                for metric in output["metrics"].keys()
            }
            stuff_to_log = losses_to_log | metrics_to_log
        else:
            stuff_to_log = losses_to_log

        for k, v in stuff_to_log.items():
            self.log(
                k,
                v,
                on_step=None if split == "train" else False,
                on_epoch=None if split == "train" else True,
                prog_bar=True,
            )

    def _init_lr_scheduler(self):
        if self.lr_scheduler_class == SchedulerLR.LINEAR:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.001,
                total_iters=self.n_steps,
            )
        elif self.lr_scheduler_class == SchedulerLR.ONECYCLE:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lr,
                total_steps=self.n_steps,
            )
        elif self.lr_scheduler_class == SchedulerLR.CONSTANT:
            # We use a dummy scheduler (1.0 factor) to maintain the same code structure
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,  # effectively, constant lr
                end_factor=1.0,
                total_iters=self.n_steps,
            )
        elif self.lr_scheduler_class == SchedulerLR.WARMUP_CONSTANT:
            warmup_steps = min(
                max(1, getattr(self.config.trainer, "lr_warmup_steps", 500)),
                max(1, self.n_steps),
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=0.000001,
                        end_factor=1.0,
                        total_iters=warmup_steps,
                    ),
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=1.0,
                        end_factor=1.0,
                        total_iters=max(1, self.n_steps - warmup_steps),
                    ),
                ],
                milestones=[warmup_steps],
            )
        elif self.lr_scheduler_class == SchedulerLR.WARMUP_CONSTANT_LINEAR:
            warmup_steps = getattr(self.config.trainer, "lr_warmup_steps", 1_000)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=0.000001,
                        end_factor=1.0,
                        total_iters=warmup_steps,
                    ),
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer, start_factor=1.0, end_factor=1.0
                    ),
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=1.0,
                        end_factor=0.0001,
                        total_iters=2_000,
                    ),
                ],
                milestones=[
                    warmup_steps,
                    warmup_steps + int((self.n_steps - 3_000) * 0.7),
                ],
            )

    def configure_optimizers(self):
        # init optimizer
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            betas=(0.9, 0.95),
            lr=self.lr,
        )
        self._init_lr_scheduler()

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,  # Check after each step
            },
        }

    def on_before_optimizer_step(self, optimizer):
        # to show grad norm
        total_grad_norm = 0
        for _, param in self.named_parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item() ** 2
        total_grad_norm = total_grad_norm**0.5
        self.log(
            "train_grad_norm",
            total_grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        # Log learning rate
        lr = optimizer.param_groups[0]["lr"]
        self.log(
            "lr",
            lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )


class BioRangoTango(BaseModel):
    """
    Rank based model, self-supervised learning strategy is not masked language modeling but gene ranking.
    It starts from the set of the top k most expressed genes (k is the context length of the model),
    and it assigns the rank for every gene in the sequence.

    We employ a differentiable ranking operation, which allows us to make the gradients flow through this model.
    """

    def __init__(self, config: ScalingConfig):
        super().__init__(config)

        if self.pipeline == Pipeline.DEFAULT:
            raise NotImplementedError(
                "BioRangoTango needs rank information, so default pipeline is not yet ok"
            )

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            n_heads=config.model.transformer.n_heads,
            dropout=self.dropout,
            gating=config.model.transformer.gating,
            n_layers=config.model.transformer.n_layers,
            norm_first=config.model.transformer.norm_first,
        )

        self.ranker = nn.Linear(self.d_model, 1)

    def get_encoder_output(self, g, x, position_ids=None):
        total_embs = self.embedder(g)
        total_embs = self.positional(total_embs, positions=position_ids)
        return self.encoder(total_embs, attention_mask=self.get_attention_mask(g))

    def get_encoder_output_old(self, g, x, position_ids=None):
        """Legacy ranking encoder path without a padding attention mask."""
        total_embs = self.embedder(g)
        total_embs = self.positional(total_embs, positions=position_ids)
        return self.encoder.forward_old(total_embs)

    def step_geneformer(self, batch, batch_idx, split="train"):
        expr, gene_ids, mask, _, _, _, position_ids = unpack_batch(batch)
        batch_size = expr.shape[0]
        seq_length = expr.shape[1]

        output = self.get_encoder_output(gene_ids, expr, position_ids=position_ids)
        rank_logits = self.ranker(output).squeeze(-1)
        # from [B, r, 1] to [B, r]
        pred_ranks = torchsort.soft_rank(rank_logits)
        # ranks start at 1... damned mathemagicians
        true_ranks = 1 + torch.tensor(
            [i for i in range(seq_length)], device=pred_ranks.device
        )
        true_ranks = torch.vstack(batch_size * [true_ranks])
        loss = torch.square(pred_ranks - true_ranks)
        loss = loss.mean()

        with torch.no_grad():
            if split != "train" and self.compute_spearman:
                full_spearman, masked_spearman = compute_spearman_ranked_ids(
                    output.argmax(dim=-1), gene_ids, mask, batch_size=expr.shape[0]
                )
                self.log_spearman(full_spearman, masked_spearman)

            if split != "train" and batch_idx == 0:
                print(" ")
                print("-------LOGITS------")
                print(rank_logits[:4])
                print("--------PRED-------")
                print(pred_ranks[:4])
                print("--------TRUE-------")
                print(true_ranks[:4])

        return {"losses": {"rank": loss}}


# Transformer
class TransformerModel(BaseModel):
    """
    Base model with a GeneExprEmbedding, ExprDecoder and a vanilla transformer encoder.
    """

    def __init__(self, config: ScalingConfig):
        super().__init__(config)

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            n_heads=config.model.transformer.n_heads,
            dropout=self.dropout,
            gating=config.model.transformer.gating,
            n_layers=config.model.transformer.n_layers,
            norm_first=config.model.transformer.norm_first,
        )


# BioFormer
class BioFormerModel(BaseModel):
    """
    Base model with the bioformer encoder.
    """

    def __init__(self, config: ScalingConfig):
        super().__init__(config)

        # Bioformer Encoder
        self.encoder = BioFormerStack(
            d_model=self.d_model,
            d_z=config.model.bioformer.d_z,
            d_opm=config.model.bioformer.d_opm,
            n_heads=config.model.bioformer.n_heads,
            n_layers=config.model.bioformer.n_layers,
            pair_updates=config.model.bioformer.pair_updates,
            dropout=self.dropout,
            norm_first=config.model.bioformer.norm_first,
            gating=config.model.bioformer.gating,
            chunk_size=config.model.bioformer.chunk_size,
        )


# FFNN
class FFNN(BaseModel):
    """
    A simple neural network.
    """

    def __init__(self, config: ScalingConfig):
        super().__init__(config)

        # Bioformer Encoder
        self.encoder = Transition(d_model=self.d_model, dropout=self.dropout)


# BERT Model
class BERTModel(pl.LightningModule):
    """
    A vanilla encoder-only BERT-like model using PyTorch's default modules.
    """

    def __init__(self, config: ScalingConfig, _legacy_behavior: bool = False):
        super().__init__()

        assert (
            config.metadata.pipeline == Pipeline.GENEFORMER
            or config.metadata.pipeline == Pipeline.GENECORPUS
            or config.metadata.pipeline == Pipeline.TOKENIZED_TEXT
        ), f"BERTModel is not compatible with pipeline {config.metadata.pipeline}"
        # Extract parameters from ScalingConfig
        self.lr = config.model.lr
        self.n_steps = config.trainer.n_steps
        self.total_genes = config.model.total_genes
        self.d_model = config.model.d_model
        self.context_length = config.model.context_length
        self.pipeline = config.metadata.pipeline
        _legacy_behavior = (
            _legacy_behavior or training_behavior(config) != TrainingBehavior.CORRECTED
        )
        self._legacy_behavior = _legacy_behavior

        if (
            config.metadata.pipeline == Pipeline.GENEFORMER
            or config.metadata.pipeline == Pipeline.GENECORPUS
        ):
            self.mask_token = self.total_genes + 1
        else:
            self.mask_token = 1
        self.pad_token = (
            0
            if (
                config.metadata.pipeline == Pipeline.TOKENIZED_TEXT
                and not _legacy_behavior
            )
            else self.total_genes
        )

        # Define BERT configuration
        bert_config = BertConfig(
            vocab_size=self.total_genes + 2,  # Include padding and mask tokens
            hidden_size=self.d_model,
            num_hidden_layers=config.model.transformer.n_layers,
            num_attention_heads=config.model.transformer.n_heads,
            intermediate_size=self.d_model * 4,
            max_position_embeddings=self.context_length,
            hidden_dropout_prob=config.model.dropout,
            attention_probs_dropout_prob=config.model.dropout,
            pad_token_id=self.pad_token,
        )

        # Initialize BERT model
        self.bert = BertModel(bert_config)

        # Decoder for token prediction
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(self.d_model, self.total_genes + 2),  # Predict token IDs
        )

    @classmethod
    def from_config_old(cls, config: ScalingConfig):
        """Build a BERT model with the complete historical masking behavior."""
        return cls(config, _legacy_behavior=True)

    def log_spearman(self, full_spear, mask_spear):
        # TODO find a better way than copying method from BaseModel
        # log spearman, differentiate bw train and val,
        # thus avoid writing multiple functions
        if self.training:
            mode = "train"
        else:
            mode = "val"
        self.log(f"{mode}_spearman_full", full_spear, prog_bar=True)
        self.log(f"{mode}_spearman_mask", mask_spear, prog_bar=True)

    def forward(self, gene_ids, attention_mask=None):
        """
        Forward pass through the BERT model.

        Args:
            gene_ids: Tensor of shape [B, r] containing gene token IDs.
            attention_mask: Optional tensor of shape [B, r] indicating which tokens to attend to.

        Returns:
            A dictionary containing:
                - "logits": Predicted token logits [B, r, vocab_size].
        """
        # Pass through BERT
        bert_output = self.bert(input_ids=gene_ids, attention_mask=attention_mask)

        # Use the last hidden state for decoding
        hidden_states = bert_output.last_hidden_state  # [B, r, d_model]

        # Decode token predictions
        logits = self.decoder(hidden_states)  # [B, r, vocab_size]

        return {"logits": logits}

    def forward_old(self, gene_ids):
        """Legacy BERT forward path without a padding attention mask."""
        return self.forward(gene_ids, attention_mask=None)

    def step(self, batch, split="train"):
        """
        Shared logic for training and validation steps.
        """
        if self._legacy_behavior:
            return self.step_old(batch, split=split)

        _, gene_ids, mask, _, _, _, _ = unpack_batch(batch)

        # Apply mask to gene IDs
        gene_ids_masked = gene_ids.clone()
        gene_ids_masked[mask] = self.mask_token  # Mask token

        # MLM-mask tokens remain visible; only padding is excluded from attention.
        attention_mask = gene_ids != self.pad_token
        output = self.forward(gene_ids_masked, attention_mask=attention_mask)

        # Compute Spearman
        if split != "train":
            with torch.no_grad():
                ranked_pred = output["logits"].argmax(dim=-1)
                full_spearman, mask_spearman = compute_spearman_ranked_ids(
                    ranked_pred, gene_ids, mask, batch_size=gene_ids.shape[0]
                )
                self.log_spearman(full_spearman, mask_spearman)

        # Compute loss (Cross-Entropy) and accuracy on masked tokens
        logits = output["logits"].reshape(-1, self.total_genes + 2)
        targets = gene_ids.reshape(-1)

        masked_logits = logits[mask.reshape(-1)]
        masked_targets = targets[mask.reshape(-1)]

        loss = (
            nn.functional.cross_entropy(masked_logits, masked_targets)
            if masked_targets.numel()
            else logits.sum() * 0.0
        )
        # subset_non_masked = (~mask) & (torch.rand_like(mask, dtype=torch.float) <= 0.1)
        # loss += nn.functional.cross_entropy(logits[subset_non_masked.view(-1)], targets[subset_non_masked.view(-1)])
        with torch.no_grad():
            accuracy = (
                (masked_logits.argmax(dim=-1) == masked_targets).float().mean()
                if masked_targets.numel()
                else logits.new_zeros(())
            )
        return loss, accuracy

    def step_old(self, batch, split="train"):
        """Legacy BERT step that did not pass a padding attention mask."""
        _, gene_ids, mask, _, _, _, _ = unpack_batch(batch)
        gene_ids_masked = gene_ids.clone()
        gene_ids_masked[mask] = self.mask_token
        output = self.forward(gene_ids_masked)

        if split != "train":
            with torch.no_grad():
                ranked_pred = output["logits"].argmax(dim=-1)
                full_spearman, mask_spearman = compute_spearman_ranked_ids(
                    ranked_pred, gene_ids, mask, batch_size=gene_ids.shape[0]
                )
                self.log_spearman(full_spearman, mask_spearman)

        logits = output["logits"].reshape(-1, self.total_genes + 2)
        targets = gene_ids.reshape(-1)
        masked_logits = logits[mask.reshape(-1)]
        masked_targets = targets[mask.reshape(-1)]
        loss = nn.functional.cross_entropy(masked_logits, masked_targets)
        with torch.no_grad():
            accuracy = (masked_logits.argmax(dim=-1) == masked_targets).float().mean()
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.step(batch, split="train")
        self.log("train_bce", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def training_step_old(self, batch, batch_idx):
        """Legacy BERT Lightning training entry point."""
        loss, accuracy = self.step_old(batch, split="train")
        self.log("train_bce", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.step(batch, split="val")
        self.log("val_bce", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step_old(self, batch, batch_idx):
        """Legacy BERT Lightning validation entry point."""
        loss, accuracy = self.step_old(batch, split="val")
        self.log("val_bce", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.001,
            total_iters=self.n_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# Custom Modules
class PositionalEncoding(nn.Module):
    """
    From https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.register_buffer("div_term", div_term)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if positions is None:
            x = x + self.pe[: x.size(1)]
        else:
            safe_positions = positions.clamp(min=0)
            pos_emb = x.new_zeros((*positions.shape, self.d_model))
            angles = safe_positions.unsqueeze(-1).to(x.dtype) * self.div_term.to(
                x.dtype
            )
            pos_emb[..., 0::2] = torch.sin(angles)
            pos_emb[..., 1::2] = torch.cos(angles)
            pos_emb = pos_emb * (positions >= 0).unsqueeze(-1)
            x = x + pos_emb
        return self.dropout(x)


class GeneExprEmbedding(nn.Module):
    """
    Encodes a pair of RNA expression values and Gene Tokens into a single vector.
    """

    def __init__(self, n_tokens, d_model, pad_id):
        super().__init__()

        # Gene Encoder
        self.emb_g = nn.Embedding(
            # TODO parametrize extra embeddings
            num_embeddings=n_tokens + 10,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )
        self.ln_g = nn.LayerNorm(d_model)

        # Expression Encoder
        self.emb_x = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.ln_x = nn.LayerNorm(d_model)

    def forward(
        self,
        g: torch.Tensor,  # [B, r]
        x: torch.Tensor,  # [B, r]
    ) -> dict:
        # Encode Genes
        g = self.emb_g(g)
        g = self.ln_g(g)  # [B, r, C]

        # Encode Expressions
        x = self.emb_x(x.unsqueeze(-1))  # [B, r] -> [B, r, 1] -> [B, r, C]
        x = self.ln_x(x)  # [B, r, C]
        # NOTE: scGPT adds dropout here after layer norm as final stage

        # Sum
        return g + x


class ExprDecoder(nn.Module):
    """
    A decoder to predict gene expression and probability of active gene.
    """

    def __init__(
        self,
        d_model,
        dropout: float = 0.2,
    ):
        super().__init__()

        # To predict masked expression
        self.mlm_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )  # input [B, r, C] -> returns [B, r, 1]

        # To predict binary probability
        self.active_logits_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            # nn.Sigmoid()
        )  # input [B, r, C] -> returns [B, r, 1]

    def forward(
        self,
        transformer_encoder_output,
    ):
        # Prepare output dict
        output = {}
        output["mlm"] = self.mlm_decoder(transformer_encoder_output).squeeze(
            -1
        )  # [B, S] one predicted expression per each symbol in sequence
        output["active_logits"] = self.active_logits_decoder(
            transformer_encoder_output
        ).squeeze(-1)  # [B, S] one predicted expression per each symbol in sequence

        return output


class GeneDecoder(nn.Module):
    """
    A decoder to predict gene tokens.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # To predict masked expression
        self.gene_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )  # input [B, r, C] -> returns [B, r, vocab_size]

    def forward(
        self,
        transformer_encoder_output,
    ):
        # Prepare output dict
        output = {  # [B, S, vocab_size] one predicted expression per each symbol in sequence
            "gene_logits": self.gene_decoder(transformer_encoder_output)
        }

        return output


class MultiHeadAttentionWithPairBias(nn.Module):
    """
    Compute a given sequence self-attention, optionally using provided biases and gating.
    """

    def __init__(
        self,
        c_in: int,
        n_heads: int,
        gating: bool,
        pair_bias: bool = True,
        d_z: int = None,
    ) -> None:
        """
        Args:
            c_in (int): Input channel dimension

            c_hidden (int): Per-head hidden channel dimension

            n_heads (int): Number of attention heads

            pair_bias (bool): Whether to use pair embedding bias

            d_z (int): Pair embedding channel dimension. Ignored unless pair_bias is true

            gating (bool): Whether to use gated attention or not
        """
        super(MultiHeadAttentionWithPairBias, self).__init__()

        self.c_in = c_in
        self.c_hidden = int(c_in / n_heads)
        self.n_heads = n_heads
        self.pair_bias = pair_bias
        self.d_z = d_z
        self.gating = gating

        # Initial Normalization
        # self.layer_norm_m = nn.LayerNorm(self.c_in)

        # Bias
        # self.layer_norm_z = nn.LayerNorm(self.d_z) if self.pair_bias else None
        self.linear_z = (
            nn.Linear(self.d_z, self.n_heads, bias=False) if self.pair_bias else None
        )

        # Queries, Keys, Values
        self.linear_q = nn.Linear(
            self.c_in, self.c_in, bias=False
        )  # project to a singular vector and then reshape into multiple heads
        self.linear_k = nn.Linear(self.c_in, self.c_in, bias=False)
        self.linear_v = nn.Linear(self.c_in, self.c_in, bias=False)

        # Gating
        self.linear_g = nn.Linear(self.c_in, self.c_in) if self.gating else None

        # Final projection
        self.linear_o = nn.Linear(self.c_in, self.c_in)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): [B, r, c] Input embedding
            z (torch.Tensor): [B, r, r, d_z] pair embedding. Required only if pair_bias is True
            attention_mask (torch.Tensor): [B, r] boolean validity mask. False is padding.
        """
        if z is None and self.pair_bias is True:
            raise ValueError("z required when pair bias is true")

        # [B, r, c]
        # m = self.layer_norm_m(m)

        # [B, r, c_hid * H]
        q = self.linear_q(m)
        k = self.linear_k(m)
        v = self.linear_v(m)

        # [*, r, H, c_hid], where H = n_heads
        q = q.view(q.shape[:-1] + (self.n_heads, -1))
        k = k.view(k.shape[:-1] + (self.n_heads, -1))
        v = v.view(v.shape[:-1] + (self.n_heads, -1))

        # [*, H, r, c_hid]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)  # scaled attn

        # [*, H, c_hid, r]
        k = torch.permute(k, (0, 1, 3, 2))  # flip last two dims for matmul

        # [*, H, r, r]
        a = torch.matmul(q, k)

        if self.pair_bias:
            # [B, r, r, d_z]
            # z = self.layer_norm_z(z)
            # [B, r, r, H]
            z = self.linear_z(z)
            # [B, H, r, r]
            z = torch.permute(z, (0, 3, 1, 2))
            # [B, H, r, r]
            a = a + z
            # print('Pair Bias')

        key_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=a.device, dtype=torch.bool)
            key_mask = attention_mask[:, None, None, :]
            a = a.masked_fill(~key_mask, torch.finfo(a.dtype).min)

        # [B, H, r, r]
        a = nn.functional.softmax(a, dim=-1)
        if key_mask is not None:
            # Keep an all-padding row finite and empty instead of uniform.
            a = a.masked_fill(~key_mask, 0.0)

        # [B, H, r, c_hid]
        a = torch.matmul(a, v)

        # [B, r, H, c_hid]
        a = a.transpose(-2, -3)

        if self.gating:
            # [B, r, c_hid * H]
            g = nn.functional.sigmoid(self.linear_g(m))
            # [B, r, H, c_hid]
            g = g.view(g.shape[:-1] + (self.n_heads, -1))
            # [B, r, H, c_hid]
            a = a * g

        # [B, r, H * c_hid]
        a = a.reshape(a.shape[:-2] + (-1,))  # flatten H dim

        # [B, r, d_model]
        a = self.linear_o(a)

        if attention_mask is not None:
            a = a * attention_mask.unsqueeze(-1).to(a.dtype)

        return a

    def forward_old(
        self,
        m: torch.Tensor,
        z: torch.Tensor = None,
    ) -> torch.Tensor:
        """Legacy path in which padding participated in attention."""
        return self.forward(m, z, attention_mask=None)


class Transition(nn.Module):
    """
    Feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.2,
        n: int = 4,
        chunk_size: int = -1,
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.ffnn = nn.Sequential(
            nn.Linear(d_model, n * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.ffnn(x)
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(output.dtype)
        return output

    def forward_old(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy path without zeroing padded positions."""
        return self.ffnn(x)
        # if len(x.shape) == 8:
        #     assert self.chunk_size > 0
        #     # out = torch.zeros_like(x)
        #     B, r, r, d_z = x.shape
        #     for i in range(0, r, self.chunk_size):
        #         i_end = min(i+self.chunk_size, r)
        #         for j in range(0, r, self.chunk_size):
        #             j_end = min(j+self.chunk_size, r)

        #             x_chunk = x[:, i:i_end, j:j_end, :]

        #             x[:, i:i_end, j:j_end] = self.ffnn(x_chunk)

        # else:
        #     x = self.ffnn(x)

        # return x


class TransformerEncoderLayer(nn.Module):
    """
    Custom implementation of one layer of the transformer encoder.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        norm_first: bool,
        gating: bool,
    ):
        super().__init__()

        self.norm_first = norm_first

        # Multi Head Attention
        self.mha = MultiHeadAttentionWithPairBias(
            c_in=d_model,
            n_heads=n_heads,
            gating=gating,
            pair_bias=False,
        )

        # Add & Norm
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

        # FFNN
        self.ffnn = Transition(d_model)

    def forward(self, x, attention_mask: torch.Tensor | None = None):
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        if self.norm_first:
            # x -> norm -> sublayer -> residual (+x)

            # 1. MHA
            x = x + self.dropout1(self.mha(self.ln1(x), attention_mask=attention_mask))

            # 2. FFNN
            x = x + self.dropout2(self.ffnn(self.ln2(x)))

        else:
            # x -> sublayer -> residual (+x) -> norm

            # 1. MHA
            x = x + self.dropout1(self.mha(x, attention_mask=attention_mask))
            x = self.ln1(x)

            # 2. FFNN
            x = x + self.dropout2(self.ffnn(x))
            x = self.ln2(x)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        return x

    def forward_old(self, x):
        """Legacy transformer layer without padding-aware attention."""
        return self.forward(x, attention_mask=None)


class TransformerEncoder(nn.Module):
    """
    Custom implementation of the transformer encoder.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        norm_first: bool,
        gating: bool,
        n_layers: int,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(n_layers):
            block = TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                gating=gating,
                norm_first=norm_first,
            )
            self.blocks.append(block)

    def forward(self, x, attention_mask: torch.Tensor | None = None):
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        return x

    def forward_old(self, x):
        """Legacy transformer stack without a padding attention mask."""
        return self.forward(x, attention_mask=None)


# BioFormer Modules
class OuterProductMean(nn.Module):
    """

    Compute outer product mean of a given tensor.

    """

    def __init__(self, d_model: int, d_z: int, d_opm: int, chunk_size: int):
        """
        Args:
            d_model (int): Input embedding channel dimension

            d_z (int): Pair embedding channel dimension

            d_opm (int): Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.d_model = d_model
        self.d_z = d_z
        self.d_opm = d_opm
        self.chunk_size = chunk_size

        # self.layer_norm = nn.LayerNorm(d_model)
        self.linear_1 = nn.Linear(d_model, d_opm)
        self.linear_2 = nn.Linear(d_model, d_opm)
        self.linear_out = nn.Linear(d_opm**2, d_z)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): [B, r, c] Input embedding

        Returns:
            torch.Tensor: [B, r, r, d_z] pair embedding update
        """

        # [B, r, d_model]
        # m = self.layer_norm(m)

        a = self.linear_1(m)
        b = self.linear_2(m)
        B, r, d_opm = a.shape

        # [B, r, r, d_z]
        outer = torch.zeros((B, r, r, self.d_z), device=a.device, dtype=a.dtype)

        # Compute OPM in chunks
        for i in range(0, r, self.chunk_size):
            i_end = min(i + self.chunk_size, r)
            for j in range(0, r, self.chunk_size):
                j_end = min(j + self.chunk_size, r)

                # Extract chunks
                a_chunk = a[:, i:i_end, :]
                b_chunk = b[:, j:j_end, :]

                # Compute OPM
                # [B, chunk_size, chunk_size, d_opm, d_opm]
                outer_chunk = torch.einsum("...ab,...cd->...acbd", a_chunk, b_chunk)

                # Flatten last two dims
                # [B, chunk_size, chunk_size, d_opm * d_opm]
                outer_chunk = outer_chunk.reshape(outer_chunk.shape[:-2] + (-1,))

                # Project to d_z
                outer_chunk = self.linear_out(outer_chunk)

                # Place in the output
                outer[:, i:i_end, j:j_end, :] = outer_chunk

        # Project output
        # [B, r, r, d_z]
        # outer = self.linear_out(outer)
        return outer


class BioFormerBlock(nn.Module):
    """
    A single BioFormer block.
    It performs gated-self attention with pair bias, transition and outer-product mean with residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_z: int,
        d_opm: int,
        n_heads: int,
        pair_updates: bool,
        norm_first: bool,
        dropout: float,
        gating: bool,
        chunk_size: int,
    ):
        """
        Args:
            d_model:
                Input channel dimension
            d_z:
                Pair embedding channel dimension
            d_opm:
                Hidden channel dimension
            n_heads:
                Number of attention heads
            do_opm:
                Whether to compute the outer product mean update
            do_pair_bias:
                Whether to use the pair bias in the self-attention layer
        """
        super(BioFormerBlock, self).__init__()

        self.pair_updates = pair_updates
        self.norm_first = norm_first

        # Norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = (
            nn.LayerNorm(d_model) if (self.pair_updates and self.norm_first) else None
        )
        self.lnz1 = nn.LayerNorm(d_z) if self.pair_updates else nn.Identity()
        self.lnz2 = nn.LayerNorm(d_z) if self.pair_updates else None
        self.lnz3 = (
            nn.LayerNorm(d_z) if (self.pair_updates and not self.norm_first) else None
        )

        # MHA
        self.mha = MultiHeadAttentionWithPairBias(
            c_in=d_model,
            n_heads=n_heads,
            pair_bias=self.pair_updates,
            d_z=d_z,
            gating=gating,
        )
        self.dropout1 = nn.Dropout(dropout)

        # FFNN
        self.ffnn = Transition(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # z
        self.opm = (
            OuterProductMean(
                d_model=d_model,
                d_z=d_z,
                d_opm=d_opm,
                chunk_size=chunk_size,
            )
            if self.pair_updates
            else None
        )
        self.pair_ffnn = (
            Transition(
                d_model=d_z,
                chunk_size=chunk_size,
            )
            if self.pair_updates
            else None
        )

    def forward(self, m, z, attention_mask: torch.Tensor | None = None):
        """
        Args:
            m:
                [B, r, d_model] RNA-seq input
            z:
                [B, r, r, d_z] pair representation
        Returns:
            m:
                [B, r, d_model] updated RNA-seqc
            z:
                [B, r, r, d_z] updated pair representation
        """

        pair_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=m.device, dtype=torch.bool)
            m = m * attention_mask.unsqueeze(-1).to(m.dtype)
            pair_mask = attention_mask[:, :, None] & attention_mask[:, None, :]
            z = z * pair_mask.unsqueeze(-1).to(z.dtype)

        if self.norm_first:
            # 1. MHA
            m = m + self.dropout1(
                self.mha(
                    self.ln1(m),
                    self.lnz1(z),
                    attention_mask=attention_mask,
                )
            )

            # 2. FFNN
            m = m + self.dropout2(self.ffnn(self.ln2(m)))

        else:
            # 1. MHA
            m = m + self.dropout1(self.mha(m, z, attention_mask=attention_mask))
            m = self.ln1(m)

            # 2. FFNN
            m = m + self.dropout2(self.ffnn(m))
            m = self.ln2(m)

        if self.pair_updates:
            if self.norm_first:
                # 1. OPM
                # [B, r, r, d_z]
                z = z + self.opm(self.ln3(m))

                # 2. FFNN
                # [B, r, r, d_z]
                z = z + self.pair_ffnn(self.lnz2(z))

            else:
                # 1. OPM
                z = z + self.opm(m)
                z = self.lnz2(z)

                # 2. FFNN
                z = z + self.pair_ffnn(z)
                z = self.lnz3(z)

        if attention_mask is not None:
            m = m * attention_mask.unsqueeze(-1).to(m.dtype)
            z = z * pair_mask.unsqueeze(-1).to(z.dtype)
        return m, z

    def forward_old(self, m, z):
        """Legacy BioFormer block without padding-aware attention."""
        return self.forward(m, z, attention_mask=None)


class BioFormerStack(nn.Module):
    """
    A stack of BioFormer blocks.
    """

    def __init__(
        self,
        d_model: int,
        d_z: int,
        d_opm: int,
        n_heads: int,
        n_layers: int,
        pair_updates: bool,
        norm_first: bool,
        dropout: float,
        gating: bool,
        chunk_size: int,
    ):
        super().__init__()

        self.d_z = d_z

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            block = BioFormerBlock(
                d_model=d_model,
                d_z=d_z,
                d_opm=d_opm,
                n_heads=n_heads,
                pair_updates=pair_updates,
                norm_first=norm_first,
                dropout=dropout,
                gating=gating,
                chunk_size=chunk_size,
            )
            self.blocks.append(block)

    def forward(self, m, attention_mask: torch.Tensor | None = None):
        B, r, c = m.shape

        # Init empty z
        z = torch.randn(size=(B, r, r, self.d_z), device=m.device, dtype=m.dtype)

        # Pass through all blocks
        for block in self.blocks:
            m, z = block(m, z, attention_mask=attention_mask)

        return m

    def forward_old(self, m):
        """Legacy BioFormer stack without a padding attention mask."""
        B, r, _ = m.shape
        z = torch.randn(size=(B, r, r, self.d_z), device=m.device)
        for block in self.blocks:
            m, z = block.forward_old(m, z)
        return m


# Superclass of our two sanity check models
class SanityCheckModel(BaseModel):
    def __init__(self, config: ScalingConfig):
        pl.LightningModule.__init__(self)  # specify whose constructor to call

        # Attr
        self.model_class = config.model.model_class
        self.total_genes = config.model.total_genes
        self.pipeline = config.metadata.pipeline
        # dummy lr just because
        self.lr = config.model.lr

        # Dummy param (just for sanity check)
        self.dummy_param = nn.Embedding(3, 3)

        # using registered buffers so when model is moved then also these tensor are
        self.register_buffer(
            "expr_tally", torch.zeros(self.total_genes + 2)
        )  # +1 is for PAD gene in active
        self.register_buffer(
            "active_tally", torch.zeros(self.total_genes + 2)
        )  # +1 is for PAD gene in active
        self.register_buffer(
            "n_samples", torch.zeros(self.total_genes + 2)
        )  # +1 is for PAD gene in active

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def step_default(self, batch, batch_idx, split):
        # Read in the batch, types are (float, int, bool, _)
        expr, gene_ids, mask, _, _, _, _ = unpack_batch(batch)
        batch_size = expr.shape[0]

        assert mask.dtype == torch.bool, (
            f"Expected mask dtype torch.bool, got {mask.dtype}"
        )

        expr = expr.to(dtype=torch.float)
        if self.training:
            # update tallies
            for b in range(batch_size):
                # function depends on whether is null or tally model
                self.expr_tally[gene_ids[b]] += self.update_expr_tally(expr[b])
                self.active_tally[gene_ids[b]] += self.update_active_tally(expr[b])
                # this is common instead
                self.n_samples[gene_ids[b]] += 1

        # MSE
        expr_preds = torch.zeros_like(expr).to(device=self.device)
        expr_preds += self.expr_tally[gene_ids] / self.n_samples[gene_ids]
        # if default pipeline MSE makes sense, for geneformer no
        if self.pipeline == Pipeline.DEFAULT:
            mse_loss = F.mse_loss(expr_preds[mask], expr[mask])
        elif (
            self.pipeline == Pipeline.GENEFORMER
            or self.pipeline == Pipeline.GENECORPUS
            or self.pipeline == Pipeline.TOKENIZED_TEXT
        ):
            mse_loss = torch.tensor(0.0)
        else:
            raise NotImplementedError(f"Pipeline {self.pipeline} not implemented")
        # require grad otherwise lightning gets angry :(
        mse_loss.requires_grad_()

        # Compute Spearman
        full_spearman, batch_spearman = compute_spearman_expressions(
            expr_preds, expr, mask, batch_size
        )
        self.log_spearman(full_spearman, batch_spearman)

        # Accuracy
        active_preds = torch.zeros_like(expr).to(device=self.device)
        active_preds += self.active_tally[gene_ids] / self.n_samples[gene_ids]
        accuracy = ((expr[mask] > 0) == (active_preds[mask] > 0.5)).sum() / mask.sum()
        return {"losses": {"mse": mse_loss}, "metrics": {"accuracy": accuracy}}

    def step_geneformer(self, batch, batch_idx, split="train"):
        return self.step_default(batch, batch_idx, split)


# Tally Model (running avg.)
class TallyModel(SanityCheckModel):
    def __init__(self, config: ScalingConfig):
        super().__init__(config)

    def update_expr_tally(self, expr):
        return expr

    def update_active_tally(self, expr):
        return expr > 0


# Null Model (no running avg.)
class NullModel(SanityCheckModel):
    def __init__(self, config: ScalingConfig):
        super().__init__(config)

    def update_expr_tally(self, expr):
        return 0

    def update_active_tally(self, expr):
        return False


# Classifier
class CellTypeClassifier(pl.LightningModule):
    """
    Implements the cell type classification task on top of trained embeddings.
    """

    def __init__(
        self,
        trained_encoder,
        lr: float,
        n_steps: int,
        path_to_celltypes: str,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # For easier access
        self.lr = lr
        self.n_steps = n_steps

        # Get vocab_size (number of cell types to predict)
        import pandas as pd

        self.vocab_size = pd.read_csv(path_to_celltypes).cell_type.unique().shape[0]

        # The trained model to produce embeddings
        self.trained_encoder = trained_encoder

        #  Freeze if required
        if freeze_encoder:
            for param in self.trained_encoder.parameters():
                param.requires_grad = False

        # Change the head
        # We can keep the same object and just change the shape
        # because by default pytorch broadcasts the input
        # i.e., it handles both [B, r, C] (gene-level classification)
        # and [B, C] (cell-level classification) inputs natively
        self.decoder = GeneDecoder(
            d_model=self.trained_encoder.d_model, vocab_size=self.vocab_size
        )

        # We don't need the pre-training decoder
        del self.trained_encoder.decoder

    def forward(
        self,
        g: torch.Tensor,  # [B, r]
        x: torch.Tensor,  # [B, r]
    ) -> dict:
        # Check dtypes
        assert x.dtype == torch.float, f"Expected expr dtype torch.float, got {x.dtype}"
        assert g.dtype == torch.long, (
            f"Expected gene_ids dtype torch.long, got {g.dtype}"
        )

        # To check dimensions
        B, r = x.shape
        C = self.trained_encoder.d_model

        # Get logits (trained model + classification head)
        # [B, r, C]
        encoder_output = self.trained_encoder.get_encoder_output(g, x)
        assert encoder_output.shape == (B, r, C)

        # Aggregate embeddings
        # [B, C]
        cell_embeddings = encoder_output.mean(dim=1)
        assert cell_embeddings.shape == (B, C)

        # Get cell-level prediction
        output = self.decoder(cell_embeddings)
        logits = output["gene_logits"]
        assert logits.shape == (B, self.vocab_size)

        return logits

    def step(self, batch, batch_idx, stage: str):
        assert stage in {"train", "val", "test"}

        expr, gene_ids, mask, cell_type, _ = batch

        logits = self.forward(gene_ids, expr)

        B, vocab_size = logits.shape
        assert cell_type.shape == (B,)
        # logits = logits.view(B * r, vocab_size)
        # targets = cell_type.view(B * r)
        ce_loss = torch.nn.functional.cross_entropy(logits, cell_type)

        with torch.no_grad():
            accuracy = (logits.argmax(dim=-1) == cell_type).float().mean()

        # Log based on stage
        if stage == "train":
            self.log("train_ce", ce_loss, prog_bar=True)
            self.log("train_accuracy", accuracy, prog_bar=True)
        else:
            self.log(
                f"{stage}_ce", ce_loss, on_step=False, on_epoch=True, prog_bar=True
            )
            self.log(
                f"{stage}_accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return ce_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            betas=(0.9, 0.9),
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.n_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # "monitor": "val_mse",  # Metric to track
                "interval": "step",
                "frequency": 1,  # Check after each epoch
            },
        }

    def on_before_optimizer_step(self, optimizer):
        # to show grad norm
        total_grad_norm = 0
        for _, param in self.named_parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item() ** 2
        total_grad_norm = total_grad_norm**0.5
        self.log(
            "train_grad_norm",
            total_grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        # Log learning rate
        lr = optimizer.param_groups[0]["lr"]
        self.log(
            "lr",
            lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )


def load_model_from_config(config: ScalingConfig, path_to_ckpt: str = None):
    assert isinstance(config, ScalingConfig)

    # If we changed something
    config.check_attributes()

    # Get model class and instantiate
    model_class = config.model.model_class

    if model_class == ModelClass.TRANSFORMER:
        model_class = TransformerModel
    elif model_class == ModelClass.FFNN:
        model_class = FFNN
    elif model_class == ModelClass.BIOFORMER:
        model_class = BioFormerModel
    elif model_class == ModelClass.NULL:
        model_class = NullModel
    elif model_class == ModelClass.TALLY:
        model_class = TallyModel
    elif model_class == ModelClass.BERT:
        model_class = BERTModel
    elif model_class == ModelClass.BIORANGOTANGO:
        model_class = BioRangoTango
    else:
        raise NotImplementedError(f"Unknown model class: {model_class}")

    if path_to_ckpt:
        print(f"Loading {model_class.__name__} from {path_to_ckpt}...")
        return model_class.load_from_checkpoint(path_to_ckpt, config=config)
    else:
        print(f"Loading new instance of {model_class.__name__}...")
        return model_class(config)
