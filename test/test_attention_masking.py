import torch
import torch.nn as nn

from core.models import (
    BERTModel,
    BioFormerStack,
    MultiHeadAttentionWithPairBias,
    TransformerEncoder,
)


def test_custom_attention_ignores_padding_keys_and_zeros_padding_queries():
    attention = MultiHeadAttentionWithPairBias(
        c_in=4,
        n_heads=1,
        gating=False,
        pair_bias=False,
    )
    with torch.no_grad():
        identity = torch.eye(4)
        attention.linear_q.weight.copy_(identity)
        attention.linear_k.weight.copy_(identity)
        attention.linear_v.weight.copy_(identity)
        attention.linear_o.weight.copy_(identity)
        attention.linear_o.bias.zero_()

    first = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [10.0] * 4]]
    )
    second = first.clone()
    second[:, 2] = -10.0
    valid_tokens = torch.tensor([[True, True, False]])

    first_output = attention(first, attention_mask=valid_tokens)
    second_output = attention(second, attention_mask=valid_tokens)

    assert torch.allclose(first_output[:, :2], second_output[:, :2])
    assert torch.equal(first_output[:, 2], torch.zeros_like(first_output[:, 2]))


def test_transformer_stack_does_not_leak_padding_into_valid_tokens():
    encoder = TransformerEncoder(
        d_model=4,
        n_heads=1,
        dropout=0.0,
        norm_first=True,
        gating=False,
        n_layers=2,
    ).eval()
    first = torch.randn(1, 3, 4)
    second = first.clone()
    second[:, 2] = torch.randn(1, 4) * 100
    valid_tokens = torch.tensor([[True, True, False]])

    first_output = encoder(first, attention_mask=valid_tokens)
    second_output = encoder(second, attention_mask=valid_tokens)

    assert torch.allclose(first_output[:, :2], second_output[:, :2], atol=1e-6)
    assert torch.equal(first_output[:, 2], torch.zeros_like(first_output[:, 2]))


def test_bioformer_stack_does_not_leak_padding_into_valid_tokens():
    encoder = BioFormerStack(
        d_model=4,
        d_z=2,
        d_opm=2,
        n_heads=1,
        n_layers=2,
        pair_updates=True,
        norm_first=True,
        dropout=0.0,
        gating=False,
        chunk_size=1,
    ).eval()
    first = torch.randn(1, 3, 4)
    second = first.clone()
    second[:, 2] = torch.randn(1, 4) * 100
    valid_tokens = torch.tensor([[True, True, False]])

    torch.manual_seed(0)
    first_output = encoder(first, attention_mask=valid_tokens)
    torch.manual_seed(0)
    second_output = encoder(second, attention_mask=valid_tokens)

    assert torch.allclose(first_output[:, :2], second_output[:, :2], atol=1e-6)
    assert torch.equal(first_output[:, 2], torch.zeros_like(first_output[:, 2]))


def test_bert_step_attends_to_mlm_tokens_but_not_padding():
    model = object.__new__(BERTModel)
    nn.Module.__init__(model)
    model.total_genes = 4
    model.mask_token = 5
    model.pad_token = 4
    model._legacy_behavior = False
    captured = {}

    def fake_forward(gene_ids, attention_mask=None):
        captured["gene_ids"] = gene_ids.clone()
        captured["attention_mask"] = attention_mask.clone()
        return {
            "logits": torch.zeros(
                (*gene_ids.shape, model.total_genes + 2), requires_grad=True
            )
        }

    model.forward = fake_forward
    gene_ids = torch.tensor([[0, model.pad_token, 2]])
    prediction_mask = torch.tensor([[True, False, False]])
    batch = (
        torch.zeros_like(gene_ids, dtype=torch.float),
        gene_ids,
        prediction_mask,
        None,
    )

    model.step(batch)

    assert torch.equal(captured["gene_ids"], torch.tensor([[5, 4, 2]]))
    assert torch.equal(
        captured["attention_mask"], torch.tensor([[True, False, True]])
    )
