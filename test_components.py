import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass

from attention import MultiHeadedAttention, MaskedMultiHeadAttention, FastMHA, FastSelfAttn,attnconfig,AttnVariant
from utils import EnFinnishDataset, PositionalVariant, RoPE, RelativePE
from encoder import TransformerEncoderBlock, EncoderConfig,TransformerEncoder
from decoder import DecoderConfig,TransformerDecoderBlock,TransformerDecoder

ARCHIVE_PATH = "EUbookshop-1"
CONTEXT_LEN = 512
DEVICE = torch.device("cuda")

def return_all_testingconstants():
    ARCHIVE_PATH = "EUbookshop-1"
    CONTEXT_LEN = 512
    BATCH_SIZE=32
    CAUSAL_MASK=False
    EMBEDDING_DIM=128
    QUERY_DIM=128
    KEY_DIM=128
    VALUE_DIM=128
    MODEL_DIM=512
    NUM_HEADS=4
    return(ARCHIVE_PATH,CONTEXT_LEN,BATCH_SIZE,CAUSAL_MASK,EMBEDDING_DIM,QUERY_DIM,KEY_DIM,VALUE_DIM,MODEL_DIM,NUM_HEADS)
    

@pytest.mark.parametrize(
    "attention_cls, attention_type",
    [
        (MultiHeadedAttention, "cross-attention"),
        (MultiHeadedAttention, "self-attention"),
        (MaskedMultiHeadAttention, "self-attention"),
        (FastMHA, "fast-attention"),
        (FastSelfAttn, "fast-self-attention")
    ]
)
def test_attention_output_shape(attention_cls, attention_type):
    """
    Verifies that various attention mechanisms produce the correct output shape.
    (This test was already pytest-compatible and is included for completeness).
    """
    batch_size = 8
    seq_len_q = 10
    seq_len_kv = 10
    query_dim = 64
    key_dim = 64
    value_dim = 128
    model_dim = 128
    n_heads = 4

    if attention_type in ("self-attention", "fast-attention", "fast-self-attention"):
        config = attnconfig(
            query_dim=query_dim,
            key_dim=query_dim,
            value_dim=query_dim,
            model_dim=model_dim,
            n_heads=n_heads,
            context_len=seq_len_q,
        )
        attention_layer = attention_cls(config)
        input_tensor = torch.randn(batch_size, seq_len_q, query_dim)
        output = attention_layer(input_tensor, input_tensor, input_tensor)

    elif attention_type == "cross-attention":
        config = attnconfig(
            query_dim=query_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            model_dim=model_dim,
            n_heads=n_heads,
            context_len=seq_len_q,
        )
        attention_layer = attention_cls(config)
        query_vector = torch.randn(batch_size, seq_len_q, config.query_dim)
        key_vector = torch.randn(batch_size, seq_len_kv, config.key_dim)
        value_vector = torch.randn(batch_size, seq_len_kv, config.value_dim)
        output = attention_layer(query_vector, key_vector, value_vector)

    else:
        pytest.fail(f"Unsupported attention type for testing: {attention_type}")

    expected_shape = torch.Size([batch_size, seq_len_q, model_dim])
    assert output.shape == expected_shape, (
        f"Attention layer output shape is incorrect.\n"
        f"Expected: {expected_shape}, Got: {output.shape}"
    )


def test_rope_output_shape():
    """
    Tests that the RoPE (Rotary Positional Embedding) layer does not alter the
    shape of the input tensor.
    """
    batch_dim = 32
    num_tokens = 40
    embedding_dim = 512
    token_embedding = torch.randn(size=(batch_dim, num_tokens, embedding_dim))

    rope = RoPE(embedding_dim=embedding_dim, context_len=num_tokens)
    output = rope(token_embedding)

    # Assertion: RoPE should only modify the values for positional encoding, not the tensor's shape.
    assert output.shape == token_embedding.shape, (
        f"RoPE layer incorrectly changed the tensor shape.\n"
        f"Expected: {token_embedding.shape}\n"
        f"Got: {output.shape}"
    )
    # Assertion: Ensure the output is a new tensor, not just a reference to the input.
    assert not torch.equal(output, token_embedding)


def test_dataloading_and_item_shape():
    """
    Tests the instantiation of the EnFinnishDataset and verifies the structure,
    type, and shape of a single item fetched from it.
    """
    dataset = EnFinnishDataset(archive_path=ARCHIVE_PATH, context_len=CONTEXT_LEN)

    # 1. Assert that the dataset object is created and is not empty.
    assert dataset is not None, "Dataset object could not be instantiated."
    assert len(dataset) > 0, "Dataset is empty after loading."

    # 2. Retrieve a single sample to check its integrity.
    sample = dataset[0]
    assert isinstance(sample, tuple) and len(sample) == 4, f"Dataset sample should be a tuple of 4 elements, but got {type(sample)} of length {len(sample)}."
    
    en_tokens, en_mask, fin_tokens, fin_mask = sample
    
    # 3. Assert that all parts of the sample are tensors with the correct shape.
    expected_shape = [torch.Size([CONTEXT_LEN]),torch.Size([CONTEXT_LEN,CONTEXT_LEN])]*2
    checks=[("English tokens", en_tokens), ("English mask", en_mask), ("Finnish tokens", fin_tokens), ("Finnish mask", fin_mask)]
    for i,(name, tensor) in enumerate(checks):
        assert isinstance(tensor, torch.Tensor), f"{name} is not a torch.Tensor."
        assert tensor.shape == expected_shape[i], (
            f"{name} shape is incorrect.\n"
            f"Expected: {expected_shape}, Got: {tensor.shape}"
        )

@pytest.mark.parametrize("posn_class_variant", [1])  # Assuming 0 and 1 are valid enum values for PositionalVariant
def test_encoder_block_integration_output_shape(posn_class_variant: int):
    """
    Performs an end-to-end test of the TransformerEncoderBlock block. It uses a real
    batch of data from the DataLoader and asserts that the final output tensor
    has the expected shape. This test is parameterized to run with different
    positional encoding variants.
    """
    EMBEDDING_SIZE=512
    # 1. Setup: Create Dataset, DataLoader, and Encoder Configuration
    dataset = EnFinnishDataset(archive_path=ARCHIVE_PATH, context_len=CONTEXT_LEN)
    dataloader = DataLoader(dataset, batch_size=2)

    atn_cfg = attnconfig(query_dim=EMBEDDING_SIZE, key_dim=EMBEDDING_SIZE, value_dim=EMBEDDING_SIZE, model_dim=512, n_heads=4,
                         context_len=CONTEXT_LEN)
    
    encoder_cfg = EncoderConfig(
        num_heads=4,
        vocab_size=len(dataset.tokenizer),  # A more realistic vocab size
        embedding_size=EMBEDDING_SIZE,
        max_seq_len=CONTEXT_LEN,
        atn_cfg=atn_cfg,
        pos_weight=0.2,
        mlp_depth=2,
        attn_class=AttnVariant(4),  # Assuming 4 corresponds to a FastMHA variant
        posn_class=PositionalVariant(posn_class_variant),
        
    )
    
    # encoder_block = TransformerEncoderBlock(config=encoder_cfg)
    encoder_block = TransformerEncoder(config=encoder_cfg,n_blocks=1).to(device=DEVICE)
    # 2. Execution: Fetch a batch and pass it through the encoder
    try:
        en_tokens, en_mask, _, _ = next(iter(dataloader))
    except StopIteration:
        pytest.fail("DataLoader is empty. Cannot fetch a test batch.")
    
    print(f"Shape of en_tokens: {en_tokens.shape}")
    print(f"Shape of en_mask: {en_mask.shape}")

    output = encoder_block(en_tokens.to(DEVICE), en_mask.to(DEVICE))
    print(output)
    
    ## Assertion: Verify No nans exist within the output:
    print(f"Are there 0 Nans in the output: {torch.isnan(output).sum()==0}")
    assert torch.isnan(output).sum()==0
    # 3. Assertion: Verify the output shape is correct
    batch_size = en_tokens.shape[0]
    model_dim = encoder_cfg.embedding_size
    expected_shape = torch.Size([batch_size, CONTEXT_LEN, model_dim])
    assert output.shape == expected_shape, (
        f"Encoder block output shape is incorrect for posn_class variant {posn_class_variant}.\n"
        f"Expected: {expected_shape}\n"
        f"Got: {output.shape}"
    )


import torch
import pytest
from torch.utils.data import DataLoader

# Assume the following imports and functions exist from your project structure
# from your_project.constants import return_all_testingconstants
# from your_project.data import EnFinnishDataset
# from your_project.config import DecoderConfig, EncoderConfig, attnconfig
# from your_project.model import TransformerEncoderBlock, TransformerDecoderBlock
# from your_project.enums import AttnVariant, PositionalVariant

@pytest.mark.parametrize("posn_class_variant", [1])  # Assuming 0 and 1 are valid enum values for PositionalVariant
def test_encoder_decoder_block_integration_output(posn_class_variant: int):
    #############
    # CONSTANTS #
    #############
    ARCHIVE_PATH, CONTEXT_LEN, BATCH_SIZE, CAUSAL_MASK, EMBEDDING_DIM, QUERY_DIM, KEY_DIM, VALUE_DIM, MODEL_DIM, NUM_HEADS = return_all_testingconstants()
    BATCH_SIZE=8
    dataset = EnFinnishDataset(archive_path=ARCHIVE_PATH, context_len=CONTEXT_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    MODEL_DIM=128
    decoder_cfg = DecoderConfig(
        num_heads=NUM_HEADS,
        vocab_size=len(dataset.tokenizer),
        embedding_size=MODEL_DIM,
        max_seq_len=CONTEXT_LEN,
        atn_cfg=attnconfig(MODEL_DIM, MODEL_DIM, MODEL_DIM, MODEL_DIM, NUM_HEADS, False, CONTEXT_LEN),
        attn_class=AttnVariant.FAST_MULTIHEADED,
        posn_class=PositionalVariant(posn_class_variant),
    )
    encoder_cfg = EncoderConfig(
        num_heads=4,
        vocab_size=len(dataset.tokenizer),
        embedding_size=MODEL_DIM, ## this the dimensions of the output of the encoder
        max_seq_len=CONTEXT_LEN,
        atn_cfg=attnconfig(MODEL_DIM, MODEL_DIM, MODEL_DIM, MODEL_DIM, NUM_HEADS, False, CONTEXT_LEN),
        pos_weight=0.2,
        mlp_depth=2,
        attn_class=AttnVariant(4),  # Assuming 4 corresponds to a FastMHA variant
        posn_class=PositionalVariant(posn_class_variant),
    )
    
    encoder = TransformerEncoder(encoder_cfg,n_blocks=1).to(DEVICE)
    decoder = TransformerDecoder(decoder_cfg,n_blocks=1).to(DEVICE)
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    bytes_per_param = 4 
    print(f"Encoder Param Count: {encoder_params:,}")   
    print(f"Decoder Param Count: {decoder_params:,}")
    print("-" * 30)
    print(f"Encoder Memory (MB): {(encoder_params * bytes_per_param) / (1024**2):.2f}")
    print(f"Decoder Memory (MB): {(decoder_params * bytes_per_param) / (1024**2):.2f}")
    
    for en_tokens, en_mask, fin_tokens, fin_mask in dataloader:
        # 1. Get the encoder's output (memory for the decoder)
        encoder_outputs = encoder(en_tokens.to(DEVICE), en_mask.to(DEVICE))
        integrated_outputs = decoder(fin_tokens.to(DEVICE), encoder_outputs.to(DEVICE), fin_mask.to(DEVICE))
        break

    ##################
    # VALIDATION #
    ##################
    expected_shape = (BATCH_SIZE, CONTEXT_LEN, len(dataset.tokenizer))
    assert integrated_outputs.shape == expected_shape, \
        f"Decoder output shape is incorrect. Expected {expected_shape}, but got {integrated_outputs.shape}"
    assert not torch.isnan(integrated_outputs).any(), "Decoder output contains NaNs."
    assert not torch.isinf(integrated_outputs).any(), "Decoder output contains infinite values."
    assert torch.sum(torch.abs(integrated_outputs)) > 1e-6, "Decoder output is likely all zeros, which is unexpected."