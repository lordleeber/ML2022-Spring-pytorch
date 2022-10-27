# # HINT: transformer architecture
# from fairseq.models.transformer import (
#     TransformerEncoder,
#     TransformerDecoder,
# )


# For strong baseline, please refer to the hyperparameters for *transformer-base* in Table 3 in [Attention is all you need](#vaswani2017)
# HINT: these patches on parameters for Transformer
def add_transformer_args(args):
    args.encoder_attention_heads =4
    args.encoder_normalize_before =True

    args.decoder_attention_heads =4
    args.decoder_normalize_before =True

    args.activation_fn ="relu"
    args.max_source_positions =1024
    args.max_target_positions =1024

    # patches on default parameters for Transformer (those not set above)
    from fairseq.models.transformer import base_architecture
    base_architecture(arch_args)

# add_transformer_args(arch_args)
