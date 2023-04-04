import torch.nn as nn
from models.informer.attention import FullAttention, ProbSparseAttention, AttentionLayer
from models.informer.embedding import DataEmbedding
from models.informer.encoder import (
    Encoder,
    EncoderLayer,
    EncoderStack,
    SelfAttentionDistil,
)
from models.informer.decoder import Decoder, DecoderLayer


class BaseInformer(nn.Module):
    def __init__(
        self,
        enc_in=7,
        dec_in=7,
        c_out=7,
        out_len=24,
        factor=5,
        d_model=512,
        n_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=1,
        d_ff=2048,
        dropout=0.05,
        attention_type="prob",
        embedding_type="fixed",
        frequency="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix_attention=False,
        **kwargs
    ):
        super(BaseInformer, self).__init__()
        self.pred_len = out_len
        self.attention_type = attention_type
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, embedding_type, frequency, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embedding_type, frequency, dropout)

        Attention = ProbSparseAttention if attention_type == "prob" else FullAttention

        self.encoder = None

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                        mix=mix_attention,
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_decoder_layers)
            ],
            nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, c_out)

    def forward(
        self,
        x_enc,
        x_enc_mark,
        x_dec,
        x_dec_mark,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        enc_out = self.enc_embedding(x_enc, x_enc_mark)
        enc_out, attentions = self.encoder(enc_out, attention_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_dec_mark)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attentions
        return dec_out[:, -self.pred_len :, :]


class Informer(BaseInformer):
    def __init__(
        self,
        enc_in=7,
        dec_in=7,
        c_out=7,
        out_len=24,
        factor=5,
        d_model=512,
        n_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=1,
        d_ff=2048,
        dropout=0.05,
        attention_type="prob",
        embedding_type="fixed",
        frequency="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix_attention=False,
        **kwargs
    ):
        super(Informer, self).__init__(
            enc_in,
            dec_in,
            c_out,
            out_len,
            factor=factor,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            attention_type=attention_type,
            embedding_type=embedding_type,
            frequency=frequency,
            activation=activation,
            output_attention=output_attention,
            distil=distil,
            mix_attention=mix_attention,
        )
        Attention = ProbSparseAttention if attention_type == "prob" else FullAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_encoder_layers)
            ],
            [SelfAttentionDistil(d_model) for _ in range(num_encoder_layers - 1)] if distil else None,
            nn.LayerNorm(d_model),
        )


class InformerStack(BaseInformer):
    def __init__(
        self,
        enc_in=7,
        dec_in=7,
        c_out=7,
        out_len=24,
        factor=5,
        d_model=512,
        n_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=1,
        d_ff=2048,
        dropout=0.05,
        attention_type="prob",
        embedding_type="fixed",
        frequency="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix_attention=False,
        **kwargs
    ):
        super(InformerStack, self).__init__(
            enc_in,
            dec_in,
            c_out,
            out_len,
            factor=factor,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            attention_type=attention_type,
            embedding_type=embedding_type,
            frequency=frequency,
            activation=activation,
            output_attention=output_attention,
            distil=distil,
            mix_attention=mix_attention,
        )
        Attention = ProbSparseAttention if attention_type == "prob" else FullAttention
        stacks = list(range(num_encoder_layers, 2, -1))  # customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model,
                            n_heads,
                            mix=False,
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for _ in range(el)
                ],
                [SelfAttentionDistil(d_model) for _ in range(el - 1)] if distil else None,
                nn.LayerNorm(d_model),
            )
            for el in stacks
        ]
        self.encoder = EncoderStack(encoders)
