import math
import torch
from torch import nn

class FixedPositionalEncoding(nn.Module):
    """
    Fixed positional encoding (embedding layer) from sine and cosine functions
    of different frequencies according to https://arxiv.org/abs/1706.03762
    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
    """

    def __init__(self, hidden_size, max_sequence_length=512):
        super().__init__()

        pos_enc = torch.zeros(max_sequence_length, hidden_size)
        position = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        coef = -math.log(10000.0) / hidden_size
        div_term = torch.exp(coef * torch.arange(0.0, hidden_size, 2))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc.div_(math.sqrt(hidden_size))
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, position_ids):
        return torch.embedding(self.pos_enc, position_ids)

class TransformerEmbedding(nn.Module):
    """
    Embedding from token and position embeddings.
    Optionally add token_type embedding (e.g. type of the sentence in BERT).
    Args:
        vocab_size: size of the vocabulary
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
        num_token_types: number of different token types
            (e.g. tokens of sentence A and tokens of sentence B in BERT)
        embedding_dropout: probability of dropout applied to embeddings
        learn_positional_encodings: whether to learn positional encodings or
            use fixed (sine-cosine) ones
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_sequence_length=512,
        num_token_types=2,
        embedding_dropout=0.0,
        learn_positional_encodings=False,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if learn_positional_encodings:
            self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        else:
            self.position_embedding = FixedPositionalEncoding(hidden_size, max_sequence_length)
        if num_token_types > 0:
            self.token_type_embedding = nn.Embedding(num_token_types, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(embedding_dropout)

    def restore_from(self,chk_path):
        chk = torch.load(chk_path,map_location='cpu')
        tok_emb_weight = chk['decoder._embedding.token_embedding.weight']
        #pos_enc = chk['decoder._embedding.position_embedding.pos_enc']
        type_emb = chk['decoder._embedding.token_type_embedding.weight']
        lm_weight = chk['decoder._embedding.layer_norm.weight']
        lm_bias = chk['decoder._embedding.layer_norm.bias']

        self.token_embedding.weight = torch.nn.Parameter(tok_emb_weight)
        #self.position_embedding = pos_enc
        self.token_type_embedding = torch.nn.Parameter(type_emb)
        self.layer_norm.weight = torch.nn.Parameter(lm_weight)
        self.layer_norm.bias = torch.nn.Parameter(lm_bias)


    def forward(self, input_ids, token_type_ids=None, start_pos=0):
        seq_length = input_ids.size(1)
        if seq_length > self.max_sequence_length:
            raise ValueError(
                f"Input sequence is longer than maximum allowed sequence length for positional encoding. "
                f"Got {seq_length} and {self.max_sequence_length}"
            )
        position_ids = torch.arange(
            start=start_pos, end=start_pos + seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        embeddings = self.layer_norm(embeddings)

        return embeddings

if __name__=='__main__':
    embedding = TransformerEmbedding(
            vocab_size = 32000,
            hidden_size = 1024,
            max_sequence_length = 512,
            num_token_types = 2,
            learn_positional_encodings = False,
        )
    embedding.restore_from('../model_bin/model_weights.ckpt')
    #x = torch.randint(0,100,(2,5),dtype=torch.int64).to(torch.device('cpu'))
    x = torch.tensor([[   0],
        [3406],
        [   0],
        [   3],
        [4983],
        [3547],
        [4983],
        [3065]],dtype=torch.int64).to(torch.device('cpu'))
    x_emb = embedding(x)
    print('x:',x)
    print('x embedding:',x_emb)
    print('x type',x.dtype)
