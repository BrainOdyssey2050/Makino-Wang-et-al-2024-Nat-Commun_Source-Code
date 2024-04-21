from typing import Tuple

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, in_channels: int, fn: nn.Module):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    '''
    The feed-forward network introduces additional capacity to the model without relying on the sequence's attention-based interactions. 
    It operates on each position (i.e., each token or patch) independently. 
    In essence, while the attention mechanism helps the model understand the relationship between different parts of the input, 
    the feed-forward network allows the model to perform complex transformations on the features of each individual part.
    '''

    def __init__(self,
                 in_channels: int,
                 hid_channels: int,
                 dropout: float = 0.):
        
        super(FeedForward, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                 nn.GELU(), 
                                 nn.Dropout(dropout),
                                 nn.Linear(hid_channels, in_channels),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 hid_channels: int,
                 heads: int = 8,
                 head_channels: int = 64,
                 dropout: float = 0.):
        super(Attention, self).__init__()
        inner_channels = head_channels * heads

        # The project_out boolean determines whether a linear projection is needed. 
        # If there's only one attention head and the dimension of that head (head_channels) is equal to the desired output dimension (hid_channels), 
        # then no projection is required, and self.to_out is set to an identity function.
        # Otherwise, a linear transformation followed by dropout is applied, in self.to_out.
        project_out = not (heads == 1 and head_channels == hid_channels)

        self.heads = heads
        self.scale = head_channels**-0.5  # see forward

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # feed each patch vector (hid_channels) to the linear layer,
        # get the q vector (head_channels) for the patch, 
        # there are num of 'heads', so we need 'head_channels * heads = inner_channels' output for all q vectors. of the patch.
        # since we also need k, v vectors, we need to multiply 3 by inner_channels.
        self.to_qkv = nn.Linear(hid_channels, inner_channels * 3, bias=False)

        # This linear projection at the end of the Attention class is a crucial step, 
        # designed to transform the concatenated multi-head outputs back into a suitable representation for subsequent layers or operations. 
        self.to_out = nn.Sequential(
            nn.Linear(inner_channels, hid_channels),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor, save_attn_matrices=False) -> torch.Tensor:
        # the input x has a shape of (batch_size, num_patches, num_features) = (b,17,128)
        # each patch is a vector of 128 features, let say shape = 128*1
        # the patch passed to the linear layer, and get the output with shape 'head_channels * heads * 3 = 64*4*3 = 768'
        # I interpret it as 'q1, q2, q3, q4,   k1, k2, k3, k4,   v1, v2, v3, v4'
        # 
        # .chunk(3, dim=-1) splits the tensor into 3 chunks along the last dimension (which is of size 768)
        # The result is a tuple containing three tensors, each of the shape (b,17,256) corresponding to the queries (Q), keys (K), and values (V).
        # they all saved in tuple qkv
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # - map applies the given lambda function to each tensor in the qkv tuple.
        # - The lambda function uses rearrange to reshape each tensor.
        # - The reshaping is guided by the pattern 'b n (h d) -> b h n d'.
        #   b: batch size, n: sequence length (which is 17 in our case), h: number of heads (4), d: head channels (64)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # q, k, and v will each have a shape of: (batch_size, 4, 17, 64) 
        #                                      = (batch_size, heads, num_patches, the num of features in each head)

        # ---------- Calculate the attention score ----------
        # 1. torch.matmul(q, k.transpose(-1, -2))
        #    q.shape = (batch_size, num_heads, num_patches, head_channels) = (b,4,17,64) = k.shape
        #    k.transpose(-1, -2).shape = (batch_size, num_heads, head_channels, num_patches) = (b,4,64,17)
        #    basically, we calculate the dot product of q and k in the last two dimensions, resulting in a tensor of shape (b,4,17,17)
        #    In this matrix, each row corresponds to a query, and each column corresponds to a key.
        #    The value in the i-th row and j-th column is the raw attention score between the i-th query and the j-th key.
        #       Key 1  2  3  4 ... i ... 17
        #    Query
        #       1
        #       2
        #       ...
        #       j                    ij
        #       ...
        #       17
        #    Each row corresponds to one patch.
        #
        # 2. * self.scale
        #    We scale each raw attention score by the square root of the head dimension, sqrt(64) = 8
        #    Because when you do a dot product, the result is proportional to the dim of the vectors.
        #    therefore, we need to 'normalize' the dot product.
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # self.attend defines as nn.Softmax(dim=-1) above
        # for dots, we apply softmax on the last dimension, which is the dimension of the keys
        # Namely, for each query, the softmax function will convert its raw attention scores (with respect to each key) into a probability distribution
        # over all keys. This ensures that all the attention scores for a given query sum up to 1.
        attn = self.attend(dots)

        # when run trainer.test(), 
        # in classifier.py, test_step() run for each batch with save_attn_matrices=True
        # then, in LFP_Transformer.py, LFP_Transformer.forward() will pass save_attn_matrices=True to Transformer.forward()
        # in Transformer.forward() layer_idx == 0 spercifically wants the first transformer block's attention
        # Transformer.forward() will pass save_attn_matrices=True to Attention.forward()
        # Finally, here, we save the first row of each attention matrix of the FIRST transformer block, which is
        # the attention between the cls token and all the patches.
        # attn.shape = (batch_size, num_heads, num_patches, num_patches) = (b,4,17,17)
        if save_attn_matrices:
            attn_matrices = attn[:, :, 0, :].detach().cpu().numpy()
        else:
            attn_matrices = None

        # adding dropout to the attention weights, the model is encouraged to avoid relying too heavily on any particular part of the input. 
        attn = self.dropout(attn)
        # -----------------------------------------------------

        # attn.shape = (batch_size, num_heads, Query, Key) = (b,4,17,17)
        # v.shape = (batch_size, num_heads, num_patches, head_channels) = (b,4,17,64)
        # out.shape = (batch_size, num_heads, num_patches, head_channels) = (b,4,17,64)
        #             regarding the last two dimensions, each row is the output for one query, corresponding to one patch.
        out = torch.matmul(attn, v)

        # after rearrange, for each position in the sequence, 
        # we now have a single representation of dimension 256 (combining the outputs of all 4 heads).
        # out.shape = (batch_size, num_patches, num_heads*head_channels) = (b,17,256)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # self.to_out is a linear layer + dropout, which transforms the concatenated multi-head outputs back into
        # a suitable representation for subsequent layers or operations.
        # from (b,17,256) to (b,17,128)
        return self.to_out(out), attn_matrices


class Transformer(nn.Module):
    '''
    This Class presents the system architecture of the Transformer module, as each block (3 in total) shows in the forward function.
    '''

    def __init__(self,
                 hid_channels: int,
                 depth: int,
                 heads: int,
                 head_channels: int,
                 mlp_channels: int,
                 dropout: float = 0.):
        
        super(Transformer, self).__init__()

        self.attn_matrices_each_batch = None

        self.layers = nn.ModuleList([])
        # There are a number of 'depth' transformer blocks saved in self.layers
        # Each block contains two sublayers: attention and feedforward
        # Both Attention and FeedForward are wrapped in PreNorm
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(hid_channels,
                            Attention(hid_channels,
                                      heads=heads,
                                      head_channels=head_channels,
                                      dropout=dropout)),
                    PreNorm(hid_channels,
                            FeedForward(hid_channels, 
                                        mlp_channels,
                                        dropout=dropout))
                                        ]))

    def forward(self, x: torch.Tensor, save_attn_matrices=False) -> torch.Tensor:
        '''
        for each iteration (block), 

        input -> PreNorm -> Attention ----> PreNorm -> FeedForward ----> 
          |                           | |                           |
          |____residual connetion_____| |____residual connetion_____|
        '''

        for layer_idx, (attn, ff) in enumerate(self.layers):

            # then we calculate the first transformer block, i.e., layer_idx == 0
            save_attn_matrices = (layer_idx == 0 and save_attn_matrices)

            attn_layer_out, attn_matrices = attn(x, save_attn_matrices=save_attn_matrices)
            x = attn_layer_out + x

            ff_layer_out = ff(x)
            x = ff_layer_out + x

            if save_attn_matrices:
                self.attn_matrices_each_batch = attn_matrices

        return x


class LFP_Transformer_LARGE(nn.Module):
    r'''
    Args:
        num_electrodes (int): The number of electrodes. (default: 3, because 3 regions)
        chunk_size (int): Number of data points included in each EEG chunk. (default: 4096, simple rate=1600Hz, 4096 points = 2.56s)
        t_patch_size (int): The size of each input patch at the temporal (chunk size) dimension. (default: 256)
        hid_channels (int): The num of features after patch embedding process (default: 128)
                            each transformer block outputs 'hid_channels' features
        depth (int): The number of the transformer blocks. (default: :obj:`3`)
                     Each block:
                        input -> PreNorm -> Attention ----> PreNorm -> FeedForward ----> 
                        |                           | |                           |
                        |___________________________| |___________________________|
        heads (int): The number of attention heads for each attention layer. (default: 4)
        head_channels (int): The feature dimension output of each attention head for each attention layer. (default: 64)
        mlp_channels (int): The number of hidden nodes in the fully connected layer of each transformer block. (default: 64)
        num_classes (int): The number of classes to predict. (default: 2)
        embed_dropout (float): Probability of an element to be zeroed in the dropout layers of the embedding layers. (default: 0.0)
        dropout (float): Probability of an element to be zeroed in the dropout layers of the transformer layers. (default: 0.0)
        pool_func (str): The pool function before the classifier, optionally including cls and mean,  (default: cls)
                         where cls represents selecting classification-related token
                               mean represents the average pooling.                 
    '''
    def __init__(self,
                 num_electrodes: int = 3,
                 chunk_size: int = 4096,
                 t_patch_size: int = 256,
                 hid_channels: int = 128,
                 depth: int = 4,
                 heads: int = 4,
                 head_channels: int = 64,
                 mlp_channels: int = 64,
                 num_classes: int = 2,
                 embed_dropout: float = 0.,
                 dropout: float = 0.,
                 pool_func: str = 'cls'):
        super(LFP_Transformer_LARGE, self).__init__()
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.t_patch_size = t_patch_size
        self.hid_channels = hid_channels
        self.depth = depth
        self.heads = heads
        self.head_channels = head_channels
        self.mlp_channels = mlp_channels
        self.num_classes = num_classes
        self.embed_dropout = embed_dropout
        self.dropout = dropout
        self.pool_func = pool_func

        assert chunk_size % t_patch_size == 0, f'EEG chunk size {chunk_size} must be divisible by the temporal patch size {t_patch_size}.'

        num_patches = chunk_size // t_patch_size
        patch_channels = num_electrodes * t_patch_size  # the flattened dimension of each patch

        assert pool_func in {'cls', 'mean'}, 'pool_func must be either cls (cls token) or mean (mean pooling)'

        # ---------- step 1: patch embedding ----------
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (w p) -> b w (c p)', p=t_patch_size),
            nn.Linear(patch_channels, hid_channels),
        )
        # b: batch size, c: channels, w: num of patches, p: patch size in time dimension

        # ---------- step 2: position embedding ----------
        # learnable positional embeddings added to the patch embeddings to retain sequential information.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hid_channels))
        # A learnable token that is prepended to the sequence of patches. It's typically used to gather global information.
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_channels))

        self.dropout = nn.Dropout(embed_dropout)

        # ---------- step 3: transformer ----------
        self.transformer = Transformer(hid_channels, 
                                       depth, 
                                       heads,
                                       head_channels, 
                                       mlp_channels, 
                                       dropout)

        self.pool_func = pool_func

        self.mlp_head = nn.Sequential(nn.LayerNorm(hid_channels),
                                      nn.Linear(hid_channels, num_classes))

    def forward(self, x: torch.Tensor, save_attn_matrices=False) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the default input shape is [b, 3, 4096]
            save_attn_matrices (bool): Whether to save the attention matrices of the FIRST transformer block
                                       Only equals True in the test period (test_step)
        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''

        # ---------- step 1: patch embedding ----------
        x = self.to_patch_embedding(x)
        x = rearrange(x, 'b ... d -> b (...) d')
        b, n, _ = x.shape

        # ---------- step 2: position embedding ----------
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # ---------- step 3: transformer ----------
        x = self.transformer(x, save_attn_matrices=save_attn_matrices)
        # output x.shape = (batch_size, num_patches, hid_channels) = (b,17,128)

        # ---------- step 4: pooling for final classification ----------
        x = x.mean(dim=1) if self.pool_func == 'mean' else x[:, 0]
        # 'mean': computes the mean of the representations across the patch dimension (dim=1). shape = (b,128)
        # 'cls': selects the cls token. shape = (b,128).
        #        note that x[:,0] selects the representation corresponding to the first position (index 0) in the patch dimension 

        # ---------- step 5: classification ----------
        # - nn.LayerNorm(hid_channels): normalizes the features of the representation across the hid_channels dimension.
        # - nn.Linear(hid_channels, num_classes)
        return self.mlp_head(x)