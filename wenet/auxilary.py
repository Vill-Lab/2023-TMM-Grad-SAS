import torch
import warnings
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import numpy as np
import cv2
from matplotlib import pyplot as plt
# from librosa.feature.inverse import mel_to_audio
import torchaudio
import os

# We define this function as _pad because it takes an argument
# named pad, which clobbers the recursive reference to the pad
# function needed for __torch_function__ support
pad = F._pad

# This class exists solely for Transformer; it has an annotation stating
# that bias is never None, which appeases TorchScript
class _LinearWithBias(torch.nn.Linear):
    bias: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)

def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None,
                                 attention_probs_forward_hook = None,
                                 attention_probs_backwards_hook = None,
                                 ) -> Tuple[Tensor, Optional[Tensor]]:
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and F.has_torch_function(tens_ops):
            return F.handle_torch_function(
                multi_head_attention_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    # use hooks for the attention weights if necessary
    if attention_probs_forward_hook is not None and attention_probs_backwards_hook is not None:
        attention_probs_forward_hook(attn_output_weights)
        attn_output_weights.register_hook(attention_probs_backwards_hook)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    im=attn_output_weights.sum(0)/8
    im=im.cpu().detach().numpy()
    cv2.imwrite('att.jpg',(im-im.min())/(im.max()-im.min())*255)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(torch.nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, attention_probs_forward_hook=None, attention_probs_backwards_hook=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                attention_probs_forward_hook=attention_probs_forward_hook,
                attention_probs_backwards_hook=attention_probs_backwards_hook)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                attention_probs_forward_hook=attention_probs_forward_hook,
                attention_probs_backwards_hook=attention_probs_backwards_hook)


class TransformerEncoderLayer(torch.nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     **factory_kwargs)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.attn_probs = None
        self.attn_grad = None

    def set_attn_probs(self, attn_probs):
        self.attn_probs = attn_probs

    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # src2 = self.self_attn(src, src, src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src, src, src, need_weights=False, attn_mask=src_mask, attention_probs_forward_hook=self.set_attn_probs,
                         attention_probs_backwards_hook=self.set_attn_grad)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))




def interpret(model, outputBatch, predIdx, index=None, character=None, focus=False, data=None):
    attn_blocks = list(dict(model.encoder.encoders.named_children()).values())
    one_hot = torch.zeros(outputBatch.shape, dtype=torch.float32)
    for i, (f_oh, idx_out) in enumerate(zip(one_hot[0], predIdx[0])):
        f_oh[idx_out] = 1
    one_hot = torch.sum(one_hot.cuda() * outputBatch)
    one_hot_all = torch.sum(outputBatch)
    one_hot.requires_grad_(True)
    one_hot_all.requires_grad_(True)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
    
    plot = viz_attblk(attn_blocks, predIdx[0], 'A01.jpg', 'mean', reqgrad=False, mult=True)
    plot = viz_attblk(attn_blocks, predIdx[0], 'A11.jpg', 'mean', reqgrad=True, mult=True)
    plot = viz_attblk(attn_blocks, predIdx[0], 'A10.jpg', 'mean', reqgrad=True, mult=False)
    plot = viz_attblk(attn_blocks, predIdx[0], 'A00.jpg', 'mean', reqgrad=False, mult=False)
    model.zero_grad()
    one_hot_all.backward(retain_graph=True)
    viz_attblk(attn_blocks, predIdx[0], 'A11_all.jpg', 'heads', reqgrad=True, mult=True)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A11_all.jpg', 'mean', reqgrad=True, mult=True)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A01_all.jpg', 'mean', reqgrad=False, mult=True)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A10_all.jpg', 'mean', reqgrad=True, mult=False)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A00_all.jpg', 'mean', reqgrad=False, mult=False)
    # visual_attn_blocks = list(dict(model.video_encoder.encoders.named_children()).values())
    # decode_attn_blocks = list(dict(model.decoder.encoders.named_children()).values())
    plot = viz_attblk(attn_blocks, predIdx[0], 'A.jpg', 'mean', grad=True)
    file = '/home/stl/LibriSpeech/test-clean/' + '/'.join(data[0][0].split('-')[:-1]) + '/' + data[0][0] + '.flac'
    wave, sr = torchaudio.load(file)
    weights_interp = torch.nn.functional.interpolate(torch.tensor(weights).unsqueeze(0).unsqueeze(0), size=wave.shape[1], scale_factor=None, mode='nearest')
    wave_revised = wave * (weights_interp > thresh).squeeze(0)
    torchaudio.save(data[0][0] + '.wav', wave, sr)
    torchaudio.save(data[0][0] + '_revized.wav', wave_revised, sr)
    viz_attblk(audio_attn_blocks, predIdx[0], 'A.jpg', 'heads')
    viz_attblk(visual_attn_blocks, predIdx[0], 'V.jpg', 'heads')
    # viz_attblk(decode_attn_blocks, predIdx[0], 'D.jpg', 'heads')
    print(1)

def viz_attblk(blocks, pred_labels, name, method, reqgrad=True, mult=True, fsize=None, lw=0.5, save=True, word=None, data=None):
    if word is None:
        pred_labels_nonzero = [x for x in pred_labels if x != '<blank>']
        pred_labels_nonzero_id = [i for i, x in enumerate(pred_labels) if x != '<blank>']
    else:
        pred_labels_nonzero = [pred_labels[word]]
        pred_labels_nonzero_id = [word]
    num_tokens_audio = blocks[0].self_attn.attn_probs.shape[-1]
    R_audios = torch.eye(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype)
    # R_audios = torch.zeros(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
    R0 = R_audios.clone()
    if method == 'mean':
        # R_audios = []
        # R_audios = torch.zeros(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
        # R_audios = (blocks[0].self_attn.attn_probs*blocks[0].self_attn.attn_grad).clamp(min=0).squeeze().mean(dim=0)
        # for blk in blocks[1:]:
        for blk in blocks:
            grad = blk.self_attn.attn_grad
            cam = blk.self_attn.attn_probs
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            if reqgrad:
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            # cam = cam.mean(dim=0)
            if mult:
                R_audios += torch.matmul(cam, R_audios)
            else:
                R_audios += cam
            
        image_relevance_audio = R_audios - R0
        # image_relevance_audio = R_audios.clone()
        # image_relevance_audio = image_relevance_audio - image_relevance_audio.min(1, keepdims=True)[0]
        image_relevance_audio = image_relevance_audio / image_relevance_audio.max(1, keepdims=True)[0].clamp(min=1e-6)
        # image_relevance_audio = image_relevance_audio - image_relevance_audio.min()
        # image_relevance_audio = image_relevance_audio / image_relevance_audio.max().clamp(min=1e-6)
        image_relevance_audio = image_relevance_audio.detach().cpu().numpy()
        # image_relevance_audio = image_relevance_audio[pred_labels_nonzero_id, :]
        
        if save:
            try:
                fig = plt.figure(figsize=(fsize, 2 * (2+int(len(pred_labels_nonzero)/len(pred_labels)*fsize))))
                waveform, sample_rate = torchaudio.load(data[0][0])
                ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
                time = np.arange(0, waveform.shape[1]) * (1.0 / sample_rate)
                ax1.plot(time, waveform[0])
                # new_ticks = np.linspace(0, waveform.shape[1] * (1.0 / sample_rate))
                # plt.xticks(new_ticks)
                plt.xlim(0, waveform.shape[1] * (1.0 / sample_rate))
                ax2 = fig.add_axes([0.1, 0.2, 0.8, 0.4])
            except:
                fig = plt.figure(figsize=(fsize, fsize))
                ax2 = fig.add_axes([0, 0, 1, 1])
                pass            
            try:
                plt.xticks(ticks=range(num_tokens_audio),labels=[x.replace('<blank>', '')+'-'+str(i) for i, x in enumerate(pred_labels)],rotation=90, ha='left', fontsize=int(fsize/4))
            except:
                pass
            # plt.yticks(ticks=range(num_tokens_audio),labels=[x.replace('<blank>', '<b>')+'-'+str(i) for i, x in enumerate(pred_labels)], ha='right', va='top', fontsize=int(fsize/4))
            # plt.yticks(ticks=range(len(pred_labels_nonzero)),labels=pred_labels_nonzero, ha='right', va='top', fontsize=int(fsize/4))
            plt.yticks(ticks=range(len(pred_labels)),labels=pred_labels, ha='right', va='top', fontsize=int(fsize/4))
            ax2.set_aspect('equal')
            plt.gca().invert_yaxis()
            ax2.pcolormesh(image_relevance_audio, edgecolors='k', linewidths=lw)
            # ax2.plot([x+0.5 for x in pred_labels_nonzero_id], [x+0.5 for x in range(len(pred_labels_nonzero_id))], 'r+', markersize=4)
            plt.savefig(name, bbox_inches='tight')
            plt.close()
        return image_relevance_audio#.mean(0)
    elif method == 'heads':
        # R_audios = 0
        for idx, blk in enumerate(blocks):
            grad = blk.self_attn.attn_grad
            cam = blk.self_attn.attn_probs
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            plt.imsave('cam.jpg', cuda2np(cam.clamp(min=0).mean(dim=0)))
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)# + R_audios
            # R_audios = cam
            plt.imsave('grad.jpg', cuda2np(grad.clamp(min=0).mean(dim=0)))
            plt.imsave('grad_cam.jpg', cuda2np(cam))
            plt.imsave('R_audios_before.jpg', cuda2np(R_audios))
            R_audios += torch.matmul(cam, R_audios)
            plt.imsave('R_audios_after.jpg', cuda2np(R_audios))
            image_relevance_audio = R_audios.clone() #R_audios - torch.eye(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
            image_relevance_audio = image_relevance_audio - image_relevance_audio.min(1, keepdims=True)[0]
            image_relevance_audio = image_relevance_audio / image_relevance_audio.max(1, keepdims=True)[0].clamp(min=1e-6)
            image_relevance_audio = image_relevance_audio.detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(fsize,fsize))
            plt.xticks(ticks=range(num_tokens_audio),labels=pred_labels,rotation=90, ha='left')
            plt.yticks(ticks=range(num_tokens_audio),labels=pred_labels,ma='right')
            plt.gca().invert_yaxis()
            ax.pcolormesh(image_relevance_audio, edgecolors='k', linewidths=lw)
            # ax.plot(range(num_tokens_audio), range(num_tokens_audio), 'wo', markersize=1)
            plt.savefig((str(idx)+'.').join(name.split('.')))  
            plt.close()
    else: raise NotImplementedError

def interpret_old(model, outputBatch, predIdx, index=None, character=None, focus=False, data=None):
    attn_blocks = list(dict(model.encoder.encoders.named_children()).values())
    one_hot = torch.zeros(outputBatch.shape, dtype=torch.float32)
    for i, (f_oh, idx_out) in enumerate(zip(one_hot[0], predIdx[0])):
        f_oh[idx_out] = 1
    one_hot = torch.sum(one_hot.cuda() * outputBatch)
    one_hot_all = torch.sum(outputBatch)
    one_hot.requires_grad_(True)
    one_hot_all.requires_grad_(True)
    # model.zero_grad()
    # one_hot.backward(retain_graph=True)
    # plot = viz_attblk(attn_blocks, predIdx[0], 'A11.jpg', 'mean', reqgrad=True, mult=True)
    # plot = viz_attblk(attn_blocks, predIdx[0], 'A01.jpg', 'mean', reqgrad=False, mult=True)
    # plot = viz_attblk(attn_blocks, predIdx[0], 'A10.jpg', 'mean', reqgrad=True, mult=False)
    # plot = viz_attblk(attn_blocks, predIdx[0], 'A00.jpg', 'mean', reqgrad=False, mult=False)
    model.zero_grad()
    one_hot_all.backward(retain_graph=True)
    viz_attblk(attn_blocks, predIdx[0], 'A11_all.jpg', 'heads', reqgrad=True, mult=True)
    
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A01_all.jpg', 'mean', reqgrad=False, mult=True)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A11_all.jpg', 'mean', reqgrad=True, mult=True)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A10_all.jpg', 'mean', reqgrad=True, mult=False)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A00_all.jpg', 'mean', reqgrad=False, mult=False)
    # visual_attn_blocks = list(dict(model.video_encoder.encoders.named_children()).values())
    # decode_attn_blocks = list(dict(model.decoder.encoders.named_children()).values())
    plot = viz_attblk(attn_blocks, predIdx[0], 'A.jpg', 'mean', grad=True)
    file = '/home/stl/LibriSpeech/test-clean/' + '/'.join(data[0][0].split('-')[:-1]) + '/' + data[0][0] + '.flac'
    wave, sr = torchaudio.load(file)
    weights_interp = torch.nn.functional.interpolate(torch.tensor(weights).unsqueeze(0).unsqueeze(0), size=wave.shape[1], scale_factor=None, mode='nearest')
    wave_revised = wave * (weights_interp > thresh).squeeze(0)
    torchaudio.save(data[0][0] + '.wav', wave, sr)
    torchaudio.save(data[0][0] + '_revized.wav', wave_revised, sr)
    viz_attblk(audio_attn_blocks, predIdx[0], 'A.jpg', 'heads')
    viz_attblk(visual_attn_blocks, predIdx[0], 'V.jpg', 'heads')
    # viz_attblk(decode_attn_blocks, predIdx[0], 'D.jpg', 'heads')
    print(1)

# def viz_attblk(blocks, pred_labels, name, method, reqgrad=True, mult=True, fsize=20, lw=0.5):
#     num_tokens_audio = blocks[0].self_attn.attn_probs.shape[-1]
#     R_audios = torch.eye(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
#     # R_audios = torch.zeros(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
#     R0 = R_audios.clone()
#     if method == 'mean':
#         # R_audios = []
#         # R_audios = torch.zeros(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
#         # R_audios = (blocks[0].self_attn.attn_probs*blocks[0].self_attn.attn_grad).clamp(min=0).squeeze().mean(dim=0)
#         # for blk in blocks[1:]:
#         for blk in blocks:
#             grad = blk.self_attn.attn_grad
#             cam = blk.self_attn.attn_probs
#             cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
#             grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
#             if reqgrad:
#                 cam = grad * cam
#             cam = cam.clamp(min=0).mean(dim=0)
#             if mult:
#                 R_audios += torch.matmul(cam, R_audios)
#             else:
#                 R_audios += cam
            
#         image_relevance_audio = R_audios - R0
#         # image_relevance_audio = R_audios.clone()
#         image_relevance_audio = image_relevance_audio - image_relevance_audio.min(1, keepdims=True)[0]
#         image_relevance_audio = image_relevance_audio / image_relevance_audio.max(1, keepdims=True)[0].clamp(min=1e-6)
#         image_relevance_audio = image_relevance_audio.detach().cpu().numpy()
#         fig, ax = plt.subplots(figsize=(fsize,fsize))
#         plt.xticks(ticks=range(num_tokens_audio),labels=pred_labels,rotation=90, ha='left')
#         plt.yticks(ticks=range(num_tokens_audio),labels=pred_labels,ma='right')
#         plt.gca().invert_yaxis()
#         ax.pcolormesh(image_relevance_audio, edgecolors='k', linewidths=lw)
#         # ax.plot(range(num_tokens_audio), range(num_tokens_audio), 'wo', markersize=1)
#         plt.savefig(name)
#         plt.close()
#         return image_relevance_audio#.mean(0)
#     elif method == 'heads':
#         # R_audios = 0
#         for idx, blk in enumerate(blocks):
#             grad = blk.self_attn.attn_grad
#             cam = blk.self_attn.attn_probs
#             cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
#             grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
#             cam = grad * cam
#             cam = cam.clamp(min=0).mean(dim=0)# + R_audios
#             # R_audios = cam
#             plt.imsave('grad_cam.jpg', cuda2np(cam))
#             plt.imsave('R_audios_before.jpg', cuda2np(R_audios))
#             R_audios += torch.matmul(cam, R_audios)
#             plt.imsave('R_audios_after.jpg', cuda2np(R_audios))
#             image_relevance_audio = R_audios.clone()
#             # image_relevance_audio = R_audios - R0
#             image_relevance_audio = image_relevance_audio - image_relevance_audio.min(1, keepdims=True)[0]
#             image_relevance_audio = image_relevance_audio / image_relevance_audio.max(1, keepdims=True)[0].clamp(min=1e-6)
#             image_relevance_audio = image_relevance_audio.detach().cpu().numpy()
#             fig, ax = plt.subplots(figsize=(fsize,fsize))
#             plt.xticks(ticks=range(num_tokens_audio),labels=pred_labels,rotation=90, ha='left')
#             plt.yticks(ticks=range(num_tokens_audio),labels=pred_labels,ma='right')
#             plt.gca().invert_yaxis()
#             ax.pcolormesh(image_relevance_audio, edgecolors='k', linewidths=lw)
#             # ax.plot(range(num_tokens_audio), range(num_tokens_audio), 'wo', markersize=1)
#             plt.savefig((str(idx)+'.').join(name.split('.')))  
#             plt.close()
#     else: raise NotImplementedError

def cuda2np(t):
    # t = (t-t.min())/(t.max()-t.min())
    t = t - t.min(1, keepdims=True)[0]
    t = t / t.max(1, keepdims=True)[0].clamp(min=1e-6)
    return np.uint8(255*t.detach().cpu().numpy())
# def viz_attblk_for_all(blocks, pred_labels, name, method, reqgrad=True, mult=True, fsize=20, lw=0.5):
#     num_tokens_audio = blocks[0].self_attn.attn_probs.shape[-1]
#     # R_audios = torch.eye(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
#     R_audios = torch.zeros(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
#     R0 = R_audios.clone()
#     if method == 'mean':
#         # R_audios = []
#         # R_audios = torch.zeros(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
#         R_audios = (blocks[0].self_attn.attn_probs*blocks[0].self_attn.attn_grad).clamp(min=0).squeeze().mean(dim=0)
#         # for blk in blocks[1:]:
#         for blk in blocks[1:]:
#             grad = blk.self_attn.attn_grad
#             cam = blk.self_attn.attn_probs
#             cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
#             grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
#             if reqgrad:
#                 cam = grad * cam
#             cam = cam.clamp(min=0).mean(dim=0)
#             if mult:
#                 R_audios += torch.matmul(cam, R_audios)
#             else:
#                 R_audios += cam
            
#         image_relevance_audio = R_audios - R0
#         # image_relevance_audio = R_audios.clone()
#         image_relevance_audio = image_relevance_audio - image_relevance_audio.min(1, keepdims=True)[0]
#         image_relevance_audio = image_relevance_audio / image_relevance_audio.max(1, keepdims=True)[0].clamp(min=1e-6)
#         image_relevance_audio = image_relevance_audio.detach().cpu().numpy()
#         fig, ax = plt.subplots(figsize=(fsize,fsize))
#         plt.xticks(ticks=range(num_tokens_audio),labels=pred_labels,rotation=90, ha='left')
#         plt.yticks(ticks=range(num_tokens_audio),labels=pred_labels,ma='right')
#         plt.gca().invert_yaxis()
#         ax.pcolormesh(image_relevance_audio, edgecolors='k', linewidths=lw)
#         # ax.plot(range(num_tokens_audio), range(num_tokens_audio), 'wo', markersize=1)
#         plt.savefig(name)
#         plt.close()
#         return image_relevance_audio#.mean(0)

def get_mask(model):
    blocks = list(dict(model.encoder.encoders.named_children()).values())
    # one_hot_all = torch.sum(outputBatch)
    # one_hot_all.requires_grad_(True)
    # model.zero_grad()
    # one_hot_all.backward(retain_graph=True)
    num_tokens_audio = blocks[0].self_attn.attn_probs.shape[-1]
    R_audios = torch.eye(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
    R0 = R_audios.clone()
    for blk in blocks:
        grad = blk.self_attn.attn_grad
        cam = blk.self_attn.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R_audios += torch.matmul(cam, R_audios)
    image_relevance_audio = R_audios - R0
    image_relevance_audio = image_relevance_audio - image_relevance_audio.min(1, keepdims=True)[0]
    image_relevance_audio = image_relevance_audio / image_relevance_audio.max(1, keepdims=True)[0].clamp(min=1e-6)
    image_relevance_audio = image_relevance_audio.detach()
    plt.imsave('train.jpg', np.uint8(image_relevance_audio.cpu().numpy()*255))
    return image_relevance_audio.mean(0)

def get_mask_nograd(model):
    raise
    blocks = list(dict(model.encoder.encoders.named_children()).values())
    num_tokens_audio = blocks[0].self_attn.attn_probs.shape[-1]
    R_audios = torch.eye(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
    R0 = R_audios.clone()
    for blk in blocks:
        # grad = blk.self_attn.attn_grad
        cam = blk.self_attn.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        # grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        # cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        # R_audios += torch.matmul(cam, R_audios)
        R_audios += cam
    image_relevance_audio = R_audios - R0
    image_relevance_audio = image_relevance_audio - image_relevance_audio.min(1, keepdims=True)[0]
    image_relevance_audio = image_relevance_audio / image_relevance_audio.max(1, keepdims=True)[0].clamp(min=1e-6)
    image_relevance_audio = image_relevance_audio.detach()
    plt.imsave('train.jpg', np.uint8(image_relevance_audio.cpu().numpy()*255))
    return image_relevance_audio.mean(0)


def get_diags(model, diags):
    blocks = list(dict(model.encoder.encoders.named_children()).values())
    num_tokens_audio = blocks[0].self_attn.attn_probs.shape[-1]
    R_audios = torch.eye(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
    R0 = R_audios.clone()
    for blk in blocks:
        grad = blk.self_attn.attn_grad#.detach()
        cam = blk.self_attn.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R_audios = torch.matmul(cam, R_audios) + R_audios
    image_relevance = R_audios - R0
    image_relevance = image_relevance - image_relevance.detach().min(1, keepdims=True)[0]
    image_relevance = image_relevance / image_relevance.detach().max(1, keepdims=True)[0].clamp(min=1e-6)
    s = 0
    for d in diags:
        s = image_relevance.diag(d).sum() + s
    return s/image_relevance.sum()


def interpret_ctc(model, loss, predIdx, index=None, character=None, focus=False, data=None):
    attn_blocks = list(dict(model.encoder.encoders.named_children()).values())
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    plot = viz_attblk(attn_blocks, predIdx[0], 'A01.jpg', 'mean', reqgrad=False, mult=True)
    plot = viz_attblk(attn_blocks, predIdx[0], 'A11.jpg', 'mean', reqgrad=True, mult=True)
    plot = viz_attblk(attn_blocks, predIdx[0], 'A10.jpg', 'mean', reqgrad=True, mult=False)
    plot = viz_attblk(attn_blocks, predIdx[0], 'A00.jpg', 'mean', reqgrad=False, mult=False)
    model.zero_grad()
    one_hot_all.backward(retain_graph=True)
    viz_attblk(attn_blocks, predIdx[0], 'A11_all.jpg', 'heads', reqgrad=True, mult=True)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A11_all.jpg', 'mean', reqgrad=True, mult=True)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A01_all.jpg', 'mean', reqgrad=False, mult=True)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A10_all.jpg', 'mean', reqgrad=True, mult=False)
    plot_all = viz_attblk(attn_blocks, predIdx[0], 'A00_all.jpg', 'mean', reqgrad=False, mult=False)
    # visual_attn_blocks = list(dict(model.video_encoder.encoders.named_children()).values())
    # decode_attn_blocks = list(dict(model.decoder.encoders.named_children()).values())
    plot = viz_attblk(attn_blocks, predIdx[0], 'A.jpg', 'mean', grad=True)
    file = '/home/stl/LibriSpeech/test-clean/' + '/'.join(data[0][0].split('-')[:-1]) + '/' + data[0][0] + '.flac'
    wave, sr = torchaudio.load(file)
    weights_interp = torch.nn.functional.interpolate(torch.tensor(weights).unsqueeze(0).unsqueeze(0), size=wave.shape[1], scale_factor=None, mode='nearest')
    wave_revised = wave * (weights_interp > thresh).squeeze(0)
    torchaudio.save(data[0][0] + '.wav', wave, sr)
    torchaudio.save(data[0][0] + '_revized.wav', wave_revised, sr)
    viz_attblk(audio_attn_blocks, predIdx[0], 'A.jpg', 'heads')
    viz_attblk(visual_attn_blocks, predIdx[0], 'V.jpg', 'heads')
    # viz_attblk(decode_attn_blocks, predIdx[0], 'D.jpg', 'heads')
    print(1)


def get_mask_ctc_reg(model):
    blocks = list(dict(model.encoder.encoders.named_children()).values())
    num_tokens_audio = blocks[0].self_attn.attn_probs.shape[-1]
    R_audios = torch.eye(num_tokens_audio, num_tokens_audio, dtype=blocks[0].self_attn.attn_probs.dtype).cuda()
    R0 = R_audios.clone()
    for blk in blocks:
        grad = blk.self_attn.attn_grad.detach()
        cam = blk.self_attn.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R_audios += torch.matmul(cam, R_audios)
    image_relevance_audio = R_audios - R0
    # image_relevance_audio = image_relevance_audio - image_relevance_audio.min(1, keepdims=True)[0]
    # image_relevance_audio = image_relevance_audio / image_relevance_audio.max(1, keepdims=True)[0].clamp(min=1e-6)
    # image_relevance_audio = image_relevance_audio.detach()
    # plt.imsave('train.jpg', np.uint8(image_relevance_audio.cpu().numpy()*255))
    return image_relevance_audio.mean(1)


def interpret_by_word(model, outputBatch, predIdx, char_dict, data, tgtWord, base_folder):
    attn_blocks = list(dict(model.encoder.encoders.named_children()).values())
    for i, word in enumerate(predIdx[0]):
        if word in tgtWord:
            model.zero_grad()
            # ((outputBatch[0, i, :].shape[-1] - 1) * outputBatch[0, i, word] - outputBatch[0, i, :word].sum() - outputBatch[0, i, (word+1):].sum()).backward(retain_graph=True)
            (outputBatch[0, i, word] - outputBatch[0, i, 2055]).backward(retain_graph=True) # why hear rather than here?
            # (outputBatch[0, i, word] - outputBatch[0, i, 0]).backward(retain_graph=True) # why hear rather than here?
            # outputBatch[0, i, word].sum().backward(retain_graph=True)
            with torch.no_grad():
                plot = viz_attblk(attn_blocks, [char_dict[x] for x in predIdx[0]], data[0][0]+'.jpg', 'mean', reqgrad=True, mult=True, save=False, word=i, data=data)
                print(i)
                print('done')
            
            pred_labels = [char_dict[x] for x in predIdx[0]]
            pred_labels_nonzero = [x for x in pred_labels if x != '<blank>']
            # pred_labels_nonzero_id = [i for i, x in enumerate(pred_labels) if x != '<blank>']
            num_tokens_audio = attn_blocks[0].self_attn.attn_probs.shape[-1]
            fsize = num_tokens_audio

            # fig = plt.figure(figsize=(fsize, 2+int(len(pred_labels_nonzero)/len(pred_labels)*fsize)))
            fig = plt.figure(figsize=(fsize, fsize))
            ax = fig.add_axes([0, 0, 1, 1])
            plt.xticks(ticks=range(num_tokens_audio),labels=[x.replace('<blank>', '')+'-'+str(i) for i, x in enumerate(pred_labels)],rotation=90, ha='left', fontsize=50)
            plt.yticks(ticks=range(1),labels=[pred_labels[i]], ha='right', va='top', fontsize=50)
            ax.set_aspect('equal')
            plt.gca().invert_yaxis()
            ax.pcolormesh(np.expand_dims(plot[i],0), edgecolors='k', linewidths=0.5)
            ax.plot(i+0.5, 0.5, 'r+', markersize=4)
            folder = os.path.join(base_folder, char_dict[word])
            if not os.path.exists(folder):
                os.mkdir(folder)
            plt.savefig(os.path.join(folder, data[0][0]+'.jpg'), bbox_inches='tight')
            plt.close()