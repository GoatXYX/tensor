# copy from https://github.com/kimiyoung/transformer-xl
from log_uniform_sampler import LogUniformSampler, sample_logits
from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

sys.path.append('utils')


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    '''
    PositionwiseFF 是实现 Transformer 中 前馈网络层（Feed-Forward Network, FFN） 的类。
    它用于对序列中的每个位置分别进行非线性变换，增强模型的表达能力。
    '''

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiLinearAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=30, ext_len=None, mem_len=None, pre_lnorm=False, rand=None, core_nums=1):
        super(MultiLinearAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.R = self.d_head if rand is None else rand
        self.core_nums = core_nums

        self.qkv_net = nn.Linear(
            d_model, 3 * n_head * d_head, bias=False)  # 线性投影层

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        # memory
        mem_tar_len = tgt_len + mem_len

        self.o_net = nn.Linear(mem_tar_len * mem_tar_len,
                               d_model, bias=False)  # 输出层
        self.core_value = nn.Parameter(F.softmax(torch.FloatTensor(
            self.core_nums, self.R), dim=-1), requires_grad=True)  # 核心值权重

        self.layer_norm = nn.LayerNorm(d_model)  # 层归一化

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    # 生成平行四边形掩码、支持双向（left=True）或单向（left=False）的上下文截断。用于掩盖不需要的位置
    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    # 用于调整查询和键之间的偏移，结合掩码矩阵对多头注意力进行局部化处理。
    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    # 实现相对位置编码的偏移调整。
    # 当 zero_triu=True 时，对上三角区域进行掩码操作，确保因果关系。
    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class BlockTensorAttn(MultiLinearAttn):
    def __init__(self, *args, **kwargs):
        super(BlockTensorAttn, self).__init__(*args, **kwargs)

        # 线性投影层，将输入嵌入维度（d_model）映射到每个头的维度（n_head * d_head）。
        self.r_net = nn.Linear(
            self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        '''
        w: 输入的查询序列。
        r: 相对位置嵌入序列。
        r_w_bias 和 r_r_bias: 相对位置的查询和键偏置。
        attn_mask: 掩码，用于截断序列。
        mems: 上一步的记忆，用于处理长序列。
        流程：
        输入序列与记忆拼接。
        生成查询、键和值，并结合偏置。
        通过 torch.einsum 高效计算注意力得分。
        应用残差连接与层归一化。
        '''
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        # 构建QKV
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        # 维度调整
        # qlen x bsz x n_head x d_head
        w_head_q = w_head_q.view(qlen, bsz, self.n_head*self.d_head)
        # klen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head*self.d_head)
        # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head*self.d_head)

        # qlen x n_head x d_head
        r_head_k = r_head_k.view(rlen, self.n_head*self.d_head)

        # 加偏置
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        rr_head_q = w_head_q + r_r_bias

        # 计算注意力分数
        full_matrixs = 0
        for i in range(self.core_nums):
            full_matrix_1 = torch.einsum('h, ibh,jbh,kbh->ibjk',
                                         [self.core_value[i], rw_head_q, w_head_k, w_head_v]).contiguous().view(qlen, bsz, -1)

            full_matrix_2 = torch.einsum('h, ibh,jh,kbh->ibjk',
                                         [self.core_value[i], rr_head_q, r_head_k, w_head_v]).contiguous().view(qlen, bsz, -1)
            # 计算了基于相对位置嵌入的注意力分数。
            full_matrixs += (full_matrix_1 + full_matrix_2)

        # linear projection
        full_matrixs.mul_(1/self.core_nums)
        # 输出线性映射
        attn_out = self.o_net(full_matrixs)

        attn_out = self.drop(attn_out)
        # 残差连接与层归一化
        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class TensorizedDecoderLayer(nn.Module):
    '''
    自注意力机制（通过 self.dec_attn）：
        计算序列元素之间的相关性。
        引入了相对位置嵌入，增强了位置关系建模能力。
    位置前馈神经网络（通过 self.pos_ff）：
        提供非线性建模能力。
        提升解码器对上下文信息的处理。
    '''

    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(TensorizedDecoderLayer, self).__init__()

        self.dec_attn = BlockTensorAttn(
            n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Module):
    '''
    AdaptiveEmbedding 类实现了一种自适应输入嵌入层，主要用于高效处理大规模词汇表（如上百万词汇）。
    自适应嵌入维度：
        根据词汇表的词频将词划分为多个组（由参数 cutoffs 指定分组界限）。
        频率更高的词被分配较小的嵌入维度，频率较低的词分配更大的嵌入维度，从而降低计算成本。
    统一输出维度：
        无论每组的嵌入维度如何，最终都会通过一个线性变换投影到统一的输出维度 d_proj，以便与后续网络处理对接。
    扩展性强：
        可以支持分组的自适应分布（div_val 控制分组之间的嵌入维度变化）。
        支持稀疏 Softmax 的优化（通过 sample_softmax 参数）。
    '''
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0)
            )
            if d_proj != d_embed:
                # output:d_proj
                self.emb_projs.append(nn.Parameter(
                    torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(
                    torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                                   dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class TensorizedTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1):
        '''
        定义模型参数和模块组件，包括嵌入层、解码层和输出层。
        支持多种嵌入技术和注意力机制类型。
        配置记忆长度和输出方式。
        '''
        super(TensorizedTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layer):
                self.layers.append(
                    TensorizedDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        # 初始化可学习的偏置和位置编码参数。
        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                # empty = torch.empty(0, dtype=param.dtype, device=param.device)
                empty = torch.zeros(
                    [self.tgt_len, bsz, self.d_model], dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # 更新记忆机制。
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        # 执行前向传播核心逻辑。
        qlen, bsz = dec_inp.size()
        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0

        # @@@@@@@@@@@@@@@@@@@@
        klen = mlen + qlen

        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]  # -1
        else:
            # dec_attn_mask = torch.triu(
            #     word_emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]
            dec_attn_mask_one = torch.triu(
                torch.ones(qlen, qlen))
            dec_attn_mask = torch.stack(
                [dec_attn_mask_one for i in range(qlen)]).cuda().float()

        hids = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                                 self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems):
        # 包含输入到输出的完整流程，包括记忆管理和损失计算。
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems:
            mems = self.init_mems(data.size(-1))

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                                  self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(
                pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems
