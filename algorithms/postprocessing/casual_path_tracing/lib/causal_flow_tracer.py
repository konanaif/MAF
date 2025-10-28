import copy
import torch
from torch import nn
import numpy as np
from collections import defaultdict

from typing import Optional, Tuple, Union

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from transformers.models.gpt_neox.modeling_gpt_neox import (
    eager_attention_forward as eager_attention_forward_gpt_neox,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from MAF.algorithms.postprocessing.casual_path_tracing.lib.utils import make_inputs


class CausalFlowTracer(object):
    def __init__(self, args):
        self.traced_paths = {}
        self.stwd_ids = None
        if args.except_stopword:
            self.stwd_ids = args.stwd_ids
            self.stwd_mask = None

    def init_node(self):
        self.in_term1 = None
        self.in_term2 = None
        self.in_term3_list = None
        self.in_term4_list = None

    def trace_normal(self, model, inp):

        self.feats_normal = None
        self.feats_k_normal = None
        self.feats_v_normal = None
        self.scores_normal = None

        inputs_block = {"input": []}
        outputs = {"output": [], "output_k": [], "output_v": []}

        def make_hook_fn(block_idx):
            def hook_fn(module, input, output):
                if block_idx == 0:
                    inputs_block["input"].append(input[0].detach())

                if len(output) == 1:
                    outputs["output"].append(output[0].detach())
                elif len(output) == 2:
                    outputs["output"].append(output[0].detach())
                    outputs["output_k"].append(output[1][0].detach())
                    outputs["output_v"].append(output[1][1].detach())
                else:
                    import pdb

                    pdb.set_trace()

            return hook_fn

        forward_hooks = []
        if hasattr(model, "transformer"):
            for block_idx, block in enumerate(model.transformer.h):
                forward_hook = block.register_forward_hook(make_hook_fn(block_idx))
                forward_hooks.append(forward_hook)
        elif hasattr(model, "gpt_neox"):
            for block_idx, block in enumerate(model.gpt_neox.layers):
                forward_hook = block.register_forward_hook(make_hook_fn(block_idx))
                forward_hooks.append(forward_hook)

        with torch.no_grad():
            outputs_exp = model(**inp)

        for forward_hook in forward_hooks:
            forward_hook.remove()

        scores_normal = torch.softmax(outputs_exp.logits[:, -1, :], dim=1)[0]
        if self.stwd_ids is None:
            answer_t = torch.max(scores_normal, dim=0).indices.unsqueeze(0)
        else:
            if self.stwd_mask is None:
                self.stwd_mask = torch.ones(
                    scores_normal.shape[0],
                    dtype=torch.bool,
                    device=scores_normal.device,
                )
                self.stwd_mask[self.stwd_ids] = False

            desc_idx = torch.argsort(scores_normal, dim=0, descending=True)
            sorted_stwd_mask = self.stwd_mask[desc_idx]
            answer_t = desc_idx[sorted_stwd_mask][0].unsqueeze(0)

        all_traced_normal = outputs["output"]
        if len(outputs["output_k"]) != 0:
            all_traced_k_normal = outputs["output_k"]
            self.feats_k_normal = copy.deepcopy(all_traced_k_normal)
        if len(outputs["output_v"]) != 0:
            all_traced_v_normal = outputs["output_v"]
            self.feats_v_normal = copy.deepcopy(all_traced_v_normal)

        # torch.clone makes different outputs -> please use
        self.feats_normal_init = copy.deepcopy(inputs_block["input"][0])
        self.feats_normal = copy.deepcopy(all_traced_normal)
        self.scores_normal = copy.deepcopy(scores_normal)

        return answer_t

    def trace_corrupted(
        self,
        model,
        tokenizer,
        prompt,
        noise_level=0.1,
        rand_seed=0,
        noise_type="other",
        num_noise_sample=3,
    ):

        self.feats_corrupted = None
        self.feats_k_corrupted = None
        self.feats_v_corrupted = None
        self.scores_corrupted = None

        prng = np.random.RandomState(
            rand_seed
        )  # For reproducibility, use pseudorandom noise

        def add_noise_to_embedding(module, input, output):
            noise = (
                noise_level
                * torch.from_numpy(
                    prng.randn(output.shape[0], output.shape[1], output.shape[2])
                )
                .to(output.device)
                .float()
            )
            return output + noise
            # return output

        if noise_type == "emb_added":
            inp = make_inputs(tokenizer, [prompt] * (num_noise_sample))
            embedding_layer = model.transformer.wte
            noise_hook_handle = embedding_layer.register_forward_hook(
                add_noise_to_embedding
            )
        elif noise_type == "other":
            enc_prompt = tokenizer.encode(prompt)
            available_tokens = list(set(range(tokenizer.vocab_size)) - set(enc_prompt))
            selected_tokens = prng.choice(
                available_tokens, size=num_noise_sample * len(enc_prompt)
            )
            inp = make_inputs(
                tokenizer,
                selected_tokens.reshape(num_noise_sample, len(enc_prompt)).tolist(),
                pass_encoding=True,
            )
        else:
            import pdb

            pdb.set_trace()

        inputs_block = {"input": []}
        outputs = {"output": [], "output_k": [], "output_v": []}

        def make_hook_fn(block_idx):
            def hook_fn(module, input, output):
                if block_idx == 0:
                    inputs_block["input"].append(input[0].detach())

                if len(output) == 1:
                    outputs["output"].append(output[0].detach())
                elif len(output) == 2:
                    outputs["output"].append(output[0].detach())
                    outputs["output_k"].append(output[1][0].detach())
                    outputs["output_v"].append(output[1][1].detach())
                else:
                    import pdb

                    pdb.set_trace()

            return hook_fn

        forward_hooks = []
        if hasattr(model, "transformer"):
            for block_idx, block in enumerate(model.transformer.h):
                forward_hook = block.register_forward_hook(make_hook_fn(block_idx))
                forward_hooks.append(forward_hook)
        elif hasattr(model, "gpt_neox"):
            for block_idx, block in enumerate(model.gpt_neox.layers):
                forward_hook = block.register_forward_hook(make_hook_fn(block_idx))
                forward_hooks.append(forward_hook)

        with torch.no_grad():
            outputs_exp = model(**inp)

        for forward_hook in forward_hooks:
            forward_hook.remove()

        if noise_type == "emb_added":
            noise_hook_handle.remove()

        probs_corrupted = (
            torch.softmax(outputs_exp.logits[:, -1, :], dim=1).mean(dim=0).unsqueeze(0)
        )

        if self.stwd_ids is None:
            corrupted_answer_t = torch.max(probs_corrupted, dim=1).indices
        else:
            if self.stwd_mask is None:
                self.stwd_mask = torch.ones(
                    probs_corrupted.shape[0],
                    dtype=torch.bool,
                    device=probs_corrupted.device,
                )
                self.stwd_mask[self.stwd_ids] = False

            desc_idx = torch.argsort(probs_corrupted, dim=1, descending=True)
            sorted_stwd_mask = self.stwd_mask[desc_idx]
            corrupted_answer_t = desc_idx[sorted_stwd_mask][0].unsqueeze(0)

        all_traced_corrupted = outputs["output"]
        if len(outputs["output_k"]) != 0:
            all_traced_k_corrupted = outputs["output_k"]
            self.feats_k_corrupted = copy.deepcopy(all_traced_k_corrupted)
        if len(outputs["output_v"]) != 0:
            all_traced_v_corrupted = outputs["output_v"]
            self.feats_v_corrupted = copy.deepcopy(all_traced_v_corrupted)

        self.feats_corrupted_init = copy.deepcopy(inputs_block["input"][0])
        self.feats_corrupted = copy.deepcopy(all_traced_corrupted)
        self.scores_corrupted = copy.deepcopy(probs_corrupted)
        return corrupted_answer_t, inp["input_ids"]


def get_lm_inp_mu_std(x, eps):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    std = torch.sqrt(var + eps)
    return mu, std


def make_layernorm2mm(x, layernorm_layer, prefix_mu=None, prefix_std=None):
    # x.shape == [batch, token, hidden]
    if (prefix_mu is None) | (prefix_std is None):
        mu, std = get_lm_inp_mu_std(x, eps=layernorm_layer.eps)
    else:
        mu = prefix_mu
        std = prefix_std

    w_ = layernorm_layer.weight / std
    b = -layernorm_layer.weight * mu / std + layernorm_layer.bias
    w = torch.diag_embed(w_)

    return w, b


def ln_mm(x, W, b=None):
    out = x.unsqueeze(-2) @ W.transpose(-1, -2)
    out = out.squeeze(-2)
    if b is not None:
        out = out + b
    return out


def layernorm2mm(x, layernorm_layer, prefix_mu=None, prefix_std=None, bias_divide=None):
    # x.shape == [batch, token, hidden]
    a, c = make_layernorm2mm(x, layernorm_layer, prefix_mu, prefix_std, bias_divide)
    out = x * a + c  # Broadcasting: [batch, token, hidden] * [hidden] + [hidden]
    return out


def make_homoneneous_coord(x):
    ones = torch.ones_like(x[..., :1])
    x_ext = torch.cat([x, ones], dim=-1)
    return x_ext


def subsetidx2pathnodeidx(subsetidx, num_heads):
    subsetidx_list = list(subsetidx)
    target_idx_term1 = []
    target_idx_term2 = []
    target_idx_term3 = []
    target_idx_term4 = []

    for idx in subsetidx_list:
        if idx == 0:
            target_idx_term1.append(idx)
        elif idx == 1:
            target_idx_term2.append(idx - 1)
        elif (idx >= 2) & (idx < (2 + num_heads)):
            target_idx_term3.append(idx - 2)
        elif idx >= (2 + num_heads):
            target_idx_term4.append(idx - 2 - num_heads)

    other_idx_term1 = np.setdiff1d([0], target_idx_term1).tolist()
    other_idx_term2 = np.setdiff1d([0], target_idx_term2).tolist()
    other_idx_term3 = np.setdiff1d(np.arange(num_heads), target_idx_term3).tolist()
    other_idx_term4 = np.setdiff1d(np.arange(num_heads), target_idx_term4).tolist()

    return (target_idx_term1, target_idx_term2, target_idx_term3, target_idx_term4), (
        other_idx_term1,
        other_idx_term2,
        other_idx_term3,
        other_idx_term4,
    )


def path_level_intervention_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]
]:

    if hasattr(self, "jwon_save_mode"):
        if self.jwon_save_mode:
            if hasattr(self, "jwon_feats_corrupted_init"):
                hidden_states = copy.deepcopy(self.jwon_feats_corrupted_init)

    residual = hidden_states

    # -------for Path Division-------#
    residual_in = copy.deepcopy(residual)
    # -------for Path Division-------#

    hidden_states = self.ln_1(hidden_states)

    attn_outputs, head_wise_attn_output = self.attn(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]

    # residual connection
    hidden_states = attn_output + residual

    if encoder_hidden_states is not None:
        # add one self-attention block for cross-attention
        if not hasattr(self, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                "cross-attention layers by setting `config.add_cross_attention=True`"
            )
        residual = hidden_states
        hidden_states = self.ln_cross_attn(hidden_states)
        cross_attn_outputs = self.crossattention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = cross_attn_outputs[0]
        # residual connection
        hidden_states = residual + attn_output
        outputs = (
            outputs + cross_attn_outputs[2:]
        )  # add cross attentions if we output attention weights

    residual = copy.deepcopy(hidden_states.detach())

    # -------for Path Division-------#
    ln2_mu, ln2_std = get_lm_inp_mu_std(
        copy.deepcopy(hidden_states.detach()), self.ln_2.eps
    )
    W_ln_mlp, b_ln_mlp = make_layernorm2mm(
        [], self.ln_2, prefix_mu=ln2_mu, prefix_std=ln2_std
    )
    # -------for Path Division-------#

    hidden_states_ln2 = self.ln_2(hidden_states)

    feed_forward_hidden_states, gelu_input_states = self.mlp(hidden_states_ln2)

    # -------for Path Division-------#
    D_a = torch.where(
        gelu_input_states.abs() > 1e-4,
        nn.functional.gelu(gelu_input_states) / gelu_input_states,
        torch.full_like(gelu_input_states, 0.5),
    )
    # -------for Path Division-------#

    hidden_states = residual + feed_forward_hidden_states

    # -------for Path Division-------#
    org_output = copy.deepcopy(hidden_states)
    # -------for Path Division-------#

    # -------Path Division-------#
    in_term1 = residual_in
    in_term2 = self.mlp(
        ln_mm(residual_in, W_ln_mlp, b=None),
        custom_gelu=D_a,
        bias_divide=None,
        ignore_bias=True,
    )[0]

    in_term3_list, in_term4_list = [], []
    # debug_attn_out_proj = []
    for h_i in range(self.attn.num_heads):
        # Note: nn.Linear -> x*W^T, but, Conv1d -> x*W
        h_attn_out_proj = nn.functional.linear(
            head_wise_attn_output[:, h_i],
            weight=self.attn.c_proj.weight.T[
                :, self.attn.head_dim * (h_i) : self.attn.head_dim * (h_i + 1)
            ],
            bias=self.attn.c_proj.bias / self.attn.num_heads,
        )
        in_term3_i = h_attn_out_proj
        in_term3_list.append(in_term3_i.unsqueeze(0))

        in_term4_i = self.mlp(
            ln_mm(copy.deepcopy(h_attn_out_proj.detach()), W_ln_mlp, b=None),
            custom_gelu=D_a,
            bias_divide=None,
            ignore_bias=True,
        )[0]
        in_term4_list.append(in_term4_i.unsqueeze(0))

    in_term3_list = torch.cat(in_term3_list, dim=0)
    in_term4_list = torch.cat(in_term4_list, dim=0)
    # -------Path Division-------#

    # -------Floating-point Error Correction-------#
    mlp_bias_term1 = ((b_ln_mlp @ self.mlp.c_fc.weight) * D_a) @ self.mlp.c_proj.weight
    mlp_bias_term2 = (self.mlp.c_fc.bias * D_a) @ self.mlp.c_proj.weight
    mlp_bias_term3 = self.mlp.c_proj.bias
    mlp_bias_term_all = mlp_bias_term1 + mlp_bias_term2 + mlp_bias_term3

    in_term2 += mlp_bias_term_all / (self.attn.num_heads + 1)
    in_term4_list += (mlp_bias_term_all / (self.attn.num_heads + 1)).unsqueeze(0)

    divided_output = (
        in_term1
        + in_term2
        + torch.sum(in_term3_list, dim=0)
        + torch.sum(in_term4_list, dim=0)
    )

    comp_err = divided_output - org_output
    per_comp_err = comp_err / (2 * self.attn.num_heads + 2)

    # in_term1 += per_comp_err
    # in_term2 += per_comp_err
    # in_term3_list += per_comp_err.unsqueeze(0)
    # in_term4_list += per_comp_err.unsqueeze(0)
    # -------Floating-point Error Correction-------#

    if hasattr(self, "jwon_save_mode"):
        if self.jwon_save_mode:
            self.jwon_flow_tracer.in_term1 = copy.deepcopy(in_term1)
            self.jwon_flow_tracer.in_term2 = copy.deepcopy(in_term2)
            self.jwon_flow_tracer.in_term3_list = copy.deepcopy(in_term3_list)
            self.jwon_flow_tracer.in_term4_list = copy.deepcopy(in_term4_list)
            self.jwon_flow_tracer.org_output = copy.deepcopy(org_output)

    if hasattr(self, "jwon_trace_mode"):
        if self.jwon_trace_mode:
            target_idx_term, other_idx_term = subsetidx2pathnodeidx(
                self.jwon_curr_subset, self.attn.num_heads
            )
            if self.jwon_cond in ["path", "contingency"]:
                # interv
                if len(target_idx_term[0]) == 0:
                    in_term1 = copy.deepcopy(self.jwon_corrupted_feats.in_term1)
                if len(target_idx_term[1]) == 0:
                    in_term2 = copy.deepcopy(self.jwon_corrupted_feats.in_term2)
                if len(other_idx_term[2]) != 0:
                    in_term3_list[other_idx_term[2]] = copy.deepcopy(
                        self.jwon_corrupted_feats.in_term3_list[other_idx_term[2]]
                    )
                if len(other_idx_term[3]) != 0:
                    in_term4_list[other_idx_term[3]] = copy.deepcopy(
                        self.jwon_corrupted_feats.in_term4_list[other_idx_term[3]]
                    )

            elif self.jwon_cond in ["counterfactual"]:
                in_term1 = copy.deepcopy(self.jwon_corrupted_feats.in_term1)
                in_term2 = copy.deepcopy(self.jwon_corrupted_feats.in_term2)
                in_term3_list = copy.deepcopy(self.jwon_corrupted_feats.in_term3_list)
                in_term4_list = copy.deepcopy(self.jwon_corrupted_feats.in_term4_list)
            else:
                import pdb

                pdb.set_trace()

    # # Unfolded output
    hidden_states = (
        in_term1
        + in_term2
        + torch.sum(in_term3_list, dim=0)
        + torch.sum(in_term4_list, dim=0)
    )

    if hasattr(self, "jwon_trace_mode"):
        if self.jwon_trace_mode:
            if self.jwon_cond in ["path", "contingency"]:
                other_idx_term_len = [len(i) for i in other_idx_term]
                if (
                    sum(other_idx_term_len) == 0
                ):  # If the target subset is equal to the entire set, compensates for the error.
                    hidden_states = org_output
            elif self.jwon_cond in ["counterfactual"]:
                hidden_states = copy.deepcopy(self.jwon_corrupted_feats.org_output)
            else:
                import pdb

                pdb.set_trace()

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def eager_attention_forward(
    module, query, key, value, attention_mask, head_mask=None, **kwargs
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [],
            value.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full(
            [], mask_value, dtype=attn_weights.dtype, device=attn_weights.device
        )
        attn_weights = torch.where(
            causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        )

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


def custom_linear_forward(x, layer, bias_divide=1, ignore_bias=False):
    size_out = x.size()[:-1] + (layer.nf,)
    if ignore_bias:
        x = torch.addmm(
            torch.zeros_like(layer.bias), x.view(-1, x.size(-1)), layer.weight
        )
    else:
        x = torch.addmm(layer.bias / bias_divide, x.view(-1, x.size(-1)), layer.weight)
    x = x.view(size_out)
    return x


def custom_mlp_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    custom_gelu=None,
    bias_divide=None,
    ignore_bias=False,
) -> torch.FloatTensor:
    if (bias_divide == None) & (ignore_bias == False):
        hidden_states = self.c_fc(hidden_states)
    else:
        hidden_states = custom_linear_forward(
            hidden_states, self.c_fc, bias_divide=bias_divide, ignore_bias=ignore_bias
        )

    gelu_input_states = copy.deepcopy(hidden_states.detach())
    if custom_gelu is None:
        hidden_states = self.act(hidden_states)
    else:
        hidden_states = hidden_states * custom_gelu

    if (bias_divide == None) & (ignore_bias == False):
        hidden_states = self.c_proj(hidden_states)
    else:
        hidden_states = custom_linear_forward(
            hidden_states, self.c_proj, bias_divide=bias_divide, ignore_bias=ignore_bias
        )
    hidden_states = self.dropout(hidden_states)
    return hidden_states, gelu_input_states


def custom_attn_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        query_states = self.q_attn(hidden_states)
        key_states, value_states = self.c_attn(encoder_hidden_states).split(
            self.split_size, dim=2
        )
        attention_mask = encoder_attention_mask
    else:
        query_states, key_states, value_states = self.c_attn(hidden_states).split(
            self.split_size, dim=2
        )

    shape_q = (*query_states.shape[:-1], -1, self.head_dim)
    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

    query_states = query_states.view(shape_q).transpose(1, 2)
    key_states = key_states.view(shape_kv).transpose(1, 2)
    value_states = value_states.view(shape_kv).transpose(1, 2)

    if layer_past is not None:
        past_key, past_value = layer_past
        key_states = torch.cat((past_key, key_states), dim=-2)
        value_states = torch.cat((past_value, value_states), dim=-2)

    if use_cache is True:
        present = (key_states, value_states)
    else:
        present = None

    is_cross_attention = encoder_hidden_states is not None
    is_causal = (
        attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention
    )

    using_eager = self.config._attn_implementation == "eager"
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and (
            output_attentions or head_mask is not None
        ):
            using_eager = True
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
            # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
            # not necessarily to eager (if mentionned options are provided).
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    if using_eager and self.reorder_and_upcast_attn:
        attn_output, attn_weights = self._upcast_and_reordered_attn(
            query_states, key_states, value_states, attention_mask, head_mask
        )
    else:
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            head_mask=head_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
            **kwargs,
        )

    head_wise_attn_output = copy.deepcopy(
        attn_output.permute(0, 2, 1, 3).contiguous().detach()
    )
    # batch, num_head, token, head_dim

    attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs, head_wise_attn_output  # a, present, (attentions)


def org_mlp_forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]):
    hidden_states = self.c_fc(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.c_proj(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


def org_attn_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
):
    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        query_states = self.q_attn(hidden_states)
        key_states, value_states = self.c_attn(encoder_hidden_states).split(
            self.split_size, dim=2
        )
        attention_mask = encoder_attention_mask
    else:
        query_states, key_states, value_states = self.c_attn(hidden_states).split(
            self.split_size, dim=2
        )

    shape_q = (*query_states.shape[:-1], -1, self.head_dim)
    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

    query_states = query_states.view(shape_q).transpose(1, 2)
    key_states = key_states.view(shape_kv).transpose(1, 2)
    value_states = value_states.view(shape_kv).transpose(1, 2)

    if layer_past is not None:
        past_key, past_value = layer_past
        key_states = torch.cat((past_key, key_states), dim=-2)
        value_states = torch.cat((past_value, value_states), dim=-2)

    if use_cache is True:
        present = (key_states, value_states)
    else:
        present = None

    is_cross_attention = encoder_hidden_states is not None
    is_causal = (
        attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention
    )

    using_eager = self.config._attn_implementation == "eager"
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and (
            output_attentions or head_mask is not None
        ):
            using_eager = True
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
            # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
            # not necessarily to eager (if mentionned options are provided).
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    if using_eager and self.reorder_and_upcast_attn:
        attn_output, attn_weights = self._upcast_and_reordered_attn(
            query_states, key_states, value_states, attention_mask, head_mask
        )
    else:
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            head_mask=head_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
            **kwargs,
        )
    attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)


def org_block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]
]:

    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]
    # residual connection
    hidden_states = attn_output + residual

    if encoder_hidden_states is not None:
        # add one self-attention block for cross-attention
        if not hasattr(self, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                "cross-attention layers by setting `config.add_cross_attention=True`"
            )
        residual = hidden_states
        hidden_states = self.ln_cross_attn(hidden_states)
        cross_attn_outputs = self.crossattention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = cross_attn_outputs[0]
        # residual connection
        hidden_states = residual + attn_output
        outputs = (
            outputs + cross_attn_outputs[2:]
        )  # add cross attentions if we output attention weights

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)

    feed_forward_hidden_states = self.mlp(hidden_states)

    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def pass_block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]
]:

    B, _, _ = hidden_states.shape
    attn_outputs = (
        copy.deepcopy(self.jwon_feats_k_normal),
        copy.deepcopy(self.jwon_feats_v_normal),
    )
    outputs = (copy.deepcopy(self.jwon_feats_normal), attn_outputs)

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def pass_corrupted_block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]
]:

    B, _, _ = hidden_states.shape
    attn_outputs = (
        copy.deepcopy(self.jwon_feats_k_corrupted),
        copy.deepcopy(self.jwon_feats_v_corrupted),
    )
    outputs = (copy.deepcopy(self.jwon_feats_corrupted), attn_outputs)

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def pass_block_forward_gpt_neox(
    self,
    hidden_states: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):
    outputs = (copy.deepcopy(self.jwon_feats_normal),)
    return outputs


def pass_corrupted_block_forward_gpt_neox(
    self,
    hidden_states: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):
    outputs = (copy.deepcopy(self.jwon_feats_corrupted),)
    return outputs


def path_level_intervention_forward_gpt_neox(
    self,
    hidden_states: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):
    if hasattr(self, "jwon_save_mode"):
        if self.jwon_save_mode:
            if hasattr(self, "jwon_feats_corrupted_init"):
                hidden_states = copy.deepcopy(self.jwon_feats_corrupted_init)

    # -------for Path Division-------#
    residual_in = copy.deepcopy(hidden_states)
    # -------for Path Division-------#

    attn_output, attn_weights, head_wise_attn_output = self.attention(
        self.input_layernorm(hidden_states),
        attention_mask=attention_mask,
        position_ids=position_ids,
        layer_past=layer_past,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    attn_output = self.post_attention_dropout(attn_output)

    if self.use_parallel_residual:
        # pseudocode:
        # x = x + attn(ln1(x)) + mlp(ln2(x))
        mlp_output, gelu_input_states = self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        mlp_output = self.post_mlp_dropout(mlp_output)
        mlp_output_copy = copy.deepcopy(mlp_output)
        hidden_states = mlp_output + attn_output + hidden_states
    else:
        # pseudocode:
        # x = x + attn(ln1(x))
        # x = x + mlp(ln2(x))
        attn_output = attn_output + hidden_states
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output

    org_output = copy.deepcopy(hidden_states)

    in_term3_list = []
    for h_i in range(self.attention.config.num_attention_heads):
        # Note: nn.Linear -> x*W^T, but, Conv1d -> x*W
        h_attn_out_proj = nn.functional.linear(
            head_wise_attn_output[:, h_i],
            weight=self.attention.dense.weight[
                :,
                self.attention.head_size * (h_i) : self.attention.head_size * (h_i + 1),
            ],
            bias=self.attention.dense.bias / self.attention.config.num_attention_heads,
        )
        in_term3_list.append(h_attn_out_proj.unsqueeze(0))

    in_term1 = residual_in
    in_term2 = mlp_output_copy
    in_term3_list = torch.cat(in_term3_list, dim=0)

    if hasattr(self, "jwon_save_mode"):
        if self.jwon_save_mode:
            self.jwon_flow_tracer.in_term1 = copy.deepcopy(in_term1)
            self.jwon_flow_tracer.in_term2 = copy.deepcopy(in_term2)
            self.jwon_flow_tracer.in_term3_list = copy.deepcopy(in_term3_list)
            self.jwon_flow_tracer.org_output = copy.deepcopy(org_output)

    if hasattr(self, "jwon_trace_mode"):
        if self.jwon_trace_mode:
            target_idx_term, other_idx_term = subsetidx2pathnodeidx(
                self.jwon_curr_subset, self.attention.config.num_attention_heads
            )
            if self.jwon_cond in ["path", "contingency"]:
                # interv
                if len(target_idx_term[0]) == 0:
                    in_term1 = copy.deepcopy(self.jwon_corrupted_feats.in_term1)
                if len(target_idx_term[1]) == 0:
                    in_term2 = copy.deepcopy(self.jwon_corrupted_feats.in_term2)
                if len(other_idx_term[2]) != 0:
                    in_term3_list[other_idx_term[2]] = copy.deepcopy(
                        self.jwon_corrupted_feats.in_term3_list[other_idx_term[2]]
                    )
            elif self.jwon_cond in ["counterfactual"]:
                in_term1 = copy.deepcopy(self.jwon_corrupted_feats.in_term1)
                in_term2 = copy.deepcopy(self.jwon_corrupted_feats.in_term2)
                in_term3_list = copy.deepcopy(self.jwon_corrupted_feats.in_term3_list)
            else:
                import pdb

                pdb.set_trace()

    # # Unfolded output
    hidden_states = in_term1 + in_term2 + torch.sum(in_term3_list, dim=0)

    if hasattr(self, "jwon_trace_mode"):
        if self.jwon_trace_mode:
            if self.jwon_cond in ["path", "contingency"]:
                other_idx_term_len = [
                    len(i) for i in other_idx_term[:3]
                ]  # since gpt neox use parallel residual, so we don't need the consideration of term4
                if (
                    sum(other_idx_term_len) == 0
                ):  # If the target subset is equal to the entire set, compensates for the error.
                    hidden_states = org_output
            elif self.jwon_cond in ["counterfactual"]:
                hidden_states = copy.deepcopy(self.jwon_corrupted_feats.org_output)
            else:
                import pdb

                pdb.set_trace()

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def custom_attn_forward_gpt_neox(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)

    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Cache QKV values
    if layer_past is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "partial_rotation_size": self.rotary_ndims,
            "cache_position": cache_position,
        }
        key_states, value_states = layer_past.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # Checking for fallbacks in case an unsupported feature is requested
    attention_type = self.config._attn_implementation
    if (
        output_attentions or head_mask is not None
    ) and self.config._attn_implementation in [
        "sdpa",
        "flash_attention_2",
    ]:
        logger.warning_once(
            f"Setting `attention_type` to `eager` because `{attention_type}` does not support"
            f" `output_attentions=True` or `head_mask`."
        )
        attention_type = "eager"

    elif (
        self.training
        and self.attention_dropout > 0
        and self.config._attn_implementation == "flex_attention"
    ):
        logger.warning_once(
            f"Setting `attention_type` to `eager` because `dropout` is not supported in `{attention_type}`."
        )
        attention_type = "eager"

    attention_interface: Callable = eager_attention_forward_gpt_neox
    attention_interface = (
        ALL_ATTENTION_FUNCTIONS[attention_type]
        if attention_type != "eager"
        else attention_interface
    )

    # Compute attention
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        head_mask=head_mask,
        **kwargs,
    )
    head_wise_attn_output = copy.deepcopy(
        attn_output.permute(0, 2, 1, 3).contiguous().detach()
    )

    # Reshape outputs and final projection
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)

    return attn_output, attn_weights, head_wise_attn_output


def custom_linear_forward_gpt_neox(x, layer, bias_divide=1, ignore_bias=False):
    size_out = x.size()[:-1]
    if ignore_bias:
        x = torch.addmm(
            torch.zeros_like(layer.bias), x.view(-1, x.size(-1)), layer.weight.T
        )
    else:
        x = torch.addmm(
            layer.bias / bias_divide, x.view(-1, x.size(-1)), layer.weight.T
        )
    x = x.view(size_out[0], size_out[1], -1)
    return x


def custom_mlp_forward_gpt_neox(
    self, hidden_states, custom_gelu=None, bias_divide=None, ignore_bias=False
):
    if (bias_divide == None) & (ignore_bias == False):
        hidden_states = self.dense_h_to_4h(hidden_states)
    else:
        hidden_states = custom_linear_forward_gpt_neox(
            hidden_states,
            self.dense_h_to_4h,
            bias_divide=bias_divide,
            ignore_bias=ignore_bias,
        )

    gelu_input_states = copy.deepcopy(hidden_states.detach())
    if custom_gelu is None:
        hidden_states = self.act(hidden_states)
    else:
        hidden_states = hidden_states * custom_gelu

    if (bias_divide == None) & (ignore_bias == False):
        hidden_states = self.dense_4h_to_h(hidden_states)
    else:
        hidden_states = custom_linear_forward_gpt_neox(
            hidden_states,
            self.dense_4h_to_h,
            bias_divide=bias_divide,
            ignore_bias=ignore_bias,
        )
    return hidden_states, gelu_input_states


def org_block_forward_gpt_neox(
    self,
    hidden_states: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):
    attn_output, attn_weights = self.attention(
        self.input_layernorm(hidden_states),
        attention_mask=attention_mask,
        position_ids=position_ids,
        layer_past=layer_past,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    attn_output = self.post_attention_dropout(attn_output)

    if self.use_parallel_residual:
        # pseudocode:
        # x = x + attn(ln1(x)) + mlp(ln2(x))
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output + hidden_states
    else:
        # pseudocode:
        # x = x + attn(ln1(x))
        # x = x + mlp(ln2(x))
        attn_output = attn_output + hidden_states
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def org_attn_forward_gpt_neox(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)

    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Cache QKV values
    if layer_past is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "partial_rotation_size": self.rotary_ndims,
            "cache_position": cache_position,
        }
        key_states, value_states = layer_past.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # Checking for fallbacks in case an unsupported feature is requested
    attention_type = self.config._attn_implementation
    if (
        output_attentions or head_mask is not None
    ) and self.config._attn_implementation in [
        "sdpa",
        "flash_attention_2",
    ]:
        logger.warning_once(
            f"Setting `attention_type` to `eager` because `{attention_type}` does not support"
            f" `output_attentions=True` or `head_mask`."
        )
        attention_type = "eager"

    elif (
        self.training
        and self.attention_dropout > 0
        and self.config._attn_implementation == "flex_attention"
    ):
        logger.warning_once(
            f"Setting `attention_type` to `eager` because `dropout` is not supported in `{attention_type}`."
        )
        attention_type = "eager"

    attention_interface: Callable = eager_attention_forward_gpt_neox
    attention_interface = (
        ALL_ATTENTION_FUNCTIONS[attention_type]
        if attention_type != "eager"
        else attention_interface
    )

    # Compute attention
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        head_mask=head_mask,
        **kwargs,
    )

    # Reshape outputs and final projection
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)

    return attn_output, attn_weights


def org_mlp_forward_gpt_neox(self, hidden_states):
    hidden_states = self.dense_h_to_4h(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.dense_4h_to_h(hidden_states)
    return hidden_states


FUNCTION_MAP = {
    "gpt2": {
        "org_block_forward": org_block_forward,
        "org_attn_forward": org_attn_forward,
        "org_mlp_forward": org_mlp_forward,
        "intervention_forward": path_level_intervention_forward,
        "custom_attn_forward": custom_attn_forward,
        "custom_mlp_forward": custom_mlp_forward,
        "pass_block_forward": pass_block_forward,
        "pass_corrupted_block_forward": pass_corrupted_block_forward,
    },
    "gpt_neox": {
        "org_block_forward": org_block_forward_gpt_neox,
        "org_attn_forward": org_attn_forward_gpt_neox,
        "org_mlp_forward": org_mlp_forward_gpt_neox,
        "intervention_forward": path_level_intervention_forward_gpt_neox,
        "custom_attn_forward": custom_attn_forward_gpt_neox,
        "custom_mlp_forward": custom_mlp_forward_gpt_neox,
        "pass_block_forward": pass_block_forward_gpt_neox,
        "pass_corrupted_block_forward": pass_corrupted_block_forward_gpt_neox,
    },
}


def return_forward_method_dict(args):
    if "gpt2" in args.model:
        return FUNCTION_MAP["gpt2"]
    elif "pythia" in args.model:
        return FUNCTION_MAP["gpt_neox"]
