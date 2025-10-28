import torch
import json
import random
import os, sys
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.optim import AdamW
from transformers import (
    XLMRobertaTokenizer,
    RobertaTokenizer,
    BertModel,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaLMHead,
)

from torch import nn
import pandas as pd
import logging
from typing import Optional
from scipy.stats import spearmanr

## Customized packages
from MAF.datamodule.concse.concse_dataset import LoadNLI, LoadSTSB
from MAF.datamodule.concse.concse_dataloader import (
    Iterator,
    CustomCollator,
    CustomDataLoader,
)
from MAF.loss.loss import TripletLoss
from MAF.utils.common import fix_seed


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, model_config):
        super().__init__()
        self.dense = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], (
            "unrecognized pooling type %s" % self.pooler_type
        )

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state  # encoder의 last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, args, model_config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = args.pooler_type
    cls.pooler = Pooler(cls.pooler_type)
    if cls.pooler_type == "cls":
        cls.mlp = MLPLayer(model_config)
    cls.sim = Similarity(temp=args.temp)
    if cls.args.method == "ConCSE":  # TripletLoss
        cls.triplet = TripletLoss(cls.args.margin)

    cls.init_weights()  # BertModel,Pooler, MLPLayer, Similarity 모두 random initialize


def cl_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)

    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None

    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view(
        (-1, attention_mask.size(-1))
    )  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(
            (-1, token_type_ids.size(-1))
        )  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=(
            True if cls.args.pooler_type in ["avg_top2", "avg_first_last"] else False
        ),
        return_dict=True,
    )

    # Pooling
    pooler_output = cls.pooler(
        attention_mask, outputs
    )  # encoder(bs*num_sent, max_len, 768)

    # (bs*num_ent, hidden) => (bs, num_sent, hidden)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.args.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
    # Separate representation
    z1, z2 = (
        pooler_output[:, 0],
        pooler_output[:, 1],
    )  # positive pair(sentence1,sentence1)

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))  # z1_z3_cos = [64,64]
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)  # cos_sim = [64,128]
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.args.hard_negative_weight

        weights = torch.tensor(
            [
                [0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1))
                + [0.0] * i
                + [z3_weight]
                + [0.0] * (z1_z3_cos.size(-1) - i - 1)
                for i in range(z1_z3_cos.size(-1))
            ]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1)
        )
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


## cross_forward is ConCSE
def cross_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 6: ours
    num_sent = input_ids.size(1)

    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view(
        (-1, attention_mask.size(-1))
    )  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(
            (-1, token_type_ids.size(-1))
        )  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=(
            True if cls.args.pooler_type in ["avg_top2", "avg_first_last"] else False
        ),
        return_dict=True,
    )
    # Pooling
    pooler_output = cls.pooler(
        attention_mask, outputs
    )  # encoder(bs*num_sent, max_len, 768)

    # (bs*num_ent, hidden) => (bs, num_sent, hidden)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.args.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    e0, e1 = pooler_output[:, 0], pooler_output[:, 1]  # en_positive pair
    e2 = pooler_output[:, 2]  # en_hard_negative
    c0, c1 = pooler_output[:, 3], pooler_output[:, 4]  # cs_positive pair
    c2 = pooler_output[:, 5]  # cs_hard_negative

    if cls.args.method == "ConCSE":

        e0_e1_cos = cls.sim(e0.unsqueeze(1), e1.unsqueeze(0))
        e0_e2_cos = cls.sim(e0.unsqueeze(1), e2.unsqueeze(0))
        c0_c1_cos = cls.sim(c0.unsqueeze(1), c1.unsqueeze(0))
        c0_c2_cos = cls.sim(c0.unsqueeze(1), c2.unsqueeze(0))
        e0_c0_cos = cls.sim(e0.unsqueeze(1), c0.unsqueeze(0))
        e1_c1_cos = cls.sim(e1.unsqueeze(1), c1.unsqueeze(0))
        e2_c2_cos = cls.sim(e2.unsqueeze(1), c2.unsqueeze(0))

        e0_c2_cos = cls.sim(e0.unsqueeze(1), c2.unsqueeze(0))
        c0_e2_cos = cls.sim(c0.unsqueeze(1), e2.unsqueeze(0))
        e1_c2_cos = cls.sim(e1.unsqueeze(1), c2.unsqueeze(0))

        sim1 = torch.cat([e0_e1_cos, e0_e2_cos], 1)
        sim2 = torch.cat([c0_c1_cos, c0_c2_cos], 1)
        sim3 = torch.cat([e0_e1_cos, e0_c2_cos], 1)
        sim4 = torch.cat([c0_c1_cos, c0_e2_cos], 1)
        sim5 = torch.cat([e0_c0_cos, c0_e2_cos], 1)
        sim6 = torch.cat([e1_c1_cos, e1_c2_cos], 1)

        cat_labels = torch.arange(sim1.size(0)).long().to(cls.device)
        e0_c0_labels = torch.arange(e0_c0_cos.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()
        e2_weight = cls.args.hard_negative_weight
        weights = torch.tensor(
            [
                [0.0] * (sim1.size(-1) - e0_e2_cos.size(-1))
                + [0.0] * i
                + [e2_weight]
                + [0.0] * (e0_e2_cos.size(-1) - i - 1)
                for i in range(e0_e2_cos.size(-1))
            ]
        ).to(cls.device)

        sim1 = sim1 + weights
        sim2 = sim2 + weights
        sim3 = sim3 + weights
        sim4 = sim4 + weights
        sim5 = sim5 + weights
        sim6 = sim6 + weights

        ## Cross Contrastive loss
        sim1_loss = loss_fct(sim1, cat_labels)
        sim2_loss = loss_fct(sim2, cat_labels)
        sim3_loss = loss_fct(sim3, cat_labels)
        sim4_loss = loss_fct(sim4, cat_labels)
        sim5_loss = loss_fct(sim5, cat_labels)
        sim6_loss = loss_fct(sim6, cat_labels)

        ## Neg_align_loss
        cs_pos_loss = loss_fct(e2_c2_cos, e0_c0_labels)  # cs_pos_loss

        ce_loss = (
            sim1_loss
            + sim2_loss
            + sim3_loss
            + sim4_loss
            + sim5_loss
            + sim6_loss
            + cs_pos_loss
        )

        ## Cross Triplet loss
        sim1_tri_loss = cls.triplet(e0, e1, e2)
        sim2_tri_loss = cls.triplet(c0, c1, c2)
        sim3_tri_loss = cls.triplet(e0, e1, c2)
        sim4_tri_loss = cls.triplet(c0, c1, e2)
        sim5_tri_loss = cls.triplet(e0, c0, e2)
        sim6_tri_loss = cls.triplet(e1, c1, c2)

        tri_loss = (
            sim1_tri_loss
            + sim2_tri_loss
            + sim3_tri_loss
            + sim4_tri_loss
            + sim5_tri_loss
            + sim6_tri_loss
        )

        total_loss = ce_loss + cls.args.triplet * tri_loss
        total_logits = None

    return SequenceClassifierOutput(
        loss=total_loss,
        logits=total_logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=(
            True if cls.args.pooler_type in ["avg_top2", "avg_first_last"] else False
        ),
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    """
    if cls.args.pooler_type == "cls" and not cls.args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)
    """
    if cls.args.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    def __init__(self, pretrained_model_name_or_path, args, model_config):
        super().__init__(model_config)
        self.args = args
        self.model_config = model_config
        self.model_name = self.args.model
        self.pooler_type = self.args.pooler_type
        self.hard_negative_weight = self.args.hard_negative_weight

        self.bert = BertModel(self.model_config, add_pooling_layer=False)
        cl_init(self, self.args, self.model_config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        simcse=False,
        cross=False,
    ):

        if sent_emb:
            # Encoder
            return sentemb_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif simcse:
            # SimCSE
            return cl_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
        elif cross:
            # ConCSE
            return cross_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class RobertaForCL(RobertaPreTrainedModel):
    def __init__(self, pretrained_model_name_or_path, args, model_config):
        super().__init__(model_config)
        self.args = args
        self.model_config = model_config
        self.model_name = self.args.model
        self.pooler_type = self.args.pooler_type
        self.hard_negative_weight = self.args.hard_negative_weight

        self.roberta = RobertaModel(self.model_config, add_pooling_layer=False)
        cl_init(self, self.args, self.model_config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        simcse=False,
        cross=False,
    ):

        if sent_emb:
            # Encoder
            return sentemb_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif simcse:
            # SimCSE
            return cl_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
        elif cross:
            # ConCSE
            return cross_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class TaskSpecificEvaluator:
    def __init__(self, model, ckpt_savepath, test_collator, device, args):

        self.model = model
        self.ckpt_savepath = ckpt_savepath
        self.test_collator = test_collator
        self.device = device
        self.args = args

    def test_model(self):
        self.test_collator.data_loaders.reset()
        try:
            print("Load .pt in ", self.ckpt_savepath + ".pt")
            self.model.load_state_dict(
                torch.load(self.ckpt_savepath + ".pt", map_location=self.device)
            )
        except FileNotFoundError:
            print(self.ckpt_savepath + ".pt")
        self.model.eval()
        test_total_examples = 0
        predicted_scores, true_scores = [], []
        cos_sim_fct = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            for data in tqdm(
                self.test_collator, total=len(self.test_collator), desc="Test"
            ):
                labels = data["labels"]
                sent1_input_ids = data["input_ids"][:, 0, :].to(self.device)
                sent1_attention_mask = data["attention_mask"][:, 0, :].to(self.device)
                sent1_token_type_ids = data["token_type_ids"][:, 0, :].to(self.device)
                bs = sent1_input_ids.size(0)
                sent1_outputs = self.model(
                    input_ids=sent1_input_ids,
                    attention_mask=sent1_attention_mask,
                    token_type_ids=sent1_token_type_ids,
                    output_hidden_states=True,
                    return_dict=True,
                    sent_emb=True,
                )
                sent2_input_ids = data["input_ids"][:, 1, :].to(self.device)
                sent2_attention_mask = data["attention_mask"][:, 1, :].to(self.device)
                sent2_token_type_ids = data["token_type_ids"][:, 1, :].to(self.device)
                sent2_outputs = self.model(
                    input_ids=sent2_input_ids,
                    attention_mask=sent2_attention_mask,
                    token_type_ids=sent2_token_type_ids,
                    output_hidden_states=True,
                    return_dict=True,
                    sent_emb=True,
                )
                z1_pooler_output = sent1_outputs["pooler_output"].cpu()
                z2_pooler_output = sent2_outputs["pooler_output"].cpu()
                sys_score = (
                    cos_sim_fct(z1_pooler_output, z2_pooler_output).detach().numpy()
                )
                predicted_scores.extend(sys_score)
                true_scores.extend(labels.numpy())
                test_total_examples += bs
            print("test_total_examples : ", test_total_examples)
            spearman_corr, pvalue = spearmanr(predicted_scores, true_scores)
            return spearman_corr, pvalue


class TrainingConfig:
    def __init__(self, config):
        self.epochs = config.get("EPOCHS")
        self.patience = config.get("PATIENCE")
        self.batch_size = config.get("BATCH_SIZE")
        self.max_length = config.get("MAX_SEQ_LEN")


class Arguments:
    def __init__(self, model: str):
        parent_dir = os.environ["PYTHONPATH"]
        self.random_seed = 11111
        self.model = model
        self.lang_type = "cross2cross"
        self.ckpt_dir = parent_dir + "/MAF/model/"
        self.pooler_type = "cls"
        self.hard_negative_weight = 0
        self.method = "ConCSE"
        self.task = "stsb"
        self.eval_type = "transfer"
        self.temp = 0.05
        self.schedular = "linear"
        self.warmup_step = 0.1
        self.lr = 5e-5
        self.eps = 1e-8
        self.margin = 1.0
        self.triplet = 1.2
        save_dir = os.path.join(
            self.ckpt_dir,
            self.method,
            self.model,
            self.task,
            self.lang_type,
            "temp" + str(self.temp),
        )
        save_filename = (
            self.schedular
            + "_warm_"
            + str(self.warmup_step)
            + "_lr_"
            + str(self.lr)
            + "_margin_"
            + str(self.margin)
            + "_triplet_"
            + str(self.triplet)
        )
        self.ckpt_savepath = os.path.join(save_dir, save_filename)


def set_config(base_model: str):
    parent_dir = os.environ["PYTHONPATH"]
    if base_model == "mbert_uncased" or base_model == "xlmr_base":
        with open(
            parent_dir + "/MAF/algorithms/config/concse/concse_base.json", "r"
        ) as f:
            train_config = json.load(f)
    elif base_model == "xlmr_large":
        with open(
            parent_dir + "/MAF/algorithms/config/concse/concse_large.json", "r"
        ) as f:
            train_config = json.load(f)
    return train_config


def mitigate_concse(base_model: str):
    logging.disable(logging.WARNING)
    args = Arguments(model=base_model)
    train_config = TrainingConfig(set_config(base_model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fix_seed(args.random_seed)

    ## mBERT
    if args.model == "mbert_uncased":
        pretrained_model_name_or_path = "bert-base-multilingual-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

    ## XLM-R
    if args.model == "xlmr_base":
        pretrained_model_name_or_path = "xlm-roberta-base"
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    if args.model == "xlmr_large":
        pretrained_model_name_or_path = "xlm-roberta-large"
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

    train_data = LoadNLI.cross_train()
    valid_data = LoadSTSB.cross_valid()
    test_data = LoadSTSB.cross_test()

    print("len(test): ", len(test_data), test_data.columns)

    test_iter = Iterator(test_data, tokenizer, train_config, args)
    test_data_loaders = CustomDataLoader(
        test_iter, train_config, args, iter_type="test"
    )
    test_collator = CustomCollator(test_data_loaders, args, iter_type="test")
    print("===========Collator&DataLoader_Examples==========")
    print(f"lang_type : {args.lang_type}_{args.method}")
    print("=====Test_collator=====")
    print(tokenizer.batch_decode(test_collator.__next__()["input_ids"][0]))
    if args.model == "mbert_uncased":
        model = BertForCL.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            args=args,
            model_config=model_config,
        )
        print("Start to load model")
        model.load_state_dict(
            torch.load(args.ckpt_savepath + ".pt", map_location=device)
        )

    if "xlmr" in args.model:
        model = RobertaForCL.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            args=args,
            model_config=model_config,
        )
        print("Start to load model")
        model.load_state_dict(
            torch.load(args.ckpt_savepath + ".pt", map_location=device)
        )

    model.to(device)
    taskSpecificEvaluator = TaskSpecificEvaluator(
        model=model,
        ckpt_savepath=args.ckpt_savepath,
        test_collator=test_collator,
        device=device,
        args=args,
    )

    ##Evaluation
    spearman_corr, pvalue = taskSpecificEvaluator.test_model()
    print(f"Spearman_corr: {spearman_corr}")
    print(f"P-Value: {pvalue}")
    return spearman_corr


if __name__ == "__main__":
    mitigate_concse(base_model="mbert_uncased")
