import os
import json
import types
import argparse
import random
import tqdm
import torch
import numpy as np
from pathlib import Path
import nltk

nltk.download("stopwords")

from MAF.algorithms.postprocessing.casual_path_tracing.lib.utils import (
    LamatrexDataset,
    ModelAndTokenizer,
    set_utils,
    predict_from_normal_and_noise_input,
    get_stopwords,
    make_inputs,
    predict_from_input,
)
from MAF.algorithms.postprocessing.casual_path_tracing.lib.causal_flow_tracer import (
    CausalFlowTracer,
    return_forward_method_dict,
)
from MAF.algorithms.postprocessing.casual_path_tracing.lib.search_func import (
    top_down_causal_flow_trace,
    get_detailed_corrupted_flow_tracer_one_block,
    block_initializer,
)


class CausalPathTracing:
    def __init__(self, args):
        self.args = args
        self.dataset_type = args.dataset_type
        self.model_name = args.model

        # ---Fix Seed---#
        random_seed = 0
        torch.set_grad_enabled(False)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # ---Fix Seed---#

        self.load_and_postprocess_data()
        self.load_model()
        self.logger, self.func_time_saver = set_utils(args)

    def load_and_postprocess_data(self):
        dataset_loaders = {"lama_trex": LamatrexDataset("MAF/data/casual_path_tracing")}
        if self.dataset_type not in dataset_loaders:
            raise ValueError(f"Unsupported dataset: {self.dataset_type}")
        print("Loading..")
        self.dataset_loader = dataset_loaders[self.dataset_type]

    def load_model(self):
        model_loaders = {
            "pythia-1b": "EleutherAI/pythia-1b",
            "pythia-14m": "EleutherAI/pythia-14m",
        }
        if self.model_name not in model_loaders:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.model = ModelAndTokenizer(
            model_loaders[self.model_name],
            low_cpu_mem_usage=False,
            torch_dtype=(
                torch.float16 if "20b" in model_loaders[self.model_name] else None
            ),
        )

        if self.args.except_stopword:
            get_stopwords(self.args, self.model)

    def fairness_algorithm_fit(self):
        # ---Correct Sample Check First---#
        if self.args.correct_check_first:
            self.logger.info("[Correct Check Start]")
            self.func_time_saver.logging(
                self.logger,
                header="\n\n===============Correct Check Log===============",
            )
            # Data Iteration
            correct_ids = []
            for li, statement in tqdm.tqdm(
                enumerate(self.dataset_loader), desc="Correct Check First"
            ):
                prompt = statement["prompt"]
                y = statement["attribute"]

                flow_tracer = CausalFlowTracer(self.args)
                correctness = predict_from_normal_and_noise_input(
                    prompt,
                    flow_tracer,
                    y,
                    self.logger,
                    self.model,
                    0.1,
                    end_symbol=self.args.end_symbol,
                    out_num=1,
                    num_normal_sample=self.args.num_noise_sample,
                    num_noise_sample=self.args.num_noise_sample,
                    noise_type=self.args.noise_type,
                    correct_check_only=True,
                )
                if correctness:
                    correct_ids.append(li)
            self.logger.info("\t Correct Data Num.: {}".format(len(correct_ids)))
            np.savetxt(
                os.path.join(self.args.save_root, "correct_data_idx.txt"),
                np.asarray(correct_ids),
                fmt="%s",
            )
        # ---Correct Sample Check First---#

        causal_path_result_zip = {}
        for li, statement in tqdm.tqdm(enumerate(self.dataset_loader)):
            prompt = statement["prompt"]
            y = statement["attribute"]

            flow_tracer = CausalFlowTracer(self.args)
            (
                flow_tracer,
                answer,
                curr_total_token_num,
                curr_total_block_num,
                normal_inp,
            ) = predict_from_normal_and_noise_input(
                prompt,
                flow_tracer,
                y,
                self.logger,
                self.model,
                0.1,
                end_symbol=self.args.end_symbol,
                out_num=1,
                num_normal_sample=self.args.num_noise_sample,
                num_noise_sample=self.args.num_noise_sample,
                noise_type=self.args.noise_type,
            )

            if isinstance(answer, list) is False:
                if answer == -1:
                    log_line = "--------------------"
                    self.logger.info(log_line)
                    log_line = "[Incorrect Answer Pass] Index:{}, Prompt:{}".format(
                        li, prompt
                    )
                    self.logger.info(log_line)
                    continue

            log_line = "--------------------"
            self.logger.info(log_line)
            log_line = "[Searching Start] Index:{}, Prompt:{}, Ans:{}".format(
                li, prompt, y
            )
            self.logger.info(log_line)
            self.args.curr_save_dir = os.path.join(
                self.args.save_result_root, "R{:04d}".format(li)
            )
            os.makedirs(self.args.curr_save_dir, exist_ok=True)

            curr_meta_arg = {}
            curr_meta_arg["total_block_num"] = curr_total_block_num
            curr_meta_arg["total_token_num"] = curr_total_token_num
            curr_meta_arg["prompt"] = prompt
            curr_meta_arg["mt"] = self.model
            curr_meta_arg["logger"] = self.logger

            stop_flag = False
            stop_flag = top_down_causal_flow_trace(
                args=self.args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer
            )

            if stop_flag is False:
                success_logger = ["li:{}\nprompt:{}\ny:{}".format(li, prompt, y)]
                np.savetxt(
                    os.path.join(self.args.save_inpinfo_root, "I{:06d}.txt".format(li)),
                    success_logger,
                    fmt="%s",
                )

                json_data = {
                    str(k): [[int(i) for i in t] for t in v]
                    for k, v in flow_tracer.traced_paths.items()
                }

                with open(
                    os.path.join(self.args.curr_save_dir, "C{:06d}.json".format(li)),
                    "w",
                ) as f:
                    f.write("{\n")
                    for idx, (k, paths) in enumerate(json_data.items()):
                        f.write(f'  "{k}": [\n')
                        for i, p in enumerate(paths):
                            comma = "," if i < len(paths) - 1 else ""
                            f.write(f"    {p}{comma}\n")
                        f.write(
                            "  ]" + ("," if idx < len(json_data) - 1 else "") + "\n"
                        )
                    f.write("}\n")

                curr_result_zip = {}
                for block_idx, paths in json_data.items():
                    sets = [set(lst) for lst in paths]
                    target_paths = list(set.union(*sets))
                    curr_result_zip.update({int(block_idx): target_paths})
                causal_path_result_zip.update({li: curr_result_zip})

        return causal_path_result_zip

    def compute_metrics(self, causal_path_result_zip):

        sample_wise_result = {
            "org_decision": [],
            "path_decision": [],
            "org_output": [],
            "path_output": [],
            "org_pred_order": [],
            "path_pred_order": [],
        }
        cnt = 0
        prgs = tqdm.tqdm(enumerate(self.dataset_loader))
        for li, statement in prgs:

            if li not in list(causal_path_result_zip.keys()):
                continue

            prompt = statement["prompt"]
            y = statement["attribute"]

            flow_tracer = CausalFlowTracer(self.args)
            (
                flow_tracer,
                answer,
                curr_total_token_num,
                curr_total_block_num,
                normal_inp,
            ) = predict_from_normal_and_noise_input(
                prompt,
                flow_tracer,
                y,
                None,
                self.model,
                0.1,
                end_symbol=[],
                out_num=1,
                num_normal_sample=self.args.num_noise_sample,
                num_noise_sample=self.args.num_noise_sample,
                noise_type=self.args.noise_type,
            )

            curr_meta_arg = {}
            curr_meta_arg["total_block_num"] = curr_total_block_num
            curr_meta_arg["total_token_num"] = curr_total_token_num
            curr_meta_arg["prompt"] = prompt
            curr_meta_arg["mt"] = self.model
            inp = make_inputs(
                curr_meta_arg["mt"].tokenizer,
                [curr_meta_arg["prompt"]] * (self.args.num_noise_sample),
            )

            curr_meta_arg["inp"] = inp

            detailed_flow_tracer = {}
            for detailed_flow_tracer_idx in range(
                0, curr_meta_arg["total_block_num"], 1
            ):
                curr_detailed_corrupted_flow_tracer = (
                    get_detailed_corrupted_flow_tracer_one_block(
                        args=self.args,
                        curr_meta_arg=curr_meta_arg,
                        flow_tracer=flow_tracer,
                        detailed_flow_tracer_idx=detailed_flow_tracer_idx,
                    )
                )

                detailed_flow_tracer.update(
                    {
                        detailed_flow_tracer_idx: {
                            "corrupted": curr_detailed_corrupted_flow_tracer
                        }
                    }
                )

            curr_meta_arg["detailed_flow_tracer"] = detailed_flow_tracer

            if self.args.except_stopword is False:
                normal_p, normal_pred = torch.max(flow_tracer.scores_normal, dim=0)
                desc_idx = torch.argsort(
                    flow_tracer.scores_normal, dim=0, descending=True
                )
                normal_output = torch.index_select(
                    flow_tracer.scores_normal, 0, desc_idx
                )
                pred_ord = desc_idx
            else:
                desc_idx = torch.argsort(
                    flow_tracer.scores_normal, dim=0, descending=True
                )
                sorted_stwd_mask = flow_tracer.stwd_mask[desc_idx]
                preds = desc_idx[sorted_stwd_mask]
                normal_pred = preds[0]
                normal_p = flow_tracer.scores_normal[normal_pred]
                normal_output = torch.index_select(flow_tracer.scores_normal, 0, preds)
                pred_ord = preds

            curr_meta_arg["normal_p"] = normal_p
            curr_meta_arg["normal_pred"] = normal_pred
            curr_meta_arg["normal_output"] = normal_output
            curr_meta_arg["normal_pred_order"] = pred_ord
            if hasattr(curr_meta_arg["mt"].model.config, "use_parallel_residual"):
                curr_meta_arg["num_path_node"] = (
                    curr_meta_arg["mt"].model.config.num_attention_heads + 2
                )
            else:
                curr_meta_arg["num_path_node"] = (
                    curr_meta_arg["mt"].model.config.n_head * 2 + 2
                )

            for bidx in range(curr_meta_arg["total_block_num"]):
                if hasattr(curr_meta_arg["mt"].model, "transformer"):
                    curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
                elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
                    curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
                curr_block = block_initializer(self.args, curr_block)

                fwd_method_dict = return_forward_method_dict(self.args)

                curr_block.forward = types.MethodType(
                    fwd_method_dict["intervention_forward"], curr_block
                )
                curr_block.mlp.forward = types.MethodType(
                    fwd_method_dict["custom_mlp_forward"], curr_block.mlp
                )
                if hasattr(curr_block, "attn"):
                    curr_block.attn.forward = types.MethodType(
                        fwd_method_dict["custom_attn_forward"], curr_block.attn
                    )
                elif hasattr(curr_block, "attention"):
                    curr_block.attention.forward = types.MethodType(
                        fwd_method_dict["custom_attn_forward"], curr_block.attention
                    )

                curr_block.jwon_trace_mode = True
                curr_block.jwon_cond = "path"
                curr_block.jwon_corrupted_feats = curr_meta_arg["detailed_flow_tracer"][
                    bidx
                ]["corrupted"]

                curr_block.jwon_curr_subset = tuple(causal_path_result_zip[li][bidx])

            with torch.no_grad():
                curr_pred, curr_p, raw_out = predict_from_input(
                    curr_meta_arg["mt"].model,
                    curr_meta_arg["inp"],
                    multipred=(self.args.out_num != 1),
                    end_symbol=self.args.end_symbol,
                    use_mean=True,
                    stwd_mask=flow_tracer.stwd_mask,
                )
                raw_out_sftmx = (
                    torch.softmax(raw_out[:, -1], dim=1).mean(dim=0).unsqueeze(0)
                )
                desc_idx = torch.argsort(raw_out_sftmx, dim=1, descending=True)

                if self.args.except_stopword is False:
                    output = torch.index_select(
                        raw_out_sftmx, 1, desc_idx.squeeze(0)
                    ).squeeze(0)
                    pred_order = desc_idx.squeeze(0)
                else:
                    sorted_stwd_mask = flow_tracer.stwd_mask[desc_idx]
                    preds = desc_idx[sorted_stwd_mask]
                    output = torch.index_select(raw_out_sftmx, 1, preds).squeeze(0)

                    pred_order = preds

            for bidx in range(curr_meta_arg["total_block_num"]):
                if hasattr(curr_meta_arg["mt"].model, "transformer"):
                    curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
                elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
                    curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
                curr_block = block_initializer(self.args, curr_block)

            token_id_to_index = {token.item(): i for i, token in enumerate(pred_order)}
            sorted_pred_order = torch.tensor(
                [
                    token_id_to_index[i.item()]
                    for i in curr_meta_arg["normal_pred_order"]
                ]
            )

            sample_wise_result["org_decision"].append(
                curr_meta_arg["normal_pred"].item()
            )
            sample_wise_result["path_decision"].append(curr_pred.item())
            sample_wise_result["org_output"].append(
                curr_meta_arg["normal_output"].cpu()
            )
            sample_wise_result["path_output"].append(output[sorted_pred_order].cpu())
            # sort following normal prediction!!!!!!!!!!!!!!!!!!
            cnt += 1
            prgs.set_description(
                "Processing line {:4d}/{:4d}".format(
                    cnt, len(list(causal_path_result_zip.keys()))
                )
            )

        sample_wise_result["org_decision"] = torch.tensor(
            sample_wise_result["org_decision"]
        )
        sample_wise_result["path_decision"] = torch.tensor(
            sample_wise_result["path_decision"]
        )
        sample_wise_result["org_output"] = torch.vstack(
            sample_wise_result["org_output"]
        )
        sample_wise_result["path_output"] = torch.vstack(
            sample_wise_result["path_output"]
        )

        sample_wise_result["metric"] = {}
        sample_wise_result["metric"]["hit_rate"] = (
            (sample_wise_result["org_decision"] == sample_wise_result["path_decision"])
            .float()
            .mean()
            .item()
        )
        sample_wise_result["metric"]["top1_faithful"] = (
            (
                sample_wise_result["path_output"][:, 0]
                / sample_wise_result["org_output"][:, 0]
            )
            .mean()
            .item()
        )

        return {
            "HitRate": sample_wise_result["metric"]["hit_rate"],
            "Faithfulness": sample_wise_result["metric"]["top1_faithful"],
        }

    def run(self):
        causal_path_result_zip = self.fairness_algorithm_fit()
        return self.compute_metrics(causal_path_result_zip)


def set_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="6", type=str, help="gpu")
    parser.add_argument("--end_symbol", default=[".", "?"])
    parser.add_argument(
        "--save_root",
        default="MAF/algorithms/postprocessing/casual_path_tracing/jobs/debug",
        type=str,
    )

    parser.add_argument("--num_noise_sample", default=100, type=int)
    parser.add_argument("--model", default="pythia-14m", type=str)
    parser.add_argument(
        "--noise_type",
        default="other",
        choices=["other", "emb_added", "zero", "mean"],
        help="noise type",
    )
    parser.add_argument(
        "--dataset_type",
        default="lama_trex",
        choices=["lama_trex"],
        help="dataset_type",
    )

    parser.add_argument("--subset_search", default="minimality", type=str)
    parser.add_argument(
        "--efficient_mode",
        default=True,
        help="not check counterfactual",
        action="store_true",
    )
    parser.add_argument(
        "--slightly_quiet_mode",
        default=True,
        help="slightly_quiet_mode",
        action="store_true",
    )

    parser.add_argument(
        "--except_stopword", default=True, help="except_stopword", action="store_true"
    )
    parser.add_argument(
        "--correct_check_first",
        default=True,
        help="correct_check_first",
        action="store_true",
    )
    args = parser.parse_args()
    args.out_num = 1
    return args


def mitigate_cpt():
    """
    for MAF-DEMO
    """

    class args:
        def __init__(self):
            self.gpu = "0"
            self.end_symbol = [".", "?"]
            self.save_root = (
                "MAF/algorithms/postprocessing/casual_path_tracing/jobs/debug"
            )
            self.num_noise_sample = 100
            self.model = "pythia-14m"
            self.noise_type = "other"
            self.dataset_type = "lama_trex"
            self.subset_search = "minimality"
            self.efficient_mode = True
            self.slightly_quiet_mode = True
            self.except_stopword = True
            self.correct_check_first = True
            self.out_num = 1

    CPT = CausalPathTracing(args=args())
    debiased_data = CPT.run()
    return debiased_data


if __name__ == "__main__":
    debiased_data = mitigate_cpt()
    print(debiased_data)
