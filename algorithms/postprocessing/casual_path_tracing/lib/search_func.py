import copy
import types
import time
import pickle
import itertools

from MAF.algorithms.postprocessing.casual_path_tracing.lib.utils import *
from MAF.algorithms.postprocessing.casual_path_tracing.lib.causal_flow_tracer import CausalFlowTracer, return_forward_method_dict

def block_initializer(args, curr_block, symbolic_name="jwon"):
    fwd_method_dict = return_forward_method_dict(args)
    
    ### INITIALIZE
    # restore the original function
    curr_block.forward = types.MethodType(fwd_method_dict["org_block_forward"], curr_block)
    curr_block.mlp.forward = types.MethodType(fwd_method_dict["org_mlp_forward"], curr_block.mlp)
    if hasattr(curr_block, "attn"):
        curr_block.attn.forward = types.MethodType(fwd_method_dict["org_attn_forward"], curr_block.attn)
    elif hasattr(curr_block, "attention"):
        curr_block.attention.forward = types.MethodType(fwd_method_dict["org_attn_forward"], curr_block.attention)
    else:
        import pdb; pdb.set_trace()
        
    # remove previous used values
    dellist = [k for k in curr_block.__dict__.keys() if "jwon" in k]
    _ = [ delattr(curr_block, k) for k in dellist]
    dellist = [k for k in curr_block.mlp.__dict__.keys() if "jwon" in k]
    _ = [ delattr(curr_block.mlp, k) for k in dellist]
    
    if hasattr(curr_block, "attn"):
        dellist = [k for k in curr_block.attn.__dict__.keys() if "jwon" in k]
        _ = [ delattr(curr_block.attn, k) for k in dellist]
    elif hasattr(curr_block, "attention"):
        dellist = [k for k in curr_block.attention.__dict__.keys() if "jwon" in k]
        _ = [ delattr(curr_block.attention, k) for k in dellist]
    else:
        import pdb; pdb.set_trace()
    ### INITIALIZE
    return curr_block

def get_detailed_corrupted_flow_tracer_one_block(
        args, curr_meta_arg, flow_tracer, detailed_flow_tracer_idx):
    fwd_method_dict = return_forward_method_dict(args)

    # just for saving corrupted features
    detailed_flow_tracer = CausalFlowTracer(args)
    detailed_flow_tracer.init_node()

    # Noise RUN
    for bidx in range(curr_meta_arg["total_block_num"]):
        if hasattr(curr_meta_arg["mt"].model, "transformer"):
            curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
        elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
            curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
        curr_block = block_initializer(args, curr_block)

        # change target function
        if bidx < detailed_flow_tracer_idx-1:
            curr_block.forward = types.MethodType(fwd_method_dict["pass_block_forward"], curr_block)
            curr_block.jwon_feats_normal = flow_tracer.feats_normal[bidx]
            if flow_tracer.feats_k_normal is not None:
                curr_block.jwon_feats_k_normal = flow_tracer.feats_k_normal[bidx]
            if flow_tracer.feats_k_normal is not None:
                curr_block.jwon_feats_v_normal = flow_tracer.feats_v_normal[bidx]
        elif bidx == detailed_flow_tracer_idx-1:
            # prepare corrupted features
            curr_block.forward = types.MethodType(fwd_method_dict["pass_corrupted_block_forward"], curr_block)          

            curr_block.jwon_feats_corrupted = flow_tracer.feats_corrupted[bidx]
            if flow_tracer.feats_k_corrupted is not None:
                curr_block.jwon_feats_k_corrupted = flow_tracer.feats_k_corrupted[bidx]
            if flow_tracer.feats_v_corrupted is not None:
                curr_block.jwon_feats_v_corrupted = flow_tracer.feats_v_corrupted[bidx]
        elif bidx == detailed_flow_tracer_idx:
            if bidx == 0: 
                # Since the process of preparing the corrupted feature is missing when bidx==0, replace it with this line instead.
                curr_block.jwon_feats_corrupted_init = flow_tracer.feats_corrupted_init

            # save corrupted features
            curr_block.forward = types.MethodType(fwd_method_dict["intervention_forward"], curr_block)
            curr_block.mlp.forward = types.MethodType(fwd_method_dict["custom_mlp_forward"], curr_block.mlp)
            if hasattr(curr_block, "attn"):
                curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)
            elif hasattr(curr_block, "attention"):
                curr_block.attention.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attention)

            curr_block.jwon_flow_tracer = detailed_flow_tracer
            curr_block.jwon_save_mode = True
        else:
            curr_block.forward = types.MethodType(fwd_method_dict["pass_block_forward"], curr_block)

            curr_block.jwon_feats_normal = flow_tracer.feats_normal[bidx]
            if flow_tracer.feats_k_normal is not None:
                curr_block.jwon_feats_k_normal = flow_tracer.feats_k_normal[bidx]
            if flow_tracer.feats_v_normal is not None:
                curr_block.jwon_feats_v_normal = flow_tracer.feats_v_normal[bidx]

    with torch.no_grad():
        _, _, _ = predict_from_input(curr_meta_arg["mt"].model, curr_meta_arg["inp"], multipred=(args.out_num!=1), end_symbol=args.end_symbol)

    # Restore orignal module
    for bidx in range(curr_meta_arg["total_block_num"]):
        if hasattr(curr_meta_arg["mt"].model, "transformer"):
            curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
        elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
            curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
        curr_block = block_initializer(args, curr_block)

    return detailed_flow_tracer


def cause_edge_classifier(
    args, curr_meta_arg, flow_tracer
):  


    cond_bool = []
    raw_outs = []
    preds = []
    ps = []
    for cond in ["counterfactual", "contingency"]:
        if args.efficient_mode:
            if cond == "counterfactual":
                cond_bool.append(True)
                preds.append(copy.deepcopy(curr_meta_arg["normal_pred"]))
                ps.append(copy.deepcopy(curr_meta_arg["normal_p"]))
                continue
       
        for bidx in range(curr_meta_arg["total_block_num"]):
            if hasattr(curr_meta_arg["mt"].model, "transformer"):
                curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
            elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
                curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
            curr_block = block_initializer(args, curr_block)
            
            fwd_method_dict = return_forward_method_dict(args)

            # change target function
            if bidx < curr_meta_arg["curr_block_idx"]:
                curr_block.forward = types.MethodType(fwd_method_dict['pass_block_forward'], curr_block)
                curr_block.jwon_feats_normal = flow_tracer.feats_normal[bidx]
                if flow_tracer.feats_k_normal is not None:
                    curr_block.jwon_feats_k_normal = flow_tracer.feats_k_normal[bidx]
                if flow_tracer.feats_v_normal is not None:
                    curr_block.jwon_feats_v_normal = flow_tracer.feats_v_normal[bidx]
            elif bidx == curr_meta_arg["curr_block_idx"]:
                curr_block.forward = types.MethodType(fwd_method_dict['intervention_forward'], curr_block)
                curr_block.mlp.forward = types.MethodType(fwd_method_dict['custom_mlp_forward'], curr_block.mlp)
                if hasattr(curr_block, "attn"):
                    curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)
                elif hasattr(curr_block, "attention"):
                    curr_block.attention.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attention)

                curr_block.jwon_trace_mode = True
                curr_block.jwon_cond = cond
                curr_block.jwon_corrupted_feats = curr_meta_arg["detailed_flow_tracer"][bidx]["corrupted"]
                curr_block.jwon_curr_subset = curr_meta_arg["curr_subset"]
            elif bidx > curr_meta_arg["curr_block_idx"]:
                curr_block.forward = types.MethodType(fwd_method_dict['intervention_forward'], curr_block)
                curr_block.mlp.forward = types.MethodType(fwd_method_dict['custom_mlp_forward'], curr_block.mlp)
                if hasattr(curr_block, "attn"):
                    curr_block.attn.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attn)
                elif hasattr(curr_block, "attention"):
                    curr_block.attention.forward = types.MethodType(fwd_method_dict["custom_attn_forward"], curr_block.attention)

                curr_block.jwon_trace_mode = True
                curr_block.jwon_cond = "path"
                curr_block.jwon_corrupted_feats = curr_meta_arg["detailed_flow_tracer"][bidx]["corrupted"]
                curr_block.jwon_curr_subset = tuple(set().union(*flow_tracer.traced_paths[bidx]))

        with torch.no_grad():
            curr_pred, curr_p, raw_out = predict_from_input(
                curr_meta_arg["mt"].model, curr_meta_arg["inp"], 
                multipred=(args.out_num!=1), end_symbol=args.end_symbol, use_mean=True, stwd_mask=flow_tracer.stwd_mask)
        
        out = curr_meta_arg["mt"].model(**curr_meta_arg["inp"])["logits"]
        probs1 = torch.softmax(out[:, -1], dim=1).mean(dim=0).unsqueeze(0)
        probs2 = torch.softmax(out[:, -1], dim=1)[0:1]
        
        preds.append(copy.deepcopy(curr_pred))
        ps.append(copy.deepcopy(curr_p))
        # raw_outs.append(raw_out)
        curr_pred_unq = torch.unique(curr_pred)
        change_bool = curr_pred_unq==curr_meta_arg["normal_pred"]
        if (change_bool.sum()!=change_bool.shape[0]).item():
            if cond == "counterfactual":
                cond_bool.append(True)
            elif cond == "contingency":
                cond_bool.append(False)
        else:
            if cond == "counterfactual":
                cond_bool.append(False)
            elif cond == "contingency":
                cond_bool.append(True)
    cond_bool = torch.tensor(cond_bool)

    actual_cause_bool = (cond_bool==True).all().item()
    if args.slightly_quiet_mode:
        if actual_cause_bool:
            log_line = "\t\t ({})[Subset: {}] | org:{}({}) path:CF-{}({}), CT-{}({})".format(
                "Causal" if actual_cause_bool else "Non-causal",
                curr_meta_arg["curr_subset"],
                curr_meta_arg["normal_pred"].item(), 
                round(curr_meta_arg["normal_p"].item(), 5), 
                preds[0].item() if args.efficient_mode is False else "",
                round(ps[0].item(), 5)if args.efficient_mode is False else "PASS",
                preds[1].item(), 
                round(ps[1].item(), 5),
            )
            curr_meta_arg["logger"].info(log_line)
    else:
        log_line = "\t\t ({})[Subset: {}] | org:{}({}) path:CF-{}({}), CT-{}({})".format(
            "Causal" if actual_cause_bool else "Non-causal",
            curr_meta_arg["curr_subset"],
            curr_meta_arg["normal_pred"].item(), 
            round(curr_meta_arg["normal_p"].item(), 5), 
            preds[0].item() if args.efficient_mode is False else "",
            round(ps[0].item(), 5)if args.efficient_mode is False else "PASS",
            preds[1].item(), 
            round(ps[1].item(), 5),
        )
        curr_meta_arg["logger"].info(log_line)
    
    decision_save = {
        "CF":  [str(preds[0].item()), str(ps[0].item())] if args.efficient_mode is False else ["None", "None"],
        "CT":  [str(preds[1].item()), str(ps[1].item())]
    }
    
    return actual_cause_bool, cond_bool, decision_save


def minimality_search(
    args, curr_meta_arg, flow_tracer):  
    curr_meta_arg["logger"].info("[MinSearch Start]")

    detailed_flow_tracer = {}
    for detailed_flow_tracer_idx in range(
            curr_meta_arg["curr_block_idx"], curr_meta_arg["total_block_num"], 1):
        curr_detailed_corrupted_flow_tracer = get_detailed_corrupted_flow_tracer_one_block(
            args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer,
            detailed_flow_tracer_idx=detailed_flow_tracer_idx)
        
        detailed_flow_tracer.update({
            detailed_flow_tracer_idx: {"corrupted": curr_detailed_corrupted_flow_tracer}
            })

    curr_meta_arg["detailed_flow_tracer"] = detailed_flow_tracer

    if args.except_stopword is False:
        normal_p, normal_pred = torch.max(flow_tracer.scores_normal, dim=0)
    else:
        desc_idx = torch.argsort(flow_tracer.scores_normal, dim=0, descending=True)
        sorted_stwd_mask = flow_tracer.stwd_mask[desc_idx]
        normal_pred = desc_idx[sorted_stwd_mask][0]
        normal_p = flow_tracer.scores_normal[normal_pred]

    curr_meta_arg["normal_p"] = normal_p
    curr_meta_arg["normal_pred"] = normal_pred
    if hasattr(curr_meta_arg['mt'].model.config, "use_parallel_residual"):
        curr_meta_arg["num_path_node"] = curr_meta_arg['mt'].model.config.num_attention_heads + 2
    else:
        curr_meta_arg["num_path_node"] = curr_meta_arg['mt'].model.config.n_head *2 + 2

    Xs_index = np.arange(curr_meta_arg["num_path_node"])
    min_search_memory = {}
    selected_subset = []
    step_subset = {}
    step_subset_cond_bool = {}
    stop_step = curr_meta_arg["num_path_node"]
    
    minsear_iter = sorted(np.arange(curr_meta_arg["num_path_node"])+1)
    for curr_step in minsear_iter:
        curr_step_subset = list(itertools.combinations(Xs_index, curr_step))

        if args.subset_search == "minimality":
            subset_candidate = exclude_subsets(curr_step_subset, selected_subset)
        else:
            subset_candidate = copy.deepcopy(curr_step_subset)
        
        if len(subset_candidate) == 0:
            stop_step = curr_step-1
        
        buffer_cond_bool = []
        if len(subset_candidate) != 0:
            curr_meta_arg["logger"].info("\t MinSearch Step: ({}/{}) -> Target Num.: {}".format(curr_step, max(minsear_iter), len(subset_candidate) ))
        curr_meta_arg["curr_step"] = curr_step
        

        for subidx, curr_subset in enumerate(subset_candidate):
            curr_meta_arg["curr_subset"] = curr_subset

            actual_cause_bool, cond_bool, decision_save = cause_edge_classifier(
                args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer
            )
            
            buffer_cond_bool.append(cond_bool.tolist())
            if actual_cause_bool:
                selected_subset.append(curr_subset)
                    
        step_subset[curr_step] = subset_candidate
        step_subset_cond_bool[curr_step] = buffer_cond_bool
        
            
    stop_flag = False
    if len(selected_subset)==0:
        curr_meta_arg["logger"].info("[Error] Selected subsets are empty!")
        stop_flag = True
        

    min_search_memory["selected_subset"] = selected_subset
    min_search_memory["stop_step"] = stop_step
    min_search_memory["step_subset"] = step_subset
    min_search_memory["step_subset_cond_bool"] = step_subset_cond_bool

    # Restore orignal module
    for bidx in range(curr_meta_arg["total_block_num"]):
        if hasattr(curr_meta_arg["mt"].model, "transformer"):
            curr_block = curr_meta_arg["mt"].model.transformer.h[bidx]
        elif hasattr(curr_meta_arg["mt"].model, "gpt_neox"):
            curr_block = curr_meta_arg["mt"].model.gpt_neox.layers[bidx]
        curr_block = block_initializer(args, curr_block)

    return min_search_memory, stop_flag

def top_down_causal_flow_trace(
    args, curr_meta_arg, flow_tracer):
    
    inp = make_inputs(
        curr_meta_arg["mt"].tokenizer, [curr_meta_arg["prompt"]] * (args.num_noise_sample))
    
    curr_meta_arg["inp"] = inp

    stop_flag = False
    for curr_block_idx in tqdm.tqdm(range(curr_meta_arg["total_block_num"]-1, -1, -1)):
        start_block_time = time.time()
        curr_meta_arg["curr_block_idx"] = curr_block_idx
        curr_meta_arg["logger"].info("-----------Block Change----------- Block:{}".format( curr_meta_arg["curr_block_idx"]))
        
        min_search_memory, stop_flag = minimality_search(args=args, curr_meta_arg=curr_meta_arg, flow_tracer=flow_tracer)
        if stop_flag:
            curr_meta_arg["logger"].info("[STOP] Stopping early at block {} due to empty selection".format(curr_block_idx))
            break
        else:
            flow_tracer.traced_paths.update({curr_block_idx: copy.deepcopy(min_search_memory["selected_subset"])})
       

    return stop_flag