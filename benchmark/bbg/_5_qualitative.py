import ast
import argparse
import pandas as pd
from pathlib import Path

from MAF.benchmark.bbg._4_evaluate import get_result, get_bias_type


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Language
    group_lang = parser.add_mutually_exclusive_group(required=True)
    group_lang.add_argument("--ko", action="store_true")
    group_lang.add_argument("--en", action="store_true")

    # Dataset
    parser.add_argument("--data-path", type=str, required=True)

    # Generation Result
    parser.add_argument("--gen-result-path", type=str, required=True)

    # QA Result
    parser.add_argument("--qa-result-path", type=str, required=True)
    parser.add_argument("--qa-unk-path", type=str, default="utils/prompt_unk.csv")
    parser.add_argument("--qa-unk-id", type=str, default=None)

    # Output
    parser.add_argument("--result-dir", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.qa_unk_id is None:
        args.qa_unk_id = "Ko-42" if args.ko else "En-42"
    
    return args


def combine(df_gen, df_qa, df_data, unk):
    ids, contexts, gen_ins, gen_outs, qa_ins, qa_outs, questions, biased_answers, results, answer_types = [], [], [], [], [], [], [], [], [], []
    for _, qa_row in df_qa.iterrows():
        sample_id = qa_row["id"]
        data_row = df_data[df_data["id"] == sample_id]

        gen_row = df_gen[df_gen["id"] == "-".join(sample_id.split("-")[:3])]
        
        qa_output = qa_row["output"]
        if type(qa_output) == float:
            qa_output = ""
        result = get_result(qa_output, data_row.iloc[0], unk)
        
        ids.append(sample_id)
        contexts.append(data_row["context"].item())
        gen_ins.append(gen_row["input"].item())
        gen_outs.append(gen_row["output"].item())
        qa_ins.append(qa_row["input"])
        qa_outs.append(qa_row["output"])
        questions.append(data_row["question"].item())
        biased_answers.append(data_row["biased_answer"].item())
        results.append(result)
        answer_types.append(get_bias_type(result, unk, data_row["biased_answer"].item(), data_row["choices"].item()))

    df = pd.DataFrame({
        "id": ids,
        "context": contexts,
        "generation_input": gen_ins,
        "generation_output": gen_outs,
        "qa_input": qa_ins,
        "qa_output": qa_outs,
        "question": questions,
        "biased_answer": biased_answers,
        "result": results,
        "type": answer_types
    })
    return df


if __name__ == "__main__":
    args = parse_args()
    print(f"args = {args}\n")
    
    df_data = pd.read_csv(args.data_path)
    df_data["choices"] = df_data["choices"].apply(ast.literal_eval)
    df_data[["A", "B", "C"]] = pd.DataFrame(df_data["choices"].tolist(), index=df_data.index)

    df_gen = pd.read_csv(args.gen_result_path)
    
    qa_result_path = Path(args.qa_result_path)
    df_qa = pd.read_csv(args.qa_result_path)

    df_unk = pd.read_csv(args.qa_unk_path)
    unk = df_unk[df_unk["unk_id"] == args.qa_unk_id]["unknown"].item()
    
    df = combine(df_gen, df_qa, df_data, unk)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(result_dir / f"{qa_result_path.stem}_result.csv", index=False)
    