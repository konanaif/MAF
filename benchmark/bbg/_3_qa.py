import ast
import argparse
import pandas as pd
from pathlib import Path

from MAF.benchmark.bbg.utils.utils import original_bbq_data
from MAF.benchmark.bbg.utils.model_inference import model_inference


def parse_args():
    parser = argparse.ArgumentParser()
    group_lang = parser.add_mutually_exclusive_group(required=True)
    group_lang.add_argument("--ko", action="store_true")
    group_lang.add_argument("--en", action="store_true")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--generation-result-path", type=str, default=None)
    parser.add_argument("--qa-model", type=str, default="gpt-4-0613")
    parser.add_argument("--prompt-instruction-path", type=str, default="utils/prompt_qa.csv")
    parser.add_argument("--prompt-instruction-id", type=str, default=None)
    parser.add_argument("--prompt-unk-path", type=str, default="utils/prompt_unk.csv")
    parser.add_argument("--prompt-unk-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    if args.prompt_instruction_id is None:
        args.prompt_instruction_id = "Ko-42" if args.ko else "En-42"
    if args.prompt_unk_id is None:
        args.prompt_unk_id = "Ko-42" if args.ko else "En-42"

    return args


def make_input(gen_or_origin, df_data, inst_row, unk=None, df_gen=None):
    
    def make_prompt(data):
        if gen_or_origin == "gen":
            context = data["amb_context"].strip() + ' ' + data["obfdis_context"].strip() + '\n\n'
            print("data: ",data)
            print("df_gen: ",df_gen)
            output = df_gen[df_gen["id"] == data["template_id"]]["output"].item()
            if type(output) == str:
                context += output.strip() + '\n'
            else:
                print(f"[{data['template_id']}] empty output: {output}")
                return None
        
        else: # "origin"
            context = data["bbq_context"]
        
        prompt = f"{inst_row['instruction'].item().strip()}\n\n"
        prompt += f"{inst_row['context'].item().strip()} {context}\n"
        prompt += f"{inst_row['question'].item().strip()} {data['question']}\n"
        prompt += f"{inst_row['a'].item().strip()} {data['A']}\n"
        prompt += f"{inst_row['b'].item().strip()} {data['B']}\n"
        prompt += f"{inst_row['c'].item().strip()} {data['C']}\n"
        prompt += inst_row['answer'].item().strip()
        return prompt
        
    if gen_or_origin == "origin":
        df_data = original_bbq_data(df_data, unk)
    
    df = pd.DataFrame({"id": df_data["id"]})
    df["input"] = df_data.apply(make_prompt, axis=1)

    if df.isnull().values.any():
        print("NaN exists")
        print(df.isnull().sum())
        df = df.dropna(subset=["input"])
    
    return df


if __name__ == "__main__":
    args = parse_args()
    print(f"args = {args}\n")

    data_path = Path(args.data_path)
    df_data = pd.read_csv(data_path)
    df_data["choices"] = df_data["choices"].apply(ast.literal_eval)
    df_data[["A", "B", "C"]] = pd.DataFrame(df_data["choices"].tolist(), index=df_data.index)

    inst_df = pd.read_csv(args.prompt_instruction_path)
    inst_row = inst_df[inst_df["prompt_id"] == args.prompt_instruction_id]

    if args.generation_result_path:
        gen_or_origin = "gen"
        gen_path = Path(args.generation_result_path)
        df_gen = pd.read_csv(gen_path)
        unk = None
        output_path = Path(args.output_dir) / f"{gen_path.stem}_qa_{args.qa_model}_{args.prompt_instruction_id}.csv"
    else: # original BBQ task
        gen_or_origin = "origin"
        df_gen = None
        df_unk = pd.read_csv(args.prompt_unk_path)
        unk = df_unk[df_unk["unk_id"] == args.prompt_unk_id]["unknown"].item()
        output_path = Path(args.output_dir) / f"{data_path.stem}_qa_{args.qa_model}_{args.prompt_instruction_id}.csv"

    df_input = make_input(gen_or_origin, df_data, inst_row, unk, df_gen)
    model_inference(args.qa_model, df_input, output_path)

    