import argparse
import pandas as pd
from pathlib import Path

from MAF.benchmark.bbg.utils.model_inference import model_inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--instruction-path", type=str, default=None)
    parser.add_argument("--instruction-id", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    return args


def make_input(df, instruction):
    
    def make_prompt(row):
        prompt = instruction + row["amb_context"].strip()
        prompt += ' ' + row["obfdis_context"].strip()
        return prompt
    
    df["input"] = df.apply(make_prompt, axis=1)
    return pd.DataFrame({
        "id": df["template_id"],
        "input": df["input"]
    })


if __name__ == "__main__":
    args = parse_args()
    print(f"args = {args}\n")

    data_path = Path(args.data_path)
    df_data = pd.read_csv(data_path).drop_duplicates(subset=["template_id"])

    if args.instruction_path and args.instruction_id != None:
        df_instruction = pd.read_csv(args.instruction_path)
        instruction = df_instruction[df_instruction["id"] == args.instruction_id]["prompt"].item()
        print(f"instruction = {instruction}\n")
    else:
        instruction = ''
    
    df_input = make_input(df_data, instruction)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{data_path.stem}_gen_{args.model}_{args.instruction_id}.csv"
    
    model_inference(args.model, df_input, output_path)
    