import ast
import argparse
import types
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import os

# Step modules
import MAF.benchmark.bbg._1_build_data as s1
import MAF.benchmark.bbg._2_generate as s2
import MAF.benchmark.bbg._3_qa as s3
import MAF.benchmark.bbg._4_evaluate as s4
import MAF.benchmark.bbg._5_qualitative as s5

from MAF.benchmark.bbg.utils.model_inference import model_inference

parent_dir = os.environ["PYTHONPATH"]
data_dir = parent_dir + "/MAF/data/bbg/"
# -------------------------
# Shared argument schema
# -------------------------
@dataclass
class SharedConfig:
    # language flags
    ko: bool = True
    en: bool = False

    # build-data
    use_all: bool = False
    gen_context: bool = True
    random_seed: int = 42
    prompt_unk_path: str = parent_dir + "/MAF/benchmark/bbg/utils/prompt_unk.csv"
    prompt_unk_id: str = "Ko-42"
    data_dir: str = data_dir

    # generation
    instruction_path: str = parent_dir + "/MAF/benchmark/bbg/utils/prompt_gen.csv"
    instruction_id: str = "Ko-1"
    model: str = "gpt-3.5-turbo-0125"

    # QA
    qa_model: str = "gpt-3.5-turbo-0125"
    prompt_instruction_path: str = parent_dir + "/MAF/benchmark/bbg/utils/prompt_qa.csv"
    prompt_instruction_id: str = "Ko-42"

    # I/O
    output_dir: str = "outputs"

# -------------------------
# BBG row selection
# -------------------------
@dataclass
class BBGArguments:
    category: str
    stereotype: str
    template_id: str
    data_path: str = data_dir + "/KoBBG_templates.csv"

    def load_rows(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        filtered = df[
            (df["Category"].str.strip() == self.category.strip()) &
            (df["Stereotype"].str.strip() == self.stereotype.strip()) &
            (df["Template_ID"].astype(str).str.strip() == str(self.template_id).strip())
        ]
        if len(filtered) == 0:
            raise ValueError(
                f"No matching rows for {self.category}, {self.stereotype}, {self.template_id}"
            )
        return filtered
    
# -------------------------
# Utility helpers
# -------------------------
def _set_module_args(module, **kwargs):
    module.args = types.SimpleNamespace(**kwargs)

# -------------------------
# Step runners
# -------------------------
def run_step1_build_data(cfg: SharedConfig, bbg_args: BBGArguments) -> Path:
    # Load filtered row
    df_input = bbg_args.load_rows()
    for col in ["Lexical_diversity", "N1_obfdis_adj", "N2_obfdis_adj", "N1_ans_post", "N2_ans_post", "N1_obfdis_post", "N2_obfdis_post"]:
        if col in df_input.columns:
            df_input[col] = df_input[col].fillna('').astype(str)

    _set_module_args(
        s1,
        ko=cfg.ko,
        en=cfg.en,
        all=cfg.use_all,
        random_seed=cfg.random_seed,
        prompt_unk_path=cfg.prompt_unk_path,
        prompt_unk_id=cfg.prompt_unk_id,
        output_dir=str(Path(cfg.data_dir).resolve()),
        gen_context=cfg.gen_context,
    )

    import random, numpy as np
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    df_unk = pd.read_csv(cfg.prompt_unk_path)
    unk = df_unk[df_unk["unk_id"] == cfg.prompt_unk_id]["unknown"].item()

    print("BBQ to BBG... (Step 1)")
    df_output = s1.bbq2bbg(df_input, unk)
    return df_output


def run_step2_generate_df(cfg: SharedConfig, df_data: pd.DataFrame) -> pd.DataFrame:
    df_data = df_data.drop_duplicates(subset=["template_id"])
    if cfg.instruction_path and cfg.instruction_id:
        df_instruction = pd.read_csv(cfg.instruction_path)
        instruction = df_instruction[df_instruction["id"] == cfg.instruction_id]["prompt"].item()
    else:
        instruction = ""
    df_input = s2.make_input(df_data, instruction)
    print("Generate context continuations... (Step 2)")
    df_output = model_inference(model_name = cfg.model, df = df_input, return_df=True)
    return df_output


def run_step3_qa_df(cfg: SharedConfig, df_data: pd.DataFrame, df_gen: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df_data["choices"].iloc[0], str):
        df_data["choices"] = df_data["choices"].apply(ast.literal_eval)

    df_data[["A", "B", "C"]] = pd.DataFrame(df_data["choices"].tolist(), index=df_data.index)

    inst_df = pd.read_csv(cfg.prompt_instruction_path)
    inst_row = inst_df[inst_df["prompt_id"] == cfg.prompt_instruction_id]

    df_input = s3.make_input("gen", df_data, inst_row, None, df_gen)
    print("Run QA over generated continuations... (Step 3)")
    df_output = model_inference(cfg.qa_model, df_input, return_df=True)
    return df_output


def run_step4_evaluate_df(cfg: SharedConfig, df_data: pd.DataFrame, df_qa: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df_data["choices"].iloc[0], str):
        df_data["choices"] = df_data["choices"].apply(ast.literal_eval)
        
    df_data[["A", "B", "C"]] = pd.DataFrame(df_data["choices"].tolist(), index=df_data.index)

    df_unk = pd.read_csv(cfg.prompt_unk_path)
    unk = df_unk[df_unk["unk_id"] == cfg.prompt_unk_id]["unknown"].item()

    print("Evaluate QA results... (Step 4)")
    df_scores = s4.evaluate(df_data, df_qa, unk, gen_or_qa="gen", return_df=True)
    return df_scores


def run_step5_qualitative_df(cfg: SharedConfig, df_data: pd.DataFrame, df_gen: pd.DataFrame, df_qa: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df_data["choices"].iloc[0], str):
        df_data["choices"] = df_data["choices"].apply(ast.literal_eval)

    df_data[["A", "B", "C"]] = pd.DataFrame(df_data["choices"].tolist(), index=df_data.index)

    df_unk = pd.read_csv(cfg.prompt_unk_path)
    unk = df_unk[df_unk["unk_id"] == cfg.prompt_unk_id]["unknown"].item()

    print("Build qualitative sheet... (Step 5)")
    df = s5.combine(df_gen, df_qa, df_data, unk)
    df = df[['id', 'context', 'generation_input', 'generation_output', 'qa_input', 'qa_output', 'question', 'biased_answer', 'result', 'type']]
    return df


def run_bbg_pipeline(args: BBGArguments):
    cfg = SharedConfig()

    # Step1: Build data
    df_data = run_step1_build_data(cfg, args)

    # Step2: Generation
    df_gen = run_step2_generate_df(cfg, df_data)

    # Step3: QA
    df_qa = run_step3_qa_df(cfg, df_data, df_gen)

    # Step4: Evaluate
    df_scores = run_step4_evaluate_df(cfg, df_data, df_qa)

    # Step5: Qualitative
    df_result = run_step5_qualitative_df(cfg, df_data, df_gen, df_qa)
    print("df_scores", df_scores)
    print("df_result", df_result)
    print("df_result_columns", df_result.columns)
    return {
        "gen_text": df_gen.to_dict(orient="records"),
        "scores": df_scores.to_dict(orient="records"),
        "qualitative": df_result.to_dict(orient="records"),
    }
    
if __name__ == "__main__":
    args = BBGArguments(
        category="Age",
        stereotype="기술 사용의 어려움",
        template_id="1"
    )
    result = run_bbg_pipeline(args)