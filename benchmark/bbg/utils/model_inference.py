import csv
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


GPT_NAMES = ["gpt-3.5-turbo-0125"]
AZURE_NAMES = ["gpt-4-0613", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13", "gpt-4o-2024-11-20"]
HCX_NAMES = ["HCX-003", "HCX-DASH-001"]
CLAUDE_NAMES = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"]
GEMINI_NAMES = ["gemini-2.0-flash-001"]
TOGETHERAI_NAMES = ["Llama-3.3-70B-Instruct-Turbo", "Qwen2.5-72B-Instruct-Turbo"]

        
def model_inference(model_name, df, output_path=None, return_df=False, **model_kwargs):
    '''
    df: pandas.DataFrame with columns named "id" and "input"
    '''
    results = []

    if not return_df:
        if output_path is None:
            raise ValueError("output_path must be provided when return_df=False")
        output_path = Path(output_path)

        if output_path.is_file():
            print(f'Continue on {output_path}')
            done_ids = pd.read_csv(output_path)["id"].to_list()
        else:
            done_ids = []
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "id", "input", "output"])
    else:
        done_ids = []
    
    sleep = 0
    if model_name in GPT_NAMES:
        from .gpt_utils import GPT
        model = GPT()
    elif model_name in AZURE_NAMES:
        from .azure_utils import AZURE
        model = AZURE()
    elif model_name in HCX_NAMES:
        from .hcx_utils import HCX
        model = HCX()
    elif model_name in CLAUDE_NAMES:
        from .claude_utils import CLAUDE
        model = CLAUDE()
    elif model_name in GEMINI_NAMES:
        from .gemini_utils import GEMINI
        model = GEMINI()
    elif model_name in TOGETHERAI_NAMES:
        from .togetherai_utils import TogetherAI
        model = TogetherAI()
    else:
        raise ValueError(model_name)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=model_name):
        if row["id"] in done_ids:
            continue

        if sleep > 0:
            time.sleep(sleep)
        output = model.get_response(
            model_name,
            row["input"],
            **model_kwargs
        )
        if return_df:
            results.append({
                "id": row["id"],
                "input": row["input"],
                "output": output
            })
        else:
            open_trial = 0
            while True:
                if open_trial >= 10:
                    raise Exception(f"[{datetime.now()}] Open Fail Exception")
                    
                try:
                    with open(output_path, 'a', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now(), row["id"], row["input"], output])
                    break
                except KeyboardInterrupt:
                    raise Exception("Keyboard Interrupt")
                except:
                    print("Open Failed")
                    open_trial += 1
                    continue
    if return_df:
        return pd.DataFrame(results)
