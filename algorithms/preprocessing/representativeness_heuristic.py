import pandas as pd
from openai import OpenAI
import sys, os


def get_gpt4_response(prompt: str):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ["OPENAI_API_KEY"],
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


class RepresentativenessHeuristicMitigator:
    def __init__(self):
        self.prompt = {
            "1": "considering the probability of A and B",
            "2": "considering the inclusion relationship of A and B",
        }

    def run(self, question: str, prompt_no: str):
        baseline_output = get_gpt4_response(question)
        prompt_fair = f"{self.prompt[prompt_no]}\n\n{question}"
        mitigation_output = get_gpt4_response(prompt_fair)
        return {
            "baseline": baseline_output,
            "prompt_fair": prompt_fair,
            "mitigation_output": mitigation_output,
        }

    def run_on_excel_data(
        self,
        excel_path: str = os.environ["PYTHONPATH"]
        + "/MAF/data/RH/"
        + "RH_dataset.xlsx",
        save_path: str = os.environ["PYTHONPATH"] + "/MAF/data/RH/" + "RH_output.xlsx",
    ):
        data = pd.read_excel(excel_path)

        responses_baseline = []
        responses_fair = []

        for index, row in data.iterrows():
            question = row["question"]

            output_baseline = get_gpt4_response(question)
            responses_baseline.append(output_baseline)

            prompt_fair = f"{self.prompt[int(row['category'])]}\n\n{question}"
            output_fair = get_gpt4_response(prompt_fair)
            responses_fair.append(output_fair)

            print(f"Question {index + 1}: {question}")
            print(f"GPT-4 Response (Baseline): {output_baseline}")
            print(f"GPT-4 Response (Fairness): {output_fair}")
            print("-" * 50)

        result = data.copy()
        result["baseline_response"] = responses_baseline
        result["fair_response"] = responses_fair
        result.to_excel(save_path, index=False)


if __name__ == "__main__":
    rhm = RepresentativenessHeuristicMitigator()
    rhm.run_on_excel_data()
    print(
        rhm.run(
            question="A person comes to the movies alone. Which is more probable? A) This person is a man. B) This person is a single man. The answer is",
            prompt_no=1,
        )
    )
