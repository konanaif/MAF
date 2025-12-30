import json
import time
from pathlib import Path
from openai import AzureOpenAI, BadRequestError

from .gpt_utils import GPT


class AZURE:
    def __init__(self):
        info = json.load(open(Path(__file__).parent / "info.json"))
        self.client = AzureOpenAI(
            api_key=info["azure"]["api_key"],
            api_version="2024-05-01-preview",
            azure_endpoint=info["azure"]["endpoint"]
        )
    
    def get_response(self, model_name, prompt, max_try=30, seed=42, temperature=0, **model_kwargs):
        n_try = 0
        while True:
            if n_try == max_try:
                raise Exception("Something Wrong")
            
            try:
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    seed=seed,
                    temperature=temperature,
                    **model_kwargs
                )
                response = completion.choices[0].message.content
                break
                
            except KeyboardInterrupt:
                raise Exception("KeyboardInterrupt")

            except BadRequestError as e:
                print(e)
                print(prompt)
                if e.code == "content_filter":
                    print("Get OpenAI API Response...")
                    model = GPT()
                    response = model.get_response(
                        model_name,
                        prompt,
                        seed=seed, temperature=temperature,
                        **model_kwargs
                    )
                    print(response)
                    break
                else:
                    raise e
                
            except Exception as e:
                print(e)
                print("Exception: Sleep for 5 sec")
                time.sleep(5)
                n_try += 1
                continue
        
        return response    


if __name__ == "__main__":
    model = AZURE()
    response = model.get_response("gpt-4o-2024-11-20", "안녕")
    print(response)
