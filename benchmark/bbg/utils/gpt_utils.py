import json
import time
from pathlib import Path
from openai import OpenAI


class GPT:
    def __init__(self):
        info = json.load(open(Path(__file__).parent / "info.json"))
        self.client = OpenAI(
            # organization=info["openai"]["organization"],
            api_key=info["openai"]["api_key"]
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
                    # seed=seed,
                    # temperature=temperature,
                    **model_kwargs
                )
                response = completion.choices[0].message.content
                break
                
            except KeyboardInterrupt:
                raise Exception("KeyboardInterrupt")
                
            except Exception as e:
                print(e)
                print("Exception: Sleep for 5 sec")
                time.sleep(5)
                n_try += 1
                continue
        
        return response


if __name__ == "__main__":
    model = GPT()
    response = model.get_response("gpt-3.5-turbo-0125", "안녕")
    print(response)
