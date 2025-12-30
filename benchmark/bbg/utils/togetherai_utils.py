import json
import time
from pathlib import Path
from together import Together


class TogetherAI:
    def __init__(self):
        info = json.load(open(Path(__file__).parent / "info.json"))
        self.client = Together(
            api_key=info["togetherai"]["api_key"]
        )
        self.model_dict = {
            "Llama-3.3-70B-Instruct-Turbo": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen2.5-72B-Instruct-Turbo": "Qwen/Qwen2.5-72B-Instruct-Turbo"
        }

    def get_response(self, model_name_key, prompt, max_try=30, seed=42, temperature=0, max_tokens=4096, **model_kwargs):
        model_name = self.model_dict[model_name_key]
        
        n_try = 0
        while True:
            if n_try == max_try:
                raise Exception("Something Wrong")
            
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **model_kwargs
                )
            
                response = response.choices[0].message.content
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
    model = TogetherAI()
    for model_name in ["Llama-3.3-70B-Instruct-Turbo", "Qwen2.5-72B-Instruct-Turbo"]:
        print(model_name)
        response = model.get_response(model_name, "안녕")
        print(response)
