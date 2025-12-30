import json
import time
import anthropic
from pathlib import Path


class CLAUDE:
    def __init__(self):
        info = json.load(open(Path(__file__).parent / "info.json"))
        self.client = anthropic.Anthropic(
            api_key=info["anthropic"]["api_key"]
        )

    def get_response(self, model_name, prompt, max_try=30, temperature=0, max_tokens=4096, **model_kwargs):
        n_try = 0
        while True:
            if n_try == max_try:
                raise Exception("Something Wrong")
            
            try:
                message = self.client.messages.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response = message.content[0].text
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
    model = CLAUDE()
    response = model.get_response("claude-3-haiku-20240307", "안녕")
    print(response)
