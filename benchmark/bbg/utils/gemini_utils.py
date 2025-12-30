import json
import time
from pathlib import Path

import google.generativeai as gemini
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class GEMINI:
    def __init__(self):
        info = json.load(open(Path(__file__).parent / "info.json"))
        gemini.configure(api_key=info["google_genai"]["api_key"])

    def get_response(self, model_name, prompt, max_try=30, temperature=0, max_tokens=4096, **model_kwargs):
        n_try = 0
        while True:
            if n_try == max_try:
                raise Exception("Something Wrong")
            
            try:
                model = gemini.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=gemini.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        **model_kwargs
                    ),
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                    }
                )
                break
                
            except KeyboardInterrupt:
                raise Exception("KeyboardInterrupt")
                
            except Exception as e:
                print(e)
                print("Exception: Sleep for 5 sec")
                time.sleep(5)
                n_try += 1
                continue
                
        try:
            response = response.text
        except Exception as e:
            print(e)
            print(prompt)
            print(response)
            print("Return an empty string")
            response = ''
        
        return response


if __name__ == "__main__":
    model = GEMINI()
    response = model.get_response("gemini-1.5-flash-001", "안녕")
    print(response)
