import json
import time
import requests
from pathlib import Path


class HCX:
    def __init__(self):
        info = json.load(open(Path(__file__).parent / "info.json"))
        self._host = "https://clovastudio.apigw.ntruss.com"
        self._api_key = info["naver"]["api_key"]
        self._api_key_primary_val = info["naver"]["api_key_primary_val"]
        self._request_id = info["naver"]["request_id"]

    def get_response(self, model_name, prompt, max_try=30, seed=42, aifilter=False, **model_kwargs):
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
            "Content-Type": "application/json; charset=utf-8"
        }

        request_data = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "seed": seed,
            "includeAiFilters": aifilter,
            **model_kwargs
        }

        n_try = 0
        while True:
            if n_try == max_try:
                raise Exception("Something Wrong")
            
            try:
                response = requests.post(
                    self._host + "/testapp/v1/chat-completions/" + model_name,
                    headers=headers,
                    json=request_data
                )
                response = response.json()["result"]["message"]["content"]
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
    model = HCX()
    response = model.get_response("HCX-DASH-001", "안녕")
    print(response)
