def original_bbq_data(df_data, unk):
    import pandas as pd
    
    df_amb = pd.DataFrame({
        "bbq_context": df_data["amb_context"]
    })
    df_amb = pd.concat([df_data, df_amb], axis=1)
    df_amb["id"] = df_amb["id"].apply(lambda x: x + "-amb")
    df_amb["answer"] = unk
    
    df_dis = pd.DataFrame({
        "bbq_context": df_data.apply(lambda x: x["amb_context"].strip() + " " + x["dis_context"].strip(), axis=1)
    })
    df_dis = pd.concat([df_data, df_dis], axis=1)
    df_dis["id"] = df_dis["id"].apply(lambda x: x + "-dis")
    
    df = pd.concat([df_amb, df_dis], ignore_index=True)
    return df


def raw2prediction(raw, choices):
    import re
    
    try:
        prediction = re.search('^\s*\(?(?P<raw>[^\.\n]*)\s*', raw).groupdict()['raw']
    except:
        prediction = ''
        
    prediction = prediction.replace('없습니다', '없음').replace('입니다', '')
    
    prediction_upper = prediction.upper()

    if prediction_upper and (prediction_upper[0] in choices.keys()): # starts with A, B, C
        try:
            choice = re.search('[:)]\s*(?P<choice>.*)\s*', prediction_upper).groupdict()['choice'].strip()
            
            if len(choice) == 0:
                raise Exception
            
            if choices[prediction_upper[0]] == choice.lower():
                return prediction_upper[0]
            elif sum(prediction_upper.count(alphabet) for alphabet in choices.keys()) == 1:
                prediction = prediction_upper[0]
                return prediction
            elif sum(prediction_upper.count(choice.upper()) > 0 for choice in choices.values()) > 1: # out-of-choice
                return prediction
            else:
                # print(f"'{prediction_upper[0]}' should be '{choices[prediction_upper[0]]}', but '{prediction}' found")
                return prediction_upper[0]
            
        except:
            if sum(prediction_upper.count(alphabet) for alphabet in choices.keys()) == 1:
                prediction = prediction_upper[0]
                return prediction
        
    if prediction.lower() in choices.values(): # one of choices
        return list(choices.keys())[list(choices.values()).index(prediction.lower())]
    
    else:
        try:
            raw = re.search('\*\*[\'\"]?(?P<answer>[^\.\n\*\'\"]*)\s*', raw).groupdict()['answer']
            return raw2prediction(raw, choices)
        except:
            pass
        
        try:
            raw = re.search('답변?[은:]\s*[\'\"]?(?P<answer>[^\.\n\*\'\"]*)\s*', raw).groupdict()['answer']
            return raw2prediction(raw, choices)
        except:
            pass
        
        try:
            raw = re.search('[\'\"](?P<answer>[^\.\n\*\'\"]*)\s*', raw).groupdict()['answer']
            return raw2prediction(raw, choices)
        except:
            pass
        
        raw_upper = raw.upper()
        count = 0
        count_choice = 0
        answer_alphabet = ''
        for alphabet, choice in choices.items():
            if len(re.findall(f'{alphabet}[:).]', raw_upper)) > 0:
                answer_alphabet = alphabet
                count += 1
            elif len(re.findall(choice.upper(), raw_upper)) > 0:
                answer_alphabet = alphabet
                count_choice += 1
        if count == 1:
            return answer_alphabet
        if count_choice == 1:
            return answer_alphabet
        
        return raw
    