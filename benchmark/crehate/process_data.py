import re


def raw2prediction(x, hn_or_nh):
    if hn_or_nh == "hn":
        choices = ["Hate", "Non-hate"]
    elif hn_or_nh == "nh":
        choices = ["Non-hate", "Hate"]
    choice_alphabets = ["a", "b"]
    choice2idx = {"a": 0, "b": 1}

    try:
        raw = x.strip()
    except:
        print("Exception in raw2prediction", x)

    if (raw.startswith("a") and "a\n" in raw) or "A is the correct answer" in raw:
        return choices[choice2idx["a"]]

    if raw.startswith("b") and "b\n" in raw:
        return choices[choice2idx["b"]]

    if (
        "post is not hate" in raw.lower()
        or "it's non-hate" in raw.lower()
        or "is not hate" in raw.lower()
        or "it's not hate" in raw.lower()
        or "it's not a hate" in raw.lower()
        or "don't think this is hate" in raw.lower()
        or "is not a hate" in raw.lower()
        or "would not call it hate" in raw.lower()
        or "would not consider it as a hate" in raw.lower()
    ):
        return "Non-hate"

    if (
        raw.startswith("Hate.")
        or raw.startswith("Hate Speech")
        or raw.startswith("Hate\n")
        or "it is hate" in raw.lower()
        or "that's a hate" in raw.lower()
        or "think it's hate" in raw.lower()
        or "this is a hate" in raw.lower()
    ):
        return "Hate"

    if raw.startswith("Non-hate.") or raw.startswith("Non-hate\n"):
        return "Non-hate"

    if raw.startswith("Comment:"):
        raw = raw.replace("Comment:", "")

    if "Classify this text as" in raw:
        raw = re.sub("Classify this text as[\s\S]+", "", raw)

    if "answer:" in raw:
        raw = re.sub("[\s\S]+answer:\s", "", raw)

    if "Answer:" in raw:
        raw = re.sub("[\s\S]+Answer:\s", "", raw)

    if "can be classified as " in raw:
        raw = re.sub("[\s\S]+can be classified as ", "", raw)

    if "can be categorized as " in raw:
        raw = re.sub("[\s\S]+can be categorized as ", "", raw)

    if "is classified as " in raw:
        raw = re.sub("[\s\S]+is classified as ", "", raw)

    raw = raw.replace("hateful", "Hate")
    raw = raw.replace("hate speech", "Hate")
    raw = raw.replace("Hate speech", "Hate")
    raw = raw.replace("###", "")

    if "can be considered hateful" in raw:
        return "Hate"

    if "does not contain hate" in raw:
        return "Non-hate"

    try:
        raw = re.search("\*\*\s*(?P<raw>.*)\s*\*\*", raw).groupdict()["raw"]
    except:
        pass

    if "answer is" in raw:
        regex = "answer is\s*(?P<answer>[^\.\n<]*)"
    else:
        regex = "Answer\s*:\s*(?P<answer>[^\.\n<]*)"

    try:
        prediction = re.search(regex, raw).groupdict()["answer"]
    except:
        prediction = raw

    try:
        if raw.strip()[0] == "(":
            regex = "(Option|option|[\*\s]*)\s*(?P<answer>[^\.\n\*<]*)"
        else:
            regex = "(Option|option|[\*\s]*)\s*(?P<answer>[^\.\n\*(<]*)"
    except:
        if raw.strip() == "(":
            regex = "(Option|option|[\*\s]*)\s*(?P<answer>[^\.\n\*<]*)"
        else:
            regex = "(Option|option|[\*\s]*)\s*(?P<answer>[^\.\n\*(<]*)"
    try:
        prediction = re.search(regex, prediction).groupdict()["answer"]
    except:
        prediction = re.search("\s*(?P<answer>[^\.\n\*(<]*)", prediction).groupdict()[
            "answer"
        ]

    prediction = re.sub("[^\S ]", "", prediction.strip(" \n\t'\"()"))

    if len(prediction) == 0:
        prediction = raw
    if len(prediction) == 0:
        return ""
    prediction_upper = prediction.lower()

    if prediction_upper[0] in choice_alphabets:
        try:
            choice = re.search(
                "[:)\-]\s*(?P<choice>[^(]*)", prediction_upper
            ).groupdict()["choice"]
        except:
            if (
                prediction_upper.count(choice_alphabets[0])
                + prediction_upper.count(choice_alphabets[1])
                == 1
            ):
                prediction = prediction_upper[0]

            if prediction in choice_alphabets:
                prediction = choices[choice2idx[prediction]]

            if prediction.startswith("non"):
                prediction = "Non-hate"
            if prediction in choice_alphabets:
                prediction = choices[choice2idx[prediction]]
            if prediction in [
                "hate",
                "H",
                "h",
                "H for Hate",
                'a" Hate',
                "Yes, a: Hate",
                "This post would be considered as a Hate",
                "[a] Hate",
                "b, Hate",
                "b Hate",
                "h: Hate",
                "B  Hate",
                "b  Hate",
                "Hate Speech",
                "b  Hate",
                'a" Hate',
                "This text is hate",
                "HATE",
                "I think it's hate",
                "I think it's a hate post",
                "\\strong{Hate}",
                "I think this is a hate post",
                'hate"?',
            ]:
                prediction = "Hate"
            if prediction in [
                "Non-Hate Speech",
                "The post provided does not contain Hate",
                "Non-hate",
                "I: Non-hate",
                "Not-hate",
                "n",
                "N",
                "No, b: Non-hate",
                'Not enough context to determine if it is Hate or not, "b: Non-hate',
                "Non-hate a",
                "Non-hate Speech",
                "Non-Hate",
                "It is not hate",
                "I think it's non-hate",
                "I think it's a non-hate post",
                "It's not hate",
                "Non",
                "This is non-hate",
                "Non-",
            ]:
                prediction = "Non-hate"
            if prediction == "1":
                prediction = choices[0]

            return prediction

        if choices[choice2idx[prediction_upper[0]]].lower() == choice.lower():
            prediction = prediction_upper[0]
        else:
            prediction = prediction_upper[0]
    if prediction.startswith("non"):
        prediction = "Non-hate"
    if prediction in choice_alphabets:
        prediction = choices[choice2idx[prediction]]
    if prediction in [
        "hate",
        "H",
        "h",
        "H for Hate",
        'a" Hate',
        "Yes, a: Hate",
        "This post would be considered as a Hate",
        "[a] Hate",
        "b, Hate",
        "b Hate",
        "h: Hate",
        "B  Hate",
        "b  Hate",
        "Hate Speech",
        "b  Hate",
        'a" Hate',
        "This text is hate",
        "HATE",
        "I think it's hate",
        "I think it's a hate post",
        "\\strong{Hate}",
        "I think this is a hate post",
        'hate"?',
    ]:
        prediction = "Hate"
    if prediction in [
        "Non-Hate Speech",
        "The post provided does not contain Hate",
        "Non-hate",
        "I: Non-hate",
        "Not-hate",
        "n",
        "N",
        "No, b: Non-hate",
        'Not enough context to determine if it is Hate or not, "b: Non-hate',
        "Non-hate a",
        "Non-hate Speech",
        "Non-Hate",
        "It is not hate",
        "I think it's non-hate",
        "I think it's a non-hate post",
        "It's not hate",
        "Non",
        "This is non-hate",
        "Non-",
    ]:
        prediction = "Non-hate"
    if prediction == "1":
        prediction = choices[0]

    return prediction


def prediction_2_label(prediction, ab2label):
    prediction_lower = prediction.lower()
    if prediction_lower in [
        "a",
        ab2label["a"].lower(),
        ab2label["a"].lower().replace("-", ""),
        f"{ab2label['a'].lower()} (a)",
        f"a ({ab2label['a'].lower()})",
        f"a: {ab2label['a'].lower()}",
        f"a. {ab2label['a'].lower()}",
    ]:
        return ab2label["a"]
    if prediction_lower in [
        "b",
        ab2label["b"].lower(),
        ab2label["b"].lower().replace("-", ""),
        f"{ab2label['b'].lower()} (b)",
        f"b ({ab2label['b'].lower()})",
        f"b: {ab2label['b'].lower()}",
        f"b. {ab2label['b'].lower()}",
    ]:
        return ab2label["b"]

    if prediction not in ["Hate", "Non-hate"]:
        print(prediction)

    return prediction


def make_prompt(
    instance,
    ab2label,
    persona=False,
    country=None,
    simple=False,
    definition=False,
    prompt_num=None,
):
    if prompt_num == None:
        prompt_num = 0

    prompts = [
        "Answer if this post is hate or not with a single alphabet letter among given answer choices a and b.",
        "Given a POST from online, answer if it is 'Hate Speech' or 'Non-Hate Speech'. Answer with only one single alphabet between 'a' and 'b', with no other description.",
        "Would you consider this post as a hate speech? Give an answer in a single alphabet only.",
        "Do you think this is a hate speech? Answer with only one single alphabet between 'a' and 'b', with no other description.",
        "Consider yourself as a very typical person. Would you consider this post as a hate speech? Give an answer in a single alphabet only.",
    ]

    prompts_w_persona = [
        f"Answer if this post is hate or not in {country} with a single alphabet letter among given answer choices a and b.",
        f"Given a POST from online, answer if it is 'Hate Speech' or 'Non-Hate Speech' in {country}. Answer with only one single alphabet between 'a' and 'b', with no other description.",
        f"As a person raised up in {country}, would you consider this post as a hate speech? Give an answer in a single alphabet only.",
        f"You are from {country}. Do you think this is a hate speech? Answer with only one single alphabet between 'a' and 'b', with no other description.",
        f"Consider yourself as a very typical person in {country}. Would you consider this post as a hate speech? Give an answer in a single alphabet only.",
    ]

    p = ""

    if simple:
        p = f'Classify this text as {ab2label["a"]} or {ab2label["b"]}. {instance}. Answer:'
        return p

    if definition:
        p = f"Definition of Hate Speech:\n\nHate speech refers to offensive discourse targeting a group or an individual based on inherent characteristics such as race, religion, sexual orientation, gender, or any other factors that may threaten social peace.\n\n"

    if persona:
        p += prompts_w_persona[prompt_num]
    else:
        p += prompts[prompt_num]

    p += "\n\n"

    p += f"POST: {instance}\n"
    p += f'a: {ab2label["a"]}\n'
    p += f'b: {ab2label["b"]}\n'
    p += "answer:"

    return p


def check_gpt_input_list(history):
    check = True
    for i, u in enumerate(history):
        if not isinstance(u, dict):
            check = False
            break

        if not u.get("role") or not u.get("content"):
            check = False
            break

    return check
