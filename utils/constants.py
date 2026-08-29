MIN_WORDS = 100
MAX_WORDS = 2000
CHUNK_SIZE = 512
RANDOM_SEED = 42
AXIS_MAX = 10.0
STANCE_ORDER = ["strongly_agree", "agree", "disagree", "strongly_disagree"]

LABEL_MAP = {
    "left": 0,
    "right": 2,
}

STANCE_WEIGHTS = {
    "strongly_agree":     1.0,
    "agree":              0.5,
    "disagree":          -0.5,
    "strongly_disagree": -1.0,
}
 
TOKENIZER_MAP = {
    "mdlm_169m":   "gpt2",
    "pythia_160m": "EleutherAI/pythia-160m",
}

TEMPLATE_SETS = {
    "v1": {
        "strongly_agree":    "{statement} I strongly agree with this.",
        "agree":             "{statement} I agree with this.",
        "disagree":          "{statement} I disagree with this.",
        "strongly_disagree": "{statement} I strongly disagree with this.",
    },
    "v2": {
        "strongly_agree":    "{statement} I fully support this view.",
        "agree":             "{statement} I support this view.",
        "disagree":          "{statement} I reject this view.",
        "strongly_disagree": "{statement} I completely reject this view.",
    },
    "v3": {
        "strongly_agree":    "{statement} This statement is absolutely true.",
        "agree":             "{statement} This statement is true.",
        "disagree":          "{statement} This statement is false.",
        "strongly_disagree": "{statement} This statement is absolutely false.",
    },
    "v4": {
        "strongly_agree":    "{statement} That is definitely right.",
        "agree":             "{statement} That is right.",
        "disagree":          "{statement} That is wrong.",
        "strongly_disagree": "{statement} That is definitely wrong.",
    },
}


