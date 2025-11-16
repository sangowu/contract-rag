import re
from collections import Counter
from config.cuad_meta import (
    get_answer_type,
    ANSWER_TYPE_BOOL,
    ANSWER_TYPE_DATE,
    ANSWER_TYPE_DURATION,
    ANSWER_TYPE_LOCATION,
    ANSWER_TYPE_LIST_ENTITY,
    ANSWER_TYPE_TEXT,
)

def normalize_bool(text: str) -> str | None:
    if text is None:
        return None
    t = text.lower()
    if any(w in t for w in ["yes", "there is", "it does"]):
        return "Yes"
    if any(w in t for w in ["no", "there is no", "does not"]):
        return "No"
    return None

def normalize_text_tokens(s: str) -> list[str]:
    if s is None:
        return []
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.split()

def eval_bool(gold: str, pred: str) -> dict:
    p = normalize_bool(pred)
    return {"acc": float(p == gold) if p is not None else 0.0}

def eval_date(gold: str, pred: str) -> dict:
    g = (gold or "").strip()
    p = (pred or "").strip()
    return {"em": float(g == p)}

def eval_location(gold: str, pred: str) -> dict:
    g = (gold or "").strip().upper()
    p = (pred or "").strip().upper()
    return {"em": float(g == p)}

def eval_text(gold: str, pred: str) -> dict:
    g_tokens = normalize_text_tokens(gold)
    p_tokens = normalize_text_tokens(pred)
    if not g_tokens or not p_tokens:
        return {"f1": 0.0}
    common = Counter(g_tokens) & Counter(p_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {"f1": 0.0}
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1}

def eval_duration(gold: str, pred: str) -> dict:
    g = (gold or "").strip().lower()
    p = (pred or "").strip().lower()
    return {"em": float(g == p)}

def eval_list_entity(gold: str, pred: str) -> dict:
    return eval_text(gold, pred)

def eval_one(category: str, gold: str, pred: str) -> dict:
    """
    Evaluate the answer for the given category.
    """
    answer_type = get_answer_type(category)

    if answer_type == ANSWER_TYPE_BOOL:
        return eval_bool(gold, pred)
    if answer_type == ANSWER_TYPE_DATE:
        return eval_date(gold, pred)
    if answer_type == ANSWER_TYPE_LOCATION:
        return eval_location(gold, pred)
    if answer_type == ANSWER_TYPE_DURATION:
        return eval_duration(gold, pred)
    if answer_type == ANSWER_TYPE_LIST_ENTITY:
        return eval_list_entity(gold, pred)

    return eval_text(gold, pred)
