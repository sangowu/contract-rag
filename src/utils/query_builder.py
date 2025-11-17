import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.cuad_meta import (
    get_answer_type,
    ANSWER_TYPE_BOOL,
    ANSWER_TYPE_DATE,
    ANSWER_TYPE_DURATION,
    ANSWER_TYPE_LOCATION,
    ANSWER_TYPE_LIST_ENTITY,
    ANSWER_TYPE_TEXT,
)

def build_query(category: str) -> str:
    """
    Build the query for the given category.
    """
    t = get_answer_type(category)

    if t == ANSWER_TYPE_BOOL:
        return (
            f"Does this agreement include any '{category}' clause? "
            "Answer strictly with 'Yes' or 'No'."
        )

    if t == ANSWER_TYPE_DATE:
        return (
            f"What is the {category} of this agreement? "
            "Answer strictly in the format mm/dd/yyyy."
        )

    if t == ANSWER_TYPE_DURATION:
        return (
            f"What is the {category} in this agreement? "
            "Answer with a number and unit, such as '3 years' or '12 months'."
        )

    if t == ANSWER_TYPE_LOCATION:
        return (
            f"What is the {category} of this agreement? "
            "Answer with the name of a state or country only."
        )

    if t == ANSWER_TYPE_LIST_ENTITY:
        return (
            "Who are the parties to this agreement? "
            "List the entity or individual names separated by semicolons."
        )

    return f"What is the {category} in this agreement?"
