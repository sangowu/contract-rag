
ANSWER_TYPE_TEXT = "TEXT"           
ANSWER_TYPE_LIST_ENTITY = "LIST_ENTITY" 
ANSWER_TYPE_DATE = "DATE"            
ANSWER_TYPE_DURATION = "DURATION"    
ANSWER_TYPE_LOCATION = "LOCATION"    
ANSWER_TYPE_BOOL = "BOOL"           

CATEGORY_META = {
    # 1
    "Document Name": {
        "answer_type": ANSWER_TYPE_TEXT,
        "group": None,
    },
    # 2
    "Parties": {
        "answer_type": ANSWER_TYPE_LIST_ENTITY,
        "group": None,
    },
    # 3
    "Agreement Date": {
        "answer_type": ANSWER_TYPE_DATE,
        "group": 1,
    },
    # 4
    "Effective Date": {
        "answer_type": ANSWER_TYPE_DATE,
        "group": 1,
    },
    # 5
    "Expiration Date": {
        "answer_type": ANSWER_TYPE_DATE,
        "group": 1,
    },
    # 6
    "Renewal Term": {
        "answer_type": ANSWER_TYPE_DURATION,
        "group": 1,
    },
    # 7
    "Notice to Terminate Renewal": {
        "answer_type": ANSWER_TYPE_DURATION,
        "group": 1,
    },
    # 8
    "Governing Law": {
        "answer_type": ANSWER_TYPE_LOCATION,
        "group": None,
    },
    # 9
    "Most Favored Nation": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 10
    "Non-Compete": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 2,
    },
    # 11
    "Exclusivity": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 2,
    },
    # 12
    "No-Solicit of Customers": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 2,
    },
    # 13
    "Competitive Restriction Exception": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 2,
    },
    # 14
    "No-Solicit of Employees": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 15
    "Non-Disparagement": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 16
    "Termination for Convenience": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 17
    "Right of First Refusal, Offer or Negotiation (ROFR/ROFO/ROFN)": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 18
    "Change of Control": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 3,
    },
    # 19
    "Anti-Assignment": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 3,
    },
    # 20
    "Revenue/Profit Sharing": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 21
    "Price Restriction": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 22
    "Minimum Commitment": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 23
    "Volume Restriction": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 24
    "IP Ownership Assignment": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 25
    "Joint IP Ownership": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 26
    "License Grant": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 4,
    },
    # 27
    "Non-Transferable License": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 4,
    },
    # 28
    "Affiliate IP License-Licensor": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 4,
    },
    # 29
    "Affiliate IP License-Licensee": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 4,
    },
    # 30
    "Unlimited/All-You-Can-Eat License": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 31
    "Irrevocable or Perpetual License": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 4,
    },
    # 32
    "Source Code Escrow": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 33
    "Post-Termination Services": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 34
    "Audit Rights": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 35
    "Uncapped Liability": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 5,
    },
    # 36
    "Cap on Liability": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": 5,
    },
    # 37
    "Liquidated Damages": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 38
    "Warranty Duration": {
        "answer_type": ANSWER_TYPE_DURATION,
        "group": None,
    },
    # 39
    "Insurance": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 40
    "Covenant Not to Sue": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
    # 41
    "Third Party Beneficiary": {
        "answer_type": ANSWER_TYPE_BOOL,
        "group": None,
    },
}


def get_answer_type(category: str) -> str:
    meta = CATEGORY_META.get(category)
    if meta is None:
        raise KeyError(f"Unknown CUAD category: {category!r}")
    return meta["answer_type"]
