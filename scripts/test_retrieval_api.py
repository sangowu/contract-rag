import requests

def main():
    url = "http://127.0.0.1:8000/api/retrieval/search"
    payload = {
        "query": "Does this contract have a non-compete clause?",
        "top_k_retrieval": 5,
        "file_name": None,
    }

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    print("ok:", data["ok"])
    print("num_docs:", len(data["data"]))
    if data["data"]:
        print("first doc:", data["data"][0])

if __name__ == "__main__":
    main()