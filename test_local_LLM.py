import requests, json

url = "http://localhost:1234/v1/chat/completions"
payload = {
  "model": "<PUT_YOUR_MODEL_ID_HERE>",  # e.g., "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
  "messages": [
    {"role":"system","content":"Return only JSON."},
    {"role":"user","content":"{\"ping\":1}"}
  ],
  "temperature": 0
}
resp = requests.post(url, json=payload, timeout=60)
print(resp.status_code)
print(resp.json()["choices"][0]["message"]["content"][:200])
