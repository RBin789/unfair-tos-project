from google import genai

API_KEY = "AQ.Ab8RN6Kk-VBwQdCHemT5i9bGNFoibzK8dnrSik589vEOAbtuRw"
client = genai.Client(api_key=API_KEY)

print("Models that look batch-capable:\n")
for model in client.models.list():
    actions = [str(a) for a in (getattr(model, "supported_actions", None) or [])]
    action_text = " | ".join(actions).lower()
    if "batch" in action_text:
        print(f"{model.name} -> {actions}")