"""
Groq API Diagnostic Script
===========================
Runs step-by-step checks to find exactly why Groq isn't working.

Run from the project root:
    python scripts/diagnose_groq.py
"""

import os
import sys

# ── Force UTF-8 on Windows ──────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Load .env ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  GROQ DIAGNOSTIC TOOL")
print("=" * 60)

# Step 1: Load .env
print("\n[1/7] Loading .env file...")
try:
    from dotenv import load_dotenv
    loaded = load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
    print(f"  ✅ python-dotenv loaded .env: {loaded}")
except ImportError:
    print("  ⚠️  python-dotenv not installed — reading .env manually")
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        print("  ✅ Manually loaded .env")
    else:
        print("  ❌ .env file not found")

# Step 2: Check API key
print("\n[2/7] Checking GROQ_API_KEY...")
api_key = os.getenv("GROQ_API_KEY", "")
if not api_key:
    print("  ❌ GROQ_API_KEY is NOT set in environment or .env")
    sys.exit(1)

masked = api_key[:8] + "..." + api_key[-4:]
print(f"  ✅ Key found: {masked}")
print(f"  ℹ️  Length: {len(api_key)} chars")

if not api_key.startswith("gsk_"):
    print("  ⚠️  Key doesn't start with 'gsk_' — may be invalid format")
else:
    print("  ✅ Key format looks correct (starts with gsk_)")

# Step 3: Check groq package
print("\n[3/7] Checking groq package...")
try:
    import groq
    print(f"  ✅ groq package installed, version: {groq.__version__}")
except ImportError:
    print("  ❌ groq package NOT installed")
    print("     Fix: pip install groq")
    sys.exit(1)
except AttributeError:
    print("  ✅ groq package installed (version attribute not available)")

# Step 4: Create client
print("\n[4/7] Creating Groq client...")
try:
    client = groq.Groq(api_key=api_key)
    print("  ✅ Groq client created successfully")
except Exception as e:
    print(f"  ❌ Failed to create client: {type(e).__name__}: {e}")
    sys.exit(1)

# Step 5: List available models
print("\n[5/7] Fetching available models...")
try:
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    print(f"  ✅ {len(model_ids)} models available:")
    for mid in sorted(model_ids):
        marker = "  👉" if "llama" in mid.lower() else "    "
        print(f"  {marker} {mid}")

    target_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if target_model in model_ids:
        print(f"\n  ✅ Target model '{target_model}' is available")
    else:
        print(f"\n  ⚠️  Target model '{target_model}' NOT in available list")
        llama_models = [m for m in model_ids if "llama" in m.lower()]
        if llama_models:
            print(f"     Available Llama models: {llama_models}")
            print(f"     Suggestion: set GROQ_MODEL={llama_models[0]} in .env")

except Exception as e:
    print(f"  ❌ Failed to list models: {type(e).__name__}: {e}")
    print(f"     This usually means the API key is invalid or network issue")
#GROQ_API_KEY=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
# Step 6: Minimal chat completion
print("\n[6/7] Sending minimal test message...")
target_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
try:
    response = client.chat.completions.create(
        model=target_model,
        messages=[{"role": "user", "content": "Reply with just the word: OK"}],
        max_tokens=10,
        temperature=0,
        timeout=10.0,
    )
    reply = response.choices[0].message.content.strip()
    print(f"  ✅ Got response: '{reply}'")
    print(f"  ℹ️  Model used: {response.model}")
    print(f"  ℹ️  Tokens used: {response.usage.total_tokens}")
except groq.AuthenticationError as e:
    print(f"  ❌ Authentication failed — API key is INVALID or EXPIRED")
    print(f"     Error: {e}")
    print(f"     Fix: Get a new key from https://console.groq.com/keys")
except groq.RateLimitError as e:
    print(f"  ❌ Rate limit hit")
    print(f"     Error: {e}")
except groq.APIConnectionError as e:
    print(f"  ❌ Network/connection error — check internet connection")
    print(f"     Error: {e}")
except groq.APIStatusError as e:
    print(f"  ❌ API error: HTTP {e.status_code}")
    print(f"     Body: {e.body}")
except Exception as e:
    print(f"  ❌ Unexpected error: {type(e).__name__}: {e}")

# Step 7: Full vehicle recommendation call
print("\n[7/7] Testing full vehicle recommendation call...")
try:
    response = client.chat.completions.create(
        model=target_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a vehicle maintenance advisory AI. "
                    "Respond ONLY with a valid JSON object with exactly these keys: "
                    "recommendation_service_priority (one of: critical/high/medium/low/normal), "
                    "recommendation_suggested_action (string), "
                    "recommendation_safe_operating_limit_km (integer)."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Vehicle Diagnostics:\n"
                    "- Failure probability (next 7 days): 82%\n"
                    "- Engine RUL: 35.0%\n"
                    "- Brake RUL: 18.0%\n"
                    "- Primary fault: BRAKE_THERMAL_SATURATION\n"
                    "- Contributing factors: high_thermal_stress_index, low_brake_rul_pct\n\n"
                    "Produce the JSON recommendation."
                ),
            },
        ],
        temperature=0.3,
        max_tokens=300,
        timeout=10.0,
    )
    raw = response.choices[0].message.content
    print(f"  ✅ Raw Groq response:\n")
    print(f"  {raw}\n")

    import json
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    print(f"  ✅ Parsed successfully:")
    print(f"     Priority : {parsed.get('recommendation_service_priority')}")
    print(f"     Action   : {parsed.get('recommendation_suggested_action')}")
    print(f"     Limit    : {parsed.get('recommendation_safe_operating_limit_km')} km")

except groq.AuthenticationError:
    print("  ❌ Authentication failed — key is invalid")
except Exception as e:
    print(f"  ❌ {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("  DIAGNOSIS COMPLETE")
print("=" * 60 + "\n")
