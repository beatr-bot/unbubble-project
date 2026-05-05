import os
import json
import time
import re
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Configuration ────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL_NAME = "gemini-3.1-flash-lite-preview"
MAX_RETRIES = 5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unbubble")


def gemini_call(prompt):
    RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return (response.text or "").strip()
        except Exception as e:
            err_str = str(e)
            if any(code in err_str for code in RETRYABLE):
                wait = 15 * attempt
                match = re.search(r'retry in ([\d.]+)', err_str, re.IGNORECASE)
                if match:
                    wait = max(float(match.group(1)) + 2, wait)
                time.sleep(wait)
            else:
                raise e
    raise Exception(f"Gemini API failed after {MAX_RETRIES} retries.")


def step1_classify(claim):
    prompt = f"""You are a claim classifier. Your ONLY job is to decide if a claim is a FACT or an OPINION.

CLAIM: "{claim}"

RULES:
- A FACT is a statement that can be verified as true or false using evidence (dates, numbers, events, statistics).
- An OPINION is a subjective judgment, personal belief, or value statement that cannot be objectively proven.

KEY INDICATORS OF AN OPINION:
- Words like "most", "best", "worst", "greatest", "most impressive", "boldest", "most effective", "heartbreaking", "masterclass", "most controversial", "most poorly managed"
- Superlative comparisons that express a personal evaluation

Respond with ONLY one word: FACT or OPINION"""
    try:
        text = gemini_call(prompt).upper()
        if "OPINION" in text:
            return "OPINION"
        return "FACT"
    except Exception as e:
        logger.error(f"Step 1 error: {e}")
        return "FACT"


def step2_search(claim):
    try:
        search = tavily_client.search(query=claim, max_results=3)
        results = search.get("results", [])
        if not results:
            return "No relevant search results found.", ["N/A"]
        evidence_parts = []
        sources = []
        for r in results:
            evidence_parts.append(r.get("content", ""))
            sources.append(r.get("url", "N/A"))
        evidence = "\n\n---\n\n".join(evidence_parts)
        return evidence, sources
    except Exception as e:
        return f"Search error: {e}", ["N/A"]


def step3_verdict(claim, evidence, sources):
    sources_text = "\n".join(f"- {s}" for s in sources)
    prompt = f"""You are a professional fact-checker for 'Unbubble'.

CLAIM: "{claim}"

EVIDENCE GATHERED FROM THE WEB:
{evidence}

SOURCES:
{sources_text}

Based ONLY on the evidence above, classify this claim as one of:
- TRUE: the evidence clearly supports the claim
- PARTIALLY TRUE: the evidence supports parts of the claim but some details are inaccurate
- FAKE NEWS: the evidence contradicts the claim, or no credible evidence supports it

You MUST respond in EXACTLY this format (3 lines, no extra text):
VERDICT: [TRUE or PARTIALLY TRUE or FAKE NEWS]
EXPLANATION: [One sentence explaining your reasoning]
SUMMARY: [If PARTIALLY TRUE or FAKE NEWS: write 1-2 sentences stating what actually happened. If TRUE: write "The claim is accurate."]"""
    try:
        return gemini_call(prompt)
    except Exception as e:
        return f"VERDICT: ERROR\nEXPLANATION: {e}\nSUMMARY: Unable to determine."


def parse_verdict_response(raw_text):
    result = {"verdict": "UNKNOWN", "explanation": "", "summary": ""}
    for line in raw_text.strip().splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            if "FAKE" in val:
                result["verdict"] = "FAKE NEWS"
            elif "PARTIALLY" in val:
                result["verdict"] = "PARTIALLY TRUE"
            elif "TRUE" in val:
                result["verdict"] = "TRUE"
            else:
                result["verdict"] = val
        elif upper.startswith("EXPLANATION:"):
            result["explanation"] = line.split(":", 1)[1].strip()
        elif upper.startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()
    return result


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    claim = data.get("claim", "").strip()
    if not claim:
        return jsonify({"error": "No claim provided"}), 400

    try:
        # Step 1: Classify
        claim_type = step1_classify(claim)
        time.sleep(1)

        if claim_type == "OPINION":
            return jsonify({
                "type": "OPINION",
                "verdict": "OPINION",
                "explanation": "This is a subjective opinion and cannot be fact-checked.",
                "summary": "This is a subjective opinion and cannot be fact-checked.",
                "sources": []
            })

        # Step 2: Search
        evidence, sources = step2_search(claim)
        time.sleep(1)

        # Step 3: Verdict
        raw_verdict = step3_verdict(claim, evidence, sources)
        parsed = parse_verdict_response(raw_verdict)

        return jsonify({
            "type": "FACT",
            "verdict": parsed["verdict"],
            "explanation": parsed["explanation"],
            "summary": parsed["summary"] or "The claim is accurate.",
            "sources": sources
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)