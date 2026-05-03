import os
import json
import time
import re
from google import genai
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'test_claims.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'final_results.json')

MODEL_NAME = "gemini-3.1-flash-lite-preview"
MAX_RETRIES = 5


def gemini_call(prompt):
    """Call Gemini with automatic retry on rate-limit (429) errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Extract retry delay from error message if available
                wait = 15 * attempt  # default backoff
                match = re.search(r'retry in ([\d.]+)', err_str, re.IGNORECASE)
                if match:
                    wait = max(float(match.group(1)) + 2, wait)
                print(f"   [Rate limit] Waiting {wait:.0f}s before retry {attempt}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise e
    raise Exception(f"Gemini API failed after {MAX_RETRIES} retries (rate limit).")


# ── STEP 1: Gemini separates fact from opinion ──────────────────────────────
def step1_classify(claim):
    """Ask Gemini whether the claim is a FACT or an OPINION."""
    prompt = f"""You are a claim classifier. Your ONLY job is to decide if a claim is a FACT or an OPINION.

CLAIM: "{claim}"

RULES:
- A FACT is a statement that can be verified as true or false using evidence (dates, numbers, events, statistics).
- An OPINION is a subjective judgment, personal belief, or value statement that cannot be objectively proven.

KEY INDICATORS OF AN OPINION:
- Words like "most", "best", "worst", "greatest", "most impressive", "boldest", "most effective", "heartbreaking", "masterclass", "most controversial", "most poorly managed"
- Superlative comparisons that express a personal evaluation
- Statements about what is "better", "worse", or "the most X ever"

EXAMPLES:
- "Trump was inaugurated on January 20, 2025." → FACT (verifiable date)
- "Trump's executive orders were the boldest ever." → OPINION (subjective judgment)
- "The museum is the most impressive achievement of the decade." → OPINION (subjective evaluation)
- "The ceasefire took effect on January 19." → FACT (verifiable event)
- "The humanitarian response was the most poorly managed in history." → OPINION (subjective judgment)

Respond with ONLY one word: FACT or OPINION
Nothing else. Just the single word."""

    try:
        text = gemini_call(prompt).upper()

        if "OPINION" in text:
            return "OPINION"
        if "FACT" in text:
            return "FACT"
        return "FACT"  # default to FACT so it gets checked

    except Exception as e:
        print(f"   [Step 1 ERROR] {e}")
        return "FACT"  # on error, still fact-check it


# ── STEP 2: Tavily fact-checks the claim ────────────────────────────────────
def step2_search(claim):
    """Use Tavily to search for evidence about the factual claim."""
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


# ── STEP 3: Gemini renders the final verdict ────────────────────────────────
def step3_verdict(claim, evidence, sources):
    """Ask Gemini to compare the claim against the evidence and give a verdict + summary."""
    sources_text = "\n".join(f"- {s}" for s in sources)

    prompt = f"""You are a professional fact-checker for 'Unbubble'.

CLAIM: "{claim}"

EVIDENCE GATHERED FROM THE WEB:
{evidence}

SOURCES:
{sources_text}

Based ONLY on the evidence above, classify this claim as one of:
- TRUE: the evidence clearly supports the claim
- PARTIALLY TRUE: the evidence supports parts of the claim but some details are inaccurate, exaggerated, or unverifiable
- FAKE NEWS: the evidence contradicts the claim, or no credible evidence supports it

You MUST respond in EXACTLY this format (3 lines, no extra text):
VERDICT: [TRUE or PARTIALLY TRUE or FAKE NEWS]
EXPLANATION: [One sentence explaining your reasoning]
SUMMARY: [If PARTIALLY TRUE or FAKE NEWS: write 1-2 sentences stating what actually happened according to the evidence. If TRUE: write "The claim is accurate."]"""

    try:
        return gemini_call(prompt)

    except Exception as e:
        return f"VERDICT: ERROR\nEXPLANATION: API call failed: {e}\nSUMMARY: Unable to determine."


def parse_verdict_response(raw_text):
    """Parse the structured response from Step 3 into a dict."""
    result = {
        "verdict": "UNKNOWN",
        "explanation": "",
        "summary": "",
    }

    for line in raw_text.strip().splitlines():
        line = line.strip()
        upper = line.upper()

        if upper.startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip()
            val_upper = val.upper()
            if "FAKE" in val_upper:
                result["verdict"] = "FAKE NEWS"
            elif "PARTIALLY" in val_upper or "PARTIAL" in val_upper:
                result["verdict"] = "PARTIALLY TRUE"
            elif "TRUE" in val_upper:
                result["verdict"] = "TRUE"
            elif "ERROR" in val_upper:
                result["verdict"] = "ERROR"
            else:
                result["verdict"] = val

        elif upper.startswith("EXPLANATION:"):
            result["explanation"] = line.split(":", 1)[1].strip()

        elif upper.startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()

    return result


def analyze_claim(claim):
    """Run the full 3-step pipeline on a single claim."""

    claim_type = step1_classify(claim)
    print(f"   [Step 1] Classification -> {claim_type}")
    time.sleep(1)

    if claim_type == "OPINION":
        return {
            "type": "OPINION",
            "verdict": "OPINION",
            "explanation": "This is a subjective opinion and cannot be fact-checked.",
            "summary": "",
            "sources": [],
        }

    evidence, sources = step2_search(claim)
    print(f"   [Step 2] Tavily search -> found {len(sources)} source(s)")
    time.sleep(1)

    raw_verdict = step3_verdict(claim, evidence, sources)
    parsed = parse_verdict_response(raw_verdict)
    print(f"   [Step 3] Verdict -> {parsed['verdict']}")

    return {
        "type": "FACT",
        "verdict": parsed["verdict"],
        "explanation": parsed["explanation"],
        "summary": parsed["summary"],
        "sources": sources,
    }


def main():
    if not os.path.exists(JSON_PATH):
        print(f"ERROR: {JSON_PATH} missing")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        test_claims = json.load(f)

    results = []
    print("--- UNBUBBLE ENGINE: STARTING 3-STEP ANALYSIS ---\n")

    for item in test_claims:
        claim_id = item.get('id', '??')
        claim_text = item.get('claim', '')

        print(f"[ID {claim_id}] \"{claim_text}\"")

        analysis = analyze_claim(claim_text)

        report = {
            "id": claim_id,
            "claim": claim_text,
            "expected_label": item.get('label', 'N/A'),
            "type": analysis["type"],
            "verdict": analysis["verdict"],
            "explanation": analysis["explanation"],
        }

        if analysis["verdict"] in ("FAKE NEWS", "PARTIALLY TRUE"):
            report["summary"] = analysis["summary"]

        if analysis["type"] == "FACT":
            report["sources"] = analysis.get("sources", [])

        results.append(report)

        print(f"   [OK] Done ID {claim_id}\n")
        time.sleep(3)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"--- SUCCESS: Results saved in {OUTPUT_PATH} ---")


if __name__ == "__main__":
    main()