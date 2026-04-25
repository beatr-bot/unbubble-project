import os
import json
from google import genai
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# Inizializzazione corretta per la nuova libreria google-genai
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

USE_REAL_TAVILY = True 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'test_claims.json')

def analyze_claim(claim):
    check_prompt = f"Is the following statement a factual claim that can be verified? Answer only 'YES' or 'NO'. Statement: '{claim}'"
    
    # Correzione: rimosso l'import genai vecchio e usato il client nuovo correttamente
    is_sourceable_res = client.models.generate_content(
        model_name = "gemini-2.0-flash", 
        contents=check_prompt
    )
    is_sourceable = is_sourceable_res.text.strip().upper()

    if "YES" not in is_sourceable:
        return "[SKIPPED]: Statement identified as opinion or non-verifiable.", None

    context = "No context found."
    url = "N/A"

    if USE_REAL_TAVILY:
        try:
            search_res = tavily_client.search(query=claim, max_results=1)
            if search_res['results']:
                context = search_res['results'][0]['content']
                url = search_res['results'][0]['url']
        except Exception as e:
            context = f"Search error: {e}"

    verdict_prompt = f"""
    CLAIM: '{claim}'
    SOURCE TEXT: '{context}'
    
    Compare the claim against the source text found on the internet. 
    Categorize the claim as:
    - CORRECT: Matches entirely.
    - PARTIAL MATCH: Main event true, but details (dates/names/numbers) are wrong.
    - CONTRADICTION: Fundamentally false.
    
    Provide a clear verdict and 1 sentence explanation.
    """
    
    verdict_res = client.models.generate_content(
        model="gemini-1.5-flash", 
        contents=verdict_prompt
    )
    return verdict_res.text, url

def main():
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            test_claims = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {JSON_PATH}")
        return

    print("--- UNBUBBLE PIPELINE START ---")

    for item in test_claims:
        print(f"\n[ID {item['id']}] TESTING CLAIM: {item['claim']}")
        
        try:
            verdict, source_url = analyze_claim(item['claim'])
            if source_url:
                print(f"PRIMARY SOURCE: {source_url}")
            print(f"AI ANALYSIS:\n{verdict}")
        except Exception as e:
            print(f"ERROR processing claim: {e}")
            
        print(f"EXPECTED LABEL: {item['label']}")
        print("-" * 60)

    print("\n--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    main()