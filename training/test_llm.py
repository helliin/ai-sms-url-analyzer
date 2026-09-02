import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


from backend.ai.llm_analyzer import analyze_with_llm


message = """
Your account will be suspended today.
Click the link immediately to verify your account.
"""


result = analyze_with_llm(message)


print("\nLLM sonucu:")
print(result)