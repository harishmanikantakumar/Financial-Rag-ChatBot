import os
import warnings

from dotenv import load_dotenv
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(override=True)

_LLM = None


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"
TEMPERATURE = 0


# ============================================================
# GET GROQ LLM
# ============================================================

def get_llm():
    """
    Return the cached Groq LLM.

    The model is initialized only once and reused for
    subsequent questions.
    """

    global _LLM

    if _LLM is not None:
        return _LLM

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found.\n"
            "Make sure your .env file contains:\n"
            "GROQ_API_KEY=your_groq_api_key"
        )

    _LLM = ChatGroq(
        groq_api_key=api_key,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE
    )

    return _LLM


# ============================================================
# BUILD FINANCIAL PROMPT
# ============================================================

def build_llm_prompt(context: str, query: str) -> str:
    """
    Build a strict financial QA prompt.

    The model is instructed to use ONLY the retrieved
    financial document context.
    """

    return f"""
You are a precise financial document assistant.

Answer the user's question using ONLY the information
contained in the provided context.

IMPORTANT RULES:

1. Never use outside knowledge.
2. Never rely on your pretrained knowledge when the answer
   is not explicitly supported by the context.
3. Never guess.
4. Never invent financial numbers.
5. Use the exact company requested.
6. Use the exact requested year.
7. Prefer values from financial tables.
8. If the context contains a value in millions, you may
   express it in billions when appropriate.
9. Do not convert percentages, ratios, EPS, or per-share values.
10. Do not mix values from different years.
11. Do not mix values from different companies.
12. If the exact requested value exists in the context,
    answer directly.
13. If the answer cannot be found in the context, respond:
    "The provided context does not contain enough information
    to answer this question."
14. Do not mention your knowledge cutoff.
15. Do not say that you lack real-time information.
16. Keep the final answer concise.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""


# ============================================================
# QUERY LLM
# ============================================================

def query_llm(context: str, query: str) -> str:
    """
    Send retrieved financial context + user question to Groq.

    This function is useful independently or from another
    RAG module.
    """

    if not context or not context.strip():
        return (
            "The provided context does not contain enough "
            "information to answer this question."
        )

    if not query or not query.strip():
        return "Please provide a financial question."

    try:

        llm = get_llm()

        prompt = build_llm_prompt(
            context=context,
            query=query
        )

        response = llm.invoke(prompt)

        if response is None:
            return (
                "The provided context could not be processed."
            )

        answer = response.content

        if isinstance(answer, list):

            answer = " ".join(
                str(item)
                for item in answer
            )

        answer = str(answer).strip()

        if not answer:
            return (
                "The provided context does not contain enough "
                "information to answer this question."
            )

        return answer

    except Exception as e:

        print("\n===== GROQ ERROR =====")
        print(str(e))

        return (
            "The provided context could not be processed."
        )


# ============================================================
# TEST LLM CONNECTION
# ============================================================

def test_llm():
    """
    Test whether the Groq connection is working.
    """

    try:

        llm = get_llm()

        response = llm.invoke(
            "Reply with exactly: Groq connection successful."
        )

        print("\n===== GROQ CONNECTION TEST =====")
        print(response.content)

        return True

    except Exception as e:

        print("\n===== GROQ CONNECTION FAILED =====")
        print(str(e))

        return False


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_llm()