import os
import re

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever

from app.llm import get_llm
from app.vector_store import load_vector_store


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(override=True)


# ============================================================
# DEBUG / CHUNKING CONSTANTS
# ============================================================

# Toggle verbose chunk debugging with environment variable:
# DEBUG_CHUNKS=1 -> prints chunk verification and fallback-match lines
DEBUG_CHUNKS = os.getenv("DEBUG_CHUNKS", "0") == "1"

# Chunking parameters (adjust if your embeddings/chunking use other sizes)
_CHUNK_SIZE = 1600
_CHUNK_OVERLAP = 250

# Company aliases (MS-focused changes: removed ambiguous 'ms', added 'msft')
COMPANY_ALIASES = {
    "microsoft": [
        "microsoft",
        "msft",
        "microsoft corp",
        "microsoft corporation",
        "microsoft corporation (msft)",
    ],
    "tesla": [
        "tesla",
        "tsla",
        "tesla, inc",
        "tesla inc",
        "tesla motors",
        "tesla, inc.",
    ],
    "apple": [
        "apple"
    ],
}


# ============================================================
# COMPANY DETECTION
# ============================================================
def detect_company(question):
    """
    Detect the company mentioned in the question.

    Uses COMPANY_ALIASES with whole-word matching so:
        "MSFT" -> Microsoft
        "Microsoft" -> Microsoft
        "Tesla" -> Tesla
        "TSLA" -> Tesla
        "Apple" -> Apple
    """

    q = question.lower()

    # Build alias list, sort by alias length so longer aliases match first
    alias_matches = []

    for company, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            alias_matches.append((alias, company))

    alias_matches.sort(
        key=lambda x: len(x[0]),
        reverse=True
    )

    for alias, company in alias_matches:

        # Use whole-word regex (avoid substring matches like "ms" inside "systems")
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"

        if re.search(pattern, q, flags=re.IGNORECASE):
            return company.capitalize()

    return None


# ============================================================
# YEAR DETECTION
# ============================================================

def detect_year(question):
    """
    Detect the requested financial year.
    """

    years = re.findall(
        r"\b(20\d{2})\b",
        question
    )

    if years:
        return years[0]

    return None


# ============================================================
# QUESTION NORMALIZATION
# ============================================================

def normalize_question(question):
    """
    Normalize common financial terminology without destroying
    useful retrieval terms.
    """

    replacements = {
        "earnings per share": "diluted earnings per share",
        "eps": "earnings per share",
        "profit": "net income",
    }

    normalized = question

    for informal, formal in replacements.items():

        normalized = re.sub(
            rf"\b{re.escape(informal)}\b",
            formal,
            normalized,
            flags=re.IGNORECASE
        )

    return normalized


# ============================================================
# TEXT NORMALIZATION & CHUNKING HELPERS
# ============================================================

def normalize_text(s: str) -> str:
    """
    Normalize whitespace, dashes, and other common extraction artifacts.
    """
    if not s:
        return ""

    s = s.replace("\r", " ")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"[\t\u00A0]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def chunk_text(text: str, max_chars: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP):
    """
    Split text into overlapping chunks for chunk-level inspection.
    """
    text = normalize_text(text)
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


# ============================================================
# FINANCIAL METRIC DETECTION
# ============================================================

def detect_metric(question, company=None):
    """
    Detect the financial metric being requested.

    `company` is optional. When provided and it is NOT Microsoft
    or Tesla (e.g. Apple, or unrecognized), metric detection
    behaves EXACTLY as it did before this patch -- no new
    sub-type categories are returned. This keeps the Apple flow
    (and any other/unknown company) byte-for-byte unchanged.
    """

    q = question.lower()

    # Support many ways to ask for operating cash flows
    if ("operating cash flow" in q
        or "operating cash flows" in q
        or "cash flows from operating activities" in q
        or "net cash provided by" in q):
        return "operating_cash_flow"

    # Order matters.
    # More specific metrics should be checked first.

    # --------------------------------------------------------
    # NEW sub-type detection -- ONLY active for Microsoft/Tesla.
    # Apple (and any other/unknown company) falls straight through
    # to the original checks below, unchanged.
    # --------------------------------------------------------
    ms_tesla = company is not None and company.lower() in ("microsoft", "tesla")

    if ms_tesla:

        if "adjusted" in q or "non-gaap" in q or "non gaap" in q:
            if "net income" in q:
                return "adjusted_net_income"
            if "operating income" in q:
                return "adjusted_operating_income"
            if "gross margin" in q:
                return "adjusted_gross_margin"
            if "diluted earnings per share" in q or "eps" in q:
                return "adjusted_diluted_eps"

        if "product revenue" in q or "products revenue" in q:
            return "product_revenue"

        if ("service revenue" in q
                or "service and other revenue" in q
                or "services revenue" in q
                or "service & other revenue" in q):
            return "service_revenue"

        segment_names = [
            "intelligent cloud",
            "productivity and business processes",
            "more personal computing",
        ]
        if "revenue" in q and any(seg in q for seg in segment_names):
            return "segment_revenue"

    # --------------------------------------------------------
    # ORIGINAL checks (unchanged) -- these run for every company,
    # including Apple, exactly as before.
    # --------------------------------------------------------

    if "operating income" in q:
        return "operating_income"
    
    if "operating expenses" in q:
        return "operating_expenses"

    if "net income" in q:
        return "net_income"

    if "cost of revenue" in q:
        return "cost_of_revenue"

    if "cost of sales" in q:
        return "cost_of_sales"

    if "gross profit" in q:
        return "gross_profit"

    if "gross margin" in q:
        return "gross_margin"

    if "total assets" in q:
        return "total_assets"

    if "total liabilities" in q:
        return "total_liabilities"

    if "cash and cash equivalents" in q:
        return "cash"

    if "operating activities" in q:
        return "operating_cash_flow"

    if "diluted earnings per share" in q:
        return "diluted_eps"

    if "earnings per share" in q:
        return "eps"

    if "revenue" in q:
        return "revenue"

    if "net sales" in q:
        return "revenue"

    return None


# ============================================================
# GET ALL DOCUMENTS FROM FAISS
# ============================================================

def get_all_documents(vectorstore):
    """
    Extract all documents stored inside the FAISS docstore.
    """

    docs = []

    # --------------------------------------------------------
    # Primary method
    # --------------------------------------------------------

    try:

        docs = [
            doc
            for doc in vectorstore.docstore._dict.values()
            if hasattr(doc, "page_content")
        ]

    except Exception:

        docs = []

    # --------------------------------------------------------
    # Fallback method
    # --------------------------------------------------------

    if not docs:

        try:

            for doc_id in vectorstore.index_to_docstore_id.values():

                doc = vectorstore.docstore.search(
                    doc_id
                )

                if (
                    doc
                    and
                    hasattr(doc, "page_content")
                ):

                    docs.append(doc)

        except Exception as e:

            raise RuntimeError(
                f"Could not extract FAISS documents: {e}"
            )

    if not docs:

        raise ValueError(
            "No documents found in FAISS vector store."
        )

    return docs


# ============================================================
# FILTER COMPANY DOCUMENTS
# ============================================================

def filter_company_documents(docs, company):
    """
    Keep only documents belonging to the requested company.

    Uses metadata first and source filename second.
    Falls back to content-based alias matching (for MS / Tesla), but only
    accepts fallback if alias appears near the head of the document (first 400 chars)
    or if the alias is a safe ticker (msft/tsla) and appears anywhere.
    """

    if not company:
        return docs

    company_lower = company.lower()
    filtered = []

    aliases = COMPANY_ALIASES.get(company_lower, [company_lower])

    for doc in docs:

        metadata = doc.metadata or {}

        metadata_company = str(
            metadata.get("company", "")
        ).lower().strip()

        metadata_source = str(
            metadata.get("source", "")
        ).lower()

        metadata_file = str(
            metadata.get("file_name", "")
        ).lower()

        # metadata match (strict)
        if (
            metadata_company == company_lower
            or re.search(rf"\b{re.escape(company_lower)}\b", metadata_source)
            or re.search(rf"\b{re.escape(company_lower)}\b", metadata_file)
        ):
            filtered.append(doc)
            continue

        # content fallback: match aliases as whole words (avoid substrings)
        try:
            content = getattr(doc, "page_content", "") or ""
            content_lower = content.lower()
        except Exception:
            content_lower = ""

        for alias in aliases:
            alias_clean = alias.lower().strip()
            if len(alias_clean) < 2:
                continue

            pattern = rf"\b{re.escape(alias_clean)}\b"

            # Require alias in document head (first 400 chars) to count as fallback match.
            head = content_lower[:400]
            if re.search(pattern, head):
                filtered.append(doc)
                if DEBUG_CHUNKS:
                    print(
                        f"[FALLBACK MATCH-HEAD] company={company} alias={alias} -> doc preview: {head[:120]}"
                    )
                break

            # Allow safe tickers anywhere (e.g., msft, tsla)
            if alias_clean in ("msft", "tsla"):
                if re.search(pattern, content_lower):
                    filtered.append(doc)
                    if DEBUG_CHUNKS:
                        print(
                            f"[FALLBACK MATCH-ANY] company={company} alias={alias} -> doc preview: {content_lower[:120]}"
                        )
                    break

    return filtered


# ============================================================
# BM25 RETRIEVAL
# ============================================================

def retrieve_bm25(company_docs, query, k=20):
    """
    Retrieve documents using BM25 keyword matching.
    """

    if not company_docs:
        return []

    bm25 = BM25Retriever.from_documents(
        company_docs
    )

    bm25.k = k

    return bm25.invoke(query)


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def retrieve_faiss(vectorstore, query, k=25):
    """
    Retrieve documents using semantic FAISS search.
    """

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k
        }
    )

    return retriever.invoke(query)


# ============================================================
# COMBINE DOCUMENTS
# ============================================================

def combine_documents(*document_lists):
    """
    Combine BM25 and FAISS results while removing duplicates.
    """

    combined = []
    seen = set()

    for documents in document_lists:

        for doc in documents:

            if not hasattr(doc, "page_content"):
                continue

            content = doc.page_content.strip()

            if not content:
                continue

            doc_id = content

            if doc_id in seen:
                continue

            seen.add(doc_id)

            combined.append(doc)

    return combined


# ============================================================
# FINANCIAL TABLE DETECTION
# ============================================================

def is_financial_table(text):
    """
    Detect whether a chunk looks like a financial table.
    """

    text_lower = text.lower()

    signals = [
        "in millions",
        "consolidated statements of operations",
        "consolidated statements of income",
        "consolidated balance sheets",
        "consolidated statements of cash flows",
        "total net sales",
        "total revenue",
        "net income",
        "operating income",
        "gross margin",
        "total assets",
        "total liabilities",
    ]

    matches = sum(
        1
        for signal in signals
        if signal in text_lower
    )

    return matches >= 1


# ============================================================
# FINANCIAL RERANKING
# ============================================================

def rerank_financial_documents(
    docs,
    query,
    company=None,
    top_k=8
):
    """
    Rerank retrieved documents using financial-specific
    keyword and table signals.

    `company` is optional and defaults to None, which preserves
    the exact original behavior. It is only used to gate the
    NEW Microsoft/Tesla-specific keyword and scoring additions
    below -- Apple (or any other/unknown company) is scored
    exactly as before this patch.
    """

    requested_year = detect_year(query)
    metric = detect_metric(query, company)

    ms_tesla = company is not None and company.lower() in ("microsoft", "tesla")

    scored = []

    # --------------------------------------------------------
    # Metric keyword map
    # --------------------------------------------------------

    metric_keywords = {

        "revenue": [
            "total net sales",
            "net sales",
            "total revenue",
            "revenue"
        ],

        "net_income": [
            "net income",
            "net earnings",
            "net loss",
            "consolidated statements of income",
            "consolidated statements of operations"
        ],

        "operating_expenses": [
            "total operating expenses",
            "operating expenses",
            "research and development",
            "selling, general and administrative",
            "sales and marketing",
            "general and administrative"
        ],

        "cost_of_revenue": [
            "cost of revenue",
            "cost of sales"
        ],

        "cost_of_sales": [
            "cost of sales",
            "cost of revenue"
        ],

        "gross_profit": [
            "gross profit",
            "gross margin"
        ],

        "gross_margin": [
            "gross margin",
            "gross profit"
        ],

        "total_assets": [
            "total assets",
            "consolidated balance sheets"
        ],

        "total_liabilities": [
            "total liabilities",
            "total liabilities and shareholders",
            "consolidated balance sheets"
        ],

        "cash": [
            "cash and cash equivalents",
            "consolidated balance sheets"
        ],

        "operating_cash_flow": [
            "cash flows from operating activities",
            "operating activities",
            "consolidated statements of cash flows"
        ],

        "diluted_eps": [
            "diluted earnings per share",
            "diluted eps"
        ],

        "eps": [
            "earnings per share",
            "diluted earnings per share"
        ]
    }

    # NEW: Microsoft/Tesla-only metric keyword entries. Kept in a
    # separate dict and merged in only when ms_tesla is True, so
    # the Apple keyword lookup table is identical to before.
    MS_TESLA_METRIC_KEYWORDS = {
        "product_revenue": [
            "products",
            "service and other",
            "total revenue"
        ],
        "service_revenue": [
            "service and other",
            "services and other",
            "service revenue"
        ],
        "segment_revenue": [
            "segment revenue",
            "reportable segments"
        ],
        "adjusted_net_income": [
            "adjusted net income",
            "non-gaap"
        ],
        "adjusted_operating_income": [
            "adjusted operating income",
            "non-gaap"
        ],
        "adjusted_gross_margin": [
            "adjusted gross margin",
            "non-gaap"
        ],
        "adjusted_diluted_eps": [
            "adjusted diluted earnings per share",
            "non-gaap"
        ],
    }

    if ms_tesla:
        metric_keywords = dict(metric_keywords)
        metric_keywords.update(MS_TESLA_METRIC_KEYWORDS)

    keywords = metric_keywords.get(
        metric,
        []
    )

    # --------------------------------------------------------
    # Score documents
    # --------------------------------------------------------

    for doc in docs:

        text = doc.page_content.lower()

        score = 0

        # ====================================================
        # YEAR SIGNAL
        # ====================================================

        if requested_year:

            if requested_year in text:
                score += 30

        # ====================================================
        # METRIC SIGNAL
        # ====================================================

        for keyword in keywords:

            if keyword in text:

                # Exact metric/table phrase gets stronger score
                if keyword in [
                    "total net sales",
                    "total revenue",
                    "total operating expenses",
                    "net income",
                    "total assets",
                    "total liabilities",
                    "cash and cash equivalents",
                    "gross margin",
                    "gross profit"
                ]:

                    score += 20

                else:

                    score += 10


        # ====================================================
        # FINANCIAL TABLE SIGNAL
        # ====================================================

        if is_financial_table(text):

            score += 15

        # ====================================================
        # EXACT FINANCIAL VALUE / TABLE SIGNAL
        # ====================================================
        # Prefer actual financial statement values over
        # narrative mentions of the same metric.

        metric_value_patterns = {
            "revenue": [
                r"\btotal\s+net\s+sales\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
                r"\btotal\s+revenue\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
                r"\brevenue\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
                r"\bnet\s+sales\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "gross_margin": [
                r"\bgross\s+margin\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
                r"\bgross\s+margin\b\s+\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "gross_profit": [
                r"\bgross\s+profit\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "net_income": [
                r"\bnet\s+income\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "operating_income": [
                r"\boperating\s+income\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "operating_expenses": [
                r"\btotal\s+operating\s+expenses\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
                r"\boperating\s+expenses\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "cost_of_revenue": [
                r"\bcost\s+of\s+revenue\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "cost_of_sales": [
                r"\bcost\s+of\s+sales\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "total_assets": [
                r"\btotal\s+assets\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "total_liabilities": [
                r"\btotal\s+liabilities\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "cash": [
                r"\bcash\s+and\s+cash\s+equivalents\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "operating_cash_flow": [
                r"\bnet\s+cash\s+provided\s+by\s+operating\s+activities\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
                r"\bcash\s+provided\s+by\s+operating\s+activities\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "diluted_eps": [
                r"\bdiluted\s+earnings\s+per\s+share\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],

            "eps": [
                r"\bearnings\s+per\s+share\b\s*:?\s*\$?\s*[\d,]+(?:\.\d+)?",
            ],
        }

        exact_patterns = metric_value_patterns.get(metric, [])

        for pattern in exact_patterns:

            if re.search(pattern, text, re.IGNORECASE):

                score += 40
                break

        # ====================================================
        # NARRATIVE / CONTEXTUAL MENTION PENALTY
        # ====================================================

        narrative_patterns = {
            "gross_margin": [
                r"\brelative\s+gross\s+margin\b",
            ],

            "revenue": [
                r"\brevenue\s+growth\b",
                r"\brevenue\s+increased\b",
                r"\brevenue\s+decreased\b",
            ],

            "net_income": [
                r"\bnet\s+income\s+increased\b",
                r"\bnet\s+income\s+decreased\b",
            ],

            "operating_income": [
                r"\boperating\s+income\s+increased\b",
                r"\boperating\s+income\s+decreased\b",
            ],
        }

        for pattern in narrative_patterns.get(metric, []):

            if re.search(pattern, text, re.IGNORECASE):

                score -= 30
                break

       

        # ====================================================
        # TABLE STRUCTURE
        # ====================================================

        table_terms = [
            "in millions",
            "year ended",
            "fiscal year",
            "2023",
            "2022",
            "2021"
        ]

        for term in table_terms:

            if term in text:
                score += 3

        # ====================================================
        # INCOME STATEMENT
        # ====================================================

        if metric in [
            "revenue",
            "net_income",
            "operating_income",
            "gross_profit",
            "gross_margin",
            "cost_of_revenue",
            "cost_of_sales"
        ]:

            if (
                "consolidated statements of operations"
                in text
            ):

                score += 40

            if (
                "consolidated statements of income"
                in text
            ):

                score += 40

        # ====================================================
        # TESLA OPERATING INCOME BOOST
        # ====================================================
        # Tesla's operating income is reported in the
        # Consolidated Statements of Operations.
        #
        # This is intentionally Tesla-only so existing
        # Microsoft / Apple ranking behavior is untouched.

        if (
            company is not None
            and company.lower() == "tesla"
            and metric == "operating_income"
        ):

            if "consolidated statements of operations" in text:
                score += 70

            if re.search(
                r"\boperating\s+income\b",
                text,
                re.IGNORECASE
            ):
                score += 50        

        # ====================================================
        # REVENUE SPECIFIC BOOST
        # ====================================================

        if metric == "revenue":

            if "total net sales" in text:

                score += 50

            if "net sales:" in text:

                score += 25

        # ====================================================
        # NET INCOME SPECIFIC BOOST
        # ====================================================

        if metric == "net_income":

            if "net income" in text:

                score += 40

        # ====================================================
        # NEW: MICROSOFT/TESLA PRODUCT / SERVICE REVENUE BOOST
        # (skipped entirely when ms_tesla is False, so Apple
        # scoring is unaffected -- these metric values can only
        # ever be returned by detect_metric when ms_tesla is True
        # anyway, but the explicit guard is kept for clarity and
        # safety in case this function is ever called elsewhere.)
        # ====================================================

       
        # ====================================================
        # NEW: MICROSOFT/TESLA PRODUCT / SERVICE REVENUE BOOST
        # ====================================================

        if ms_tesla and metric == "product_revenue":

            if re.search(
                r"\bproducts\b.*?\bservice\s+and\s+other\b",
                text,
                re.IGNORECASE
            ):
                score += 100

            if (
                "service and other" in text
                and "total revenue" in text
            ):
                score += 40

            if (
                "server products and cloud services" in text
                or "office products and cloud services" in text
            ):
                score -= 80

        if ms_tesla and metric == "service_revenue":

            if "service and other" in text or "services and other" in text:

                score += 60

        if ms_tesla and metric == "segment_revenue":

            if "reportable segments" in text:

                score += 40

        # ====================================================
        # MICROSOFT/TESLA ADJUSTED / NON-GAAP BOOST
        # ====================================================

        if ms_tesla and metric in (
            "adjusted_net_income",
            "adjusted_operating_income",
            "adjusted_gross_margin",
            "adjusted_diluted_eps",
        ):

            # ------------------------------------------------
            # Generic non-GAAP signal
            # ------------------------------------------------

            if "non-gaap" in text:
                score += 30

            # ------------------------------------------------
            # Exact adjusted metric signal
            # ------------------------------------------------

            adjusted_metric_patterns = {

                "adjusted_net_income": [
                    r"adjusted\s+net\s+income\s*\(non[- ]gaap\)",
                    r"adjusted\s+net\s+income"
                ],

                "adjusted_operating_income": [
                    r"adjusted\s+operating\s+income\s*\(non[- ]gaap\)",
                    r"adjusted\s+operating\s+income"
                ],

                "adjusted_gross_margin": [
                    r"adjusted\s+gross\s+margin\s*\(non[- ]gaap\)",
                    r"adjusted\s+gross\s+margin"
                ],

                "adjusted_diluted_eps": [
                    r"adjusted\s+diluted\s+earnings\s+per\s+share\s*\(non[- ]gaap\)",
                    r"adjusted\s+diluted\s+earnings\s+per\s+share"
                ],
            }

            exact_adjusted_patterns = adjusted_metric_patterns.get(
                metric,
                []
            )

            for pattern in exact_adjusted_patterns:

                if re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                ):

                    # Strong boost for the ACTUAL requested metric
                    score += 100

                    break

            # ------------------------------------------------
            # Penalize unrelated adjusted metrics
            # ------------------------------------------------

            if metric == "adjusted_net_income":

                if (
                    "adjusted diluted earnings per share" in text
                    and "adjusted net income" not in text
                ):

                    score -= 50

        # ====================================================
        # OPERATING EXPENSE BOOST
        # ====================================================

        if metric == "operating_expenses":

            if "total operating expenses" in text:

                score += 60

            component_count = sum(
                1
                for component in [
                    "research and development",
                    "selling, general and administrative",
                    "sales and marketing",
                    "general and administrative"
                ]
                if component in text
            )

            if component_count >= 2:

                score += 20

        # ====================================================
        # BALANCE SHEET BOOST
        # ====================================================

        if metric in [
            "total_assets",
            "total_liabilities",
            "cash"
        ]:

            if "consolidated balance sheets" in text:

                score += 50

        # ====================================================
        # CASH FLOW BOOST
        # ====================================================

        if metric == "operating_cash_flow":

            if (
                "consolidated statements of cash flows"
                in text
            ):

                score += 60

        # ====================================================
        # EPS BOOST
        # ====================================================

        if metric in [
            "eps",
            "diluted_eps"
        ]:

            if "earnings per share" in text:

                score += 40

        # ====================================================
        # COMPANY IN-TOP-SNIPPET BOOST (MS-focused safety)
        # ====================================================
        # Small boost if the company name or ticker appears in the first ~400 chars
        top_snippet = text[:400]
        if re.search(r"\bmicrosoft\b", top_snippet):
            score += 12
        elif re.search(r"\bmsft\b", top_snippet):
            score += 10

        scored.append(
            (score, doc)
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # Debug output
    # --------------------------------------------------------

    print(
        "\n===== FINANCIAL RERANKING ====="
    )

    print(
        "Metric:",
        metric
    )

    print(
        "Year:",
        requested_year
    )

    for rank, (score, doc) in enumerate(
        scored[:top_k],
        1
    ):

        company_name = (
            doc.metadata.get("company")
            if doc.metadata
            else None
        )

        print(
            f"{rank}. Score={score} | "
            f"Company={company_name}"
        )

        preview = (
            doc.page_content[:300]
            .replace("\n", " ")
        )

        print(
            preview
        )

    return [
        doc
        for score, doc in scored[:top_k]
    ]


# ============================================================
# CHUNK-LEVEL VERIFICATION HELPER
# ============================================================

def verify_chunks_for_metric(docs, metric, requested_year, company, max_reports=6):
    """
    Print chunk-level debug info: show which chunks contain metric keywords
    or the requested year. Helpful to diagnose why a file wasn't selected.
    """
    if not docs:
        return

    METRIC_KEYWORDS = {
        "revenue": ["total net sales", "net sales", "total revenue", "revenue"],
        "net_income": ["net income", "net earnings", "net loss", "consolidated statements of income", "consolidated statements of operations"],
        "operating_expenses": ["total operating expenses", "operating expenses", "research and development", "selling, general and administrative", "sales and marketing"],
        "cost_of_revenue": ["cost of revenue", "cost of sales"],
        "gross_profit": ["gross profit", "gross margin"],
        "total_assets": ["total assets", "consolidated balance sheets"],
        "total_liabilities": ["total liabilities", "consolidated balance sheets"],
        "cash": ["cash and cash equivalents", "consolidated balance sheets"],
        "operating_cash_flow": ["cash flows from operating activities", "operating activities", "consolidated statements of cash flows"],
        "diluted_eps": ["diluted earnings per share", "diluted eps"],
        "eps": ["earnings per share", "diluted earnings per share"],
    }

    keywords = METRIC_KEYWORDS.get(metric, [])
    if not keywords and metric:
        keywords = [metric]

    if DEBUG_CHUNKS:
        print("\n===== CHUNK VERIFICATION =====")
        print(f"Company={company} Metric={metric} Year={requested_year}")

    reports = 0

    for doc in docs:
        if reports >= max_reports:
            break

        content = getattr(doc, "page_content", "") or ""
        chunks = chunk_text(content)

        doc_reported = False
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            found_kw = [kw for kw in keywords if kw in chunk_lower]
            found_year = requested_year and (str(requested_year) in chunk_lower)

            if found_kw or found_year:
                if DEBUG_CHUNKS:
                    preview = chunk[:300].replace("\n", " ")
                    print(f"\n[DOC] preview: {preview[:180]}...")
                    print(f" chunk_index={i} keywords={found_kw} year_in_chunk={found_year}")
                doc_reported = True

        if doc_reported:
            reports += 1

    if DEBUG_CHUNKS:
        print("===== END CHUNK VERIFICATION =====\n")


# ============================================================
# ADD NEIGHBORING CHUNKS
# ============================================================

def add_neighboring_chunks(all_docs, selected_docs, neighbor_count=1):
    """
    Add nearby chunks belonging to the same document/page sequence.

    This helps when a financial table is split across multiple
    chunks and the requested value appears in the next chunk.

    Existing selected documents are preserved.
    """

    if not all_docs or not selected_docs:
        return selected_docs

    # Map object identity to its position in the FAISS document list
    doc_positions = {
        id(doc): i
        for i, doc in enumerate(all_docs)
    }

    expanded = []
    seen = set()

    for doc in selected_docs:

        doc_id = id(doc)

        if doc_id not in seen:
            expanded.append(doc)
            seen.add(doc_id)

        position = doc_positions.get(doc_id)

        if position is None:
            continue

        for offset in range(
            1,
            neighbor_count + 1
        ):

            # Previous chunk
            prev_pos = position - offset

            if prev_pos >= 0:

                prev_doc = all_docs[prev_pos]

                if id(prev_doc) not in seen:

                    expanded.append(prev_doc)
                    seen.add(id(prev_doc))

            # Next chunk
            next_pos = position + offset

            if next_pos < len(all_docs):

                next_doc = all_docs[next_pos]

                if id(next_doc) not in seen:

                    expanded.append(next_doc)
                    seen.add(id(next_doc))

    return expanded




# ============================================================
# RETRIEVE FINANCIAL CONTEXT
# ============================================================

def retrieve_financial_context(
    vectorstore,
    query,
    company,
    top_k=8
):
    """
    Hybrid retrieval:

    1. Get all documents.
    2. Filter by company.
    3. BM25 retrieval.
    4. FAISS retrieval.
    5. Combine.
    6. Financial reranking.
    """

    # --------------------------------------------------------
    # Get all documents
    # --------------------------------------------------------

    all_docs = get_all_documents(
        vectorstore
    )

    print(
        f"\nTotal FAISS documents: {len(all_docs)}"
    )

    # --------------------------------------------------------
    # Company filtering
    # --------------------------------------------------------

    company_docs = filter_company_documents(
        all_docs,
        company
    )

    print(
        f"Detected company: {company}"
    )

    print(
        f"Company documents: {len(company_docs)}"
    )

    if not company_docs:

        print(
            "WARNING: No company-specific documents found."
        )

        company_docs = all_docs

    # --------------------------------------------------------
    # Chunk-level verification for debugging/fallback
    # --------------------------------------------------------
    requested_year = detect_year(query)
    metric = detect_metric(query, company)

    # Only run the chunk debug for Microsoft / Tesla when company was requested
    # (keeps Apple flow unchanged)
    if company and company.lower() in ("microsoft", "tesla"):
        verify_chunks_for_metric(
            company_docs,
            metric,
            requested_year,
            company
        )


# --------------------------------------------------------
# BM25
# --------------------------------------------------------

    retrieval_query = query

    if (
        company
        and company.lower() in ("microsoft", "tesla")
        and metric == "cost_of_revenue"
    ):
        retrieval_query = (
            f"{query} "
            "revenue gross margin "
            "consolidated statements of operations "
            "consolidated statements of income"
        )

    elif (
        company
        and company.lower() == "microsoft"
        and metric == "product_revenue"
    ):
        retrieval_query = (
            f"{query} "
            "Products "
            "Service and other "
            "Total revenue"
        )

    elif (
        company
        and company.lower() == "microsoft"
        and metric == "service_revenue"
    ):
        retrieval_query = (
            f"{query} "
            "Products "
            "Service and other "
            "Total revenue"
        )

    elif (
        company
        and company.lower() in ("microsoft", "tesla")
        and metric == "adjusted_net_income"
    ):
        retrieval_query = (
            f"{query} "
            "adjusted net income "
            "adjusted net income non-GAAP "
            "non-GAAP financial measures "
            "reconciliation "
            "GAAP non-GAAP"
        )

    
    bm25_docs = retrieve_bm25(
        company_docs,
        retrieval_query,
        k=30
    )

    print(
        f"BM25 documents: {len(bm25_docs)}"
    )

# --------------------------------------------------------
# FAISS
# --------------------------------------------------------

    faiss_query = query

    if (
        company
        and company.lower() in ("microsoft", "tesla")
        and metric == "cost_of_revenue"
    ):
        faiss_query = (
            f"{query} "
            "revenue gross margin "
            "consolidated statements of operations "
            "consolidated statements of income"
        )

    elif (
        company
        and company.lower() == "microsoft"
        and metric == "product_revenue"
    ):
        faiss_query = (
            f"{query} "
            "Products "
            "Service and other "
            "Total revenue"
        )

    elif (
        company
        and company.lower() == "microsoft"
        and metric == "service_revenue"
    ):
        faiss_query = (
            f"{query} "
            "Products "
            "Service and other "
            "Total revenue"
        )

    elif (
        company
        and company.lower() in ("microsoft", "tesla")
        and metric == "adjusted_net_income"
    ):
        faiss_query = (
            f"{query} "
            "adjusted net income "
            "adjusted net income non-GAAP "
            "non-GAAP financial measures "
            "reconciliation "
            "GAAP non-GAAP"
        )
    
    faiss_docs = retrieve_faiss(
        vectorstore,
        faiss_query,
        k=25
    )



# Filter FAISS results by company

    if company:
        faiss_docs = filter_company_documents(
            faiss_docs,
            company
        )

    print(
        f"FAISS documents: {len(faiss_docs)}"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = combine_documents(
        bm25_docs,
        faiss_docs
    )

    print(
        f"Combined documents: {len(combined)}"
    )

    # --------------------------------------------------------
    # Rerank
    # --------------------------------------------------------

    final_docs = rerank_financial_documents(
        combined,
        query,
        company=company,
        top_k=top_k
    )

    print(
        f"Final context documents: {len(final_docs)}"
    )

    # --------------------------------------------------------
    # Add neighboring chunks
    # --------------------------------------------------------
    # A financial statement can be split across chunks.
    #
    # Example:
    #
    # Chunk 1 -> Revenue ... Gross profit
    # Chunk 2 -> Operating expenses ... Operating income
    #
    # Include adjacent chunks so the answer is not lost merely
    # because the table crossed a chunk boundary.

    expanded_docs = add_neighboring_chunks(
        all_docs,
        final_docs,
        neighbor_count=1
    )

    print(
        f"Expanded context documents: {len(expanded_docs)}"
    )

    return expanded_docs
    


# ============================================================
# BUILD PROMPT
# ============================================================

def build_revenue_rule(metric, company=None):
    """
    Returns the rule-14 instruction text for the prompt.

    For Apple (or any company that is not Microsoft/Tesla), this
    ALWAYS returns the original static rule, unchanged, regardless
    of metric -- so Apple's prompt is identical to before this
    patch. The metric-aware variants only apply to Microsoft/Tesla,
    since only those companies can ever produce the new sub-type
    metrics from detect_metric() in the first place.
    """
    ms_tesla = company is not None and company.lower() in ("microsoft", "tesla")

    original_rule = (
        "14. For revenue questions, prefer \"Total net sales\" or "
        "\"Total revenue\" when available."
    )

    if not ms_tesla:
        return original_rule

    if (
    metric == "product_revenue"
    and company is not None
    and company.lower() == "microsoft"
):
        return (
        "14. This question asks for PRODUCT revenue specifically. "
        "Use ONLY the 'Products' line from the income statement's "
        "Products / Service and other revenue split. Do NOT use "
        "Total revenue, and do NOT use the 'Revenue by product and "
        "service offering' table (Server products, Office, Windows, "
        "etc.) -- that table's Total is NOT product revenue."
    )

    if (
    metric == "product_revenue"
    and company is not None
    and company.lower() == "tesla"
):
        return (
            "14. This question asks for Tesla product revenue. "
            "Use the 'Automotive sales' line as Tesla's product "
            "revenue for this question. "
            "Do NOT use Total automotive revenues or Total revenues."
    )
    
    if metric == "service_revenue":
        return (
            "14. This question asks for SERVICE revenue specifically. "
            "Use ONLY the 'Service and other' line from the income "
            "statement's Products / Service and other revenue split. "
            "Do NOT use Total revenue."
        )
    if metric == "segment_revenue":
        return (
            "14. This question asks for a specific SEGMENT's revenue. "
            "Use only that segment's row, not the Total row."
        )
    if metric and metric.startswith("adjusted_"):
        return (
            "14. This question asks for a non-GAAP / adjusted figure. "
            "Use ONLY the row explicitly labeled '(non-GAAP)' or "
            "'Adjusted'. Do NOT substitute the GAAP figure."
        )

    return original_rule


def build_prompt(
    context,
    question,
    metric=None,
    company=None
):
    """
    Build the final financial QA prompt.

    `metric` and `company` are optional and default to None, which
    reproduces the exact original static rule 14 -- Apple's prompt
    text is unchanged by this patch.
    """

    revenue_rule = build_revenue_rule(metric, company)

    return f"""
You are an expert financial analyst answering questions
about company financial reports.

Use ONLY the provided context.

STRICT RULES:

1. Never use outside knowledge.
2. Never guess.
3. Never invent numbers.
4. Use only information explicitly supported by the context.
5. Use the exact company requested.
6. Use the exact requested year.
7. Prefer financial statement tables over narrative text.
8. If the financial statement says "In millions", preserve
   the original value or convert it correctly to billions.
9. Do NOT convert EPS, percentages, ratios, or per-share values.
10. For gross margin questions, use the exact "Gross margin"
    line from the financial statement. Do not substitute gross
    profit, revenue, or gross margin percentage. If the "Gross
    margin" line is reported as a dollar amount, return that
    dollar amount. If it is reported as a percentage, return
    the percentage.
11. Never mix values from different years.
12. Never mix values from different companies.
13. If multiple years are shown, select the requested year.
{revenue_rule}
15. For operating expense questions, prefer
    "Total operating expenses".
16. If the exact requested value is not present, say:
    "The provided context does not contain enough information
    to answer this question."
17. Do not mention your knowledge cutoff.
18. Do not say that you lack real-time information.
19. Give a concise answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# ============================================================
# NORMALIZE NUMBER
# ============================================================

def normalize_number(value):
    """
    Convert a numeric string into a float.

    Examples:

    383,285 -> 383285
    383.285 -> 383.285
    $54,847 -> 54847
    """

    if value is None:
        return None

    value = str(value)

    value = (
        value
        .replace("$", "")
        .replace(",", "")
        .strip()
    )

    try:

        return float(value)

    except ValueError:

        return None


# ============================================================
# EXTRACT HELPERS (DETERMINISTIC EXTRACTION FOR CANONICAL LINES)
# ============================================================

def find_number_near_terms(context, term_patterns, window=400):
    """
    Search the context for any of term_patterns (regex strings),
    return the first nearby numeric token and its float value plus snippet.

    FIXED (general correctness fix, applies to all companies): the
    previous version searched a window BEFORE and AFTER the matched
    term, then grabbed the first number in that whole span. For
    back-to-back table rows like:

        Revenue        $ 211,915
        Gross margin     146,052

    a "before" window on the Gross margin match re-included the
    Revenue row above it, and re.search() returns the FIRST number
    in the span -- i.e. Revenue's number, not Gross margin's. This
    is a bug in the shared helper itself (used by the pre-existing
    extract_operating_cash_flow / extract_total_assets for every
    company, including Apple), not new Microsoft/Tesla behavior, so
    it is fixed universally rather than gated by company.
    """
    if not context:
        return None, None, None

    ctx_lower = context.lower()
    forward_window = min(window, 60)
    for pat in term_patterns:
        m = re.search(pat, ctx_lower)
        if m:
            start = m.end()
            end = min(len(context), m.end() + forward_window)
            snippet = context[max(0, m.start() - 20):end]
            num_match = re.search(r"\$?\s*[\d,]+(?:\.\d+)?", context[start:end])
            if num_match:
                raw = num_match.group(0)
                val = normalize_number(raw)
                return raw.strip(), val, snippet
    return None, None, None


def extract_operating_cash_flow(context):
    patterns = [
        r"net cash provided by \(used in\) operating activities",
        r"net cash provided by operating activities",
        r"cash flows from operating activities",
        r"cash provided by operating activities",
    ]
    return find_number_near_terms(context, patterns, window=600)


def extract_total_assets(context, company=None):
    """
    Extract total assets.

    Tesla gets special handling because its context can contain
    VIE/subsidiary balance sheets before the main Tesla balance sheet.

    Microsoft / Apple keep the existing extraction logic.
    """

    # ========================================================
    # TESLA-SPECIFIC FIX
    # ========================================================

    if company and company.lower() == "tesla":

        ctx_lower = (context or "").lower()

        # Find the MAIN Tesla consolidated balance sheet
        m = re.search(
            r"tesla,\s*inc\.\s*consolidated balance sheets",
            ctx_lower,
            flags=re.IGNORECASE
        )

        if m:

            # Look inside the main Tesla balance-sheet section
            block = context[
                m.start():m.start() + 5000
            ]

            # Find Total assets inside that section
            nm = re.search(
                r"total\s+assets\s+\$?\s*"
                r"([\d,]+(?:\.\d+)?)",
                block,
                flags=re.IGNORECASE
            )

            if nm:

                raw = nm.group(1)

                val = normalize_number(raw)

                return (
                    raw,
                    val,
                    block
                )

    # ========================================================
    # ORIGINAL LOGIC
    # ========================================================
    # DO NOT CHANGE THIS PART.
    # This keeps Microsoft / Apple behavior the same.

    raw, val, snippet = find_number_near_terms(
        context,
        [r"total assets"],
        window=600
    )

    if raw:
        return raw, val, snippet

    # Fallback: consolidated balance sheets
    m = re.search(
        r"consolidated balance sheets",
        (context or "").lower()
    )

    if m:

        block = context[
            m.end():m.end() + 2000
        ]

        nm = re.search(
            r"total assets\s*\$?\s*"
            r"[\d,]+(?:\.\d+)?",
            block,
            flags=re.IGNORECASE
        )

        if nm:

            rawnum = re.search(
                r"\$?\s*[\d,]+(?:\.\d+)?",
                nm.group(0)
            ).group(0)

            val = normalize_number(
                rawnum
            )

            return (
                rawnum.strip(),
                val,
                block[:400]
            )

    return None, None, None


# ============================================================
# NEW: MICROSOFT/TESLA-ONLY DETERMINISTIC EXTRACTORS
# ============================================================
# These functions are only ever called from ask_question() when
# the detected company is Microsoft or Tesla (see the gating
# check at each call site below). They are never reached for
# Apple or any other company.

def extract_cost_of_revenue(context):
    """
    Extract TOTAL cost of revenue.

    Microsoft example:

        Cost of revenue:
            Product                 17,804
            Service and other       48,059
            Total cost of revenue   65,863

    We must return the TOTAL value: 65,863.
    """

    if not context:
        return None, None, None

    # --------------------------------------------------------
    # NORMALIZE CONTEXT
    # --------------------------------------------------------

    text = context.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    # --------------------------------------------------------
    # 1. FIRST: LOOK FOR TOTAL COST OF REVENUE
    # --------------------------------------------------------

    patterns = [
        r"\btotal\s+cost\s+of\s+revenue\b\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)",
        r"\btotal\s+cost\s+of\s+revenue\b.*?\$?\s*([\d,]+(?:\.\d+)?)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = normalize_number(match.group(1))

            if value is not None:

                snippet_start = max(0, match.start() - 150)
                snippet_end = min(len(text), match.end() + 200)

                snippet = text[snippet_start:snippet_end]

                print("\n===== COST OF REVENUE EXTRACTION =====")
                print("Matched: TOTAL cost of revenue")
                print(f"Extracted value: {value}")
                print(f"Snippet: {snippet}")

                return value, "direct_total", snippet

    # --------------------------------------------------------
    # 2. FALLBACK: REVENUE - GROSS MARGIN
    # --------------------------------------------------------

    _, revenue_val, rev_snip = find_number_near_terms(
        context,
        [r"\btotal\s+revenue\b"],
        window=60
    )

    _, margin_val, margin_snip = find_number_near_terms(
        context,
        [r"\bgross\s+margin\b"],
        window=60
    )

    if revenue_val is not None and margin_val is not None:

        derived = revenue_val - margin_val

        snippet = (
            f"Derived from Total Revenue={revenue_val} "
            f"and Gross Margin={margin_val}"
        )

        print("\n===== COST OF REVENUE EXTRACTION =====")
        print("Method: Revenue - Gross Margin")
        print(f"Revenue: {revenue_val}")
        print(f"Gross Margin: {margin_val}")
        print(f"Derived Cost of Revenue: {derived}")

        return derived, "derived", snippet

    # --------------------------------------------------------
    # 3. NOTHING FOUND
    # --------------------------------------------------------

    print("\n===== COST OF REVENUE EXTRACTION =====")
    print("No cost of revenue value found.")

    return None, None, None

def extract_product_service_revenue(context, which):
    """
    Extract Microsoft's actual Products / Service and other
    revenue split.

    IMPORTANT:
    Microsoft contains multiple tables that use the word
    "Products". We must NOT accidentally extract values from:

        - Server products and cloud services
        - Office products and cloud services
        - Office Commercial products
        - Consumer products
        - Revenue classified by significant product/service offerings
        - Windows
        - Gaming
        - LinkedIn
        - Other individual product/service offerings

    The requested table is specifically the Microsoft revenue
    split containing:

        Products
        Service and other
        Total revenue

    Example:

        Products            64,728
        Service and other   147,187
        Total revenue       211,915

    Therefore, we search for the complete table relationship
    instead of searching globally for the word "Products".
    """

    if not context:
        return None, None

    text = context.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    # ========================================================
    # FIND THE ACTUAL PRODUCTS / SERVICE AND OTHER TABLE
    # ========================================================

    table_pattern = re.compile(
        r"\bProducts\b"
        r"\s+\$?\s*([\d,]+(?:\.\d+)?)"
        r".{0,500}?"
        r"\bService\s+and\s+other\b"
        r"\s+\$?\s*([\d,]+(?:\.\d+)?)"
        r".{0,500}?"
        r"\bTotal\s+revenue\b"
        r"\s+\$?\s*([\d,]+(?:\.\d+)?)",
        re.IGNORECASE
    )

    match = table_pattern.search(text)

    if not match:

        print(
            "\n===== PRODUCT/SERVICE EXTRACTION ====="
        )

        print(
            "Actual Products / Service and other table "
            "was not found in context."
        )

        return None, None

    # ========================================================
    # EXTRACT VALUES
    # ========================================================

    product_value = normalize_number(
        match.group(1)
    )

    service_value = normalize_number(
        match.group(2)
    )

    total_value = normalize_number(
        match.group(3)
    )

    # ========================================================
    # CREATE DEBUG SNIPPET
    # ========================================================

    snippet_start = max(
        0,
        match.start() - 200
    )

    snippet_end = min(
        len(text),
        match.end() + 200
    )

    snippet = text[
        snippet_start:snippet_end
    ]

    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    print(
        "\n===== PRODUCT/SERVICE EXTRACTION ====="
    )

    print(
        f"Products: {product_value}"
    )

    print(
        f"Service and other: {service_value}"
    )

    print(
        f"Total revenue: {total_value}"
    )

    print(
        f"Snippet: {snippet}"
    )

    # ========================================================
    # RETURN REQUESTED VALUE
    # ========================================================

    if which == "product":

        return product_value, snippet

    if which == "service":

        return service_value, snippet

    return None, None

def extract_adjusted_net_income(context):
    """
    Targets the specific non-GAAP reconciliation table line
    "Adjusted net income (non-GAAP) $ X" rather than generic
    "net income" text.
    """
    raw, val, snippet = find_number_near_terms(
        context,
        [r"adjusted net income\s*\(non[- ]gaap\)",
         r"adjusted net income",
        ],
        window=300,
    )
    return val, snippet


# ============================================================
# EXTRACT ANSWER NUMBERS
# ============================================================

def extract_numbers(text):
    """
    Extract numeric values from an answer.
    """

    if not text:
        return []

    pattern = r"\$?\s*[\d,]+(?:\.\d+)?"

    matches = re.findall(
        pattern,
        text
    )

    numbers = []

    for match in matches:

        value = normalize_number(
            match
        )

        if value is not None:

            if value >= 10:

                numbers.append(value)

    return numbers


# ============================================================
# VALIDATE ANSWER
# ============================================================

def validate_answer(
    answer,
    context,
    metric
):
    """
    Validate whether numerical values in the answer are
    supported by the retrieved context.

    Supports:
        383,285 million
        383.285 billion
    """

    answer_numbers = extract_numbers(
        answer
    )

    context_numbers = extract_numbers(
        context
    )

    # No numerical answer
    if not answer_numbers:

        return True

    # --------------------------------------------------------
    # Exact numeric match
    # --------------------------------------------------------

    for answer_value in answer_numbers:

        for context_value in context_numbers:

            if abs(
                answer_value - context_value
            ) < 0.001:

                return True

    # --------------------------------------------------------
    # Million -> billion validation
    # --------------------------------------------------------

    for answer_value in answer_numbers:

        for context_value in context_numbers:

            # Context is millions, answer is billions
            if abs(
                answer_value * 1000
                - context_value
            ) < 0.01:

                return True

            # Context is billions, answer is millions
            if abs(
                answer_value
                - context_value * 1000
            ) < 0.01:

                return True

    return False


# ============================================================
# MILLIONS -> BILLIONS
# ============================================================

def convert_millions_to_billions(text):
    """
    Convert phrases such as:

        $54,847 million

    into:

        $54.847 billion

    Only values explicitly followed by "million" are converted.
    """

    if not text:

        return text

    pattern = (
        r"(\$?\s*[\d,]+(?:\.\d+)?)\s*million\b"
    )

    def convert(match):

        original = match.group(1)

        value = (
            original
            .replace("$", "")
            .replace(",", "")
            .strip()
        )

        try:

            billions = float(value) / 1000

            result = f"{billions:.3f}"

            result = (
                result
                .rstrip("0")
                .rstrip(".")
            )

            dollar = (
                "$"
                if "$" in original
                else ""
            )

            return (
                f"{dollar}{result} billion"
            )

        except ValueError:

            return match.group(0)

    return re.sub(
        pattern,
        convert,
        text,
        flags=re.IGNORECASE
    )


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(
    vectorstore,
    query
):
    """
    Main financial RAG pipeline.

    Flow:

        Question
          ↓
        Normalize
          ↓
        Detect company
          ↓
        BM25 + FAISS
          ↓
        Financial reranking
          ↓
        Context
          ↓
        Deterministic extraction (common metrics)
          ↓
        Groq
          ↓
        Number conversion
          ↓
        Validation
          ↓
        Final answer
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "QUESTION:",
        query
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query or not query.strip():

        return "Please provide a financial question."

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized_query = normalize_question(
        query
    )

    print(
        "Normalized question:",
        normalized_query
    )

    # --------------------------------------------------------
    # Detect company
    # --------------------------------------------------------

    company = detect_company(
        normalized_query
    )

    print(
        "Detected company:",
        company
    )

    # --------------------------------------------------------
    # Detect year
    # --------------------------------------------------------

    year = detect_year(
        normalized_query
    )

    print(
        "Detected year:",
        year
    )

    # --------------------------------------------------------
    # Detect metric
    # --------------------------------------------------------

    metric = detect_metric(
        normalized_query,
        company
    )

    print(
        "Detected metric:",
        metric
    )

    # --------------------------------------------------------
    # Retrieve documents
    # --------------------------------------------------------

    docs = retrieve_financial_context(
        vectorstore,
        normalized_query,
        company,
        top_k=8
    )

    if not docs:

        return (
            "No relevant information was found "
            "in the uploaded documents."
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    print(
        "\n===== CONTEXT LENGTH ====="
    )

    print(
        f"{len(context)} characters"
    )

    # --------------------------------------------------------
    # Debug context
    # --------------------------------------------------------

    print(
        "\n===== FINAL CONTEXT PREVIEW ====="
    )

    print(
        context[:3000]
    )

    # --------------------------------------------------------
    # Deterministic extraction for canonical lines (Microsoft-first)
    # --------------------------------------------------------
    # Try operating cash flow first (we saw failures here)
    if metric == "operating_cash_flow":
        raw, val, snippet = extract_operating_cash_flow(context)
        if val is not None:
            if "in millions" in (snippet or "").lower():
                converted = val / 1000.0
                answer_text = f"According to the provided context, Microsoft's net cash provided by operating activities in 2023 was ${converted:.3f} billion."
            else:
                answer_text = f"According to the provided context, Microsoft's net cash provided by operating activities in 2023 was ${val:,.2f}."
            print("\n===== QUICK EXTRACTION (OPERATING CASH FLOW) =====")
            print(answer_text)
            return answer_text

    # Try total assets (balance sheet) deterministically too
    # Try total assets (balance sheet) deterministically too
    if metric == "total_assets":
        raw, val, snippet = extract_total_assets(
            context,
            company
        )

        if val is not None:

            if "in millions" in (snippet or "").lower():
                converted = val / 1000.0

                answer_text = (
                    f"According to the provided context, "
                    f"{company}'s total assets in {year} were "
                    f"${converted:.3f} billion."
                )

            else:
                answer_text = (
                    f"According to the provided context, "
                    f"{company}'s total assets in {year} were "
                    f"${val:,.2f}."
                )

            print(
                "\n===== QUICK EXTRACTION (TOTAL ASSETS) ====="
            )

            print(answer_text)

            return answer_text

    # --------------------------------------------------------
    # NEW: Microsoft/Tesla-only deterministic extraction.
    # Explicitly gated so this code path can never be reached
    # for Apple or any other company -- metric can only equal
    # these new values when company was already Microsoft/Tesla
    # (see detect_metric), but the company check is repeated
    # here as well for extra safety and clarity.
    # --------------------------------------------------------
    ms_tesla = company is not None and company.lower() in ("microsoft", "tesla")

    if ms_tesla and metric == "cost_of_revenue":
        val, method, snippet = extract_cost_of_revenue(context)
        if val is not None:
            converted = val / 1000.0 if val > 1000 else val
            suffix = " (derived as Revenue minus Gross margin)" if method == "derived" else ""
            answer_text = f"${converted:.3f} billion{suffix}."
            print("\n===== QUICK EXTRACTION (COST OF REVENUE) =====")
            print(answer_text)
            return answer_text

    if ms_tesla and metric == "product_revenue":
        val, snippet = extract_product_service_revenue(context, "product")
        if val is not None:
            converted = val / 1000.0 if val > 1000 else val
            answer_text = f"${converted:.3f} billion."
            print("\n===== QUICK EXTRACTION (PRODUCT REVENUE) =====")
            print(answer_text)
            return answer_text

    if ms_tesla and metric == "service_revenue":
        val, snippet = extract_product_service_revenue(context, "service")
        if val is not None:
            converted = val / 1000.0 if val > 1000 else val
            answer_text = f"${converted:.3f} billion."
            print("\n===== QUICK EXTRACTION (SERVICE REVENUE) =====")
            print(answer_text)
            return answer_text

    if ms_tesla and metric == "adjusted_net_income":
        val, snippet = extract_adjusted_net_income(context)
        if val is not None:
            converted = val / 1000.0 if val > 1000 else val
            answer_text = f"${converted:.3f} billion."
            print("\n===== QUICK EXTRACTION (ADJUSTED NET INCOME) =====")
            print(answer_text)
            return answer_text

    # --------------------------------------------------------
    # Build prompt (LLM)
    # --------------------------------------------------------

    final_prompt = build_prompt(
        context,
        normalized_query,
        metric,
        company
    )

    print(
        "\n===== PROMPT LENGTH ====="
    )

    print(
        f"{len(final_prompt)} characters"
    )

    # --------------------------------------------------------
    # Get Groq
    # --------------------------------------------------------

    try:

        llm = get_llm()

    except Exception as e:

        print(
            "\n===== LLM INITIALIZATION ERROR ====="
        )

        print(
            str(e)
        )

        return (
            "The language model could not be initialized."
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    try:

        response = llm.invoke(
            final_prompt
        )

        answer = response.content

        if isinstance(answer, list):

            answer = " ".join(
                str(item)
                for item in answer
            )

        answer = str(answer).strip()

    except Exception as e:

        print(
            "\n===== GROQ GENERATION ERROR ====="
        )

        print(
            str(e)
        )

        return (
            "The language model could not generate "
            "an answer."
        )

    # --------------------------------------------------------
    # Empty response
    # --------------------------------------------------------

    if not answer:

        return (
            "The provided context does not contain enough "
            "information to answer this question."
        )

    # --------------------------------------------------------
    # Convert millions -> billions
    # --------------------------------------------------------

    answer = convert_millions_to_billions(
        answer
    )

    # --------------------------------------------------------
    # Validate answer
    # --------------------------------------------------------

    valid = validate_answer(
        answer,
        context,
        metric
    )

    print(
        "\n===== ANSWER VALIDATION ====="
    )

    print(
        "Number verified:",
        valid
    )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    print(
        "\n===== FINAL ANSWER ====="
    )

    print(
        answer
    )

    print(
        "=" * 70
    )

    return answer


# ============================================================
# OPTIONAL DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Loading FAISS vector store..."
    )

    vectorstore = load_vector_store(
        "db/faiss_index"
    )

    answer = ask_question(
        vectorstore,
        "What was Microsoft's revenue in 2023?"
    )

    print(
        "\nFINAL RESULT:"
    )

    print(
        answer
    )