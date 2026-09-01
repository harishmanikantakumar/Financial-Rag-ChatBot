# ============================================================
# FINANCIAL RAG ASSISTANT
# Streamlit Frontend
# ============================================================

import os
import sys
import textwrap

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    os.path.join(ROOT_DIR, ".env"),
    override=True
)


# ============================================================
# APPLICATION IMPORTS
# ============================================================

try:
    from app.rag import ask_question
    from app.vector_store import load_vector_store

except Exception as e:
    st.error("Failed to import the RAG application.")
    st.code(str(e))
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }


    /* ======================================================
       HEADER
       ====================================================== */

    .app-title {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .app-subtitle {
        font-size: 1rem;
        opacity: 0.72;
        margin-bottom: 1.5rem;
    }


    /* ======================================================
       WELCOME CARD
       ====================================================== */

    .welcome-card {
        padding: 2.2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem auto 2rem auto;
        border: 1px solid rgba(128, 128, 128, 0.18);
        background: rgba(128, 128, 128, 0.035);
    }

    .welcome-icon {
        font-size: 3.5rem;
        margin-bottom: 0.4rem;
    }

    .welcome-card h1 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .welcome-subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        max-width: 750px;
        margin: 0 auto 1.8rem auto;
    }


    /* ======================================================
       COMPANY CARDS
       ====================================================== */

    .company-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1.5rem 0;
    }

    .company-card {
        padding: 1.1rem 1.7rem;
        border-radius: 14px;
        min-width: 140px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.035);
    }

    .company-logo {
        font-size: 2.2rem;
        margin-bottom: 0.35rem;
    }

    .company-name {
        font-weight: 650;
        font-size: 1rem;
    }


    /* ======================================================
       DESCRIPTION
       ====================================================== */

    .welcome-description {
        margin: 1.8rem auto;
        max-width: 850px;
        line-height: 1.7;
        opacity: 0.82;
    }


    /* ======================================================
       FEATURES
       ====================================================== */

    .feature-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1.8rem 0;
    }

    .feature {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        text-align: left;
        padding: 0.95rem 1rem;
        border-radius: 14px;
        min-width: 220px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.025);
    }

    .feature > span {
        font-size: 1.5rem;
    }

    .feature strong {
        display: block;
    }

    .feature small {
        display: block;
        opacity: 0.65;
        margin-top: 0.2rem;
    }


    /* ======================================================
       EXAMPLES
       ====================================================== */

    .example-section {
        margin-top: 2rem;
    }

    .example-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .example-question {
        margin: 0.45rem auto;
        max-width: 700px;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.22);
        text-align: left;
        opacity: 0.85;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 700;
    }

    .sidebar-description {
        opacity: 0.7;
        margin-bottom: 0.5rem;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        opacity: 0.55;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD FAISS VECTOR STORE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_vector_store():

    faiss_path = os.path.join(
        ROOT_DIR,
        "db",
        "faiss_index"
    )

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(
            f"FAISS directory not found:\n{faiss_path}"
        )

    vector_store = load_vector_store(
        faiss_path=faiss_path
    )

    if vector_store is None:
        raise RuntimeError(
            "load_vector_store() returned None."
        )

    return vector_store


# ============================================================
# INITIALIZE VECTOR STORE
# ============================================================

vector_store = None
vector_store_error = None

try:

    with st.spinner(
        "Loading financial knowledge base..."
    ):

        vector_store = get_vector_store()

except Exception as e:

    vector_store_error = str(e)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-title">
        📊 Financial RAG Assistant
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-subtitle">
        Ask questions about Apple, Microsoft, and Tesla
        financial reports using hybrid retrieval and
        document-grounded financial analysis.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-title">📊 Financial RAG</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-description">
            SEC Financial Document Assistant
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()


    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.markdown("### ⚙️ System Status")

    if vector_store is not None:

        st.success(
            "FAISS Knowledge Base: Ready"
        )

    else:

        st.error(
            "FAISS Knowledge Base: Unavailable"
        )


    if os.getenv("GROQ_API_KEY"):

        st.success(
            "Groq LLM: Ready"
        )

    else:

        st.warning(
            "Groq LLM: API Key Missing"
        )

    st.divider()


    # --------------------------------------------------------
    # SUPPORTED COMPANIES
    # --------------------------------------------------------

    st.markdown("### 🏢 Companies")

    st.markdown(
        """
        🍎 **Apple**

        🪟 **Microsoft**

        🚗 **Tesla**
        """
    )

    st.divider()


    # --------------------------------------------------------
    # RETRIEVAL PIPELINE
    # --------------------------------------------------------

    st.markdown("### 🔎 Retrieval Pipeline")

    st.markdown(
        """
        **1. BM25**  
        Keyword-based retrieval

        **2. FAISS**  
        Semantic vector retrieval

        **3. Financial Reranking**  
        Metric-aware ranking

        **4. Deterministic Extraction**  
        Exact financial values

        **5. Groq LLM**  
        Context-grounded generation
        """
    )

    st.divider()


    # --------------------------------------------------------
    # EXAMPLE QUESTIONS
    # --------------------------------------------------------

    st.markdown("### 💡 Example Questions")

    examples = [
        "What was Apple's revenue in 2023?",
        "What was Microsoft's gross margin in 2023?",
        "What was Microsoft's net income in 2023?",
        "What was Tesla's operating income in 2023?",
    ]

    for index, example in enumerate(examples):

        if st.button(
            example,
            use_container_width=True,
            key=f"example_question_{index}"
        ):

            st.session_state.selected_question = example

            st.rerun()

    st.divider()


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.selected_question = None

        st.rerun()


# ============================================================
# FAISS ERROR
# ============================================================

if vector_store_error:

    st.error(
        "❌ The FAISS financial knowledge base could not be loaded."
    )

    st.markdown(
        "Please check that your FAISS index exists at:"
    )

    st.code(
        os.path.join(
            ROOT_DIR,
            "db",
            "faiss_index"
        )
    )

    with st.expander(
        "🔧 Technical details"
    ):

        st.code(
            vector_store_error
        )

    st.stop()


# ============================================================
# WELCOME SCREEN
# ============================================================
# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown("## 📊 Financial Document Assistant")

    st.markdown(
        "Intelligent financial question answering powered by "
        "hybrid retrieval and document-grounded analysis."
    )

    st.divider()

    # --------------------------------------------------------
    # COMPANIES
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🍎 Apple")
        st.caption("Financial reports")

    with col2:
        st.markdown("### 🪟 Microsoft")
        st.caption("Financial reports")

    with col3:
        st.markdown("### 🚗 Tesla")
        st.caption("Financial reports")

    st.divider()

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    st.markdown(
        """
        Ask questions about:

        - Revenue
        - Net income
        - Gross margin
        - Operating income
        - Operating expenses
        - R&D
        - Sales & Marketing
        - Other financial metrics
        """
    )

    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    st.markdown("### 🔎 Retrieval & Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**🔎 Hybrid Retrieval**")
        st.caption("BM25 + FAISS")

        st.markdown("**📐 Financial Reranking**")
        st.caption("Metric-aware retrieval")

    with col2:

        st.markdown("**⚡ Deterministic Extraction**")
        st.caption("Precise financial values")

        st.markdown("**🤖 LLM Analysis**")
        st.caption("Context-grounded answers")

    st.divider()

    # --------------------------------------------------------
    # EXAMPLE QUESTIONS
    # --------------------------------------------------------

    st.markdown("### 💡 Try asking")

    st.info(
        'What was Microsoft\'s gross margin in 2023?'
    )

    st.info(
        'What was Apple\'s net income in 2023?'
    )

    st.info(
        "What was Tesla's revenue in 2023?"
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message.get("role")

    content = message.get(
        "content",
        ""
    )

    if role not in ["user", "assistant"]:
        continue

    with st.chat_message(role):

        st.markdown(content)


# ============================================================
# GET QUESTION FROM EXAMPLE BUTTON
# ============================================================

selected_question = st.session_state.pop(
    "selected_question",
    None
)


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask a financial question..."
)


# ============================================================
# SELECT QUESTION
# ============================================================

question = (
    user_question
    if user_question
    else selected_question
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a financial question."
        )

        st.stop()


    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        answer = None

        try:

            with st.spinner(
                "🔎 Searching financial documents..."
            ):

                # ==================================================
                # IMPORTANT
                # Current RAG interface:
                #
                # ask_question(vector_store, question)
                # ==================================================

                answer = ask_question(
                    vector_store,
                    question
                )


            # ----------------------------------------------------
            # SAFETY CHECK
            # ----------------------------------------------------

            if answer is None:

                answer = (
                    "I could not generate an answer "
                    "from the financial documents."
                )

            elif not isinstance(answer, str):

                answer = str(answer)


            answer = answer.strip()


            if not answer:

                answer = (
                    "I could not find a reliable answer "
                    "in the financial documents."
                )


            # ----------------------------------------------------
            # DISPLAY ANSWER
            # ----------------------------------------------------

            st.markdown(answer)


        except Exception as e:

            answer = (
                "I couldn't process the question. "
                "Please try again."
            )

            st.error(answer)

            with st.expander(
                "🔧 Technical details"
            ):

                st.code(
                    str(e)
                )


    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        Financial RAG Assistant ·
        BM25 + FAISS ·
        Financial Reranking ·
        Deterministic Extraction ·
        Groq LLM

    </div>
    """,
    unsafe_allow_html=True
)