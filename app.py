import streamlit as st
import os
import json
import logging
import re
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pymongo import MongoClient

# --- 1. Configuration and Initialization ---
load_dotenv()

# --- Application Constants ---
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o-mini"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ato-legal-database")
MONGO_COLLECTION_NAME = "documents"
MONGO_DB_NAME = "ato_data"
TOP_K = 8

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 2. Enhanced System Prompt (v3) ---
SYSTEM_PROMPT = """
You are TaxAUmate, an expert AI assistant specializing in Australian Taxation Office (ATO) matters and Australian Legal Database. Your primary function is to provide accurate, factual, and helpful information based *only* on the provided context documents. You must operate under the following strict guidelines:

**Guideline 1: Scope of Knowledge & Disclaimers**
- Your knowledge is strictly limited to the information contained in the provided context.
- If the user asks a question that requires information not present in the context, you MUST state: "I could not find specific information about this in my knowledge base. For the most accurate details, please refer to the official ATO website."
- You are an information provider, NOT a financial advisor. For any query that asks for advice, opinions, recommendations, or "should I" type questions, you MUST respond with the following disclaimer and nothing else:
  "I cannot provide financial advice or personal recommendations. My purpose is to supply factual information based on ATO documents. For personalized financial or tax advice, please consult a registered tax agent or a licensed financial adviser."

**Guideline 2: Answering and Formatting**
- When the query is within scope and the context contains relevant information, provide a direct and comprehensive answer.
- Synthesize information from multiple sources in the context to create a cohesive response.
- **CRITICAL FORMATTING RULE:** Present ALL information, including step-by-step calculations, as standard text, paragraphs, and bullet points. **DO NOT use Markdown code blocks (```) or inline code backticks (`) for any reason.** All text, especially numbers and calculations, must render in the standard user-facing font.
- **Assume Latest Year:** If a query involves calculations (e.g., tax rates, thresholds) and the user does not specify a financial year, you must assume they are asking about the most recent completed financial year. You should state this assumption in your response (e.g., "Assuming the 2023-2024 financial year...").
- **Crucially, every piece of information or claim you make must be followed by an inline citation**, like this: (Source: [Title of Document](URL)).

**Guideline 3: Citing Sources**
- At the end of your entire response, include a "Sources" section.
- List all the unique documents you cited in your response as a bulleted list, with each item formatted as: `[Title of Document](URL)`.

**Workflow:**
1.  Analyze the user's query.
2.  Determine if it's a request for factual information that can be answered from the context.
3.  If it's a request for advice or is out-of-scope, provide the disclaimer.
4.  If it's a valid query, synthesize an answer from the provided context, following all formatting rules and citing sources inline.
5.  Conclude with a list of all sources used.
"""

# --- 3. Client and Resource Initialization ---
@st.cache_resource
def get_mongo_client():
    """Initialize and return MongoDB client."""
    try:
        client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        logger.info("MongoDB connection successful.")
        return client
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        st.error("Fatal: Could not connect to the document database.")
        st.stop()

@st.cache_resource
def get_pinecone_client():
    """Initialize and return Pinecone client."""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logger.info("Pinecone connection successful.")
        return pc
    except Exception as e:
        logger.error(f"Pinecone connection failed: {e}")
        st.error("Fatal: Could not connect to the vector index.")
        st.stop()

@st.cache_resource
def get_openai_client():
    """Initialize and return OpenAI client."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("OpenAI client initialized.")
        return client
    except Exception as e:
        logger.error(f"OpenAI client initialization failed: {e}")
        st.error("Fatal: Could not connect to OpenAI.")
        st.stop()

# --- 4. Core Application Logic ---

def sanitize_response(text: str) -> str:
    """Cleans the AI's response to fix common formatting and concatenation issues."""
    text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).replace('```', ''), text)
    text = text.replace('`', '')
    text = re.sub(r'([,\d]+)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([,\d]+)([\(\)])', r'\1 \2', text)
    return text

def retrieve_context(query: str, pinecone_index: Any, mongo_collection: Any, openai_client: OpenAI) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve relevant context from Pinecone and MongoDB."""
    if not query: return "", []
    try:
        query_embedding = openai_client.embeddings.create(input=[query], model=EMBEDDING_MODEL).data[0].embedding
        query_results = pinecone_index.query(vector=query_embedding, top_k=TOP_K, include_metadata=False)
        result_ids = [match['id'] for match in query_results.get('matches', [])]
        if not result_ids: return "", []
        mongo_results = list(mongo_collection.find({"_id": {"$in": result_ids}}))
        formatted_context = ""
        raw_context_for_display = []
        for doc in mongo_results:
            title = doc.get('title', 'Untitled')
            url = doc.get('url', 'No URL available')
            text_snippet = doc.get('text', 'No text available')
            formatted_context += f"---\nTitle: {title}\nURL: {url}\nText: {text_snippet}\n---\n\n"
            raw_context_for_display.append({"title": title, "url": url})
        return formatted_context, raw_context_for_display
    except Exception as e:
        logger.error(f"Error during context retrieval: {e}")
        st.warning(f"Error searching the database: {e}")
        return "", []


# --- 5. Streamlit User Interface ---
def main():
    # Page config is now cleaner, favicon removed.
    st.set_page_config(
        page_title="TaxAUmate",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # CSS has been significantly updated for a more professional look.
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Hide Streamlit's default toolbar */
        [data-testid="stToolbar"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }
        [data-testid="stDecoration"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }
        [data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }
        #MainMenu {
            visibility: hidden;
            height: 0%;
        }
        
        html, body, [class*="st-"], .stChatMessage p, .stChatMessage li {
            font-family: 'Inter', sans-serif !important;
        }
        .stApp { background-color: #f5f7fa; }
        .main .block-container { max-width: 850px; padding: 1.5rem 2rem 6rem 2rem; }
        
        h1 {
            font-size: 2.1rem; /* Slightly smaller H1 */
            font-weight: 650;
            color: #1a2c4e;
        }
        h3 {
            font-size: 1.0rem; /* Smaller tagline */
            font-weight: 350;
            color: #556177; 
        }
        
        .stChatMessage {
            background-color: #ffffff;
            border: 1px solid #e6e9f0;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        }
        .stChatMessage[data-testid="chat-message-container-user"] { background-color: #eef6ff; }
        
        .stChatMessage p, .stChatMessage li, .stChatMessage ol, .stChatMessage ul, .stChatMessage span {
            font-family: 'Inter', sans-serif !important;
            font-size: 14px; /* Smaller base font size */
            line-height: 1.65;
            color: #333d4f;
        }
        
        .stChatMessage a { color: #0056b3; text-decoration: none; font-weight: 500; }
        .stChatMessage a:hover { text-decoration: underline; }
        
        .stExpander { border: 1px solid #e6e9f0; border-radius: 10px; background-color: #fafbfd; }
        
        .stTextInput > div > div > input {
            background-color: #ffffff;
            border-radius: 10px;
            border: 1px solid #d1d9e4;
            padding: 10px 14px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #0056b3;
            box-shadow: 0 0 0 2px rgba(0, 86, 179, 0.2);
        }

        .welcome-message h4 {
            font-weight: 600;
            color: #1a2c4e;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }
        .welcome-message ul { margin-left: 20px; }
        .welcome-message p { font-size: 15px; }
    </style>
    """, unsafe_allow_html=True)

    mongo_client = get_mongo_client()
    pinecone_client = get_pinecone_client()
    openai_client = get_openai_client()

    if not (mongo_client and pinecone_client and openai_client):
        st.error("Application cannot start due to failed service connections.")
        st.stop()

    db = mongo_client[MONGO_DB_NAME]
    mongo_collection = db[MONGO_COLLECTION_NAME]
    pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)

    if "messages" not in st.session_state:
        welcome_message = """
        <div class="welcome-message">
            <h4>Important Information:</h4>
            <ul>
                <li>All data is sourced from official ATO documentation.</li>
                <li>This tool provides only general information and is not to be considered professional tax advice.</li>
                <li>The LLM used is a general guidance model and hence accuracy may not be perfect.</li>
                <li>This prototype model has limited features and does not perform accurate calculations yet.</li>
            </ul>
            <h4>Sample Questions:</h4>
            <ul>
                <li>What are the common tax deductions available for individuals in Australia?</li>
                <li>What is the primary purpose of the global minimum tax (Pillar Two)</li>
                <li>What is the fixed ratio test within thin capitalization rules, and what are its key components?</li>
                <li>What happens if a company does not lodge a tax return on time?</li>
            </ul>
            <hr style="margin-top: 20px; margin-bottom: 20px; border-color: #e6e9f0;">
            <p>How can I help you today?</p>
        </div>
        """
        st.session_state.messages = [{"role": "assistant", "content": welcome_message}]

    # Header updated with emoji and new structure.
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
        <div>
            <h1 style="margin: 0; line-height: 1;">🇦🇺 TaxAUmate</h1>
            <h3 style="margin: 0; line-height: 1.2;">Instant tax related answers with RAG precision</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question about Australian taxation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the ATO knowledge base..."):
                context, raw_context = retrieve_context(prompt, pinecone_index, mongo_collection, openai_client)
                if raw_context:
                    with st.expander("Search Details: Reviewing Sources", expanded=False):
                        st.markdown("**Retrieved Sources:**")
                        for doc in raw_context:
                            st.markdown(f"- [{doc.get('title', 'N/A')}]({doc.get('url', 'N/A')})")
                else:
                    with st.expander("Search Details", expanded=True):
                        st.warning("Could not find any relevant documents in the database for this query.")

            with st.spinner("Synthesizing information and generating response..."):
                try:
                    messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"}]
                    stream = openai_client.chat.completions.create(model=LLM_MODEL, messages=messages_for_api, temperature=0.1, stream=True)
                    
                    full_response = ""
                    placeholder = st.empty()
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            placeholder.markdown(sanitize_response(full_response) + "▌", unsafe_allow_html=True)
                    
                    sanitized_final_response = sanitize_response(full_response)
                    placeholder.markdown(sanitized_final_response, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({"role": "assistant", "content": sanitized_final_response})

                except Exception as e:
                    logger.error(f"Error during OpenAI API call: {e}")
                    error_message = "I apologize, but I encountered an error. Please try rephrasing your question."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Footer updated to remove the link.
    st.markdown("""
    <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #e6e9f0; text-align: center; font-size: 12px; color: #6c757d;">
        <p>TaxAUmate is an AI assistant for informational purposes and does not constitute professional tax advice.</p>
        <p>© 2025 TaxAUmate</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
