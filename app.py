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
TOP_K = 8 # Number of top results to retrieve from each Pinecone index

# Pinecone Index Names (from your build scripts)
PINECONE_INDEX_NAME_DOCS = os.getenv("PINECONE_INDEX_NAME_ATO", "ato-legal-database")
PINECONE_INDEX_NAME_LEGIS = os.getenv("PINECONE_INDEX_NAME_LEG", "ato-rag-app")

# MongoDB Database and Collection Names
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ato_data")
MONGO_COLLECTION_NAME_DOCS = os.getenv("MONGO_COLLECTION_NAME_ATO", "documents")
MONGO_COLLECTION_NAME_LEGIS = os.getenv("MONGO_COLLECTION_NAME_LEG", "legislation")


# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    # Remove markdown code blocks and backticks
    text = re.sub(r'```[\s\S]*?```', '', text) # Remove entire code blocks
    text = text.replace('`', '') # Remove inline code backticks

    # Add space between numbers/commas and letters (e.g., "1000dollars" -> "1000 dollars")
    text = re.sub(r'([,\d]+)([a-zA-Z])', r'\1 \2', text)
    # Add space between numbers/commas and parentheses (e.g., "100(a)" -> "100 (a)")
    text = re.sub(r'([,\d]+)([\(\)])', r'\1 \2', text)
    return text

def retrieve_context(query: str, pinecone_index_docs: Any, pinecone_index_legis: Any, 
                     mongo_collection_docs: Any, mongo_collection_legis: Any, 
                     openai_client: OpenAI) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve relevant context from multiple Pinecone indexes and MongoDB collections."""
    if not query: return "", []

    try:
        query_embedding = openai_client.embeddings.create(input=[query], model=EMBEDDING_MODEL).data[0].embedding
        
        # --- Query both Pinecone indexes ---
        # FIXED: Use the constant index names for logging, not .name attribute
        logger.info(f"Querying Pinecone index: {PINECONE_INDEX_NAME_DOCS}") 
        results_docs = pinecone_index_docs.query(vector=query_embedding, top_k=TOP_K, include_metadata=False)
        
        logger.info(f"Querying Pinecone index: {PINECONE_INDEX_NAME_LEGIS}") # FIXED: Use constant index name
        results_legis = pinecone_index_legis.query(vector=query_embedding, top_k=TOP_K, include_metadata=False)

        # --- Combine and sort results ---
        combined_matches = []
        if results_docs and results_docs.get('matches'):
            for match in results_docs['matches']:
                match['source_type'] = 'document' # Add a type identifier
                combined_matches.append(match)
        if results_legis and results_legis.get('matches'):
            for match in results_legis['matches']:
                match['source_type'] = 'legislation' # Add a type identifier
                combined_matches.append(match)
        
        # Sort by score in descending order and take top_k overall
        combined_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Get unique IDs from the top_k combined matches
        # Use a set to ensure uniqueness and preserve order as much as possible
        unique_result_ids = []
        seen_ids = set()
        for match in combined_matches:
            if match['id'] not in seen_ids:
                unique_result_ids.append({'id': match['id'], 'source_type': match['source_type']})
                seen_ids.add(match['id'])
            if len(unique_result_ids) >= TOP_K: # Limit to overall TOP_K
                break

        if not unique_result_ids: return "", []

        # --- Fetch full text from respective MongoDB collections ---
        formatted_context = ""
        raw_context_for_display = []
        
        # Separate IDs by source type to fetch efficiently
        doc_ids_to_fetch = [item['id'] for item in unique_result_ids if item['source_type'] == 'document']
        legis_ids_to_fetch = [item['id'] for item in unique_result_ids if item['source_type'] == 'legislation']

        mongo_results = []
        if doc_ids_to_fetch:
            mongo_results.extend(list(mongo_collection_docs.find({"_id": {"$in": doc_ids_to_fetch}})))
        if legis_ids_to_fetch:
            mongo_results.extend(list(mongo_collection_legis.find({"_id": {"$in": legis_ids_to_fetch}})))

        # Create a dictionary for faster lookup by _id
        mongo_docs_map = {doc['_id']: doc for doc in mongo_results}

        # Reconstruct context in order of relevance (from unique_result_ids)
        for item in unique_result_ids:
            doc = mongo_docs_map.get(item['id'])
            if doc:
                title = doc.get('title', 'Untitled')
                text_snippet = doc.get('text', 'No text available')
                
                # Determine URL/Source Identifier based on type
                if item['source_type'] == 'document':
                    url_or_source = doc.get('url', 'No URL available')
                    source_display_name = "Document"
                elif item['source_type'] == 'legislation':
                    url_or_source = doc.get('title', 'No Title') # Use title as the reference for legislation
                    source_display_name = "Legislation"
                else:
                    url_or_source = 'N/A'
                    source_display_name = "Unknown"

                formatted_context += f"---\nSource Type: {source_display_name}\nTitle: {title}\nLink/ID: {url_or_source}\nText: {text_snippet}\n---\n\n"
                raw_context_for_display.append({
                    "title": title,
                    "link_or_id": url_or_source,
                    "source_type": source_display_name
                })
        
        return formatted_context, raw_context_for_display

    except Exception as e:
        logger.error(f"Error during context retrieval: {e}")
        st.warning(f"Error searching the database: {e}")
        return "", []


# --- 5. Streamlit User Interface ---
def main():
    # --- 2. Enhanced System Prompt (v3) --- MOVED INSIDE MAIN
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
- **Crucially, every piece of information or claim you make must be followed by an inline citation**, like this: (Source: [Title of Document](URL/Source Identifier)).

**Guideline 3: Citing Sources**
- At the end of your entire response, include a "Sources" section.
- List all the unique documents you cited in your response as a bulleted list, with each item formatted as: `[Title of Document](URL/Source Identifier)`.
- If the source is legislation, use the Source Identifier (e.g., "TR_2012/5") if a URL is not available.

**Workflow:**
1.  Analyze the user's query.
2.  Determine if it's a request for factual information that can be answered from the context.
3.  If it's a request for advice or is out-of-scope, provide the disclaimer.
4.  If it's a valid query, synthesize an answer from the provided context, following all formatting rules and citing sources inline.
5.  Conclude with a list of all sources used.
""" # END OF SYSTEM_PROMPT MOVED INSIDE MAIN

    st.set_page_config(
        page_title="TaxAUmate",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <style>
        @import url('[https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap)');
        
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
            color: #f0f0f0; /* CHANGED: White font color */
        }
        .stApp { background-color: #1a1a1a; /* CHANGED: Dark background */ }
        .main .block-container { max-width: 850px; padding: 1.5rem 2rem 6rem 2rem; }
        
        h1 {
            font-size: 2.1rem;
            font-weight: 650;
            color: #ffffff; /* CHANGED: White H1 */
        }
        h3 {
            font-size: 1.0rem;
            font-weight: 350;
            color: #cccccc; /* CHANGED: Lighter white for tagline */
        }
        
        .stChatMessage {
            background-color: #2a2a2a; /* CHANGED: Darker grey for chat messages */
            border: 1px solid #444444; /* CHANGED: Darker border */
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); /* Adjusted shadow for dark mode */
        }
        .stChatMessage[data-testid="chat-message-container-user"] { background-color: #3a3a3a; /* CHANGED: Slightly lighter dark grey for user messages */ }
        
        .stChatMessage p, .stChatMessage li, .stChatMessage ol, .stChatMessage ul, .stChatMessage span {
            font-family: 'Inter', sans-serif !important;
            font-size: 14px;
            line-height: 1.65;
            color: #f0f0f0; /* CHANGED: White text inside chat messages */
        }
        
        .stChatMessage a { color: #87ceeb; text-decoration: none; font-weight: 500; } /* CHANGED: Light blue for links */
        .stChatMessage a:hover { text-decoration: underline; }
        
        .stExpander { 
            border: 1px solid #444444; /* CHANGED: Darker border */
            border-radius: 10px; 
            background-color: #222222; /* CHANGED: Even darker grey for expander */
        }
        
        .stTextInput > div > div > input {
            background-color: #333333; /* CHANGED: Dark input field */
            border-radius: 10px;
            border: 1px solid #555555; /* CHANGED: Darker border */
            padding: 10px 14px;
            color: #f0f0f0; /* CHANGED: White text in input */
        }
        .stTextInput > div > div > input:focus {
            border-color: #87ceeb; /* CHANGED: Light blue focus border */
            box-shadow: 0 0 0 2px rgba(135, 206, 235, 0.2); /* CHANGED: Light blue shadow */
        }

        .welcome-message h4 {
            font-weight: 600;
            color: #ffffff; /* CHANGED: White text */
            margin-top: 1rem;
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }
        .welcome-message ul { margin-left: 20px; }
        .welcome-message p { 
            font-size: 15px; 
            color: #f0f0f0; /* CHANGED: White text */
        }
        .welcome-message hr {
            border-color: #444444; /* CHANGED: Darker border for HR */
        }
    </style>
    """, unsafe_allow_html=True)

    mongo_client = get_mongo_client()
    pinecone_client = get_pinecone_client()
    openai_client = get_openai_client()

    if not (mongo_client and pinecone_client and openai_client):
        st.error("Application cannot start due to failed service connections.")
        st.stop()

    db = mongo_client[MONGO_DB_NAME]
    
    # Initialize both MongoDB collections
    mongo_collection_docs = db[MONGO_COLLECTION_NAME_DOCS]
    mongo_collection_legis = db[MONGO_COLLECTION_NAME_LEGIS]
    
    # Initialize both Pinecone index objects
    pinecone_index_docs = pinecone_client.Index(PINECONE_INDEX_NAME_DOCS)
    pinecone_index_legis = pinecone_client.Index(PINECONE_INDEX_NAME_LEGIS)


    if "messages" not in st.session_state:
        welcome_message = """
        <div class="welcome-message">
            <h4>Important Information:</h4>
            <ul>
                <li>All data is sourced from official ATO documentation and Australian legal databases.</li>
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
                # Pass both Pinecone indexes and MongoDB collections
                context, raw_context = retrieve_context(
                    prompt, 
                    pinecone_index_docs, 
                    pinecone_index_legis, 
                    mongo_collection_docs, 
                    mongo_collection_legis, 
                    openai_client
                )
                if raw_context:
                    with st.expander("Search Details: Reviewing Sources", expanded=False):
                        st.markdown("**Retrieved Sources:**")
                        # Display sources with their type and appropriate reference (URL or Title)
                        for doc in raw_context:
                            if doc.get('source_type') == 'Document':
                                # For documents, link_or_id is the URL, so make it a clickable link
                                link_text = f"[{doc.get('title', 'N/A')}]({doc.get('link_or_id', '#')})"
                            elif doc.get('source_type') == 'Legislation':
                                # For legislation, link_or_id is the title, so just display the text
                                link_text = doc.get('link_or_id', 'N/A')
                            else:
                                # Fallback for unknown types
                                link_text = doc.get('title', 'N/A')

                            st.markdown(f"- **{doc.get('source_type', 'Unknown')}:** {link_text}")
                else:
                    with st.expander("Search Details", expanded=True):
                        st.warning("Could not find any relevant documents in the database for this query.")

            with st.spinner("Synthesizing information and generating response..."):
                try:
                    # Use the SYSTEM_PROMPT defined within main()
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

    st.markdown("""
    <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #444444; /* CHANGED: Darker border */ text-align: center; font-size: 12px; color: #cccccc; /* CHANGED: Lighter white */">
        <p>TaxAUmate is an AI assistant for informational purposes and does not constitute professional tax advice.</p>
        <p>© 2025 TaxAUmate</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
