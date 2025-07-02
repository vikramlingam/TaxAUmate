# TaxAUmate

https://taxaumate.streamlit.app

## Overview

TaxAUmate is an AI-powered assistant designed to provide precise and factual information on Australian Taxation Office (ATO) matters. Built using a Retrieval-Augmented Generation (RAG) architecture, this system combines a dedicated knowledge base with an AI model to deliver context-aware answers derived from official tax documentation.

## Core Components

This repository contains the essential Python scripts that drive the TaxAUmate system:

### `index.py` (Knowledge Base Builder)

This script is responsible for the crucial initial setup and ongoing maintenance of the assistant's underlying knowledge base. It transforms raw tax information into a structured, searchable format (not shared in this repo.)

* **Process:** It loads source documents, segments them into manageable text portions (chunks), converts these texts into numerical representations (embeddings) suitable for AI processing, and then stores both the raw text chunks and their embeddings in specialized databases for efficient retrieval and storage.

### `app.py` (Streamlit Chatbot Application)

This script powers the interactive user interface of the TaxAUmate assistant, enabling users to query the system and receive information.

* **Process:** Upon receiving a user's question, the application intelligently searches the pre-built knowledge base to find the most relevant pieces of information. This retrieved context is then provided to an AI model, which uses it to synthesize a direct and accurate answer, ensuring all generated claims are grounded in the original source material. The response is presented in a user-friendly format with citations to the underlying documents.

## Requirements

This project relies on the following Python packages:
streamlit
python-dotenv
openai
pinecone
pymongo
