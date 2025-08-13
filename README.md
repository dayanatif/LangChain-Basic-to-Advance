# LangChain-Models

A collection of example code for working with **LLMs**, **Chat Models**, and **Embedding Models** from multiple providers — both **open-source** and **closed-source** — using [LangChain](https://www.langchain.com/) and the **Hugging Face API**.  

This repo demonstrates how to integrate, test, and run models like **OpenAI GPT**, **Google Gemini**, and **various Hugging Face models** in a unified workflow.

---

## ✨ Features

- **LLMs**: Examples of text generation using multiple providers.
- **Chat Models**: Conversational AI examples with context management.
- **Embedding Models**: Create vector embeddings for search, similarity, and RAG.
- **Multi-provider setup**:  
  - **Closed-source**: OpenAI, Gemini  
  - **Open-source**: Hugging Face Hub models
- **Environment variable management** with `.env`  
- **Practical LangChain usage** for different model types.

---

## 📂 Project Structure

LangChain-Models/
│
├── LLMs/                 # Code examples for language models
├── ChatModels/           # Code examples for conversational models
├── Embeddings/           # Code examples for embedding models
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variable file
└── README.md             # This file


---

## ⚡ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/LangChain-Models.git
   cd LangChain-Models

OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_hf_api_token
