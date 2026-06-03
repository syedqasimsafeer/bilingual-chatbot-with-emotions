
# Emotion-Aware Bilingual AI Chatbot

## Project Overview

This project is an interactive AI chatbot that supports English, Hinglish, and Roman Urdu-style conversation.

The chatbot uses NLP-based response retrieval, emotion detection, emoji-based replies, chat history, and a Streamlit user interface.

## Main Features

- English conversation
- Hinglish / Roman Urdu-style conversation
- Emotion detection
- Emoji-based responses
- Separate conversation and support retrieval models
- Confidence score
- Chat history
- Streamlit deployment

## Dataset

The project uses public conversation and sentiment-style datasets, including:

- Hinglish Everyday Conversations
- Roman Urdu sentiment support data
- Optional English dialogue data
- Optional empathetic dialogue data

The full training datasets are not uploaded to GitHub. Only optimized model artifacts are included for deployment.

## Model Approach

This is a retrieval-based chatbot.

The chatbot uses:

- Word-level TF-IDF
- Character-level TF-IDF
- Cosine similarity
- Source-aware retrieval
- Emotion-aware response selection

## Why Retrieval-Based?

Training a full generative chatbot from scratch requires very large GPU resources. This project uses a lightweight NLP retrieval method that is practical for Google Colab, GitHub, and Streamlit deployment.

## Developemnt
The chatbot is available at: https://bilingual-chatbot-with-emotions-tr.streamlit.app/

## Important Note

This chatbot is for educational and portfolio purposes. It may not always provide perfect or factual responses.
