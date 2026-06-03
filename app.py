
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Emotion-Aware Bilingual AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


def normalize_chat_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\?\!\.\,\']", " ", text)

    replacements = {
        "kese": "kaise",
        "kaisay": "kaise",
        "kesay": "kaise",
        "kia": "kya",
        "nai": "nahi",
        "nahin": "nahi",
        "mein": "main",
        "mai": "main",
        "hy": "hai",
        "hey": "hai",
        "hun": "hoon",
        "hu": "hoon",
        "acha": "achha",
        "boht": "bohat",
        "bht": "bohat",
        "yr": "yaar"
    }

    words = text.split()
    words = [replacements.get(word, word) for word in words]

    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_basic_intent(text):
    text = normalize_chat_text(text)

    greetings = [
        "hi", "hello", "hey", "salam", "assalam", "assalam o alaikum",
        "aoa", "kya haal hai", "kaise ho", "kaisay ho", "kesay ho"
    ]

    thanks = [
        "thanks", "thank you", "shukriya", "jazakallah", "mehrbani"
    ]

    goodbyes = [
        "bye", "goodbye", "khuda hafiz", "allah hafiz", "see you"
    ]

    if any(greet in text for greet in greetings):
        return "greeting"

    if any(word in text for word in thanks):
        return "thanks"

    if any(word in text for word in goodbyes):
        return "bye"

    return "normal"


def detect_language_style(text):
    text = normalize_chat_text(text)

    roman_markers = [
        "kya", "kaise", "tum", "aap", "main", "mera", "mere", "mujhe",
        "nahi", "haan", "achha", "acha", "bohat", "yaar", "bhai", "dost",
        "karo", "batao", "sun", "theek", "khush", "pareshan", "hun", "hoon"
    ]

    words = text.split()
    count = sum(1 for word in words if word in roman_markers)

    if count >= 2:
        return "Roman Urdu / Hinglish"
    elif count == 1:
        return "Mixed English + Roman Urdu/Hindi"
    else:
        return "English"


def detect_emotion(text):
    text = normalize_chat_text(text)

    sad_words = [
        "sad", "upset", "depressed", "tired", "alone", "lonely", "cry",
        "pareshan", "dukhi", "udaas", "tension", "mushkil", "bura", "akela"
    ]

    happy_words = [
        "happy", "great", "good", "excited", "awesome", "love", "nice",
        "khush", "acha", "achha", "zabardast", "mazay", "wah"
    ]

    angry_words = [
        "angry", "hate", "furious", "annoyed", "gussa", "ghussa",
        "bakwas", "bad", "worst"
    ]

    confused_words = [
        "confused", "samajh", "understand", "kaise", "kya", "help",
        "problem", "issue", "masla"
    ]

    if any(word in text for word in sad_words):
        return "sad"
    elif any(word in text for word in angry_words):
        return "angry"
    elif any(word in text for word in happy_words):
        return "happy"
    elif any(word in text for word in confused_words):
        return "confused"
    else:
        return "neutral"


def emotion_prefix(emotion, language_style):
    if language_style == "English":
        prefixes = {
            "sad": "I understand, that sounds difficult 🤗 ",
            "angry": "I get why you feel that way 😟 ",
            "happy": "That sounds great! 😊 ",
            "confused": "No worries, let’s make it simple 🙂 ",
            "neutral": ""
        }
    else:
        prefixes = {
            "sad": "samajh sakta hun, ye mushkil lag raha hai 🤗 ",
            "angry": "haan, samajh sakta hun tum frustrated ho 😟 ",
            "happy": "wah, ye to bohat achi baat hai 😊 ",
            "confused": "koi masla nahi, simple way mein samajhte hain 🙂 ",
            "neutral": ""
        }

    return prefixes.get(emotion, "")


@st.cache_resource
def load_chatbot_artifacts():
    conversation_vectorizer = joblib.load("models/conversation_vectorizer.joblib")
    conversation_matrix = sparse.load_npz("models/conversation_matrix.npz")
    conversation_data = joblib.load("models/conversation_data.joblib")

    support_vectorizer = joblib.load("models/support_vectorizer.joblib")
    support_matrix = sparse.load_npz("models/support_matrix.npz")
    support_data = joblib.load("models/support_data.joblib")

    return {
        "conversation_vectorizer": conversation_vectorizer,
        "conversation_matrix": conversation_matrix,
        "conversation_data": conversation_data,
        "support_vectorizer": support_vectorizer,
        "support_matrix": support_matrix,
        "support_data": support_data
    }


@st.cache_data
def load_evaluation_results():
    return pd.read_csv("artifacts/evaluation_results.csv")


@st.cache_data
def load_sample_data():
    return pd.read_csv("data/chatbot_sample.csv")


def get_best_match(query, vectorizer, matrix, response_df):
    query_clean = normalize_chat_text(query)
    query_vector = vectorizer.transform([query_clean])

    similarities = cosine_similarity(query_vector, matrix).flatten()

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    row = response_df.iloc[best_index]

    return {
        "response": row["clean_bot_response"],
        "source": row["source"],
        "score": float(best_score),
        "matched_text": row["clean_user_text"]
    }


def chatbot_reply(user_message, artifacts):
    language_style = detect_language_style(user_message)
    emotion = detect_emotion(user_message)
    intent = detect_basic_intent(user_message)

    conversation_match = get_best_match(
        user_message,
        artifacts["conversation_vectorizer"],
        artifacts["conversation_matrix"],
        artifacts["conversation_data"]
    )

    support_match = get_best_match(
        user_message,
        artifacts["support_vectorizer"],
        artifacts["support_matrix"],
        artifacts["support_data"]
    )

    use_support = False

    if emotion in ["sad", "angry", "confused"]:
        if support_match["score"] >= 0.25:
            use_support = True

    if intent in ["greeting", "thanks", "bye"]:
        use_support = False

    selected = support_match if use_support else conversation_match

    prefix = emotion_prefix(emotion, language_style)

    if selected["score"] < 0.07:
        if language_style == "English":
            response = "I am not fully sure, but I understand your point. Can you explain a little more? 🙂"
        else:
            response = "main puri tarah sure nahi hun, thora aur explain karo? 🙂"
    else:
        response = prefix + selected["response"]

    return {
        "response": response,
        "confidence": selected["score"],
        "source": selected["source"],
        "matched_text": selected["matched_text"],
        "language_style": language_style,
        "emotion": emotion,
        "intent": intent,
        "used_support_model": use_support
    }


def main():
    st.title("🤖 Emotion-Aware Bilingual AI Chatbot")

    st.write(
        "An interactive AI chatbot for English, Hinglish, and Roman Urdu-style conversations "
        "with emotion detection, emoji-based responses, chat history, and Streamlit deployment."
    )

    artifacts = load_chatbot_artifacts()
    evaluation_results = load_evaluation_results()
    sample_data = load_sample_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Languages", "English + Hinglish")
    col2.metric("Roman Urdu Style", "Supported")
    col3.metric("Response Engine", "Hybrid TF-IDF")
    col4.metric("Emotion Mode", "Enabled 😊")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chatbot",
        "📊 Dashboard",
        "🧪 Test Examples",
        "ℹ️ About"
    ])

    with tab1:
        st.subheader("Chat with the AI bot")

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! Salam! Main English, Hinglish aur Roman Urdu style mein baat kar sakta hun 😊"
                }
            ]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_input = st.chat_input("Type in English, Hinglish, or Roman Urdu...")

        if user_input:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })

            with st.chat_message("user"):
                st.write(user_input)

            result = chatbot_reply(user_input, artifacts)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"]
            })

            with st.chat_message("assistant"):
                st.write(result["response"])

                with st.expander("Response details"):
                    st.write("Confidence:", round(result["confidence"], 4))
                    st.write("Language style:", result["language_style"])
                    st.write("Emotion:", result["emotion"])
                    st.write("Intent:", result["intent"])
                    st.write("Source:", result["source"])
                    st.write("Matched text:", result["matched_text"])
                    st.write("Used support model:", result["used_support_model"])

        if st.button("Clear Chat"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. Start again 😊"
                }
            ]
            st.rerun()

    with tab2:
        st.subheader("Model Evaluation Results")

        st.dataframe(evaluation_results, use_container_width=True)

        fig = px.bar(
            evaluation_results,
            x="metric",
            y="value",
            title="Chatbot Evaluation Metrics",
            text=evaluation_results["value"].round(3)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Dataset Source Distribution")

        source_counts = sample_data["source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Count"]

        fig_source = px.bar(
            source_counts,
            x="Source",
            y="Count",
            title="Sample Dataset Source Distribution"
        )

        st.plotly_chart(fig_source, use_container_width=True)

    with tab3:
        st.subheader("Try example prompts")

        examples = [
            "Hello, how are you?",
            "I am feeling sad today",
            "kya haal hai yaar?",
            "main bohat pareshan hun",
            "mujhe samajh nahi aa raha kya karun",
            "yaar weekend par kya plan hai?",
            "thank you",
            "khuda hafiz"
        ]

        selected_example = st.selectbox("Choose an example", examples)

        if st.button("Test Example"):
            result = chatbot_reply(selected_example, artifacts)

            st.write("User:", selected_example)
            st.success("Bot: " + result["response"])

            details_df = pd.DataFrame({
                "Field": [
                    "Confidence",
                    "Language Style",
                    "Emotion",
                    "Intent",
                    "Source",
                    "Used Support Model"
                ],
                "Value": [
                    round(result["confidence"], 4),
                    result["language_style"],
                    result["emotion"],
                    result["intent"],
                    result["source"],
                    result["used_support_model"]
                ]
            })

            st.dataframe(details_df, use_container_width=True)

    with tab4:
        st.subheader("About this project")

        st.markdown(
            """
            This project is an interactive **Emotion-Aware Bilingual AI Chatbot**.

            ### What it supports

            - English conversation
            - Hinglish conversation
            - Roman Urdu-style conversation
            - Emotion detection
            - Emoji-based replies
            - Chat history
            - Confidence score
            - Source-aware retrieval

            ### How it works

            The chatbot uses a hybrid NLP retrieval system:

            - Word-level TF-IDF
            - Character-level TF-IDF
            - Cosine similarity
            - Separate conversation and emotion-support retrieval models

            ### Why this design?

            Training a full ChatGPT-like model requires large GPU resources.
            This project uses a lightweight retrieval-based NLP approach suitable for
            Google Colab, GitHub, and Streamlit deployment.
            """
        )

        st.warning(
            "This chatbot is for educational and portfolio purposes. It may not always give perfect responses."
        )


if __name__ == "__main__":
    main()
