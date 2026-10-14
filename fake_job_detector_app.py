import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom styling --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #f4f7ff 0%, #f8fbff 50%, #eefcf8 100%);
}
.hero {
    padding: 2rem 2.2rem;
    border-radius: 24px;
    color: white;
    background: linear-gradient(120deg, #172554 0%, #4338ca 55%, #0891b2 100%);
    box-shadow: 0 12px 35px rgba(49, 46, 129, .20);
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Poppins', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    margin: 0;
}
.hero p { font-size: 1.05rem; opacity: .92; margin: .6rem 0 0 0; }
.section-title {
    font-family: 'Poppins', sans-serif;
    color: #172554;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: .5rem;
}
.metric-card {
    background: white;
    padding: 1.1rem 1.2rem;
    border-radius: 18px;
    border: 1px solid #e0e7ff;
    box-shadow: 0 5px 18px rgba(30, 41, 59, .06);
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e0e7ff;
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(30, 41, 59, .05);
}
.stButton > button {
    width: 100%;
    border: 0;
    border-radius: 12px;
    padding: .75rem 1rem;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #4338ca, #0891b2);
}
.stButton > button:hover { color: white; filter: brightness(1.07); }
div[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #818cf8;
    border-radius: 14px;
    padding: 12px;
}
.small-note { color: #475569; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("""
<div class="hero">
  <h1>🛡️ Fake Job Posting Detector</h1>
  <p>Detect potentially fraudulent job advertisements using Natural Language Processing and a Simple RNN model.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Model settings")
    st.caption("Adjust the training configuration.")
    epochs = st.slider("Training epochs", 1, 10, 5)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128], value=32)
    max_num_words = st.select_slider("Vocabulary size", options=[3000, 5000, 10000, 15000], value=10000)
    seq_len = st.select_slider("Maximum sequence length", options=[30, 50, 75, 100], value=50)
    embedding_size = st.select_slider("Embedding size", options=[32, 50, 100, 128], value=100)
    st.markdown("---")
    st.markdown("**Dataset columns needed**")
    st.caption("Text fields such as title, location, description and requirements, plus a `fraudulent` target column.")
    st.markdown('<p class="small-note">Use a CSV file. The app combines available job-description fields automatically.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload fake_job_postings.csv", type=["csv"])

# Optional default file path from the user's notebook, if present on this computer
default_path = r"C:\Users\arjun\Downloads\csv_files\fake_job_postings.csv.xls"

@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes):
    import io
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_resource(show_spinner=False)
def train_rnn(data, epochs, batch_size, max_num_words, seq_len, embedding_size):
    df = data.copy()
    if "fraudulent" not in df.columns and "y" not in df.columns:
        raise ValueError("The dataset must contain a 'fraudulent' or 'y' target column.")

    target_col = "fraudulent" if "fraudulent" in df.columns else "y"
    text_candidates = [
        "title", "location", "department", "company_profile", "salary_range",
        "description", "requirements", "benefits", "industry", "required_experience"
    ]
    available_text = [c for c in text_candidates if c in df.columns]
    if not available_text:
        if "x" in df.columns:
            available_text = ["x"]
        else:
            raise ValueError("No recognized job-text columns were found in the uploaded file.")

    df["x"] = df[available_text].fillna("").astype(str).agg(" ".join, axis=1)
    df["y"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[["x", "y"]].dropna()
    df = df[df["y"].isin([0, 1])]
    df["y"] = df["y"].astype(int)

    if df["y"].nunique() != 2:
        raise ValueError("The target column must contain both classes 0 (genuine) and 1 (fraudulent).")
    if df["y"].value_counts().min() < 2:
        raise ValueError("Please upload more examples of each target class.")

    # Split before oversampling to avoid data leakage.
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["y"]
    )

    # Oversample minority class only in training data.
    class_counts = train_df["y"].value_counts()
    majority_count = int(class_counts.max())
    pieces = []
    for label in sorted(class_counts.index):
        subset = train_df[train_df["y"] == label]
        pieces.append(subset.sample(n=majority_count, replace=(len(subset) < majority_count), random_state=42))
    train_df = pd.concat(pieces).sample(frac=1, random_state=42).reset_index(drop=True)

    tokenizer = Tokenizer(num_words=max_num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["x"].tolist())
    train_x = pad_sequences(tokenizer.texts_to_sequences(train_df["x"].tolist()), maxlen=seq_len)
    test_x = pad_sequences(tokenizer.texts_to_sequences(test_df["x"].tolist()), maxlen=seq_len)
    train_y = to_categorical(train_df["y"].to_numpy(), num_classes=2)
    test_y = test_df["y"].to_numpy()

    model = Sequential([
        Embedding(input_dim=max_num_words, output_dim=embedding_size, input_length=seq_len),
        SimpleRNN(32),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_x, train_y, epochs=epochs, batch_size=batch_size,
        validation_split=0.2, verbose=0
    )
    probabilities = model.predict(test_x, verbose=0)
    predictions = probabilities.argmax(axis=1)
    cm = confusion_matrix(test_y, predictions, labels=[0, 1])
    report_dict = classification_report(
        test_y, predictions, labels=[0, 1],
        target_names=["Genuine (0)", "Fraudulent (1)"],
        output_dict=True, zero_division=0
    )
    report_text = classification_report(
        test_y, predictions, labels=[0, 1],
        target_names=["Genuine (0)", "Fraudulent (1)"],
        zero_division=0
    )
    return {
        "model": model, "tokenizer": tokenizer, "history": history,
        "accuracy": accuracy_score(test_y, predictions),
        "cm": cm, "report_dict": report_dict, "report_text": report_text,
        "available_text": available_text, "data": df,
        "class_counts": df["y"].value_counts().sort_index(),
        "test_df": test_df, "predictions": predictions,
        "probabilities": probabilities, "seq_len": seq_len,
        "max_num_words": max_num_words
    }

if uploaded_file is None:
    st.info("👆 Upload your CSV file to train the model and view the results.")
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><h3>📄 1. Prepare</h3><p>Combine job posting text fields and clean missing values.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h3>🔤 2. Convert</h3><p>Tokenize text and pad each sequence to a fixed length.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h3>🧠 3. Predict</h3><p>Train a Simple RNN and evaluate its predictions.</p></div>', unsafe_allow_html=True)
    st.stop()

try:
    data = load_uploaded_csv(uploaded_file.getvalue())
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

st.markdown('<div class="section-title">📊 Dataset overview</div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("Total rows", f"{len(data):,}")
m2.metric("Total columns", f"{len(data.columns):,}")
target_for_count = "fraudulent" if "fraudulent" in data.columns else ("y" if "y" in data.columns else None)
if target_for_count:
    m3.metric("Fraudulent postings", f"{(pd.to_numeric(data[target_for_count], errors='coerce') == 1).sum():,}")
else:
    m3.metric("Target column", "Not found")

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(10), use_container_width=True)

if st.button("🚀 Train RNN model and evaluate", type="primary"):
    try:
        with st.spinner("Preparing text and training the RNN. This can take several minutes for a large dataset..."):
            results = train_rnn(data, epochs, batch_size, max_num_words, seq_len, embedding_size)
        st.session_state["rnn_results"] = results
        st.success("Model training and evaluation completed.")
    except Exception as e:
        st.error(f"Training failed: {e}")

results = st.session_state.get("rnn_results")
if results:
    st.markdown('<div class="section-title">📈 Model performance</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Test accuracy", f"{results['accuracy'] * 100:.2f}%")
    b.metric("Test samples", f"{len(results['test_df']):,}")
    c.metric("Vocabulary limit", f"{results['max_num_words']:,}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Training history")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(results["history"].history.get("accuracy", []), marker="o", label="Training accuracy")
        ax.plot(results["history"].history.get("val_accuracy", []), marker="o", label="Validation accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### Confusion matrix")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            results["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Genuine", "Fraudulent"],
            yticklabels=["Genuine", "Fraudulent"]
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Classification report")
    st.code(results["report_text"], language="text")

    with st.expander("View class distribution"):
        counts = results["class_counts"].rename(index={0: "Genuine", 1: "Fraudulent"})
        fig, ax = plt.subplots(figsize=(7, 3.5))
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Job posting class")
        ax.set_ylabel("Number of rows")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown('<div class="section-title">🔎 Test a job posting</div>', unsafe_allow_html=True)
    job_text = st.text_area(
        "Paste the job title and description",
        height=180,
        placeholder="Example: We are hiring a data analyst. Responsibilities include..."
    )
    if st.button("Check this job posting"):
        if not job_text.strip():
            st.warning("Please enter some job-posting text first.")
        else:
            seq = results["tokenizer"].texts_to_sequences([job_text])
            padded = pad_sequences(seq, maxlen=results["seq_len"])
            probs = results["model"].predict(padded, verbose=0)[0]
            pred_label = int(np.argmax(probs))
            if pred_label == 1:
                st.error(f"⚠️ Model prediction: Potentially fraudulent ({probs[1] * 100:.1f}% model score)")
            else:
                st.success(f"✅ Model prediction: Likely genuine ({probs[0] * 100:.1f}% model score)")
            st.caption("This is a machine-learning estimate, not proof that a posting is genuine or fraudulent. Verify employers independently.")
