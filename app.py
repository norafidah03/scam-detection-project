# app.py for Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, auc, roc_auc_score
import torch

# Transformers imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

st.set_page_config(page_title="Scam Detection Dashboard", layout="wide")

# -------------------------
# Utility: text cleaning
# -------------------------
def clean_text_for_display(text):
    if not isinstance(text, str):
        return ""
    # minimal cleaning: strip whitespace
    return text.strip()

def clean_text_for_traditional(text):
    """Cleaning for TF-IDF (remove URLs, mentions, hashtags, emojis, punctuation)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    emoji_pattern = re.compile("["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\u2600-\u26FF\u2700-\u27BF]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # keep letters/numbers and spaces
    text = re.sub(r'[^0-9a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# -------------------------
# Caching model loaders
# -------------------------
@st.cache_resource(show_spinner=False)
def load_svm_and_tfidf(svm_path: str, tfidf_path: str):
    tfidf = joblib.load(tfidf_path)
    svm = joblib.load(svm_path)
    return tfidf, svm

@st.cache_resource(show_spinner=False)
def load_mbert_pipeline(model_dir: str, tokenizer_dir: str = None):
    # If tokenizer_dir is None, tokenizer is loaded from model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir or model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = 0 if torch.cuda.is_available() else -1
    nlp_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    return nlp_pipe

# -------------------------
# Prediction wrappers
# -------------------------
def predict_svm(texts, tfidf, svm):
    cleaned = [clean_text_for_traditional(t) for t in texts]
    X = tfidf.transform(cleaned)
    preds = svm.predict(X)
    return preds

def predict_mbert(texts, mbert_pipeline):
    # pipeline expects list of texts
    outs = mbert_pipeline(texts)  # returns [{'label':..., 'score':...}, ...]
    
    labels = []
    scores = [] # This needs to be the probability of Class 1
    
    for o in outs:
        lbl = o.get('label')
        raw_score = float(o.get('score', 0.0))
        
        numeric_label = 0 # Default to 0
        
        # --- 1. Standardization Logic (Same as before) ---
        if isinstance(lbl, str) and lbl.lower().startswith('label_'):
            # label_0 -> 0, label_1 -> 1
            numeric_label = int(lbl.split('_')[-1])
        elif lbl in ('0', 0, 'Legit', 'legit'):
            numeric_label = 0
        elif lbl in ('1', 1, 'Scam', 'scam'):
            numeric_label = 1
        else:
            # Fallback: if raw_score > 0.5, assume 1, else 0 (conservative)
            numeric_label = 1 if raw_score >= 0.5 else 0

        labels.append(numeric_label)

        # ROC-AUC requires the probability of the Positive Class (1).
        # If the model predicts Class 0 with 0.9 score, prob(Class 1) is 0.1.
        if numeric_label == 1:
            scores.append(raw_score)
        else:
            scores.append(1.0 - raw_score)
            
    return np.array(labels), np.array(scores)

# -------------------------
# Plot helpers
# -------------------------
def plot_confusion_matrix(y_true, y_pred, labels=("Legit", "Scam")):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig

# -------------------------
# Layout: Sidebar controls
# -------------------------
st.sidebar.title("Model & Data Config")
st.sidebar.markdown("**Model files** (set paths or leave defaults if in same folder):")
svm_path = st.sidebar.text_input("SVM model path", value="svm_tfidf_model.joblib")
tfidf_path = st.sidebar.text_input("TF-IDF path", value="tfidf_vectorizer.joblib")
mbert_model_dir = st.sidebar.text_input("mBERT model dir", value="mbert_scam_model")
mbert_tokenizer_dir = st.sidebar.text_input("mBERT tokenizer dir (optional)", value="mbert_scam_tokenizer")
use_mbert = st.sidebar.checkbox("Enable mBERT model", value=True)
use_svm = st.sidebar.checkbox("Enable SVM model", value=True)

# Load models on demand
tfidf, svm = None, None
mbert_pipe = None

if use_svm:
    try:
        tfidf, svm = load_svm_and_tfidf(svm_path, tfidf_path)
    except Exception as e:
        st.sidebar.error(f"Failed to load SVM/TF-IDF: {e}")

if use_mbert:
    try:
        mbert_pipe = load_mbert_pipeline(mbert_model_dir, mbert_tokenizer_dir or None)
    except Exception as e:
        st.sidebar.error(f"Failed to load mBERT model/tokenizer: {e}")

# Main title
st.title("AI-Driven Financial Scam Detection — Demo Dashboard")
st.caption("Interactive demo: SVM (TF-IDF) vs mBERT (fine-tuned)")

# -------------------------
# Section 1: Dataset & Quick Stats
# -------------------------
st.header("Dataset / Quick Stats")
uploaded = st.file_uploader("Upload a test CSV (columns: text,label) to evaluate models", type=["csv"])
if uploaded is not None:
    test_df = pd.read_csv(uploaded)
    test_df = test_df.dropna(subset=['label'])
    st.write("Preview of uploaded test data:")
    st.dataframe(test_df.head())
    st.write("Class distribution:")
    st.bar_chart(test_df['label'].value_counts())
else:
    st.info("No test CSV uploaded — use the model demo panels below for single or batch predictions.")

# -------------------------
# Section 2: Model Performance (evaluate on uploaded testset)
# -------------------------
st.header("Model Evaluation")

if uploaded is not None:
    y_true = test_df['label'].values
    
    # --- SVM evaluation ---
    if use_svm and tfidf is not None and svm is not None:
        with st.spinner("Running SVM predictions..."):
            # Get text data
            text_data = test_df['text'].astype(str).tolist()
            preds = predict_svm(text_data, tfidf, svm)
            
        st.subheader("SVM Results")
        
        # Calculate Accuracy
        acc = accuracy_score(y_true, preds)
        
        # Attempt to calculate ROC-AUC if decision_function exists
        roc_auc_val = None
        svm_scores = None
        if hasattr(svm, "decision_function"):
            try:
                svm_scores = svm.decision_function(tfidf.transform(text_data))
                roc_auc_val = roc_auc_score(y_true, svm_scores)
            except Exception as e:
                st.warning(f"Could not calculate ROC-AUC for SVM: {e}")

        # Display Metrics (Side-by-side for better layout)
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.4f}")
        if roc_auc_val is not None:
            col2.metric("ROC-AUC", f"{roc_auc_val:.4f}")
        
        # Classification Report & Confusion Matrix
        report = classification_report(y_true, preds, output_dict=True, zero_division=0)
        st.write(pd.DataFrame(report).transpose())
        st.pyplot(plot_confusion_matrix(y_true, preds))
        
        # Plot ROC Curve
        if svm_scores is not None:
            st.pyplot(plot_roc_curve(y_true, svm_scores, title="SVM ROC Curve"))

    # --- mBERT evaluation ---
    if use_mbert and mbert_pipe is not None:
        with st.spinner("Running mBERT predictions... (might be slow on CPU)"):
            # m_scores should be the probability/logit of the positive class
            m_labels, m_scores = predict_mbert(test_df['text'].astype(str).tolist(), mbert_pipe)
        
        st.subheader("mBERT Results")
        
        # Calculate Metrics
        acc_mbert = accuracy_score(y_true, m_labels)
        roc_auc_mbert = roc_auc_score(y_true, m_scores)

        # Display Metrics
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc_mbert:.4f}")
        col2.metric("ROC-AUC", f"{roc_auc_mbert:.4f}")
        
        # Classification Report & Confusion Matrix
        rep = classification_report(y_true, m_labels, output_dict=True, zero_division=0)
        st.write(pd.DataFrame(rep).transpose())
        st.pyplot(plot_confusion_matrix(y_true, m_labels))
        
        # Plot ROC Curve
        st.pyplot(plot_roc_curve(y_true, m_scores, title="mBERT ROC Curve"))

# -------------------------
# Section 3: Single Prediction demo
# -------------------------
st.header("Single Text Prediction")
col1, col2 = st.columns([3,1])
with col1:
    sample_text = st.text_area("Enter social media text to classify:", height=120,
                               value="This is the most trusted investment scheme in Malaysia. Join now for instant profit and rewards!")
    if st.button("Classify"):
        results = {}
        if use_svm and tfidf is not None and svm is not None:
            s_pred = predict_svm([sample_text], tfidf, svm)[0]
            results['SVM'] = "Scam" if int(s_pred)==1 else "Legit"
        if use_mbert and mbert_pipe is not None:
            m_label, m_score = predict_mbert([sample_text], mbert_pipe)
            results['mBERT'] = ( "Scam" if int(m_label[0])==1 else "Legit", float(m_score[0]) )
        st.write("**Predictions:**")
        st.json(results)

# -------------------------
# Section 4: Batch Prediction & Download
# -------------------------
st.header("Batch Prediction (Upload CSV with 'text' column)")
batch_file = st.file_uploader("Upload CSV for batch prediction (CSV must have 'text' column)", type=["csv"], key="batch")
if batch_file is not None:
    batch_df = pd.read_csv(batch_file)
    if 'text' not in batch_df.columns:
        st.error("CSV must include a 'text' column.")
    else:
        st.info("Running predictions...")
        out_df = batch_df.copy()
        if use_svm and tfidf is not None:
            out_df['svm_pred'] = predict_svm(out_df['text'].astype(str).tolist(), tfidf, svm)
        if use_mbert and mbert_pipe is not None:
            m_labels, m_scores = predict_mbert(out_df['text'].astype(str).tolist(), mbert_pipe)
            out_df['mbert_pred'] = m_labels
            out_df['mbert_score'] = m_scores
        st.dataframe(out_df.head())
        csv = out_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", csv, file_name="predictions.csv", mime="text/csv")

# -------------------------
# Section 5: Simulated Real-time Stream
# -------------------------
st.header("Simulated Real-time Stream (Demo)")
st.write("Upload a CSV and click Start to simulate live message stream with predictions.")

# 1. Initialize session state list to hold results across reruns
if 'stream_data_list' not in st.session_state:
    st.session_state['stream_data_list'] = []

stream_file = st.file_uploader("CSV for streaming (columns: id,text)", type=["csv"], key="stream")

if stream_file is not None:
    stream_df = pd.read_csv(stream_file)
    st.write("Preview:")
    st.dataframe(stream_df.head())

    placeholder = st.empty()
    
    # Layout for buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        start = st.button("Start Stream")
    with col2:
        stop = st.button("Stop Stream") # Clicking this triggers a rerun, effectively stopping the loop

    if start:
        # Reset the list when starting a new stream
        st.session_state['stream_data_list'] = []
        
        progress_bar = st.progress(0)
        total_rows = len(stream_df)

        for idx, row in stream_df.iterrows():
            # Processing logic
            text = str(row.get('text', ''))
            cleaned = clean_text_for_display(text)
            pred_result = {}
            
            svm_label = "N/A"
            mbert_label = "N/A"
            mbert_conf = 0.0

            if use_svm and tfidf is not None:
                p = predict_svm([cleaned], tfidf, svm)[0]
                svm_label = "Scam" if int(p)==1 else "Legit"
                pred_result['SVM'] = svm_label

            if use_mbert and mbert_pipe is not None:
                m_lbl, m_score = predict_mbert([cleaned], mbert_pipe)
                mbert_label = "Scam" if int(m_lbl[0])==1 else "Legit"
                mbert_conf = float(m_score[0])
                pred_result['mBERT'] = (mbert_label, mbert_conf)

            # Update Display
            placeholder.markdown(f"**ID:** {row.get('id','-')}  \n**Text:** {cleaned}  \n**Pred:** {pred_result}")
            
            # 2. CRITICAL FIX: Append to session state IMMEDIATELY inside the loop
            # This ensures data is saved even if the user clicks "Stop" (which kills the loop)
            st.session_state['stream_data_list'].append({
                'id': row.get('id', '-'),
                'text': text,
                'svm_pred': svm_label,
                'mbert_pred': mbert_label,
                'mbert_confidence': mbert_conf
            })

            progress_bar.progress((idx + 1) / total_rows)
            time.sleep(0.5)

    # 3. Display Download Button (Outside the loop)
    # This checks if ANY data exists in the session state list (whether finished or stopped early)
    if len(st.session_state['stream_data_list']) > 0:
        st.write(f"### Simulation Results ({len(st.session_state['stream_data_list'])} processed)")
        
        # Convert list to DataFrame for download
        result_df = pd.DataFrame(st.session_state['stream_data_list'])
        st.dataframe(result_df.tail(3)) # Show last few processed

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Stream Results CSV",
            data=csv,
            file_name="stream_simulation_results.csv",
            mime="text/csv"
        )