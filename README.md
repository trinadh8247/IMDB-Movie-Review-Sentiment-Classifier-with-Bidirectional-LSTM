# IMDB Movie Review Sentiment Classifier

A professional-grade Bidirectional LSTM sentiment classifier for IMDB reviews with an interactive Streamlit web application.

## Overview
- **Model**: Bidirectional LSTM with Dropout regularization
- **Dataset**: Full IMDB dataset (75,000 reviews)
- **Accuracy**: ~91% validation accuracy
- **Framework**: TensorFlow/Keras
- **UI**: Streamlit with custom CSS styling and session state management

## Project Layout
- [main.py](main.py) — Professional Streamlit web app with gradient styling and quick-test buttons
- [simplernn.ipynb](simplernn.ipynb) — Training notebook with full dataset, early stopping, and LR scheduling
- [imdb_rnn_model.h5](imdb_rnn_model.h5) — Saved trained Keras model
- [requirements.txt](requirements.txt) — Python dependencies
- [.gitignore](.gitignore) — Git exclusions for venv, cache, and logs
- [README.md](README.md) — This file

## Features

✨ **Professional UI:**
- Gradient-styled sentiment cards (green for positive, orange for negative)
- Real-time confidence scores with progress bars
- Word/character count metrics
- Quick-test buttons for sample reviews
- Session state management for smooth UX

🚀 **Robust Architecture:**
- Bidirectional LSTM layers for context understanding from both directions
- Dropout (50%, 30%) to prevent overfitting
- Early stopping with best weight restoration
- Dynamic learning rate reduction on validation plateau
- Trained on full 75,000 samples

🛡️ **Handling Edge Cases:**
- OOV (out-of-vocabulary) word mapping to avoid embedding errors
- Data type conversion (NumPy float32 → Python float for Streamlit compatibility)
- Input validation and informative error messages
- Graceful handling of sarcasm, negation, and mixed sentiment

## Quickstart (Local)

1. **Clone and setup:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or: source .venv/bin/activate  # Mac/Linux
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run main.py
   ```

3. **Open in browser:** Navigate to `http://localhost:8501`

## Training / Updating the Model

1. Open [simplernn.ipynb](simplernn.ipynb) in Jupyter
2. Run all cells sequentially
3. The notebook:
   - Loads full IMDB train + test data (75K samples total)
   - Analyzes review lengths for optimal padding (300 tokens)
   - Trains Bidirectional LSTM with Dropout
   - Uses early stopping (patience=3) and LR scheduling (ReduceLROnPlateau)
   - Saves model as `imdb_rnn_model.h5` and history as `training_history.json`
   - Plots accuracy and loss curves

4. **Important:** If you change `max_features` or `maxlen` in training, update the same constants in [main.py](main.py) (lines 13-14)

## Deployment (Streamlit Cloud)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy sentiment classifier"
   git push origin main
   ```

2. **Create Streamlit Cloud app:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app" → Select repo, branch, and [main.py](main.py)
   - Deploy!

3. **Note:** Ensure `imdb_rnn_model.h5` is committed. If too large, host on cloud storage.

## Test Sentences

**Positive Reviews:**
- "This movie was absolutely outstanding! A true masterpiece."
- "I laughed, I cried, I was completely blown away. Highly recommend!"
- "Phenomenal acting and a gripping storyline kept me on the edge of my seat."
- "One of the best films I've ever watched. Pure brilliance!"
- "Incredible direction and stunning cinematography. A must-watch!"

**Negative Reviews:**
- "Complete waste of time and money. Absolutely dreadful."
- "Boring, predictable, and poorly executed. I fell asleep halfway through."
- "The worst movie I've ever seen. Terrible acting and a ridiculous plot."
- "Painfully bad. I couldn't even finish watching it."
- "Utterly disappointing. Nothing redeeming about this film at all."

**Edge Cases (Sarcasm, Negation, Mixed):**
- "I thought I would hate it, but I didn't." (Positive negation)
- "Wow, what a thrilling experience of staring at a blank screen for two hours." (Sarcasm)
- "The first half was great, but it fell apart in the second half." (Mixed)
- "Not bad, not great, just... there." (Neutral)
- "Loved the performances, but the plot was a mess." (Mixed)

## Model Architecture

**Embedding Layer:**
- Vocabulary size: 10,000 words
- Embedding dimension: 128
- Input sequence length: 300

**Bidirectional LSTM Layers:**
1. BiLSTM(64, return_sequences=True) → Dropout(0.5)
2. BiLSTM(32) → Dropout(0.5)

**Dense Layers:**
1. Dense(64, activation='relu') → Dropout(0.3)
2. Dense(1, activation='sigmoid') — Binary classification output

**Optimization:**
- Optimizer: Adam (learning_rate=0.001)
- Loss: Binary Crossentropy
- Early Stopping (patience=3, restore_best_weights=True)
- ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-7)

## Troubleshooting

**OOV / Embedding Index Errors:**
- Words not in IMDB vocab automatically map to OOV token (index 2)
- Tokens beyond max_features (10,000) are clamped to prevent overflow
- All inputs padded to length 300 for consistency

**Type Errors with st.progress():**
- Confidence scores (NumPy float32) are automatically converted to Python float
- Prevents `StreamlitAPIException` in progress bar rendering

**Slow Model Loading:**
- TensorFlow/Keras initialization on first run may take 10-20 seconds
- Subsequent predictions are faster (~1-2 seconds)

**Memory Issues:**
- Use `tensorflow-cpu` instead of `tensorflow` to reduce memory
- Update [requirements.txt](requirements.txt) if needed

## Performance Metrics
- **Training Accuracy**: ~92-94%
- **Validation Accuracy**: ~91%
- **Training Time**: ~5-10 minutes (local) / ~2-3 minutes (Colab GPU)
- **Inference Time**: ~1-2 seconds per review

## Future Improvements
- Add attention mechanism for interpretability
- Integrate LIME/SHAP for feature attribution
- Support multi-class sentiment (1-5 stars)
- Fine-tune on domain-specific reviews
- Deploy model separately (FastAPI) for scalability
