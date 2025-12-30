# IMDB Movie Review Sentiment Classifier

Bidirectional LSTM sentiment classifier for IMDB reviews with a Streamlit front end.

## Project Layout
- [main.py](main.py) — Streamlit app for inference using the saved model.
- [simplernn.ipynb](simplernn.ipynb) — training notebook (Bidirectional LSTM + callbacks).
- [imdb_rnn_model.h5](imdb_rnn_model.h5) — saved Keras model used by the app.
- [embedding.ipynb](embedding.ipynb) / [prediction.ipynb](prediction.ipynb) — auxiliary experiments.
- [requirements.txt](requirements.txt) — pinned dependencies.

## Quickstart (local)
1) Create a virtual environment (Python 3.11 recommended) and activate it.
2) Install dependencies: `pip install -r requirements.txt`
3) Run the app: `streamlit run main.py`
4) Open the Streamlit URL shown in the terminal, type a review, and click **Predict Sentiment**.

## Training / Updating the Model
- Open [simplernn.ipynb](simplernn.ipynb) and run all cells.
- The notebook trains on the full IMDB dataset with Bidirectional LSTM, early stopping, and LR scheduling.
- It saves the model as [imdb_rnn_model.h5](imdb_rnn_model.h5). Keep this file alongside [main.py](main.py) for inference.
- If you change `max_features` or `maxlen` in training, update the same constants in [main.py](main.py).

## Deployment (Streamlit Cloud)
1) Push this repo (including [imdb_rnn_model.h5](imdb_rnn_model.h5)) to GitHub.
2) Create a new Streamlit Cloud app pointing to [main.py](main.py).
3) The platform installs from [requirements.txt](requirements.txt) automatically. Use Python 3.11.
4) If the model is too large for the repo, host it (e.g., cloud storage) and download it at startup.

## Test Reviews (edge cases)
- Negative: "Loved the performances, but the plot was a mess and dragged forever."
- Negative (sarcasm): "Wow, another masterpiece… if you enjoy watching paint dry."
- Positive (negation): "I thought I would hate it, but I didn’t."
- Neutral/Negative: "It’s not that it wasn’t good; it just wasn’t great."
- Neutral: "It was a movie. I sat there. Stuff happened."
- Negative: "THIS MOVIE IS TERRIBLE!"
- Positive: "Best movie ever 😍🔥" (emoji-friendly)

## Troubleshooting
- **OOV / vocab errors**: Inputs are clamped to the IMDB vocab; unseen words map to the OOV token to avoid embedding index errors.
- **Version mismatches**: Match TensorFlow version in [requirements.txt](requirements.txt). If you prefer CPU-only, replace `tensorflow` with `tensorflow-cpu`.
- **Slow startup on first run**: TensorFlow may take a moment to initialize.
