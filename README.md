# ChatterBot Two-Bot Dialogue Simulator

A turn-by-turn dialogue simulator between two independently trained ChatterBot instances (`Bot A` and `Bot B`), trained sequentially on `unified_corpus.csv`, served via a FastAPI backend, and displayed in a WhatsApp-styled HTML interface.

---

## Prerequisites & Environment

Make sure your virtual environment (`.venv`) is activated:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 1. Install Dependencies

Install required packages:

```powershell
pip install -r requirements.txt
```

> **Note on ChatterBot compatibility**:
> ChatterBot `1.2.x` supports Python 3.8 through modern versions. If spaCy requires its English language model, download it with:
> ```powershell
> python -m spacy download en_core_web_sm
> ```

---

## 2. Train the Bots (Run Once)

Run the sequential training script to populate separate SQLite databases for Bot A and Bot B (`bot_a.sqlite3` and `bot_b.sqlite3`):

```powershell
python train_chatterbot.py
```

*Note: Training sequentially on a list of standalone sentences is an approximation. ChatterBot pairs each statement with the line preceding it. Response quality will reflect nearest-match retrieval over this corpus.*

---

## 3. Run the Backend API

Start the FastAPI server:

```powershell
python serve_chatterbot.py
```

The server runs at `http://127.0.0.1:8000` with CORS enabled.

---

## 4. Open the Chat Interface

Double-click or open [chat.html](file:///c:/Users/farha/OneDrive/Desktop/nlp%20project/chatterbot/chat.html) in any web browser:

1. Enter an initial prompt / seed message (e.g., `"Got the job today, big W!"`).
2. Set the maximum turns (1–20).
3. Click **Start Chat**.
4. Watch Bot A and Bot B alternate turns with animated typing indicators until the turn limit is reached or repetition is detected.
