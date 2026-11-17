"""
serve_chatterbot.py - FastAPI service for the two-bot conversation simulator.

NOTE ON CONVERSATION QUALITY & REPETITION DETECTION:
ChatterBot operates by selecting statements from its SQLite storage based on nearest-match
retrieval (Levenshtein distance / spaCy embeddings). When trained on a non-dialogue corpus
(isolated sentences rather than authored Q&A or dialogue transcripts), responses reflect
shallow best-match retrieval rather than semantically coherent or context-aware conversation.
Because nearest-match retrieval can readily fall into deterministic feedback loops where bots
echo each other or repeat their own prior statement, the repetition-detection stop condition
exists specifically to identify and terminate such loops gracefully.
"""

import os
import re
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from chatterbot import ChatBot

app = FastAPI(
    title="ChatterBot Two-Bot Conversation Simulator",
    description="Orchestrates turn-by-turn dialogue between Bot A and Bot B.",
    version="1.0.0",
)

# Enable CORS for browser access from file:// or any local frontend server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BOT_A_DB = "bot_a.sqlite3"
BOT_B_DB = "bot_b.sqlite3"

bot_a: ChatBot | None = None
bot_b: ChatBot | None = None


def normalize_text(text: str) -> str:
    """Normalize text for near-identical equality checking (strips punctuation and excessive whitespace)."""
    # Remove all punctuation and symbols, normalize whitespace to single spaces, lowercase
    cleaned = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


@app.on_event("startup")
def load_bots():
    """Load both trained ChatBot instances from their respective SQLite storage files on startup."""
    global bot_a, bot_b

    db_a_exists = os.path.exists(BOT_A_DB)
    db_b_exists = os.path.exists(BOT_B_DB)

    if not db_a_exists or not db_b_exists:
        print(
            "WARNING: Storage databases bot_a.sqlite3 and/or bot_b.sqlite3 were not found."
            " Please run 'python train_chatterbot.py' before calling /converse."
        )

    # Initialize Bot A (read_only=True prevents runtime mutation during simulation)
    bot_a = ChatBot(
        "Bot A",
        storage_adapter="chatterbot.storage.SQLStorageAdapter",
        database_uri=f"sqlite:///{BOT_A_DB}",
        read_only=True,
    )

    # Initialize Bot B
    bot_b = ChatBot(
        "Bot B",
        storage_adapter="chatterbot.storage.SQLStorageAdapter",
        database_uri=f"sqlite:///{BOT_B_DB}",
        read_only=True,
    )

    print("Successfully initialized Bot A and Bot B instances.")


class ConverseRequest(BaseModel):
    seed: str = Field(..., min_length=1, description="Initial conversation prompt or question.")
    max_turns: int = Field(default=10, ge=1, le=20, description="Maximum number of turns (hard-capped at 20).")


class TurnItem(BaseModel):
    sender: Literal["Bot A", "Bot B"]
    text: str


class ConverseResponse(BaseModel):
    seed: str
    turns: list[TurnItem]
    stop_reason: Literal["max_turns_reached", "repetition_detected"]
    total_turns: int


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "ChatterBot Two-Bot Simulator",
        "endpoints": {"converse": "/converse"},
    }


@app.post("/converse", response_model=ConverseResponse)
def converse(payload: ConverseRequest):
    """
    Run an alternating conversation between Bot A and Bot B up to max_turns.
    Stops early if a bot's response is identical or near-identical to what it received
    or to its own previous response.
    """
    if bot_a is None or bot_b is None:
        raise HTTPException(status_code=500, detail="Bots are not initialized.")

    if not os.path.exists(BOT_A_DB) or not os.path.exists(BOT_B_DB):
        raise HTTPException(
            status_code=400,
            detail="Trained bot databases not found. Please execute 'python train_chatterbot.py' first.",
        )

    seed_message = payload.seed.strip()
    if not seed_message:
        raise HTTPException(status_code=400, detail="Seed message cannot be empty.")

    # Hard cap at 20
    max_turns = min(max(1, payload.max_turns), 20)

    turns: list[TurnItem] = []
    stop_reason: Literal["max_turns_reached", "repetition_detected"] = "max_turns_reached"

    current_input = seed_message
    last_response: dict[str, str] = {"Bot A": "", "Bot B": ""}

    for turn_index in range(max_turns):
        # Bot A starts on turn 0, Bot B follows on turn 1, alternating thereafter
        if turn_index % 2 == 0:
            current_sender = "Bot A"
            active_bot = bot_a
        else:
            current_sender = "Bot B"
            active_bot = bot_b

        # Generate response from active bot
        response_statement = active_bot.get_response(current_input)
        response_text = str(response_statement).strip()

        norm_response = normalize_text(response_text)
        norm_received = normalize_text(current_input)
        norm_own_last = normalize_text(last_response[current_sender])

        # Check natural early-stop conditions:
        # 1. Bot echoed what it just received
        # 2. Bot repeated its own immediate prior response
        if norm_response and (norm_response == norm_received or norm_response == norm_own_last):
            turns.append(TurnItem(sender=current_sender, text=response_text))
            stop_reason = "repetition_detected"
            break

        turns.append(TurnItem(sender=current_sender, text=response_text))
        last_response[current_sender] = response_text
        current_input = response_text

    return ConverseResponse(
        seed=seed_message,
        turns=turns,
        stop_reason=stop_reason,
        total_turns=len(turns),
    )


if __name__ == "__main__":
    uvicorn.run("serve_chatterbot:app", host="127.0.0.1", port=8000, reload=False)
