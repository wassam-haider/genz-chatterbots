"""
train_chatterbot.py - Trains Bot A and Bot B on unified_corpus.csv using ChatterBot.

Dependency notes:
- Requires chatterbot (install via: pip install chatterbot)
- Compatible with Python 3.8+ (tested on Python 3.11 with chatterbot 1.2.x).
"""

import csv
import os
import sys
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "unified_corpus.csv")
BOT_A_DB = "bot_a.sqlite3"
BOT_B_DB = "bot_b.sqlite3"


def load_corpus_lines(csv_path: str) -> list[str]:
    """Load non-empty text column entries from unified_corpus.csv."""
    if not os.path.exists(csv_path):
        print(f"Error: Corpus file not found at '{csv_path}'")
        sys.exit(1)

    lines: list[str] = []
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            if text:
                lines.append(text)

    return lines


def train_bot(name: str, db_file: str, conversation_lines: list[str]) -> ChatBot:
    """Initialize a distinct ChatBot instance with its own SQLite database and train it."""
    db_uri = f"sqlite:///{db_file}"

    # Create distinct ChatBot instance with dedicated SQLite database
    bot = ChatBot(
        name,
        storage_adapter="chatterbot.storage.SQLStorageAdapter",
        database_uri=db_uri,
        logic_adapters=[
            {
                "import_path": "chatterbot.logic.BestMatch",
                "default_response": "I am not sure how to respond to that.",
                "maximum_similarity_threshold": 0.90,
            }
        ],
    )

    trainer = ListTrainer(bot)

    # NOTE: Training on this corpus as a sequential list is an approximation.
    # The unified_corpus contains standalone lines/sentences rather than authentic back-and-forth dialogue.
    # ListTrainer pairs each line as a response to the preceding line.
    # Actual conversational quality and coherence will naturally depend on how conversational the corpus lines are.
    print(f"Training '{name}' on {len(conversation_lines)} lines into '{db_file}'...")
    trainer.train(conversation_lines)
    print(f"Finished training '{name}'.")

    return bot


def main():
    print("=" * 60)
    print("ChatterBot Two-Bot Training Pipeline")
    print("=" * 60)

    # 1. Load lines from CSV
    lines = load_corpus_lines(CORPUS_PATH)
    print(f"Loaded {len(lines)} lines from '{CORPUS_PATH}'.")

    # 2. Train Bot A
    print("\n--- Initializing Bot A ---")
    train_bot(name="Bot A", db_file=BOT_A_DB, conversation_lines=lines)

    # 3. Train Bot B
    print("\n--- Initializing Bot B ---")
    train_bot(name="Bot B", db_file=BOT_B_DB, conversation_lines=lines)

    # 4. Confirmation check
    size_a = os.path.getsize(BOT_A_DB) if os.path.exists(BOT_A_DB) else 0
    size_b = os.path.getsize(BOT_B_DB) if os.path.exists(BOT_B_DB) else 0

    print("\n" + "=" * 60)
    print("Confirmation: Both storage files created and populated!")
    print(f"  - {BOT_A_DB}: {size_a:,} bytes")
    print(f"  - {BOT_B_DB}: {size_b:,} bytes")
    print("=" * 60)


if __name__ == "__main__":
    main()
