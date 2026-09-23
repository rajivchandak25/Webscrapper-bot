from bot import QABot

# -------------------------------
# Interactive CLI Entrypoint
# -------------------------------
if __name__ == "__main__":
    bot = QABot()
    print("Q&A Bot ready! Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Bot: Goodbye!")
                break
            answer = bot.ask(query)
            print("\nBot:\n" + answer + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break