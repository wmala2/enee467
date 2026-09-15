"""Lets you type prompts to a local LLM and read its replies in the terminal.

Ollama runs the model on your own machine -- no API key, no network call, no per-token cost --
and this class just wraps its chat API so the conversation is remembered across turns.
"""

# External Libraries
import ollama


class LLMChat:
    """A terminal chat session with one local Ollama model, with memory across turns."""

    def __init__(self, model="llama3.2:3b", keep_alive="10m"):
        # The ollama model tag we want to talk to (llama3.2:3b is a good ~3B default)
        self.model = model

        # Keep the model loaded in memory between prompts so replies stay fast
        self.keep_alive = keep_alive

        # Download the model the first time, so the user never has to run `ollama pull`
        self._ensure_model_downloaded()

        # Remember the conversation so the model has context across prompts
        self.messages = []

    def _ensure_model_downloaded(self):
        # Ask the ollama server what it has, and pull our model if it's missing
        downloaded = [m.model for m in ollama.list().models]
        if not any(name.startswith(self.model) for name in downloaded):
            print(f"Downloading {self.model} (one time only, this can take a few minutes)...")
            ollama.pull(self.model)

    def prompt(self, text):
        """Sends one prompt to the model and returns its reply as a string."""
        # Add the user's words to the running conversation
        self.messages.append({"role": "user", "content": text})

        # Ask the model for a reply
        response = ollama.chat(
            model=self.model,
            messages=self.messages,
            keep_alive=self.keep_alive,
        )

        # Remember the reply too, so follow-up questions make sense
        reply = response.message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def clear_history(self):
        # Forget the conversation and start fresh
        self.messages = []


# Simple interactive demo: type prompts, get answers, Ctrl-C or empty line to quit
if __name__ == "__main__":
    chat = LLMChat()
    print(f"Chatting with {chat.model} - press Enter on an empty line to quit")
    while True:
        try:
            text = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text:
            break
        print(f"llm> {chat.prompt(text)}")
