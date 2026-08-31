"""
Gemini LLM Integration
======================

Cloud LLM integration using Google's Gemini API.

Architecture:

    User Question
          |
          v
    ChromaDB Retriever
          |
          v
    Retrieved Context
          |
          v
       Gemini
          |
          v
    Grounded Answer

Task 3:
GPT4All + Gemini + ChromaDB RAG Platform
"""

import os
from typing import Optional

from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

# Current Gemini model recommended by the API
GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# GEMINI LLM
# ============================================================

class GeminiLLM:
    """
    Wrapper around Google's Gemini Interactions API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):

        print("=" * 70)
        print("GEMINI CLOUD LLM")
        print("=" * 70)

        self.api_key = (
            api_key
            or GEMINI_API_KEY
        )

        # ----------------------------------------------------
        # Validate API key
        # ----------------------------------------------------

        if not self.api_key:

            raise ValueError(
                "GEMINI_API_KEY is not configured.\n\n"
                "Create a .env file in the project root:\n\n"
                "GEMINI_API_KEY=your_api_key"
            )

        print(
            f"Gemini model: {GEMINI_MODEL}"
        )

        print(
            "Connecting to Gemini..."
        )

        # ----------------------------------------------------
        # Create Gemini client
        # ----------------------------------------------------

        self.client = genai.Client(
            api_key=self.api_key
        )

        print(
            "Gemini client configured successfully."
        )

        print("=" * 70)

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        """
        Generate a response using Gemini.

        Args:
            prompt:
                Prompt sent to Gemini.

            temperature:
                Generation temperature.

            max_tokens:
                Maximum output tokens.

        Returns:
            Generated response text.
        """

        if not prompt or not prompt.strip():

            raise ValueError(
                "Prompt cannot be empty."
            )

        # ----------------------------------------------------
        # Gemini Interactions API
        # ----------------------------------------------------

        interaction = self.client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt,
        )

        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        response_text = (
            interaction.output_text
        )

        if not response_text:

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response_text.strip()


# ============================================================
# SIMPLE TEST
# ============================================================

def main():

    print("=" * 70)
    print("GEMINI LLM TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Gemini
    # --------------------------------------------------------

    llm = GeminiLLM()

    # --------------------------------------------------------
    # Test question
    # --------------------------------------------------------

    question = (
        "Explain Retrieval Augmented Generation "
        "in simple terms."
    )

    print("\nQuestion:")
    print(question)

    print(
        "\nGenerating Gemini response..."
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    response = llm.generate(
        prompt=question,
        temperature=0.0,
        max_tokens=300,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("GEMINI RESPONSE")
    print("-" * 70)

    print(response)

    print("-" * 70)

    print(
        "\nGemini test completed successfully."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()