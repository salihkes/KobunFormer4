# KobunFormer

KobunFormer is a dynamic LLM workflow designed to transform modern languages into authentic Classical Japanese (古文, *kobun*).

It's not just a direct translator. It's a multi-step agent that uses Retrieval-Augmented Generation (RAG) and a dynamic, user-provided knowledge base to generate stylistically appropriate Classical Japanese and *Kundoku* (訓読) readings of Classical Chinese.

### A Note on Authenticity

Generating perfect Classical Japanese is an incredibly complex task that even human experts find challenging. This project is an experimental workflow that **attempts** to form coherent, grammatically informed, and stylistically aware sentences in *kobun*. It should not be expected to produce flawless or academically perfect results. Always consult with an expert for scholarly or professional use.

### Project Motivation

During my individual testing of local large language models, i observed that local language models, even powerful ones, often struggle to generate authentic Classical Japanese from a single prompt. They tend to miss nuanced grammatical rules, lack correct stylistic voice, and fail to use appropriate classical vocabulary. This was especially an issue in Kakari Musubi.

This project was created to address this challenge by unifying several techniques into a structured workflow that guides the LLM toward a better result. The core ideas are:

*   **Decomposition:** The complex task of translation is broken down into smaller, more manageable steps (e.g., Foreign -> Modern Japanese -> Classical Japanese).
*   **Contextual Grounding (RAG):** Instead of relying on the model's internal knowledge alone, the agent retrieves relevant information at runtime. It uses two distinct RAG systems:
    1.  A **Linguistic RAG** that pulls from a knowledge base of grammatical rules (your `data/` directory).
    2.  A **Stylistic RAG** that sources examples from classical literary texts (your `works/` directory) to guide the tone and rhythm.
*   **Dynamic Prompt Engineering:** The final prompt sent to the LLM is dynamically assembled with the retrieved linguistic rules, stylistic examples, custom vocabulary, and specific instructions, providing rich context.
*   **LLM-as-Analyst:** The workflow uses the LLM not just for generation but also for intermediate reasoning steps, like analyzing the input to create targeted search queries for the RAG system.
*   **Self-Validation:** The agent includes a final validation step where the LLM is asked to critically assess its own output against a set of quality criteria, providing a score and feedback.

By combining these techniques, KobunFormer aims to provide a more robust and reliable method for generating Classical Japanese than simple, direct prompting.

## Features

-   **Dynamic Transformation Pipeline:** Follows a logical workflow (e.g., Foreign -> Modern Japanese -> Kobun) to ensure contextual accuracy.
-   **Retrieval-Augmented Generation (RAG):** Uses classical literary works from the `works/` directory for stylistic context and linguistic rules from the `data/` directory for grammatical accuracy.
-   **Dynamic Knowledge Base:** Automatically loads and uses all user-provided JSON files from the `data/` directory, allowing for easy extension of its linguistic knowledge.
-   **Multiple Transformation Modes:**
    -   Foreign Language to Classical Japanese
    -   Modern Japanese to Classical Japanese
    -   Classical Chinese to Japanese *Kundoku* reading
-   **Streaming Output:** Parses XML tags from the LLM's response in real-time to provide a structured, streaming output.
-   **Interactive & Single-Shot CLI:** Can be run in an interactive loop or used to translate a single sentence from the command line.

---

## IMPORTANT: Providing Your Own Data

This project **does not** come with the necessary data to run. Due to copyright, you must provide your own linguistic data and literary texts.

Please follow the detailed guide here: **[instructions.md](./instructions.md)**

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd KobunFormer4
    ```

2.  **Create and populate data directories:**
    -   Create a `data/` directory and fill it with your own linguistic JSON files.
    -   Create a `works/` directory and fill it with your own classical `.txt` files.
    -   See **[instructions.md](./instructions.md)** for details.

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **LLM Backend:**
    This workflow requires a running LLM backend. It is configured to work with either:
    -   **Ollama:** For running local models inside your local computer.
    -   **OpenAI-compatible API:** Such as a local `llama.cpp` server, or a remote one. The default URL is `http://localhost:8080/v1`, but you can edit this in `main.py` if your server is hosted elsewhere.

---

## Usage

The script can be run from the command line in single-shot or interactive mode.

### Single-Shot Mode

Provide the sentence to translate as a command-line argument.

**Foreign to Kobun:**
```bash
python main.py --ollama "Hello, how are you today?"
```

**Modern Japanese to Kobun:**
```bash
python main.py --openai --japanese "こんにちは、今日はお元気ですか？"
```

**Classical Chinese to Kundoku:**
```bash
python main.py --ollama --chinese "溫故而知新"
```

### Options

-   `--ollama` or `--openai`: **(Required)** Specify which LLM client to use.
-   `-j`, `--japanese`: Use this flag if your input is Modern Japanese.
-   `-c`, `--chinese`: Use this flag if your input is Classical Chinese.
-   `--anachronisms`: Allow the use of modern vocabulary if no classical equivalent exists.
-   `--keigo`: Enforce the use of honorifics (*keigo*).

### Interactive Mode

To start an interactive session, run the script without a sentence argument.

```bash
python main.py --ollama --japanese
```

The agent will then prompt you for input repeatedly.

---

## LLM Model Recommendations

The effectiveness of the transformation can vary significantly based on the LLM you use. Here are some observations:

-   **For Nuanced Grammar (like *Kakari-Musubi*)**: Reasoning-focused models, such as the default **`Qwen3:32b`**, have been observed to perform best. These models are adept at applying the specific, complex grammatical rules required for authentic *kobun*.

-   **For *Kundoku* (書き下し文)**: Extremely large models, such as **`Qwen3-235B-A22B-2507`**, often provide better results for generating *kakikudashibun* (the full written-out form of a *kundoku* reading). Their broader knowledge base seems to aid in producing more natural and contextually appropriate readings of Classical Chinese.

---

## License

This project is released under the **MIT License**. See the [LICENSE](./LICENSE) file for details.
