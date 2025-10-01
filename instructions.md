# How to Gather Your Own Data for KobunFormer

Due to copyright restrictions, this project does not include the linguistic data or classical literary texts required for the agent to function at its best. You will need to provide your own data by creating two directories in the project root: `data/` and `works/`.

This guide will help you gather and structure the necessary information.

---

## 1. Linguistic Data (in the `data/` directory)

The agent's core knowledge of Classical Japanese grammar comes from JSON files you place in the `data/` directory. The system dynamically loads **all `.json` files** from this folder, so you can add as many as you like.

### What to Collect

Your goal is to create a knowledge base of grammatical rules, vocabulary, and linguistic patterns. Here are some essential topics to research and include in your JSON files:

-   **Particles (助詞):** Case particles, binding particles, final particles, etc. This is one of the most critical areas.
-   **Verb Conjugations (活用):** The different stems and forms for various verb types.
-   **Auxiliary Verbs (助動詞):** Their meanings, conjugations, and how they attach to other words.
-   **Honorifics (敬語):** Rules for Sonkeigo (尊敬語), Kenjōgo (謙譲語), and Teineigo (丁寧語). You can create separate files for honorific verbs and non-inflected honorifics.
-   **Essential Grammatical Rules:** Create dedicated files for complex but important rules. For example, **Kakari-Musubi (係り結び)** is essential for authentic classical Japanese and should be well-documented in its own JSON file.
-   **Vocabulary:** Lists of classical vocabulary with their modern equivalents.

### How to Format the JSON

The system is flexible, but it's best to use a clear, descriptive structure. The LLM will interpret the raw JSON, so make your keys and values easy to understand.

**Example: A simple JSON for a particle**

You could create a file named `particles_ga.json`:

```json
{
  "particle": "が",
  "type": "Case Particle (格助詞)",
  "functions": [
    {
      "function_name": "Subject Marker",
      "modern_equivalent": "が",
      "explanation": "Marks the subject of a sentence. In classical Japanese, this was less common than 'の' for marking the subject.",
      "example": "風が吹く"
    },
    {
      "function_name": "Attributive Marker (possessive)",
      "modern_equivalent": "の",
      "explanation": "Connects two nouns where the first modifies the second, similar to the modern 'の'.",
      "example": "我が庵は都のたつみ"
    }
  ]
}
```

---

## 2. Literary Works (in the `works/` directory)

The agent uses full texts of classical literature to learn authentic stylistic patterns, rhythm, and tone. These texts are used by the RAG (Retrieval-Augmented Generation) system.

### What to Collect

Collect plain text (`.txt`) files of classical Japanese literature. The more variety you provide, the better the agent's stylistic range will be. Good sources for public domain texts include:

-   **Aozora Bunko (青空文庫):** A digital library of Japanese books.

Recommended starter texts:
-   `源氏物語` (The Tale of Genji)
-   `竹取物語` (The Tale of the Bamboo Cutter)
-   `伊勢物語` (The Tales of Ise)
-   `枕草子` (The Pillow Book)

### How to Format the Text Files

-   Ensure the files are saved as plain text (`.txt`).
-   Use UTF-8 encoding.
-   The content should be just the literary work itself.

---

## Summary of Directory Structure

After gathering your data, your project's root directory should look like this:

```
KobunFormer4/
├── data/
│   ├── particles.json
│   ├── honorifics.json
│   ├── kakari_musubi.json
│   └── (etc...)
├── works/
│   ├── genji_monogatari.txt
│   ├── taketori_monogatari.txt
│   └── (etc...)
├── src/
│   └── ...
├── main.py
├── instructions.md
└── ...
```

By providing your own high-quality data, you will enable KobunFormer to generate authentic and nuanced Classical Japanese.
