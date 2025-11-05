# AI Text Generator with Sentiment Analysis

✨ Generate emotion-aligned paragraphs using AI.  

This Streamlit app analyzes the sentiment of your input text (Positive, Negative, Neutral) and generates text aligned with that emotion using GPT-2.  

---

## Features

- **Sentiment Analysis**: Detects if your input text is positive, negative, or neutral.
- **Emotion-Aligned Text Generation**: Generates text that matches the detected or selected sentiment.
- **Adjustable Output Length**: Control how long the generated paragraph is.
- **Download Generated Text**: Easily save the generated content as a `.txt` file.
- **Interactive Streamlit Interface**: Easy-to-use frontend with examples and live generation.

---

## How It Works

1. **Input Analysis**: User enters a topic or statement.  
2. **Sentiment Detection**: The app detects the sentiment of the text using `cardiffnlp/twitter-roberta-base-sentiment-latest`.  
3. **Content Generation**: GPT-2 generates a paragraph aligned with the sentiment.  
4. **Output Delivery**: Generated text is displayed and can be downloaded.  

---

## Tech Stack

- **Frontend & UI**: Streamlit
- **AI Models**: 
  - Sentiment Analysis: `cardiffnlp/twitter-roberta-base-sentiment-latest`  
  - Text Generation: GPT-2 (Hugging Face Transformers)
- **Python Libraries**: `transformers`, `torch`, `streamlit`, `time`, `os`, `sys`

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Ake544/ai_text_generator.git
cd https://github.com/Ake544/ai_text_generator.git
