import os
import flask
from flask import Flask, request, jsonify
app = Flask(__name__)

# Initialize models in session state
if 'models_loaded' not in app.config:
    app.config['models_loaded'] = False
    app.config['sentiment_pipeline'] = None
    app.config['text_generator'] = None

# Load models if not already loaded
if not app.config['models_loaded']:
    # Load sentiment model
    from transformers import pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None
    )
    app.config['sentiment_pipeline'] = sentiment_pipeline
    # Load text generator
    from transformers import pipeline
    text_generator = pipeline(
        "text-generation",
        model="gpt2",
        pad_token_id=50256
    )
    app.config['text_generator'] = text_generator
    app.config['models_loaded'] = True

# Define a route for the main page
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # Get the user's input
        user_prompt = request.form['user_prompt']
        # Analyze sentiment
        sentiment, confidence, detailed_results = analyze_sentiment_comprehensive(user_prompt, app.config['sentiment_pipeline'])
        # Generate text
        generated_text = generate_emotion_aligned_text(user_prompt, sentiment, confidence, app.config['text_generator'])
        # Return the result
        return jsonify({'generated_text': generated_text})
    else:
        # Return the main page
        return "<html><body><h1>AI Text Generator with Sentiment</h1><form method='post'><input type='text' name='user_prompt' placeholder='Enter your text'><input type='submit' value='Generate'></form></body></html>"

# Define a function to analyze sentiment
def analyze_sentiment_comprehensive(text, sentiment_pipeline):
    try:
        results = sentiment_pipeline(text)[0]
        best_sentiment = max(results, key=lambda x: x['score'])
        label_mapping = {
            "positive": "POSITIVE",
            "negative": "NEGATIVE", 
            "neutral": "NEUTRAL",
            "lab_0": "NEGATIVE",
            "lab_1": "NEUTRAL",
            "lab_2": "POSITIVE"
        }
        final_sentiment = label_mapping.get(
            best_sentiment['label'], 
            best_sentiment['label'].upper()
        )
        return final_sentiment, best_sentiment['score'], results
    except Exception as e:
        return "NEUTRAL", 0.5, []

# Define a function to generate text
def generate_emotion_aligned_text(prompt, sentiment, confidence, text_generator, length=150):
    import random
    natural_prompts = {
        "POSITIVE": [
            f"Write a positive paragraph about: {prompt}. Stay on topic and do not mention anything else.",
            f"Explain why {prompt} is positive in a short paragraph, staying fully on topic."
        ],
        "NEGATIVE": [
            f"Write a negative paragraph about: {prompt}. Stay on topic and do not mention anything else.",
            f"Explain why {prompt} is negative in a short paragraph, staying fully on topic."
        ],
        "NEUTRAL": [
            f"Write a neutral paragraph about: {prompt}. Stay on topic and do not mention anything else.",
            f"Describe {prompt} neutrally in a short paragraph."
        ]
    }
    final_prompt = random.choice(natural_prompts[sentiment])
    generated_output = text_generator(
        final_prompt,
        max_new_tokens=int(length * 1.3),
        num_return_sequences=1,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=2.0,
        pad_token_id=50256
    )
    full_text = generated_output[0]['generated_text'].replace(final_prompt, "").strip()
    words = full_text.split()
    if len(words) > length:
        words = words[:length]
    full_text = " ".join(words)
    if full_text and full_text[-1] not in ['.', '!', '?']:
        full_text += '.'
    if full_text and full_text[0].islower():
        full_text = full_text[0].upper() + full_text[1:]
    return full_text


#test commit

if __name__ == '__main__':
    app.run(debug=True)