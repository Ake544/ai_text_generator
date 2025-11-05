import streamlit as st
import time
import os
import sys


st.set_page_config(
    page_title="AI Text Generator with Sentiment",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: #00C851;
        font-weight: bold;
        font-size: 1.3rem;
        background-color: #E8F5E8;
        padding: 8px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    .sentiment-negative {
        color: #ff4444;
        font-weight: bold;
        font-size: 1.3rem;
        background-color: #FFE8E8;
        padding: 8px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    .sentiment-neutral {
        color: #ffbb33;
        font-weight: bold;
        font-size: 1.3rem;
        background-color: #FFF8E1;
        padding: 8px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    .generated-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        font-size: 1.1rem;
        line-height: 1.7;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING - CLEAN IMPLEMENTATION
# ============================================================================

def load_sentiment_model():
    """
    Load sentiment analysis model with error handling
    """
    try:
        # Import inside function to avoid circular imports
        from transformers import pipeline
        
        with st.spinner("🔄 Loading Sentiment Analysis Model..."):
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=None
            )
        st.success("✅ Sentiment Analysis Model Loaded!")
        return sentiment_pipeline
    except Exception as e:
        st.error(f"❌ Failed to load sentiment model: {e}")
        return None

def load_text_generator():
    """
    Load text generation model with fallbacks
    """
    try:
        from transformers import pipeline
        
        with st.spinner("🔄 Loading Text Generation Model..."):
            # Try base GPT-2 first which is more stable on streamlit cloud
            text_generator = pipeline(
                "text-generation",
                model="gpt2",
                pad_token_id=50256
            )
        st.success("✅ Text Generation Model Loaded!")
        return text_generator
        
    except Exception as e:
        st.warning(f"⚠️ GPT-2 failed: {e}")
        
        try:
            # Fallback to custom implementation
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            import torch
            
            st.info("🔄 Loading GPT-2 with custom setup...")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            def custom_generator(prompt, **kwargs):
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=kwargs.get('max_length', 200),
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        repetition_penalty=1.3,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return [{"generated_text": generated_text}]
            
            st.success("✅ Custom GPT-2 Model Loaded!")
            return custom_generator
            
        except Exception as e2:
            st.error(f"❌ All text generation failed: {e2}")
            return None

# ============================================================================
# CORE FUNCTIONALITY
# ============================================================================

def analyze_sentiment_comprehensive(text, sentiment_pipeline):
    """
    Analyze sentiment with comprehensive results
    """
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
        st.error(f"Sentiment analysis error: {e}")
        return "NEUTRAL", 0.5, []

def generate_emotion_aligned_text(prompt, sentiment, confidence, text_generator, length=150):
   # GPT-2 UNDERSTANDS THESE PROMPTS - they mimic natural conversation
    natural_prompts = {
        "POSITIVE": [
            f"I think {prompt} is really great because",
            f"What I love about {prompt} is that",
            f"{prompt} makes me happy because",
            f"The best thing about {prompt} is"
        ],
        "NEGATIVE": [
            f"I'm concerned about {prompt} because", 
            f"The problem with {prompt} is that",
            f"What worries me about {prompt} is",
            f"{prompt} is difficult because"
        ],
        "NEUTRAL": [
            f"When I think about {prompt}, I notice that",
            f"{prompt} can be understood as",
            f"Looking at {prompt} objectively,",
            f"{prompt} involves"
        ]
    }
    
    import random
    final_prompt = random.choice(natural_prompts[sentiment])
    
    try:
        word_count_estimate = int(length * 1.5)  
        generated_output = text_generator(
            final_prompt,
            max_new_tokens=word_count_estimate,  
            num_return_sequences=1,
            temperature=0.4,  #To avoid off-topic generation 
            do_sample=True,
            repetition_penalty=1.8, 
            pad_token_id=50256,
            truncation=True,
            no_repeat_ngram_size=3,
            early_stopping=False
        )
        
        full_output = generated_output[0]['generated_text']
        
        if final_prompt in full_output:
            generated_text = full_output.replace(final_prompt, "").strip()
        else:
            generated_text = full_output
        
        generated_text = generated_text.split('\n')[0].strip()
        
        # Ensure it ends properly
        if generated_text and generated_text[-1] not in ['.', '!', '?']:
            generated_text += '.'
        
        return generated_text
        
    except Exception as e:
        return f"Generation failed: {str(e)}"

def clean_generated_text(text):
    """Clean and format generated text"""
    text = text.replace('<|endoftext|>', '').strip()
    
    if text and text[-1] not in ['.', '!', '?']:
        last_period = text.rfind('.')
        if last_period != -1:
            text = text[:last_period + 1]
        else:
            text = text + '.'
    
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    return text

# ============================================================================
# STREAMLIT APP INTERFACE
# ============================================================================

def main():
    # Header Section
    st.markdown('<div class="main-header">✨ AI Text Generator with Sentiment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Generate emotion-aligned paragraphs using AI</div>', unsafe_allow_html=True)
    
    # Initialize models in session state
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
        st.session_state.sentiment_pipeline = None
        st.session_state.text_generator = None
    
    # Load models if not already loaded
    if not st.session_state.models_loaded:
        with st.container():
            st.info("🔧 **Initializing AI Models...**")
            sentiment_pipeline = load_sentiment_model()
            text_generator = load_text_generator()
            
            if sentiment_pipeline and text_generator:
                st.session_state.sentiment_pipeline = sentiment_pipeline
                st.session_state.text_generator = text_generator
                st.session_state.models_loaded = True
                st.success("✅ All models loaded successfully!")
            else:
                st.error("❌ Failed to initialize models. Please check console for errors.")
                return
    
    # Main Application Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Your Text")
        user_prompt = st.text_area(
            "Describe your topic, idea, or statement:",
            "I love sunny days at the beach with friends",
            height=100,
            help="The AI will analyze sentiment and generate aligned text"
        )
        
        # Generation settings
        st.subheader("⚙️ Generation Settings")
        col_a, col_b = st.columns(2)
        
        with col_a:
            output_length = st.slider(
                "Output Length (words):",
                min_value=50,
                max_value=300,
                value=150,
                help="Maximum length of generated text"
            )
        
        with col_b:
            auto_detect = st.checkbox(
                "Auto-detect sentiment",
                value=True,
                help="Automatically detect sentiment from input"
            )
        
        if not auto_detect:
            selected_sentiment = st.radio(
                "Select sentiment:",
                ["POSITIVE", "NEGATIVE", "NEUTRAL"],
                horizontal=True
            )
        else:
            selected_sentiment = None
    
    with col2:
        st.subheader("ℹ️ How It Works")
        st.markdown("""
        <div class="info-box">
        <b>1. Input Analysis</b><br>
        AI analyzes text sentiment
        
        <b>2. Sentiment Detection</b><br>
        Identifies as Positive, Negative, or Neutral
        
        <b>3. Content Generation</b><br>
        GPT-2 generates aligned paragraphs
        
        <b>4. Output Delivery</b><br>
        Returns coherent, emotion-matched text
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("💡 Try These Examples")
        examples = {
            "🎓 University": "I just got accepted into my dream university!",
            "🌍 Climate": "The challenges of climate change are overwhelming",
            "🤖 Technology": "Artificial intelligence is transforming education",
            "💼 Career": "I lost my job and feel completely devastated",
            "🌞 Positive": "I love sunny days at the beach with friends"
        }
        
        for label, example in examples.items():
            if st.button(f"{label}", use_container_width=True):
                st.session_state.user_prompt = example
                st.rerun()
    
    # Generate Button
    st.markdown("---")
    generate_clicked = st.button(
        "🚀 GENERATE AI TEXT", 
        type="primary", 
        use_container_width=True
    )
    
    # Process generation when clicked
    if generate_clicked and user_prompt.strip():
        with st.container():
            st.subheader("🔍 Analysis & Generation Process")
            
            # Step 1: Sentiment Analysis
            with st.expander("📊 Step 1: Sentiment Analysis", expanded=True):
                if auto_detect:
                    sentiment, confidence, detailed_results = analyze_sentiment_comprehensive(
                        user_prompt, 
                        st.session_state.sentiment_pipeline
                    )
                    
                    sentiment_class = f"sentiment-{sentiment.lower()}"
                    st.markdown(f"<div class='{sentiment_class}'>🎯 Detected Sentiment: {sentiment}</div>", 
                               unsafe_allow_html=True)
                    st.write(f"**Confidence:** {confidence:.2%}")
                    
                    if detailed_results:
                        st.write("**Detailed Analysis:**")
                        for result in detailed_results:
                            label = result['label']
                            score = result['score']
                            st.write(f"- {label}: {score:.2%}")
                            
                else:
                    sentiment = selected_sentiment
                    confidence = 1.0
                    st.success(f"🎯 Using Selected Sentiment: {sentiment}")
            
            # Step 2: Text Generation
            with st.expander("✍️ Step 2: Text Generation", expanded=True):
                with st.spinner(f"🔄 Generating {sentiment.lower()} text..."):
                    start_time = time.time()
                    
                    generated_text = generate_emotion_aligned_text(
                        user_prompt,
                        sentiment,
                        confidence,
                        st.session_state.text_generator,
                        output_length
                    )
                    
                    generation_time = time.time() - start_time
                
                # Display results
                st.subheader("📖 Generated Text")
                st.markdown(f"<div class='generated-box'>{generated_text}</div>", 
                           unsafe_allow_html=True)
                
                # Statistics
                word_count = len(generated_text.split())
                st.caption(f"📊 Generated {word_count} words in {generation_time:.2f} seconds")
                
                # Download option
                st.download_button(
                    "💾 Download Text",
                    generated_text,
                    file_name=f"generated_{sentiment.lower()}_text.txt",
                    mime="text/plain"
                )
            
            # Step 3: Verification
            with st.expander("✅ Step 3: Sentiment Verification", expanded=True):
                verification_sentiment, verification_confidence, _ = analyze_sentiment_comprehensive(
                    generated_text, 
                    st.session_state.sentiment_pipeline
                )
                
                if verification_sentiment == sentiment:
                    st.success(f"✅ Generated text matches target sentiment: {verification_sentiment}")
                else:
                    st.warning(f"⚠️ Generated text sentiment: {verification_sentiment} (target: {sentiment})")
                
                st.write(f"**Verification Confidence:** {verification_confidence:.2%}")
    
    elif generate_clicked and not user_prompt.strip():
        st.error("❌ Please enter some text to generate content!")

# Run the application
if __name__ == "__main__":
    main()