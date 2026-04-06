const generateBtn = document.getElementById('generateBtn');
const userPrompt = document.getElementById('userPrompt');
const generatedText = document.getElementById('generatedText');
const sentimentValue = document.getElementById('sentimentValue');
const sentimentConfidence = document.getElementById('sentimentConfidence');
const btnText = document.querySelector('.btn-text');
const btnLoader = document.querySelector('.btn-loader');
const sentimentBadge = document.getElementById('sentimentBadge');

// Sentiment colors
const sentimentColors = {
    'POSITIVE': '#22c55e',
    'NEGATIVE': '#ef4444',
    'NEUTRAL': '#6b7280'
};

// Sentiment labels
const sentimentLabels = {
    'POSITIVE': 'Positive',
    'NEGATIVE': 'Negative',
    'NEUTRAL': 'Neutral'
};

// Show loading state
function setLoading(loading) {
    if (loading) {
        btnText.textContent = 'Generating...';
        btnLoader.style.display = 'block';
        generateBtn.disabled = true;
        generatedText.innerHTML = '<p class="placeholder-text">AI is generating your content...</p>';
    } else {
        btnText.textContent = 'Generate Text';
        btnLoader.style.display = 'none';
        generateBtn.disabled = false;
    }
}

// Display sentiment result
function displaySentiment(sentiment, confidence) {
    const color = sentimentColors[sentiment] || sentimentColors['NEUTRAL'];
    sentimentValue.textContent = sentimentLabels[sentiment] || sentiment;
    sentimentConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    sentimentBadge.style.backgroundColor = color;
    sentimentValue.style.color = 'white';
    sentimentConfidence.style.color = 'rgba(255,255,255,0.9)';
}

// Display generated text
function displayText(text) {
    generatedText.innerHTML = `<p>${text}</p>`;
}

// Handle form submission
generateBtn.addEventListener('click', async () => {
    const prompt = userPrompt.value.trim();
    
    if (!prompt) {
        alert('Please enter a topic or text to generate');
        return;
    }
    
    try {
        setLoading(true);
        
        const response = await fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `user_prompt=${encodeURIComponent(prompt)}`
        });
        
        if (!response.ok) {
            throw new Error('Failed to generate text');
        }
        
        const data = await response.json();
        
        // Assuming the backend returns both sentiment and text
        // If backend only returns text, you may need to adjust this
        displayText(data.generated_text || 'No text generated');
        
        // Reset sentiment display
        sentimentValue.textContent = '--';
        sentimentConfidence.textContent = '--';
        sentimentBadge.style.backgroundColor = '#6b7280';
        sentimentValue.style.color = 'white';
        sentimentConfidence.style.color = 'rgba(255,255,255,0.9)';
        
    } catch (error) {
        console.error('Error:', error);
        generatedText.innerHTML = '<p class="error-text">Error generating text. Please try again later.</p>';
        alert('An error occurred. Please check the console for details.');
    } finally {
        setLoading(false);
    }
});

// Allow Ctrl+Enter to submit
userPrompt.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        generateBtn.click();
    }
});