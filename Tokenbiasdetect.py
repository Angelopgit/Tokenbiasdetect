import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from io import BytesIO
import base64

print("TensorFlow version:", tf.__version__)

app = Flask(__name__)

# Load tokenizer and model for TensorFlow
tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model_tf = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect_bias', methods=['POST'])
def detect_bias():
    text1 = request.form.get('text1', '')
    text2 = request.form.get('text2', '')
    biased_probs = []

    # Helper function to calculate bias probability
    def get_bias_probability(text):
        if text:
            inputs = tokenizer(text, return_tensors="tf")
            outputs = model_tf(inputs)
            probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy().squeeze()
            return probabilities[1]  # Probability of being biased
        return 0

    biased_probs.append(get_bias_probability(text1))
    biased_probs.append(get_bias_probability(text2))

    # Plotting the results
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['red', 'blue']
    labels = ['Text 1', 'Text 2']
    for i, bias_prob in enumerate(biased_probs):
        ax.plot(i, bias_prob, 'o', color=colors[i], markersize=10)
        ax.text(i + 0.05, bias_prob, f'{labels[i]}: {bias_prob:.2f}', fontsize=12, ha='left', va='center')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel('Text Input', fontsize=14)
    ax.set_ylabel('Bias Probability', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True)

    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode()
    plt.close()

    return render_template('result.html', texts=[text1, text2], biased_probs=biased_probs, img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
