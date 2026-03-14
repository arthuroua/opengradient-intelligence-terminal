from flask import Flask, render_template

app = Flask(__name__)

models = [
{
"name":"Meta-Llama-3-8B-Instruct",
"description":"Large language model optimized for chat and reasoning.",
"category":"LLM",
"caps":["Chat","Reasoning","Coding"]
},
{
"name":"ETH Volatility Predictor",
"description":"Predict ETH volatility using statistical models.",
"category":"Finance",
"caps":["Forecast","Analytics"]
},
{
"name":"Crypto Sentiment AI",
"description":"Analyze social sentiment for crypto markets.",
"category":"Analytics",
"caps":["NLP","Sentiment"]
},
{
"name":"Bidirectional Attention Flow",
"description":"Deep learning architecture for question answering.",
"category":"NLP",
"caps":["QA","Context"]
}
]

@app.route("/")
def home():
    return render_template("index.html",models=models)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)
