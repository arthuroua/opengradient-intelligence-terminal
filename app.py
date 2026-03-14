from flask import Flask, render_template
import requests

app = Flask(__name__)

API_URL="https://hub.opengradient.ai/api/models?page=0&limit=20&sort_by=most_likes"

def get_models():

    try:
        r=requests.get(API_URL,timeout=10)

        data=r.json()

        models=[]

        for m in data.get("models",[]):

            models.append({
                "name":m.get("name","Unknown"),
                "description":m.get("description","OpenGradient model"),
                "likes":m.get("likes",0)
            })

        return models

    except Exception as e:

        print("API error:",e)

        return []

@app.route("/")
def home():

    models=get_models()

    return render_template("index.html",models=models)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)
