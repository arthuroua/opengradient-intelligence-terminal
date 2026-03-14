from flask import Flask, jsonify, render_template
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/models")
def models():

    url="https://hub.opengradient.ai/models?page=0&limit=10&sort_by=most_likes"

    try:

        r=requests.get(url,headers={
        "User-Agent":"Mozilla/5.0"
        })

        return jsonify({"status":"ok","html":r.text[:1000]})

    except Exception as e:

        return jsonify({"status":"error","msg":str(e)})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)
