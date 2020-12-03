from flask import Flask, render_template, request
import pandas as pd
import os
from query import main_predict


app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def mainm():
    if request.method == "POST":
        print("[maindb.py] Request received...")
        inputs = request.json
        img = main_predict(inputs["style_img"], inputs["content_img"])
        print("[maindb.py] Modle output received...")

        return render_template(
            "display.html",
            style_img=img
        )
    else:
        return "maindb.py - This is get method - try using post -- "


if __name__ == "__main__":
    print("[maindb.py] Running maindb.py now...")
    # from dask.distributed import Client
    # client = Client()
    # print("Dask client started: ", client)
    app.run(host="0.0.0.0", port=8082, debug=True)
