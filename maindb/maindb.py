from flask import Flask, render_template, request
import pandas as pd
import os
from query import magenta_predict, Magenta_Model


app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def mainm():
    if request.method == "POST":
        print("[maindb.py] Request received...")
        inputs = request.json
        print("[maindb.py] Request json: ", inputs)
        outfns = magenta_predict(magenta_model, inputs["style_img"],
                                 inputs["content_img"], inputs["style_weight"])
        print("[maindb.py] Model output received...")

        return render_template(
            "display.html",
            style_img=outfns[0],
            content_img=outfns[1],
            output_img=outfns[2]
        )
    else:
        return "maindb.py - This is get method - try using post -- "


if __name__ == "__main__":
    print("[maindb.py] Running maindb.py now...")
    global magenta_model
    magenta_model = Magenta_Model("/static/models/",
                     content_square_crop=False, style_square_crop=False,
                     style_image_size=256, content_image_size=256)

    app.run(host="0.0.0.0", port=8082, debug=True)
