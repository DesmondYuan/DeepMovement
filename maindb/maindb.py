from flask import Flask, render_template, request
import pandas as pd
import os
from query import magenta_predict, feature_predict, get_metadata
from core_vgg import Feature_Model
from core_magenta import Magenta_Model


app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def mainm():
    if request.method == "POST":
        print("[maindb.py] Request received...")
        inputs = request.json
        style_img, content_img, output_img = magenta_predict(magenta_model, inputs["style_img"],
                                 inputs["content_img"], inputs["style_weight"])
        nearest_img = feature_predict(feature_model, output_img)
        meta = get_metadata(nearest_img)
        print("[maindb.py] Model output received...")

        return render_template(
            "display.html",
            style_img=style_img,
            content_img=content_img,
            output_img=output_img,
            nearest_img="/static/data/wikiart/"+nearest_img,
            metadata=[meta.to_html(classes="data")]
        )
    else:
        return "maindb.py - This is get method - try using post -- "


if __name__ == "__main__":
    print("[maindb.py] Running maindb.py now...")
    global magenta_model
    global feature_model
    magenta_model = Magenta_Model("/static/models/",
                     content_square_crop=False, style_square_crop=False,
                     style_image_size=256, content_image_size=256)
    feature_model = Feature_Model()
    app.run(host="0.0.0.0", port=8082, debug=True)
