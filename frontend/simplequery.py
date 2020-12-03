from flask import Flask, render_template, request
import sys
import requests
from PIL import Image
import numpy as np

app = Flask(__name__)

def get_img_array(request, img_key):
    img = request.files[img_key]
    img = Image.open(img)
    img_arr = np.array(img.getdata())
    img_arr = img_arr.reshape(img.size[0], img.size[1], 3).tolist()
    return img_arr

@app.route("/", methods=["POST", "GET"])
def mainm():
    print("[simplequery.py] mainm() being called...")
    if request.method == "POST":  # User clicked submit button
        print("[simplequery.py] Request received...")

        style_img_fn = request.form["style_img_fn"]
        content_img_fn = request.form["content_img_fn"]
        print("[simplequery.py] Request texts parsed...")

        style_img = get_img_array(request, "style_img")
        content_img = get_img_array(request, "content_img")
        print("[simplequery.py] Request files parsed...")

        # send this data id to maindb.py
        print("[simplequery.py] Downstream request being made...")
        resp = requests.post(url=db_url, json={
            "style_img_fn" : style_img_fn,
            "content_img_fn" : content_img_fn,
            "style_img" : style_img,
            "content_img" : content_img
        })
        print("[simplequery.py] Response returned...")

        # return the response content
        return resp.content
    else:
        return render_template("index.html")


if __name__ == "__main__":
    print("[simplequery.py] Running simplequery.py...")
    # determine what the URL for the database should be, port is always 8082 for DB
    if len(sys.argv) == 2:
        db_url = "http://" + sys.argv[1] + ":8082"
    else:
        db_url = "http://0.0.0.0:8082/"

    app.run(host="0.0.0.0", port=8081, debug=True)
