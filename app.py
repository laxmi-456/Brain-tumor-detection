import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from flask import Flask, redirect, url_for, request, render_template


import flask

app = Flask(__name__)

best = load_model("save_model.h5")


def predict_tumor(img_path):
    print("entered predict tumor")

    img = load_img(img_path, target_size=(224, 224))

    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)
    if best.predict(img)[0][0] > 0.45:
        return "yes"
    else:
        return "no"


@app.route("/", methods=["GET"])
def welcome():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        imagefile = request.files["imagefile"]
        if imagefile:
            image_path = "./static/" + imagefile.filename
            imagefile.save(image_path)
            return render_template(
                "index.html",
                prediction=predict_tumor(image_path),
                imageloc=imagefile.filename,
            )
    return render_template(
        "index.html", prediction=predict_tumor(image_path), imageloc=None
    )


if __name__ == "__main__":
    app.run(port=8080)
