from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np
from PIL import Image

import base64
import io


app = Flask(__name__)

CORS(app)


# ==============================
# Load TensorFlow Lite Model
# ==============================

interpreter = tf.lite.Interpreter(
    model_path="plant_model.tflite"
)

interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("TFLite model loaded")


# ==============================
# Disease Classes
# ==============================

class_names = [

    'corn_blight',
    'corn_common_rust',
    'corn_gray_leaf_spot',
    'corn_healthy',

    'tomato_bacterial_spot',
    'tomato_early_blight',
    'tomato_late_blight',
    'tomato_leaf_mold',
    'tomato_septoria_leaf_spot',
    'tomato_spider_mites',
    'tomato_target_spot',
    'tomato_yellow_leaf_curl_virus',
    'tomato_mosaic_virus',
    'tomato_healthy',
    'tomato_powdery_mildew',

    'rice_bacterial_leaf_blight',
    'rice_brown_spot',
    'rice_healthy',
    'rice_leaf_blast',
    'rice_leaf_scald',
    'rice_sheath_blight',

    'coconut_healthy',
    'coconut_pest_damage',
    'coconut_yellowing',
    'coconut_leaf_spot'

]


# ==============================
# Home
# ==============================

@app.route("/")
def home():

    return "Plant Disease API Running (TensorFlow Lite)"



# ==============================
# Prediction API
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("Request received")


        data = request.get_json()


        if not data:

            return jsonify({
                "error":"No data received"
            }),400



        image_data = data["image"]


        crop = data.get(
            "crop",
            ""
        ).lower()



        if crop not in [
            "corn",
            "tomato",
            "rice",
            "coconut"
        ]:

            return jsonify({

                "error":
                "Invalid crop selected"

            }),400



        # Remove base64 header

        if "," in image_data:

            image_data = image_data.split(",")[1]



        # Decode image

        image_bytes = base64.b64decode(
            image_data
        )


        img = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        print("Image decoded")



        # Resize

        img = img.resize(
            (224,224)
        )


        img_array = np.array(
            img
        )


        # Normalize

        img_array = img_array / 255.0



        # Add batch dimension

        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        print("Prediction started")



        # ==============================
        # TensorFlow Lite Prediction
        # ==============================


        interpreter.set_tensor(

            input_details[0]["index"],

            img_array.astype(np.float32)

        )


        interpreter.invoke()



        prediction = interpreter.get_tensor(

            output_details[0]["index"]

        )[0]



        print("Prediction completed")



        # Select crop classes only

        allowed_indexes = [

            i for i,name in enumerate(class_names)

            if name.startswith(crop)

        ]



        crop_scores = prediction[allowed_indexes]


        crop_scores = (

            crop_scores /

            np.sum(crop_scores)

        )



        best_position = np.argmax(
            crop_scores
        )


        final_index = allowed_indexes[
            best_position
        ]



        disease = class_names[
            final_index
        ]



        confidence = float(

            np.max(crop_scores)*100

        )



        return jsonify({

            "crop":crop,

            "disease":disease,

            "confidence":
            round(confidence,2)

        })



    except Exception as e:


        print("ERROR:",e)


        return jsonify({

            "error":str(e)

        }),500




# ==============================

# Run

# ==============================

if __name__=="__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )