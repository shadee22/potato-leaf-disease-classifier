import os
from io import BytesIO
import numpy as np
import uvicorn
import json
from PIL import Image
from fastapi import FastAPI
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


app = FastAPI()


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_image(img):
    try:
        image = np.array(Image.open(BytesIO(img)) )
        return image
    except Exception as error:
        print("error on image to numpy ", error)


try:
    model = tf.keras.models.load_model('../trained_model/real_model')
    class_names = json.load(open('../trained_model/class_names.json', 'r'))

except Exception as e:
    model = 'model error ' + str(e)


@app.get("/predict")
async def predict(
        # file : UploadFile = File(...)
):
    # Real image
    # read_image(await file.read())

    # Local Image
    file_path = "../sources/Potato___Early_blight/0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG"
    if os.path.exists(file_path):
        image = Image.open(file_path)
        image = np.array(image)
        img_batch = np.expand_dims(image, 0)
        prediction = model.predict(img_batch)
        # fake_pred = {
        #     'shape': ((1, 256, 256, 3), (256, 256, 3)),
        #     'model': "<keras.engine.sequential.Sequential object at 0x16b841360>",
        #     'prediction': [[7.4639344e-01, 2.5355941e-01, 4.7039121e-05]],
        #     'classes': ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
        #     'Real Prediction': 'Potato___Early_blight'
        # }
        conf = np.round(100 * np.max(prediction[0]), 2)
        result = f"""
        'shape': {(img_batch.shape, image.shape)},
        'model ': {str(model)},
        'prediction': {str(prediction)},
        'classes': {str(class_names)}
        ----------------------------
        'Real Prediction' {class_names[np.argmax(prediction)]},
        'Confidence ' : {str(conf)}%
        """
        print(result)
        return result
    else:
        print("File does not exist.")

    return f"Hey Mother , Father!!!"


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000, reload_includes=["*.html", "*.css", "*.js"], reload=True)
