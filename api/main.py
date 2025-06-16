from fastapi import FastAPI,UploadFile,File
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "1", "model.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Early Blight","Late Blight",'Healthy']
# MODEL = tf.keras.models.load_model("/saved_models/1/model.h5")
# print(tf.__version__)

# @app.get("/ping")
# async def ping():
#     return "Hellow,World !"

def read_file_as_image(data) :
    img = np.array(Image.open(BytesIO(data)))
    # print(img,type(img))
    return img
    

@app.post("/predict")
async def predict(
    file:UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,axis=0)
    # print(image_batch.shape)
    
    predictions = MODEL.predict(image_batch)
    # print(predictions)
    pred_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])*100
    return {
        "class":pred_class,
        "confidence": float(confidence)
    }
    

# if __name__=="__main__":
#     uvicorn.run(app,port=8000,host='localhost')