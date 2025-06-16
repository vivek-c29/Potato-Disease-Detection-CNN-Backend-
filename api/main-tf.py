from fastapi import FastAPI,UploadFile,File
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf


app = FastAPI()

endpoint = "http://loacalhost:8501/v1/models/potatoes_model/predict"
class_names = ["Early Blight","Late Blight",'Healthy']
# print(tf.__version__)

@app.get("/ping")
async def ping():
    return "Hellow,World !"

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
    print(image_batch.shape)
    
    predictions = MODEL.predict(image_batch)
    print(predictions)
    pred_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])*100
    return {
        "class":pred_class,
        "confidence": float(confidence)
    }
    

if __name__=="__main__":
    uvicorn.run(app,port=8000,host='localhost')