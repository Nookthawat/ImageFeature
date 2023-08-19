from fastapi import FastAPI , Request
import cv2
import numpy as np
import base64
from app.hog import gethog_64

app = FastAPI()

def read_bases64(uri):
    Encode_data = uri.split(',')[1]
    np_array = np.fromstring(base64.b64decode(Encode_data), np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    return img

@app.get("/api/gethog")
async def read_Data(request : Request):
    item = await request.json()
    item_data = item['img']
    img = read_bases64(item_data)
    hog = gethog_64(img)
    return {"HOG":hog.tolist()}