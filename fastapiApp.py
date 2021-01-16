import base64
import logging
import json

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional

from fastapi.logger import logger as fastapi_logger

from image_processor import ImageProcessor

logger = logging.getLogger("FastAPI App")
logger.setLevel(logging.DEBUG)
fastapi_logger.handlers = logger.handlers

my_app = ImageProcessor()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class Process_Request(BaseModel):
  image: str
  noSegments: int = 10
  sigma: int = 3

class Process_Response_Sub(BaseModel):
  centroids: List[Tuple[int]]
  segments: List[List[int]]

class Process_Response(BaseModel):
  data: Optional[Process_Response_Sub]
  message: str

@app.post('/process', response_model=Process_Response)
async def search(body: Process_Request):  
  try:
    result = my_app(base64_image=body.image, n_segments=body.noSegments, sigma=body.sigma)
    return_obj = {
        "data": {
            "segments": result["segments"].tolist(), #numpy array to list
            "centroids": result["centroids"]
        },
        "message": "success"
    }
    return JSONResponse(content=return_obj)
  except Exception as err:
    logger.error(err)
    return_obj = {
      "message": "failed"
    }
    return JSONResponse(content=return_obj)