from fastapi import APIRouter, FastAPI

from ocr.routes.user import user_router
from ocr.routes.images.upload_image import image_router

app = FastAPI()

app.include_router(router=user_router)

app.include_router(router=image_router)
