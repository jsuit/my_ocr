from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from ocr.sql import SessionLocal
from ocr.sql import crud
from sqlalchemy.orm import Session
from ocr.pydantic_models import models as py_models

image_router = APIRouter(prefix="/images")


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@image_router.post("/{user_name}", response_model=py_models.FilesBase)
async def create_image(
    user_name: str,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    user_sql = crud.get_user_by_name(db, name=user_name)
    if user_sql is None:
        raise HTTPException(status_code=400, detail=f"name {user_name} not registered")

    filename = image.filename

    contents = await image.read()
    type(contents)

    return {"filename": filename}

    # return crud.create_user(db=db, user=user)
