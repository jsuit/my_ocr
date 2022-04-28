from fastapi import APIRouter, Depends, HTTPException
from ocr.sql import SessionLocal
from ocr.sql import crud
from sqlalchemy.orm import Session
from ocr.pydantic_models import models as py_models

user_router = APIRouter(prefix="/users")


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@user_router.post("", response_model=py_models.User)
def create_user(user: py_models.UserBase, db: Session = Depends(get_db)):

    new_user = crud.get_user_by_name(db, name=user.name)
    if new_user:
        raise HTTPException(
            status_code=401,
            detail="name already registered",
        )
    return crud.create_user(db=db, user=user)


@user_router.get("/{user_name}", response_model=py_models.User)
def get_user_by_name(user_name: str, db: Session = Depends(get_db)):

    user = crud.get_user_by_name(db, name=user_name)
    if user is None:
        raise HTTPException(status_code=401, detail="name not registered")
    return user
