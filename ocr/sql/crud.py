from sqlalchemy.orm import Session
from ocr.pydantic_models.models import UserBase, FilesBase, BBoxs
from ocr.sql.db import User as sqlite_User, File as sqlite_File, BBoxs as sqlite_BBox


def create_user(db: Session, user: UserBase) -> sqlite_User:
    user = sqlite_User(name=user.name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_name(db: Session, name: str):
    return db.query(sqlite_User).filter(sqlite_User.name == name).first()


def get_user(db: Session, user_id: int):
    return db.query(sqlite_User).filter(sqlite_User.id == user_id).first()


def create_file(
    db: Session,
    file: FilesBase,
    user_id: int,
):
    sql_file = sqlite_File(**file.dict(), user_id=user_id)
    db.add(sql_file)
    db.commit()
    db.refresh(sql_file)
    return sql_file


def create_bboxs(db: Session, bbox: BBoxs, file_id: int):
    sql_bbox = sqlite_BBox(**bbox.dict(), file_id=file_id)
    db.add(sql_bbox)
    db.commit()
    db.refresh(sql_bbox)
    return sql_bbox


def get_files_by_id(db: Session, file_id: int):
    return db.query(sqlite_File).filter(sqlite_File.id == file_id).first()
