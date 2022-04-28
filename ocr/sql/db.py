from ocr.sql import Base as database_base
from ocr.sql import TABLENAMES
from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.orm import relationship


class User(database_base):
    __tablename__ = TABLENAMES.Users.name

    id = Column(Integer, primary_key=True, index=True, nullable=False)
    name = Column(String, unique=True, index=True, nullable=False)
    files = relationship("File", back_populates="owner")


class File(database_base):
    __tablename__ = TABLENAMES.Files.name

    id = Column(Integer, primary_key=True, index=True, nullable=False)
    filename = Column(String, unique=True, index=True, nullable=False)
    description = Column(String, unique=False, index=False)
    user_id = Column(Integer, ForeignKey(f"{TABLENAMES.Users.name}.id"))
    owner = relationship("User", back_populates="files")
    bboxs = relationship("BBoxs", back_populates="file")


class BBoxs(database_base):

    __tablename__ = TABLENAMES.BBoxs.name

    id = Column(Integer, primary_key=True, index=True)
    xmin = Column(Float, nullable=False)
    ymin = Column(Float, nullable=False)
    xmax = Column(Float, nullable=False)
    ymax = Column(Float, nullable=False)
    file_id = Column(Integer, ForeignKey(f"{TABLENAMES.Files.name}.id"))

    file = relationship("File", back_populates="bboxs")
