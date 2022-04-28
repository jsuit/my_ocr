from typing import List, Optional
from pydantic import BaseModel, Field
from decimal import Decimal


class BBoxs(BaseModel):
    id: int
    xmin: Decimal = Field(..., ge=0.0, decimal_places=2)
    ymin: Decimal = Field(..., ge=0.0, decimal_places=2)
    xmax: Decimal = Field(..., ge=0.0, decimal_places=2)
    ymax: Decimal = Field(..., ge=0.0, decimal_places=2)
    file_id: int

    class Config:
        orm_mode = True


class FilesBase(BaseModel):
    filename: str = Field(..., max_length=2**16)
    description: Optional[str] = Field(None, max_length=128)


class Files(FilesBase):
    id: int
    bboxs: List[BBoxs] = Field(default=[], max_items=10e5)
    user_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):

    name: str = Field(
        ..., description="<= 3 user name <= 2048 chars", max_length=2048, min_length=3
    )


class User(UserBase):
    files: List[Files] = Field(default_factory=list, min_items=0, max_items=1e6)
    id: int = Field(..., description="unique id of user")

    class Config:
        orm_mode = True
