FROM ui_build as baseImage

EXPOSE 8000
ENV FASTAPI_PORT=8000
ENV FASTAPI_URL=fastapi
ENV SQLALCHEMY_DATABASE_URL="sqlite:///./sql_app.db"
ENTRYPOINT ["python", "-m", "uvicorn", "ocr.__main__:app", "--reload", "--host", "0.0.0.0",  "--port"] 
CMD ["8000"]