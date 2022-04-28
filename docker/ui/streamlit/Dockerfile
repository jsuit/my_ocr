
FROM ui_build as baseImage

EXPOSE 8501
ENV STREAMLIT_BROWSER_SERVER_ADDRESS="localhost"
ENV STREAMLIT_SERVER_PORT=8501
ENTRYPOINT [ "streamlit", "run" ]
CMD ["ocr/__main__.py"]