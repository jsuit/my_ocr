import streamlit as st
from streamlit_drawable_canvas import st_canvas
from hydralit import HydraHeadApp
from ocr import FAST_API_URL, FAST_API_PORT
import os
import pandas as pd
from PIL import Image
from io import BytesIO


class Canvas(HydraHeadApp):
    def __init__(self, title="Canvas", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        self._fast_api_port = os.environ.get("FASTAPI_PORT", FAST_API_PORT)
        self._fast_url = os.environ.get("FASTAPI_URL", FAST_API_URL)
        self._url = f"{self._fast_url}:{FAST_API_PORT}"
        if "color_to_label" not in st.session_state:
            st.session_state["color_to_label"] = {}

    def run(self) -> None:
        image_file = st.file_uploader(label="Upload Image", type=["jpeg", "jpg", "png"])
        if image_file:
            file_name = image_file.name
            data = image_file.getvalue()  #
            files = {"image": (file_name, data, image_file.type)}

            """with httpx.Client(base_url=self._url) as client:
                resp = client.post(f"/images/{c_user}", files=files)
                if resp.is_error:
                    st.error(
                        f"❌ Could not upload image. Got error {resp.json}. 😕 Try again."
                    )
            if resp.is_success:
                """

            # del self.session_state.image
            # Specify canvas parameters in application
            # Create a canvas component

            # for alpha from 00 to FF
            image = Image.open(BytesIO(data))
            label = st.sidebar.text_input("Label", "Default")
            mode = "rect"
            label_color = (
                st.sidebar.color_picker("Annotation color: ", "#EA1010") + "77"
            )  # for alpha from 00 to FF
            canvas_result = st_canvas(
                fill_color=label_color,
                stroke_width=3,
                background_image=image,
                height=320,
                width=512,
                drawing_mode=mode,
                key="color_annotation_app",
            )

            if canvas_result.json_data is not None:
                df = pd.json_normalize(canvas_result.json_data["objects"])
                if len(df) == 0:
                    return
                import pdb

                pdb.set_trace()
                st.session_state["color_to_label"][label_color] = label
                df["label"] = df["fill"].map(st.session_state["color_to_label"])
                st.dataframe(df[["top", "left", "width", "height", "fill", "label"]])

                # x = st.button("save_annotations")

        # del self.session_state.image
