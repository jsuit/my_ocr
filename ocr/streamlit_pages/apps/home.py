import streamlit as st
from streamlit_drawable_canvas import st_canvas
import httpx
from hydralit import HydraHeadApp
from ocr import FAST_API_URL, FAST_API_PORT
from io import BytesIO
from PIL import Image
import os


class Home(HydraHeadApp):
    """
    This is an example login application to be used to secure access within a HydraApp streamlit application.
    This application implementation uses the allow_access session variable and uses the do_redirect method if the login check is successful.

    """

    def __init__(self, title="home", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        self._fast_api_port = os.environ.get("FASTAPI_PORT", FAST_API_PORT)
        self._fast_url = os.environ.get("FASTAPI_URL", FAST_API_URL)
        self._url = f"{self._fast_url}:{FAST_API_PORT}"

    def run(self) -> None:
        """
        Application entry point.
        """

        c_user = self.session_state.current_user
        # if have images, display N of them in a grid
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
            self.session_state.image = 
            import pdb

            pdb.set_trace()


def canvas(image: Image):

    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == "point":
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=image,
        update_streamlit=realtime_update,
        height=150,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        key="canvas",
    )
    return canvas_result
