from typing import Dict
import streamlit as st
from hydralit import HydraHeadApp
import httpx
from ocr import FAST_API_URL, FAST_API_PORT
import os

# FAST_URL = f"{FAST_API_URL}:{FAST_API_PORT}"


class SignUpApp(HydraHeadApp):
    """
    This is an example signup application to be used to secure access within a HydraApp streamlit application.

    This application is an example of allowing an application to run from the login without requiring authentication.

    """

    def __init__(self, title="Signup", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        self._fast_api_port = os.environ.get("FASTAPI_PORT", FAST_API_PORT)
        self._fast_url = os.environ.get("FASTAPI_URL", FAST_API_URL)
        self._url = f"{self._fast_url}:{self._fast_api_port}"

    def run(self) -> None:
        """
        Application entry point.

        """

        st.markdown(
            "<h1 style='text-align: center;'>Signup</h1>",
            unsafe_allow_html=True,
        )

        pretty_btn = """
        <style>
        div[class="row-widget stButton"] > button {
            width: 100%;
        }
        </style>
        <br><br>
        """
        st.markdown(pretty_btn, unsafe_allow_html=True)

        form_data = self._create_signup_form()

        pretty_btn = """
        <style>
        div[class="row-widget stButton"] > button {
            width: 100%;
        }
        </style>
        <br><br>
        """
        st.markdown(pretty_btn, unsafe_allow_html=True)

        if form_data["submitted"]:
            self._do_signup(form_data)

    def _create_signup_form(self) -> Dict:

        with st.form("sign_up_form"):
            form_state = {}
            form_state["username"] = st.text_input("Username")
            # form_state["password2"] = login_form.text_input(
            #    "Confirm Password", type="password"
            # )

            form_state["submitted"] = st.form_submit_button("Sign Up")

        if st.button("Login", key="loginbtn"):
            # set access level to a negative number to allow a kick to the unsecure_app set in the parent
            self.set_access(0, None)

            # Do the kick to the signup app
            self.do_redirect()

        return form_state

    def _do_signup(self, form_data) -> None:
        if form_data["submitted"]:
            with st.spinner("🤓 now redirecting to login...."):
                username = form_data["username"]

                with httpx.Client(base_url=self._url) as client:
                    resp = client.post(url="/users", json={"name": username})
                    if resp.is_error:
                        st.error(
                            f"❌ Signup unsuccessful. Got error {resp.json()} 😕 please try a new username."
                        )
                    else:
                        st.write(f"saved {username}")

                # access control uses an int value to allow for levels of permission that can be set for each user, this can then be checked within each app seperately.
                self.set_access(0, None)

                # Do the kick back to the login screen
                self.do_redirect()

    def _save_signup(self, signup_data):
        # get the user details from the form and save somehwere

        # signup_data
        # this is the data submitted

        # just show the data we captured
        what_we_got = f"""
        captured signup details: \n
        username: {signup_data['username']} \n
        """

        st.write(what_we_got)
