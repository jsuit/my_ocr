import asyncio
import streamlit as st
import httpx
from hydralit import HydraHeadApp
from ocr import FAST_API_URL, FAST_API_PORT
import os

# fast_url = f"{FAST_API_URL}:{FAST_API_PORT}/users/"


class Login(HydraHeadApp):
    """
    This is an example login application to be used to secure access within a HydraApp streamlit application.
    This application implementation uses the allow_access session variable and uses the do_redirect method if the login check is successful.

    """

    def __init__(self, title="", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        self._fast_api_port = os.environ.get("FASTAPI_PORT", FAST_API_PORT)
        self._fast_url = os.environ.get("FASTAPI_URL", FAST_API_URL)
        self._url = f"{self._fast_url}:{FAST_API_PORT}"

    def run(self) -> None:
        """
        Application entry point.
        """

        if st.button("Signup", key="signup"):
            self.set_access(-1)

            self.do_redirect("Signup")
        with st.container():

            with st.form("login_form"):
                form_state = {}
                form_state["username"] = st.text_input("Username")
                # form_state['access_level'] = login_form.selectbox('Example Access Level',(1,2))
                form_state["submitted"] = st.form_submit_button("Login")

        if form_state["submitted"]:

            self._do_login(form_data=form_state)

    def _do_login(self, form_data) -> None:

        # access_level=0 Access denied!

        access_level = asyncio.run(self._check_login(form_data))

        if access_level > 0:
            st.success(f"✔️ Login success")
            with st.spinner("🤓 now redirecting to application...."):

                # access control uses an int value to allow for levels of permission that can be set for each user, this can then be checked within each app seperately.
                # Do the kick to the home page
                self.set_access(1, form_data["username"])
                self.do_redirect()
        else:
            self.session_state.allow_access = 0
            self.session_state.current_user = None

    async def _check_login(self, login_data) -> int:
        # this method returns a value indicating the success of verifying the login details provided and the permission level, 1 for default access, 0 no access etc.
        username = login_data["username"]

        async with httpx.AsyncClient(base_url=self._url) as client:
            resp = await client.get("/users/" + f"{username}")

            if resp.is_error:
                st.error(
                    f"❌ Login unsuccessful. Error: {resp.json()} 😕 please check your username and try again."
                )
                return 0
            else:
                return 1
