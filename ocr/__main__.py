from ocr.streamlit_pages import apps

from hydralit import HydraApp


if __name__ == "__main__":

    # fast_url = f"{FAST_API_URL}:{FAST_API_PORT}/users"

    app = HydraApp(
        title="Secure Hydralit Data Explorer",
        favicon="🐙",
        hide_streamlit_markers=True,
        navbar_animation=True,
        navbar_sticky=True,
        # navbar_theme=over_theme,
        use_navbar=True,
    )

    app.add_app("Login", app=apps.Login(), is_login=True, logout_label="Logout")
    app.add_app("Signup", app=apps.SignUpApp("Signup"), is_unsecure=True)
    app.add_app("Home", app=apps.Home(), is_home=True)
    app.add_app("Canvas", app=apps.Canvas())
    complex_nav = {"Home": ["Home"], "Canvas": ["Canvas"]}
    app.run(complex_nav=complex_nav)
