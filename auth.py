import streamlit as st

def check_password():
    """Returns `True` if the user had a correct password."""

    try:
        if 'passwords' not in st.secrets:
            return True
    except FileNotFoundError:
        return True

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    with st.container():
        if "password_correct" not in st.session_state:
            # First run, show inputs for username + password.
            st.divider()
            st.title('Login')
            st.text_input("Username", on_change=password_entered, key="username")
            st.text_input("Password", type="password", on_change=password_entered, key="password")
            st.divider()
            return False
        elif not st.session_state["password_correct"]:
            # Password not correct, show input + error.
            st.divider()
            st.title('Login')
            st.text_input("Username", on_change=password_entered, key="username")
            st.text_input("Password", type="password", on_change=password_entered, key="password")
            st.error("😕 User not known or password incorrect")
            st.divider()
            return False
        else:
            # Password correct.
            return True

