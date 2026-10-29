"""Helper class that turns a list of page functions into a sidebar menu."""
import streamlit as st


class MultiPage:
    """Collect dashboard pages and show the selected one."""

    def __init__(self, app_name):
        self.pages = []
        self.app_name = app_name

        # Must be the first Streamlit call in the app
        st.set_page_config(page_title=self.app_name, page_icon="🍒",
                           layout="wide")

    def add_page(self, title, func):
        """Register a page: its menu title and the function that draws it."""
        self.pages.append({"title": title, "function": func})

    def run(self):
        """Draw the sidebar menu and the page the user picked."""
        st.title(self.app_name)
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to page", self.pages,
                                format_func=lambda page: page["title"])
        page["function"]()
