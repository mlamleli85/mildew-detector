"""Entry point for the Mildew Detector dashboard.

Run locally with:  streamlit run app.py
"""
from app_pages.multipage import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_leaves_visualizer import page_leaves_visualizer_body
from app_pages.page_mildew_detector import page_mildew_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_body

app = MultiPage(app_name="Mildew Detector")

# Pages appear in the sidebar menu in this order
app.add_page("Project Summary", page_summary_body)
app.add_page("Leaf Visualiser", page_leaves_visualizer_body)
app.add_page("Mildew Detector", page_mildew_detector_body)
app.add_page("Project Hypotheses", page_project_hypothesis_body)
app.add_page("ML Performance", page_ml_performance_body)

app.run()
