import streamlit as st
from SliderSetManager import SliderSetManager

st.title("🎚 Save and Reuse Slider Configurations")

# Initialize manager and attach the UI
manager = SliderSetManager()
manager.ui(["a", "b", "c"])

# Define your sliders
a = st.slider("Slider A", 0, 100, 50, key="a")
b = st.slider("Slider B", 0, 100, 25, key="b")
c = st.slider("Slider C", -10, 10, 0, key="c")
