import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

st.title("Simple Streamlit Test App")

# Slider
value = st.slider("Choose a value", 0, 100, 50)

# Run simulation button
if st.button("Run simulation"):
    # Simple simulation: y = sin(x + slider_value)
    x = np.linspace(0, 2*np.pi, 200)
    y = np.sin(x + value/10)

    # Make a plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(f"Plot with slider value = {value}")

    # Convert to PNG bytes (Cloud Run safe)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Show plot
    st.image(buf, width='stretch')
    plt.close(fig)

st.write("Move the slider and click 'Run simulation' again. This must work every time.")
