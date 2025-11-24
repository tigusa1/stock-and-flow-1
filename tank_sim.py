import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

st.set_page_config(page_title="Tank & Pipe Flow Simulator", layout="wide")

col_spacer1, col1, col_spacer2, col2, col_spacer3 = st.columns([1, 2, 1, 2, 1])

with col1:
    st.title("Pipe-to-Tank Flow Visualization")

    # --- Inflow/Outflow Controls ---
    inflow = st.slider("Inflow Rate", 0.0, 2.0, 1.0, step=0.1)
    outflow = st.slider("Outflow Rate", 0.0, 2.0, 1.0, step=0.1)

with col2:
    # Simulation settings
    T         = 100  # number of animation steps
    dt        = 0.25
    max_level = 10.0
    tank_width = 8.0

    # Initialize state
    if "tank_level" not in st.session_state:
        st.session_state.tank_level = 5.0

    placeholder = st.empty()

    # Run animation loop
    for t in range(T):
        # Update tank level
        delta = (inflow - outflow) * dt
        st.session_state.tank_level = np.clip(st.session_state.tank_level + delta, 0.0, max_level)

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.set_xlim(-tank_width/2, tank_width/2)
        ax.set_ylim(-2, max_level + 2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Draw tank
        tank  = plt.Rectangle((-tank_width/2, 0), tank_width, max_level, fill=False, linewidth=2)
        water = plt.Rectangle((-tank_width/2, 0), tank_width, st.session_state.tank_level, color='skyblue')
        ax.add_patch(tank)
        ax.add_patch(water)

        # Inflow in_pipe (top)
        in_pipe_width = 0.0 + inflow / .5
        in_pipe_x     = -in_pipe_width / 2
        in_pipe_y_top = max_level + 1.5
        in_pipe = plt.Rectangle((in_pipe_x, 0),
                                in_pipe_width, max_level + 1.5, color='skyblue')
        ax.add_patch(in_pipe)

        # Outflow in_pipe (bottom)
        out_pipe_width = 0.0 + outflow / .5
        out_pipe_x     = -out_pipe_width / 2
        out_pipe_y     = -1.5
        out_pipe = plt.Rectangle((out_pipe_x, out_pipe_y),
                                 out_pipe_width, 1.5, color='skyblue')
        ax.add_patch(out_pipe)

        # Arrows
        ax.annotate("", xy=(0, max_level), # arrow tip
                    xytext=(0, max_level+1.5),  # arrow end
                    arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8))
        ax.annotate("", xy=(0, -1.5),            xytext=(0, 0),
                    arrowprops=dict(facecolor='red',   shrink=0.05, width=2, headwidth=8))

        placeholder.pyplot(fig)
        time.sleep(0.025)

# st.success("Simulation complete.")