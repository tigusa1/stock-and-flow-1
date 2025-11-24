import streamlit as st
import json
from pathlib import Path

class SliderSetManager:
    def __init__(self, file_path="slider_sets.json"):
        self.file_path = Path(file_path)
        self.sets = self._load()

    # ---------- Internal ----------
    def _load(self):
        if self.file_path.exists():
            try:
                return json.loads(self.file_path.read_text())
            except json.JSONDecodeError:
                return []
        return []

    def _save(self):
        self.file_path.write_text(json.dumps(self.sets, indent=2))

    # ---------- Core operations ----------
    def add_set(self, name, keys):
        if not name:
            st.warning("Please enter a name for the set.")
            return
        # Overwrite if name already exists
        values = {k: st.session_state[k] for k in keys}
        self.sets = [s for s in self.sets if s["name"] != name]
        self.sets.append({"name": name, "values": values})
        self._save()
        st.success(f"Saved set '{name}'!")

    def load_set(self, name):
        selected = next((s for s in self.sets if s["name"] == name), None)
        if selected:
            for k, v in selected["values"].items():
                st.session_state[k] = v
            st.success(f"Loaded set '{name}'!")
            st.rerun()

    def rename_set(self, old_name, new_name):
        if not new_name:
            st.warning("Please enter a new name.")
            return
        for s in self.sets:
            if s["name"] == old_name:
                s["name"] = new_name
                self._save()
                st.success(f"Renamed '{old_name}' to '{new_name}'")
                return
        st.warning(f"Set '{old_name}' not found.")

    def delete_set(self, name):
        self.sets = [s for s in self.sets if s["name"] != name]
        self._save()
        st.success(f"Deleted set '{name}'.")

    # ---------- UI ----------
    def ui(self, keys):
        st.subheader("🎛 Slider Set Manager")

        # Save new set
        with st.expander("💾 Save Current Set"):
            name = st.text_input("Name this set:", key="set_name")
            if st.button("Save Set"):
                self.add_set(name, keys)

        # Manage existing sets
        if self.sets:
            names = [s["name"] for s in self.sets]
            selected = st.selectbox("Choose a set", names, key="selected_set")

            cols = st.columns(3)
            with cols[0]:
                if st.button("Load", key="load_btn"):
                    self.load_set(selected)
            with cols[1]:
                new_name = st.text_input("Rename to:", key="rename_input")
                if st.button("Rename", key="rename_btn"):
                    self.rename_set(selected, new_name)
            with cols[2]:
                if st.button("Delete", key="delete_btn"):
                    self.delete_set(selected)
        else:
            st.info("No sets saved yet.")