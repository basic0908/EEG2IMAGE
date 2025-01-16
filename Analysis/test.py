import mne

# Path to the EDF file
file_path = "data/Envisioned_Speech_Recognition/Char/aashay_A.edf"

# Try loading the file with MNE
try:
    raw = mne.io.read_raw_edf(file_path, preload=True)
    print(raw.info)  # Display metadata
    raw.plot()  # Visualize the data
except Exception as e:
    print(f"Error: {e}")