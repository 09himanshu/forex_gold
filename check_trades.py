import pickle
import os

path = "models/scalping_training_data.pkl"

if not os.path.exists(path):
    print("❌ Training data file not found")
else:
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("📊 Total trades collected:", len(data))
