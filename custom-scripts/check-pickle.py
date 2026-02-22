import pickle
import os

pkl_files = [
    "/home/aniruth/Desktop/Courses/Independent - Study/D-PCC/data/semantickitti/semantickitti_train_cube_size_12.pkl",
    "/home/aniruth/Desktop/Courses/Independent - Study/D-PCC/data/semantickitti/semantickitti_test_cube_size_12.pkl",
    "/home/aniruth/Desktop/Courses/Independent - Study/D-PCC/data/semantickitti/semantickitti_val_cube_size_12.pkl"
]

def human_readable_size(size_bytes):
    """Convert bytes into a human-readable format (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

for f in pkl_files:
    if not os.path.exists(f):
        print(f"⚠️ {f} not found.\n")
        continue

    size = os.path.getsize(f)
    print(f"📦 {f}")
    print(f"   → Size: {human_readable_size(size)}")

    try:
        with open(f, "rb") as fp:
            _ = pickle.load(fp)
        print("   ✅ Loaded successfully.\n")
    except Exception as e:
        print(f"   ❌ Corrupted or unreadable: {e}\n")
