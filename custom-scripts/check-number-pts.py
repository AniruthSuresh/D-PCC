import open3d as o3d
import glob
import os

def count_points_in_folder(folder_path):
    ply_files = sorted(glob.glob(os.path.join(folder_path, "*.ply")))
    if not ply_files:
        print(f"[!] No PLY files found in {folder_path}")
        return 0

    total_points = 0
    for ply_file in ply_files:
        try:
            pcd = o3d.io.read_point_cloud(ply_file)
            n_points = len(pcd.points)
            print(f"{ply_file}: {n_points} points")
            total_points += n_points
        except Exception as e:
            print(f"[Error reading] {ply_file}: {e}")
    print(f"→ Total points in {folder_path}: {total_points}\n")
    return total_points

if __name__ == "__main__":
    base_dir = "/home/aniruth/Desktop/Courses/Independent - Study/D-PCC/output/2025-10-29T00:11:00.664377/pcd"
    # subfolders = ["merge/gt", "merge/pred", "patch/gt", "patch/pred"]

    subfolders = ["patch/pred"]

    for sub in subfolders:
        folder = os.path.join(base_dir, sub)
        if os.path.exists(folder):
            print(f"\n--- Checking {folder} ---")
            count_points_in_folder(folder)
        else:
            print(f"[!] Folder not found: {folder}")
