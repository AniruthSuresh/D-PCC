import os
import numpy as np
import open3d as o3d

# === CONFIG ===
base_dir = "/home/aniruth/Desktop/Courses/Independent - Study/D-PCC/output/2025-10-31T03:07:02.977277"
gt_merge_dir = os.path.join(base_dir, "pcd/merge/gt")
pred_merge_dir = os.path.join(base_dir, "pcd/merge/pred")
latent_dir = os.path.join(base_dir, "pcd/compressed")  # folder where all .pkl latent files are stored

# === Helper to get PLY stats ===
def get_ply_stats(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    num_points = np.asarray(pcd.points).shape[0]
    file_size = os.path.getsize(ply_path)
    bpp = (file_size * 8) / num_points if num_points > 0 else 0
    return num_points, file_size, bpp

# === 1. Collect stats for all GT and Pred merged clouds ===
def collect_ply_folder_stats(folder):
    ply_files = sorted([f for f in os.listdir(folder) if f.endswith(".ply")])
    total_points, total_size = 0, 0
    for f in ply_files:
        path = os.path.join(folder, f)
        points, size, _ = get_ply_stats(path)
        total_points += points
        total_size += size
    return total_points, total_size, ply_files

gt_points, gt_size, gt_files = collect_ply_folder_stats(gt_merge_dir)
pred_points, pred_size, pred_files = collect_ply_folder_stats(pred_merge_dir)

# === 2. Load compressed latent files (.pkl) ===
latent_files = [f for f in os.listdir(latent_dir) if f.endswith(".pkl")]
latent_total_size = sum(os.path.getsize(os.path.join(latent_dir, f)) for f in latent_files)

# === 3. Compute overall compressed bpp and ratio ===
compressed_bpp = (latent_total_size * 8) / pred_points if pred_points > 0 else 0
compression_ratio = (pred_size / latent_total_size) if latent_total_size > 0 else 0

# === 4. Display ===
print("=== Overall Compression Summary ===")
print(f"GT merged PLYs   : {len(gt_files)} files | {gt_size/1024:.2f} KB total | {gt_points:,} points")
print(f"Pred merged PLYs : {len(pred_files)} files | {pred_size/1024:.2f} KB total | {pred_points:,} points")
print(f"Compressed .pkl  : {len(latent_files)} files | {latent_total_size/1024:.2f} KB total")

print("\n=== Compression Metrics ===")
print(f"Average compressed bpp   : {compressed_bpp:.3f}")
print(f"Compression ratio (PLY→PKL): {compression_ratio:.2f}× smaller")

# === Optional: check average per point cloud ===
avg_pred_size = pred_size / len(pred_files)
avg_compressed_size = latent_total_size / len(pred_files)
print(f"\nAverage per PLY:")
print(f"  Pred PLY size      : {avg_pred_size/1024:.2f} KB")
print(f"  Compressed size    : {avg_compressed_size/1024:.2f} KB")
print(f"  Avg compression ratio: {(avg_pred_size/avg_compressed_size):.2f}× smaller")
