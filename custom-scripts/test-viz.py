import os
import numpy as np
import open3d as o3d

# === CONFIG ===
base_dir = "/home/aniruth/Desktop/Courses/Independent - Study/D-PCC/output/2025-10-31T03:07:02.977277"
gt_merge_dir = os.path.join(base_dir, "pcd/merge/gt")
pred_merge_dir = os.path.join(base_dir, "pcd/merge/pred")
latent_dir = os.path.join(base_dir, "pcd/compressed")  # folder where all .pkl latent files are stored
viz_output = os.path.join(base_dir, "pcd/merge/combined")  # folder to save combined visualization

test_number = 1  # which point cloud to visualize (0-indexed)

gt_path = os.path.join(gt_merge_dir, f"{test_number}.ply")
pred_path = os.path.join(pred_merge_dir, f"{test_number}.ply")

if os.path.exists(gt_path) and os.path.exists(pred_path):
    gt_pcd = o3d.io.read_point_cloud(gt_path)
    pred_pcd = o3d.io.read_point_cloud(pred_path)

    # Paint GT red, Pred green
    gt_pcd.paint_uniform_color([1.0, 0.0, 0.0])   # Red = ground truth
    pred_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # Green = prediction

    # Combine both
    combined = gt_pcd + pred_pcd
    combined_path = os.path.join(viz_output, f"combined_{test_number}.ply")

    # Save combined PLY
    o3d.io.write_point_cloud(combined_path, combined)
    print(f"\n✅ Combined GT+Pred saved at: {combined_path}")

    # Visualize
    o3d.visualization.draw_geometries(
        [gt_pcd, pred_pcd],
        window_name="GT (Red) vs Pred (Green)",
        width=1280,
        height=720
    )
else:
    print(f"\n⚠️ Could not find {test_number}.ply in GT or Pred folder!")
