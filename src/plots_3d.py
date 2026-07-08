import os
import logging
import numpy as np
import matplotlib
# Force headless rendering engine for server portability
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyspark.sql.functions import col

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_plots_exist(plot_filenames, plots_dir) -> bool:
    # Check whether the plots directory exists; create it if missing.
    if not plots_dir.exists():
        logging.warning(f"Plots directory '{plots_dir}' is missing. Creating directory.")
        plots_dir.mkdir(parents=True, exist_ok=True)
        return False
        
    missing = [f for f in plot_filenames if not (plots_dir / f).exists()]
    if missing:
        logging.info(f"Missing plot files: {missing}. Plot generation required.")
        return False
        
    logging.info(f"All target plots {plot_filenames} already exist in '{plots_dir}'. Skipping generation.")
    return True

def plot_attractor_geometries(spark, phone_df, plots_dir):
    # Check if the plots exist in the target directory.
    plot_filenames = ["topographic_map.png"]
    if check_plots_exist(plot_filenames, plots_dir):
        return

    logging.info("Generating phase-space attractor distributions (2D & 3D)...")
    
    # Isolate user 'a' phone telemetry data.
    user_phone = phone_df.filter(col("User") == 'a')
    
    activities = ['stand', 'walk', 'bike']
    activity_colors = {'stand': '#5c768d', 'walk': '#b85a5a', 'bike': '#4c8c72'}
    
    # Render attractor grids in 2D and 3D.
    fig = plt.figure(figsize=(14, 6))
    ax2d = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection='3d')
    
    for act in activities:
        act_df = user_phone.filter(col("gt") == act).orderBy("Creation_Time").limit(1000).select("x", "y", "z")
        rows = act_df.collect()
        
        if len(rows) > 0:
            x_vals = [r['x'] for r in rows]
            y_vals = [r['y'] for r in rows]
            z_vals = [r['z'] for r in rows]
            
            ax2d.scatter(x_vals, y_vals, label=act, color=activity_colors[act], alpha=0.4, s=8)
            ax3d.scatter(x_vals, y_vals, z_vals, label=act, color=activity_colors[act], alpha=0.3, s=6)
            
    ax2d.set_title("2D acceleration space projection (X vs Y)", fontsize=11, loc='left')
    ax2d.set_xlabel("X-vector acceleration (m/s²)")
    ax2d.set_ylabel("Y-vector acceleration (m/s²)")
    ax2d.legend(frameon=True)
    
    ax3d.set_title("3D structural attractor point cloud", fontsize=11, loc='left')
    ax3d.set_xlabel("X (m/s²)")
    ax3d.set_ylabel("Y (m/s²)")
    ax3d.set_zlabel("Z (m/s²)")
    
    plt.tight_layout()
    plt.savefig(plots_dir / "topographic_map.png", dpi=150)
    plt.close()
    logging.info("Attractor geometries plot exported successfully.")

def plot_multivariate_user_attractors_3d(spark, phone_df, plots_dir):
    # Check if the plots exist in the target directory.
    plot_filenames = ["multivariate_user_attractors_3d.png"]
    if check_plots_exist(plot_filenames, plots_dir):
        return

    logging.info("Generating 3D joint distribution attractor plot...")
    
    # Select three distinct users to contrast walk signatures.
    contrast_users = ['a', 'b', 'c']
    target_activity = 'walk'
    
    attractor_segments = []
    for u in contrast_users:
        user_segment = phone_df \
            .filter((col("User") == u) & (col("gt") == target_activity)) \
            .orderBy("Creation_Time") \
            .limit(800) \
            .select("User", "x", "y", "z")
        
        attractor_segments.append(user_segment)
        
    joint_attractor_df = attractor_segments[0]
    for next_segment in attractor_segments[1:]:
        joint_attractor_df = joint_attractor_df.union(next_segment)
        
    pdf_attractor = joint_attractor_df.toPandas()
    
    # Render the 3D phase-space plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    user_colors = {'a': '#4a7c59', 'b': '#b85a5a', 'c': '#3a5a40'}
    user_markers = {'a': 'o', 'b': '^', 'c': 's'}
    
    for u in contrast_users:
        user_data = pdf_attractor[pdf_attractor['User'] == u]
        
        ax.scatter(
            user_data['x'], 
            user_data['y'], 
            user_data['z'], 
            label=f"User '{u}' footprint", 
            color=user_colors[u],
            marker=user_markers[u],
            alpha=0.3,
            s=12,
            edgecolors='none'
        )
        
    ax.set_title(f"3D joint multi-user spatial acceleration space (Activity: {target_activity})", fontsize=11, loc='left')
    ax.set_xlabel("X-axis acceleration orientation (m/s²)", fontsize=9)
    ax.set_ylabel("Y-axis acceleration orientation (m/s²)", fontsize=9)
    ax.set_zlabel("Z-axis acceleration orientation (m/s²)", fontsize=9)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.3)
    
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    
    plt.savefig(plots_dir / "multivariate_user_attractors_3d.png", dpi=150)
    plt.close()
    logging.info("3D joint distribution plot exported successfully.")
