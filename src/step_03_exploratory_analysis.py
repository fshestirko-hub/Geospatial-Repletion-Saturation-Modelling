import os
import logging
from pathlib import Path
import numpy as np
import matplotlib
# Force headless rendering engine for server portability
# Set matplotlib backend to Agg to allow chart generation on headless server environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, var_samp, expr
from src.step_04_plots_3d import plot_multivariate_user_attractors_3d

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


def plot_linear_series(spark, phone_df, plots_dir):
    # Check if the plots exist in the target directory.
    plot_filenames = ["linear_series.png"]
    if check_plots_exist(plot_filenames, plots_dir):
        return

    logging.info("Constructing continuous signal time series plots...")
    
    # Isolate user 'a' phone telemetry data.
    user_phone = phone_df.filter(col("User") == 'a')
    
    color_x, color_y, color_z = 'firebrick', 'steelblue', 'darkslategray'
    activities = ['stand', 'walk', 'bike']
    fig, axes = plt.subplots(len(activities), 1, figsize=(11, 7), sharex=True)
    
    for i, act in enumerate(activities):
        act_df = user_phone.filter(col("gt") == act) \
                           .orderBy("Creation_Time") \
                           .limit(500) \
                           .select("x", "y", "z")
        rows = act_df.collect()
        
        if len(rows) > 0:
            x_vals = [r['x'] for r in rows]
            y_vals = [r['y'] for r in rows]
            z_vals = [r['z'] for r in rows]
            t = np.arange(len(rows))
            
            axes[i].plot(t, x_vals, label='X axis', color=color_x, alpha=0.85, linewidth=1.2)
            axes[i].plot(t, y_vals, label='Y axis', color=color_y, alpha=0.85, linewidth=1.2)
            axes[i].plot(t, z_vals, label='Z axis', color=color_z, alpha=0.85, linewidth=1.2)
            axes[i].set_title(f"Activity trace: {act} (User 'a')", fontsize=11, loc='left')
            axes[i].set_ylabel("Acceleration (m/s²)", fontsize=9)
            if i == 0:
                axes[i].legend(loc='upper right', frameon=True)
        else:
            axes[i].text(0.5, 0.5, f"No matching data slice found for: {act}", ha='center', va='center')
            
    axes[-1].set_xlabel("Relative sample sequence index", fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "linear_series.png", dpi=150)
    plt.close()
    logging.info("Linear series plot exported successfully.")


def plot_variance_comparison(spark, phone_df, watch_df, plots_dir):
    # Check if the plots exist in the target directory.
    plot_filenames = ["phone_vs_watch_comparison.png"]
    if check_plots_exist(plot_filenames, plots_dir):
        return

    logging.info("Aggregating multi-device statistical variances...")
    user_phone = phone_df.filter(col("User") == 'a')
    user_watch = watch_df.filter(col("User") == 'a')
    
    comparison_data = []
    
    for device_name, df_device in [('phone', user_phone), ('watch', user_watch)]:
        agg_df = df_device.filter(col("gt").isin(["stand", "walk"])) \
                          .groupBy("gt") \
                          .agg(var_samp("x").alias("vx"), var_samp("y").alias("vy"), var_samp("z").alias("vz")) \
                          .withColumn("total_var", col("vx") + col("vy") + col("vz"))
                          
        for r in agg_df.collect():
            comparison_data.append({
                'device': device_name,
                'activity': r['gt'],
                'variance': r['total_var']
            })
            
    comp_map = {
        'device': [item['device'] for item in comparison_data],
        'activity': [item['activity'] for item in comparison_data],
        'total variance (m/s²)': [item['variance'] for item in comparison_data]
    }
    
    plt.figure(figsize=(7, 4.5))
    sns.barplot(
        data=comp_map, 
        x='activity', 
        y='total variance (m/s²)', 
        hue='device', 
        palette=['#5c768d', '#9aaab7']
    )
    plt.title("Acceleration dispersion metrics: phone vs. watch devices", fontsize=11, loc='left')
    plt.ylabel("Total variance log scale (sum of X, Y, Z variants)")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / "phone_vs_watch_comparison.png", dpi=150)
    plt.close()
    logging.info("Variance comparison plot exported successfully.")


def plot_gravity_magnitude(spark, phone_df, plots_dir):
    # Check if the plots exist in the target directory.
    plot_filenames = ["gravity_magnitude.png"]
    if check_plots_exist(plot_filenames, plots_dir):
        return

    logging.info("Computing magnitude parameters against earth gravity baseline...")
    user_phone = phone_df.filter(col("User") == 'a')
    activity_colors = {'stand': '#5c768d', 'walk': '#b85a5a', 'bike': '#4c8c72'}
    
    plt.figure(figsize=(10, 4.5))
    
    for act in ['stand', 'walk']:
        mag_df = user_phone.filter(col("gt") == act) \
                           .orderBy("Creation_Time") \
                           .limit(500) \
                           .withColumn("magnitude", expr("sqrt(x*x + y*y + z*z)")) \
                           .select("magnitude")
        rows = mag_df.collect()
        if len(rows) > 0:
            mags = [r['magnitude'] for r in rows]
            plt.plot(mags, label=f"{act} magnitude vector", color=activity_colors[act], alpha=0.85, linewidth=1.2)
            
    plt.axhline(y=9.80665, color='firebrick', linestyle='--', alpha=0.7, label='Standard earth gravity constant (~9.81 m/s²)')
    plt.title("Resultant acceleration magnitude trace", fontsize=11, loc='left')
    plt.ylabel("Magnitude $|a| = \\sqrt{x^2 + y^2 + z^2}$ (m/s²)")
    plt.xlabel("Relative sample sequence index")
    plt.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "gravity_magnitude.png", dpi=150)
    plt.close()
    logging.info("Gravity magnitude plot exported successfully.")


def plot_population_heterogeneity(spark, phone_df, plots_dir):
    # Check if the plots exist in the target directory.
    plot_filenames = ["population_heterogeneity_violins.png"]
    if check_plots_exist(plot_filenames, plots_dir):
        return

    logging.info("Calculating acceleration magnitudes and filtering across population...")
    
    target_activities = ['stand', 'walk', 'bike']
    population_df = phone_df \
        .filter(col("gt").isin(target_activities)) \
        .withColumn("magnitude", expr("sqrt(x*x + y*y + z*z)")) \
        .select("User", "gt", "magnitude")
    
    sampled_pop_df = population_df.sample(withReplacement=False, fraction=0.002, seed=42)
    
    logging.info("Collecting sampled distribution matrix into driver memory...")
    pdf_violin = sampled_pop_df.toPandas()
    pdf_violin = pdf_violin.sort_values(by=['gt', 'User'])
    
    logging.info("Generating population heterogeneity violin matrix...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(len(target_activities), 1, figsize=(12, 9), sharex=True)
    activity_palette = {'stand': '#5c768d', 'walk': '#b85a5a', 'bike': '#4c8c72'}
    
    for i, act in enumerate(target_activities):
        sub_df = pdf_violin[pdf_violin['gt'] == act]
        
        sns.violinplot(
            ax=axes[i],
            data=sub_df,
            x='User',
            y='magnitude',
            color=activity_palette[act],
            inner='quartile',
            linewidth=1.2,
            density_norm='width'
        )
        
        axes[i].set_title(f"Kinetic signature profile density: {act}", fontsize=11, loc='left')
        axes[i].set_ylabel("Magnitude (m/s²)", fontsize=9)
        axes[i].set_xlabel("")
        axes[i].grid(True, linestyle='--', alpha=0.4)
        
    axes[-1].set_xlabel("Anonymous citizen identifier (original study participants)", fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "population_heterogeneity_violins.png", dpi=150)
    plt.close()
    logging.info("Violin matrix exported successfully.")