import os
import logging
import numpy as np
import pandas as pd
import matplotlib
# Force headless rendering engine for server portability
# Set matplotlib backend to Agg to allow chart generation on headless server environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, expr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_plots_exist(plot_filenames, plots_dir) -> bool:
    # Checks whether the plots directory exists, warns and creates it recursively if missing.
    # Verify the plots directory and individual file availability, emitting warnings when directories must be created.
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

def plot_agent_kinematics(spark, synth_df, plots_dir):
    # Check if the plots exist in the target directory.
    # Evaluate the plot presence on the filesystem to prevent redundant CPU cycles and Spark executions.
    plot_filenames = ["diagnostic_agent_kinematics.png"]
    plot_path = plots_dir / "diagnostic_agent_kinematics.png"
    
    if not check_plots_exist(plot_filenames, plots_dir):
        logging.info("Extracting telemetry slice for synthetic agent 0 (walking verification)...")
        
        # Isolate agent 0 and compute the dynamic magnitude vector.
        # Filter the telemetry for walking segments of synthetic agent ID 0 and evaluate acceleration magnitudes.
        agent_sample_df = synth_df.filter((col("Agent_ID") == 0) & (col("Activity") == "walk")) \
                                  .orderBy("Timestamp") \
                                  .limit(400) \
                                  .withColumn("magnitude", expr("sqrt(ax*ax + ay*ay + az*az)"))
                                  
        pdf_agent = agent_sample_df.toPandas()
        
        # Render component forces and magnitude traces.
        # Plot the axial component forces and resultant magnitudes relative to the standard gravity baseline.
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        t = np.arange(len(pdf_agent))
        
        ax1.plot(t, pdf_agent['ax'], label='X axis', color='firebrick', alpha=0.8, linewidth=1.1)
        ax1.plot(t, pdf_agent['ay'], label='Y axis', color='steelblue', alpha=0.8, linewidth=1.1)
        ax1.plot(t, pdf_agent['az'], label='Z axis', color='darkslategray', alpha=0.8, linewidth=1.1)
        ax1.set_title("synthetic agent kinematic signature: component forces (agent 0 - walk)", fontsize=11, loc='left')
        ax1.set_ylabel("acceleration (m/s²)", fontsize=9)
        ax1.legend(loc='upper right', frameon=True)
        
        ax2.plot(t, pdf_agent['magnitude'], color='#b85a5a', alpha=0.9, linewidth=1.3, label='resultant magnitude')
        ax2.axhline(y=9.81, color='black', linestyle=':', alpha=0.4, label='earth gravity baseline')
        ax2.set_title("synthetic agent resultant magnitude profile (verification of smooth boundary transitions)", fontsize=11, loc='left')
        ax2.set_ylabel("magnitude |a| (m/s²)", fontsize=9)
        ax2.set_xlabel("continuous streaming sample index (10ms steps)", fontsize=10)
        ax2.legend(loc='upper right', frameon=True)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info("Kinematic trace diagnostic plot exported.")
        
    # Display inline in IPython/Jupyter notebook environments.
    try:
        from IPython.display import display, Image
        display(Image(filename=str(plot_path)))
    except (ImportError, KeyError, AttributeError):
        pass

def plot_population_density_comparison(spark, phone_df, synth_df, plots_dir):
    # Check if the plots exist in the target directory.
    # Evaluate the plot presence on the filesystem to prevent redundant CPU cycles and Spark executions.
    plot_filenames = ["diagnostic_population_violins.png"]
    plot_path = plots_dir / "diagnostic_population_violins.png"
    
    if not check_plots_exist(plot_filenames, plots_dir):
        logging.info("Preparing real vs synthetic population density comparison matrices...")
        
        # 1. Isolate and down-sample the original empirical data (from phone_df).
        # Extract the empirical walk and bike magnitudes and down-sample to preserve shape.
        real_pop = phone_df.filter(col("gt").isin(["walk", "bike"])) \
                           .withColumn("magnitude", expr("sqrt(x*x + y*y + z*z)")) \
                           .select("gt", "magnitude") \
                           .withColumn("source", expr("'empirical (original 9 users)'")) \
                           .sample(False, 0.002, seed=42).toPandas()
                           
        # 2. Isolate and down-sample the new synthetic generated population.
        # Extract the synthetic walk and bike magnitudes and down-sample for plotting.
        synth_pop = synth_df.filter(col("Activity").isin(["walk", "bike"])) \
                            .withColumn("magnitude", expr("sqrt(ax*ax + ay*ay + az*az)")) \
                            .select(col("Activity").alias("gt"), "magnitude") \
                            .withColumn("source", expr("'synthetic (simulated agents)'")) \
                            .sample(False, 0.05, seed=42).toPandas()
                            
        # Combine the local records into a single verification structure.
        # Concatenate empirical and synthetic DataFrames for the Seaborn violin representation.
        diagnostic_pop_df = pd.concat([real_pop, synth_pop], axis=0)
        
        # Render the side-by-side comparison matrix.
        # Plot split violins displaying comparative distribution parameters.
        plt.figure(figsize=(10, 5.5))
        sns.violinplot(
            data=diagnostic_pop_df,
            x='gt',
            y='magnitude',
            hue='source',
            split=True,
            palette=['#5c768d', '#b85a5a'],
            inner='quartile',
            linewidth=1.2
        )
        
        plt.title("population profile verification: empirical vs. synthetic distribution properties", fontsize=11, loc='left')
        plt.ylabel("absolute acceleration magnitude (m/s²)")
        plt.xlabel("mode of infrastructure transit activity")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(title="data vector layer source", loc='upper right', frameon=True)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info("Population density validation matrix exported.")
        
    # Display inline in IPython/Jupyter notebook environments.
    try:
        from IPython.display import display, Image
        display(Image(filename=str(plot_path)))
    except (ImportError, KeyError, AttributeError):
        pass

def plot_temporal_autocorrelation(spark, phone_df, synth_df, plots_dir):
    # Check if the plots exist in the target directory.
    # Evaluate the plot presence on the filesystem to prevent redundant CPU cycles and Spark executions.
    plot_filenames = ["diagnostic_temporal_acf.png"]
    plot_path = plots_dir / "diagnostic_temporal_acf.png"
    
    if not check_plots_exist(plot_filenames, plots_dir):
        logging.info("Running temporal autocorrelation (ACF) congruence calculations...")
        
        # Extract a contiguous clean walking signal from a real user and a synthetic agent.
        # Retrieve the contiguous walking sequences from the original and synthetic records.
        real_signal = phone_df.filter((col("User") == 'a') & (col("gt") == 'walk')).orderBy("Creation_Time").limit(500).select(expr("sqrt(x*x + y*y + z*z)").alias("m")).toPandas()['m'].values
        synth_signal = synth_df.filter((col("Agent_ID") == 0) & (col("Activity") == 'walk')).orderBy("Timestamp").limit(500).select(expr("sqrt(ax*ax + ay*ay + az*az)").alias("m")).toPandas()['m'].values
        
        # Calculate Autocorrelation Function (ACF) lines manually using NumPy.
        # Define a local ACF helper function to evaluate coefficients across time lags.
        def compute_acf(signal, max_lag=120):
            mean_val = np.mean(signal)
            var_val = np.var(signal)
            norm_signal = signal - mean_val
            acf_vals = []
            for lag in range(max_lag):
                if lag == 0:
                    acf_vals.append(1.0)
                else:
                    covariance = np.mean(norm_signal[:-lag] * norm_signal[lag:])
                    acf_vals.append(covariance / var_val)
            return acf_vals
            
        max_lags = 100
        real_acf = compute_acf(real_signal, max_lags)
        synth_acf = compute_acf(synth_signal, max_lags)
        
        # Render the comparison.
        # Plot the empirical and synthetic autocorrelation coefficients side by side.
        plt.figure(figsize=(10, 4.5))
        lags = np.arange(max_lags)
        plt.plot(lags, real_acf, label='empirical stride signature (user a)', color='#5c768d', linewidth=1.5)
        plt.plot(lags, synth_acf, label='synthetic stride signature (agent 0)', color='#b85a5a', linestyle='--', linewidth=1.5)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        plt.title("temporal dependency verification: autocorrelation (acf) profile alignment", fontsize=11, loc='left')
        plt.xlabel("temporal sample lag interval (10ms bins)")
        plt.ylabel("autocorrelation coefficient (ρ)")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(frameon=True)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info("Autocorrelation diagnostics complete. Shutting down diagnostic environment.")
        
    # Display inline in IPython/Jupyter notebook environments.
    try:
        from IPython.display import display, Image
        display(Image(filename=str(plot_path)))
    except (ImportError, KeyError, AttributeError):
        pass
