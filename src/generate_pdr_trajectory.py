import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force headless Matplotlib backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType

def generate_pdr_trajectory():
    # Setup paths.
    # Determine the directories dynamically based on the repository structure.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    raw_dir = os.path.join(workspace_dir, "data", "raw", "Activity recognition exp")
    plots_dir = os.path.join(workspace_dir, "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    gif_path = os.path.join(plots_dir, "walk_pdr_trajectory_2d.gif")
    png_path = os.path.join(plots_dir, "walk_pdr_trajectory_2d.png")
    
    # Cache verification: Check if both plots exist to skip intensive processing.
    if os.path.exists(gif_path) and os.path.exists(png_path):
        print("DEBUG: PDR trajectory plots already exist. Skipping generation.")
        return
        
    acc_path = os.path.join(raw_dir, "Phones_accelerometer.csv")
    gyro_path = os.path.join(raw_dir, "Phones_gyroscope.csv")
    
    # Compile an explicit schema to skip high-latency CSV pre-scanning.
    csv_schema = StructType([
        StructField("Index", LongType(), True),
        StructField("Arrival_Time", LongType(), True),
        StructField("Creation_Time", LongType(), True),
        StructField("x", DoubleType(), True),
        StructField("y", DoubleType(), True),
        StructField("z", DoubleType(), True),
        StructField("User", StringType(), True),
        StructField("Model", StringType(), True),
        StructField("Device", StringType(), True),
        StructField("gt", StringType(), True)
    ])
    
    # Check if an active Spark session already exists or is configured in the environment.
    active_session = SparkSession.getActiveSession()
    session_created = False
    
    if active_session is not None:
        spark = active_session
        print("DEBUG: Reusing active Spark Session for PDR Trajectory.")
    else:
        print("DEBUG: Initialising new Spark Session for PDR Trajectory...")
        spark_builder = SparkSession.builder \
            .appName("HHAR-PDR-Trajectory") \
            .config("spark.sql.shuffle.partitions", "4")
        
        # Check if master is specified by environment variables or system properties.
        # If not, fall back to "local[*]" and allocate 4g memory for local development stability.
        master_url = os.environ.get("SPARK_MASTER")
        if not master_url and not any(env.startswith("SPARK_") for env in os.environ):
            spark_builder = spark_builder.master("local[*]") \
                                         .config("spark.driver.memory", "4g")
            
        spark = spark_builder.getOrCreate()
        session_created = True
        
    try:
        print("DEBUG: Reading sensor data with explicit schema...")
        df_acc_raw = spark.read.schema(csv_schema).csv(acc_path, header=True)
        df_gyro_raw = spark.read.schema(csv_schema).csv(gyro_path, header=True)
        
        # Clean columns by removing white space from headers.
        for c in df_acc_raw.columns:
            df_acc_raw = df_acc_raw.withColumnRenamed(c, c.strip())
        for c in df_gyro_raw.columns:
            df_gyro_raw = df_gyro_raw.withColumnRenamed(c, c.strip())
            
        print("DEBUG: Filtering for User 'a', Nexus 4, and Walking activity...")
        # We target a single device nexus4_1 to keep streams perfectly synchronised.
        df_acc = df_acc_raw.filter((col("User") == "a") & (col("Model") == "nexus4") & (col("Device") == "nexus4_1") & (col("gt") == "walk"))
        df_gyro = df_gyro_raw.filter((col("User") == "a") & (col("Model") == "nexus4") & (col("Device") == "nexus4_1") & (col("gt") == "walk"))
        
        # Downsample to 100 Hz bins (10 ms resolution).
        # Creation_Time is in nanoseconds, so dividing by 10,000,000 gives 10ms bins.
        print("DEBUG: Binning and aligning sensors in Spark...")
        df_acc_bin = df_acc.withColumn("time_bin", (col("Creation_Time") / 10000000).cast("long")) \
                           .groupBy("time_bin") \
                           .agg(mean("x").alias("ax"), mean("y").alias("ay"), mean("z").alias("az"))
                           
        df_gyro_bin = df_gyro.withColumn("time_bin", (col("Creation_Time") / 10000000).cast("long")) \
                             .groupBy("time_bin") \
                             .agg(mean("x").alias("gx"), mean("y").alias("gy"), mean("z").alias("gz"))
                             
        # Inner join to align timestamps.
        aligned_df = df_acc_bin.join(df_gyro_bin, ["time_bin"], "inner").orderBy("time_bin")
        
        # Pull a 15-second window (1500 samples at 100 Hz).
        print("DEBUG: Collecting 1500 samples (15 seconds) of walking data...")
        rows = aligned_df.limit(1500).collect()
        
        if len(rows) < 100:
            raise ValueError(f"Insufficient aligned rows found: {len(rows)}")
            
        print(f"DEBUG: Successfully aligned {len(rows)} samples.")
        
        # Extract columns.
        ax_vals = np.array([r['ax'] for r in rows])
        ay_vals = np.array([r['ay'] for r in rows])
        az_vals = np.array([r['az'] for r in rows])
        
        gx_vals = np.array([r['gx'] for r in rows])
        gy_vals = np.array([r['gy'] for r in rows])
        gz_vals = np.array([r['gz'] for r in rows])
        
        # Time array (10ms steps).
        dt = 0.01
        t = np.arange(len(rows)) * dt
        
        # Compute accelerometer magnitude.
        acc_mag = np.sqrt(ax_vals**2 + ay_vals**2 + az_vals**2)
        
        # Low-pass filter accelerometer magnitude to isolate stepping cycles.
        # We try to use SciPy, with a fallback rolling average if SciPy is not available.
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            print("DEBUG: Applying SciPy Butterworth low-pass filter...")
            b, a_coeff = butter(3, 3.0, fs=100)
            acc_mag_filtered = filtfilt(b, a_coeff, acc_mag)
            
            # Find peaks representing individual steps.
            # Typical step rate: 1.5 - 2.5 Hz (60-100 ms between steps).
            # Min distance: 40 samples (400 ms) to avoid double counting within one stride.
            print("DEBUG: Detecting steps via SciPy peak-finding...")
            peaks, _ = find_peaks(acc_mag_filtered, height=10.2, distance=40)
        except ImportError:
            print("DEBUG: SciPy not available. Falling back to moving average and custom peak finder...")
            # Moving average filter.
            window = 15
            acc_mag_filtered = np.convolve(acc_mag, np.ones(window)/window, mode='same')
            
            # Simple peak finder.
            peaks = []
            min_dist = 40
            last_peak = -min_dist
            for i in range(1, len(acc_mag_filtered) - 1):
                if acc_mag_filtered[i] > 10.2:
                    if acc_mag_filtered[i] > acc_mag_filtered[i-1] and acc_mag_filtered[i] > acc_mag_filtered[i+1]:
                        if i - last_peak >= min_dist:
                            peaks.append(i)
                            last_peak = i
            peaks = np.array(peaks)
            
        print(f"DEBUG: Detected {len(peaks)} steps in the 15-second window.")
        
        # Integrate gyroscope yaw rate to track heading (yaw angle).
        # For a standard phone in a pocket, rotation around Z or Y coordinates represents heading.
        # Let's integrate gz (yaw rate around Z axis) to track heading over time.
        # We apply a slight high-pass/bias filter by removing the mean to reduce integration drift.
        gz_bias_removed = gz_vals - np.mean(gz_vals)
        heading = np.cumsum(gz_bias_removed) * dt
        
        # Reconstruct 2D Trajectory.
        # Position updates only when a step occurs.
        # We step 0.70 metres (standard step length) in the direction of current heading.
        step_len = 0.70
        x_pos = np.zeros_like(t)
        y_pos = np.zeros_like(t)
        
        curr_x, curr_y = 0.0, 0.0
        peak_idx = 0
        step_coords = []
        
        for i in range(len(t)):
            if peak_idx < len(peaks) and i == peaks[peak_idx]:
                theta = heading[i]
                curr_x += step_len * np.cos(theta)
                curr_y += step_len * np.sin(theta)
                step_coords.append((curr_x, curr_y, i))
                peak_idx += 1
            x_pos[i] = curr_x
            y_pos[i] = curr_y
            
        # ----------------------------------------------------
        # Matplotlib Premium Visualisation (Dual Panel)
        # ----------------------------------------------------
        fig = plt.figure(figsize=(14, 8), facecolor='#121212')
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1])
        
        # Left Panel: Reconstructed Trajectory.
        ax_traj = fig.add_subplot(gs[:, 0], facecolor='#1e1e1e')
        ax_traj.set_title("Reconstructed Walk Trajectory (Pedestrian Dead Reckoning)", color='white', fontsize=12, pad=15)
        ax_traj.set_xlabel("X Position (metres)", color='#aaaaaa')
        ax_traj.set_ylabel("Y Position (metres)", color='#aaaaaa')
        ax_traj.tick_params(colors='#aaaaaa')
        ax_traj.grid(True, color='#333333', linestyle='--')
        
        # Right Panel Top: Accelerometer Step Detection.
        ax_acc = fig.add_subplot(gs[0, 1], facecolor='#1e1e1e')
        ax_acc.set_title("Step Detection (Accelerometer Magnitude)", color='white', fontsize=11)
        ax_acc.set_ylabel("Magnitude |a| (m/s²)", color='#aaaaaa')
        ax_acc.tick_params(colors='#aaaaaa')
        ax_acc.grid(True, color='#333333', linestyle='--')
        
        # Right Panel Bottom: Gyroscope Yaw Rate & Heading.
        ax_gyro = fig.add_subplot(gs[1, 1], facecolor='#1e1e1e')
        ax_gyro.set_title("Heading Angle & Rotational Velocity", color='white', fontsize=11)
        ax_gyro.set_xlabel("Time (seconds)", color='#aaaaaa')
        ax_gyro.set_ylabel("Heading (rad) / Yaw (rad/s)", color='#aaaaaa')
        ax_gyro.tick_params(colors='#aaaaaa')
        ax_gyro.grid(True, color='#333333', linestyle='--')
        
        # ----------------------------------------------------
        # Setup static components for static PNG output first.
        # ----------------------------------------------------
        # Trajectory Plotting.
        ax_traj.plot(x_pos, y_pos, color='#00F2FE', lw=3, label="Estimated Trajectory")
        ax_traj.scatter([0], [0], color='#FF5733', s=100, label="Start Point", zorder=5)
        
        # Step points.
        step_x = [pt[0] for pt in step_coords]
        step_y = [pt[1] for pt in step_coords]
        ax_traj.scatter(step_x, step_y, color='#39FF14', s=45, marker='o', edgecolors='white', label="Steps Registered", zorder=4)
        
        # Heading arrow at the end.
        if len(x_pos) > 0:
            last_idx = len(x_pos) - 1
            theta = heading[last_idx]
            ax_traj.quiver(x_pos[last_idx], y_pos[last_idx], np.cos(theta), np.sin(theta), 
                           color='#FF007F', scale=15, width=0.015, label="Final Heading")
                           
        ax_traj.legend(facecolor='#1e1e1e', labelcolor='white', loc='upper left')
        
        # Accelerometer Plotting.
        ax_acc.plot(t, acc_mag, color='#444444', alpha=0.5, label="Raw Acceleration")
        ax_acc.plot(t, acc_mag_filtered, color='#00F2FE', lw=2, label="Filtered Signal")
        ax_acc.scatter(t[peaks], acc_mag_filtered[peaks], color='#39FF14', s=40, label="Step Peaks", zorder=3)
        ax_acc.axhline(y=10.2, color='#FF5733', linestyle='--', alpha=0.7, label="Step Threshold")
        ax_acc.legend(facecolor='#1e1e1e', labelcolor='white', loc='upper right')
        
        # Gyroscope Plotting.
        ax_gyro.plot(t, gz_vals, color='#E28490', alpha=0.4, label="Yaw Rate (g_z)")
        ax_gyro.plot(t, heading, color='#FFCC00', lw=2, label="Yaw Heading (integrated)")
        ax_gyro.legend(facecolor='#1e1e1e', labelcolor='white', loc='upper right')
        
        # Adjust margins.
        plt.tight_layout()
        
        # Save static PNG.
        print(f"DEBUG: Saving static PDR Trajectory image to: {png_path}...")
        plt.savefig(png_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        print("DEBUG: Static PDR image saved successfully.")
        
        # ----------------------------------------------------
        # Animation Setup.
        # ----------------------------------------------------
        # Clear plots to setup dynamic artists.
        ax_traj.clear()
        ax_acc.clear()
        ax_gyro.clear()
        
        # Re-apply styles.
        ax_traj.set_facecolor('#1e1e1e')
        ax_traj.set_title("Reconstructed Walk Trajectory (Pedestrian Dead Reckoning)", color='white', fontsize=12, pad=15)
        ax_traj.grid(True, color='#333333', linestyle='--')
        ax_traj.set_xlim(min(x_pos) - 2, max(x_pos) + 2)
        ax_traj.set_ylim(min(y_pos) - 2, max(y_pos) + 2)
        
        ax_acc.set_facecolor('#1e1e1e')
        ax_acc.set_title("Step Detection (Accelerometer Magnitude)", color='white', fontsize=11)
        ax_acc.grid(True, color='#333333', linestyle='--')
        ax_acc.set_xlim(0, t[-1])
        ax_acc.set_ylim(min(acc_mag) - 1, max(acc_mag) + 1)
        
        ax_gyro.set_facecolor('#1e1e1e')
        ax_gyro.set_title("Heading Angle & Rotational Velocity", color='white', fontsize=11)
        ax_gyro.grid(True, color='#333333', linestyle='--')
        ax_gyro.set_xlim(0, t[-1])
        ax_gyro.set_ylim(min(heading) - 0.5, max(heading) + 0.5)
        
        # Artists for animation.
        # Left Panel (Trajectory).
        traj_line, = ax_traj.plot([], [], color='#00F2FE', lw=3, label="Pedestrian Trail")
        start_dot, = ax_traj.plot([0], [0], 'o', color='#FF5733', ms=8, label="Start Point")
        agent_dot, = ax_traj.plot([], [], 'o', color='#FF007F', ms=9, label="Pedestrian Position")
        arrow = ax_traj.quiver(0, 0, 1, 0, color='#FF007F', scale=10, width=0.015, zorder=5)
        step_scatter = ax_traj.scatter([], [], color='#39FF14', s=45, edgecolors='white', zorder=4, label="Registered Steps")
        
        # Right Top Panel (Accelerometer).
        acc_raw_line, = ax_acc.plot(t, acc_mag, color='#444444', alpha=0.4, label="Raw Acceleration")
        acc_filt_line, = ax_acc.plot(t, acc_mag_filtered, color='#00F2FE', lw=2, label="Filtered Signal")
        acc_scan_line = ax_acc.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        acc_step_scatter = ax_acc.scatter([], [], color='#39FF14', s=40, zorder=3, label="Detected Steps")
        
        # Right Bottom Panel (Gyroscope).
        gyro_raw_line, = ax_gyro.plot(t, gz_vals, color='#E28490', alpha=0.4, label="Yaw Rate (g_z)")
        gyro_head_line, = ax_gyro.plot(t, heading, color='#FFCC00', lw=2, label="Yaw Heading")
        gyro_scan_line = ax_gyro.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        
        # Legends.
        ax_traj.legend(facecolor='#1e1e1e', labelcolor='white', loc='upper left')
        ax_acc.legend(facecolor='#1e1e1e', labelcolor='white', loc='upper right')
        ax_gyro.legend(facecolor='#1e1e1e', labelcolor='white', loc='upper right')
        
        # Sub-sampling animation frames to run efficiently (render every 5th frame, i.e., 50ms interval).
        frame_indices = range(0, len(rows), 5)
        
        def update_anim(frame):
            idx = frame_indices[frame]
            
            # 1. Update Trajectory Panel.
            traj_line.set_data(x_pos[:idx], y_pos[:idx])
            agent_dot.set_data([x_pos[idx]], [y_pos[idx]])
            
            # Update Arrow Orientation.
            nonlocal arrow
            arrow.remove()
            theta = heading[idx]
            arrow = ax_traj.quiver(x_pos[idx], y_pos[idx], np.cos(theta), np.sin(theta), 
                                   color='#FF007F', scale=15, width=0.015, zorder=5)
                                   
            # Update step points registered so far.
            steps_so_far = [pt for pt in step_coords if pt[2] <= idx]
            if len(steps_so_far) > 0:
                s_x = [pt[0] for pt in steps_so_far]
                s_y = [pt[1] for pt in steps_so_far]
                step_scatter.set_offsets(np.column_stack((s_x, s_y)))
            else:
                step_scatter.set_offsets(np.empty((0, 2)))
                
            # 2. Update Accelerometer Panel.
            acc_scan_line.set_xdata([t[idx]])
            peaks_so_far = [p for p in peaks if p <= idx]
            if len(peaks_so_far) > 0:
                acc_step_scatter.set_offsets(np.column_stack((t[peaks_so_far], acc_mag_filtered[peaks_so_far])))
            else:
                acc_step_scatter.set_offsets(np.empty((0, 2)))
                
            # 3. Update Gyroscope Panel.
            gyro_scan_line.set_xdata([t[idx]])
            
            return traj_line, agent_dot, arrow, step_scatter, acc_scan_line, acc_step_scatter, gyro_scan_line
            
        print(f"DEBUG: Rendering PDR Trajectory animation to: {gif_path}...")
        ani = animation.FuncAnimation(fig, update_anim, frames=len(frame_indices), interval=50, blit=False)
        ani.save(gif_path, writer='pillow', fps=20)
        print("DEBUG: Dynamic PDR animation rendering complete!")
        
    except Exception as e:
        print(f"DEBUG: Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if session_created:
            print("DEBUG: Stopping local Spark Session...")
            spark.stop()
        else:
            print("DEBUG: Keeping active Spark Session alive for downstream tasks.")

if __name__ == "__main__":
    generate_pdr_trajectory()
