import os
import matplotlib
matplotlib.use('Agg') # Force headless Matplotlib backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def generate_gyro_animation(activity="walk", color="#33FF57", title_label="Walking"):
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    data_path = os.path.join(workspace_dir, "data", "raw", "Activity recognition exp", "Phones_gyroscope.csv")
    plots_dir = os.path.join(workspace_dir, "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    gif_path = os.path.join(plots_dir, f"{activity}_gyro_attractor_3d.gif")
    
    # Initialize Spark Session
    print(f"DEBUG: Initializing Spark Session for {title_label} Gyroscope animation...")
    spark = SparkSession.builder \
        .appName(f"HHAR-Gyro-Animation-{activity}") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    try:
        print("DEBUG: Loading dataset via Spark...")
        df = spark.read.csv(data_path, header=True, inferSchema=True)
        
        # Clean column names by stripping spaces
        for c in df.columns:
            df = df.withColumnRenamed(c, c.strip())
            
        print(f"DEBUG: Filtering {activity} Gyroscope telemetry for User 'a'...")
        activity_df = df.filter((col("User") == "a") & (col("gt") == activity))
        
        # Select first 200 samples sorted by time efficiently
        slice_df = activity_df.orderBy("Creation_Time").limit(200).select("x", "y", "z")
        
        print("DEBUG: Collecting records to local driver memory...")
        rows = slice_df.collect()
        
        # Gyroscope measures angular velocity (rad/s)
        x = [r['x'] for r in rows]
        y = [r['y'] for r in rows]
        z = [r['z'] for r in rows]
        
        print(f"DEBUG: Collected {len(x)} points. Preparing 3D rotational trajectory animation...")
        
        # ----------------------------------------------------
        # Matplotlib 3D Animation Setup
        # ----------------------------------------------------
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set titles and styles
        ax.set_title(f"3D {title_label} Gyroscope Rotational Attractor (User 'a' - Phone)", fontsize=14, pad=20)
        ax.set_xlabel("X Angular Velocity (rad/s)", fontsize=10)
        ax.set_ylabel("Y Angular Velocity (rad/s)", fontsize=10)
        ax.set_zlabel("Z Angular Velocity (rad/s)", fontsize=10)
        
        # Fix the axes limits to frame the rotational cycle perfectly
        ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
        ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
        ax.set_zlim(min(z) - 0.5, max(z) + 0.5)
        
        # Style grid and backgrounds
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # The line that draws the historical rotational trajectory
        line, = ax.plot([], [], [], lw=2.5, color=color, alpha=0.8, label=f"{title_label} Rotation Path")
        # The dot representing the current angular velocity vector
        point, = ax.plot([], [], [], 'ro', ms=7, color='#FF5733', label="Current Angular Velocity")
        
        # Subtle rotation animation of the camera view angle for professional look
        def update_frame(num):
            # Update path line
            line.set_data(x[:num], y[:num])
            line.set_3d_properties(z[:num])
            
            # Update current point
            if num > 0:
                point.set_data([x[num-1]], [y[num-1]])
                point.set_3d_properties([z[num-1]])
            else:
                point.set_data([], [])
                point.set_3d_properties([])
                
            # Slowly rotate camera angle to give 3D depth perception
            ax.view_init(elev=20, azim=30 + (num * 0.5))
            return line, point
            
        ax.legend(loc="upper right")
        
        print(f"DEBUG: Rendering animation to: {gif_path}...")
        ani = animation.FuncAnimation(fig, update_frame, frames=len(x), interval=25, blit=False)
        
        # Save as GIF using pillow writer
        ani.save(gif_path, writer='pillow', fps=30)
        print(f"DEBUG: 3D Gyroscope Attractor GIF rendering complete for {activity}!")
        
    except Exception as e:
        print(f"DEBUG: Error occurred: {str(e)}")
    finally:
        print("DEBUG: Stopping Spark Session...")
        spark.stop()

if __name__ == "__main__":
    # Generate walking and biking gyroscope animations sequentially
    generate_gyro_animation(activity="walk", color="#39FF14", title_label="Walking")
    generate_gyro_animation(activity="bike", color="#FFCC00", title_label="Biking")
