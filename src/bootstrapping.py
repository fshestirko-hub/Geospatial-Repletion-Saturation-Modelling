import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, mean, stddev, expr
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_project_root() -> Path:
    # locate project root relative to execution context
    # Determine the directory path to project root recursively from current working directory
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "data").exists():
            return parent
    return current

def setup_winutils(workspace_dir: Path):
    # configure Windows native binary environment variables
    # Initialize WinUtils and hadoop dll binaries dynamically on Windows hosts to skip filesystem driver warnings
    if os.name != 'nt':
        logging.info("Non-Windows OS detected. Skipping native WinUtils configuration.")
        return
        
    winutils_dir = workspace_dir / "data" / "winutils" / "bin"
    winutils_dir.mkdir(parents=True, exist_ok=True)
    
    winutils_path = winutils_dir / "winutils.exe"
    hadoop_dll_path = winutils_dir / "hadoop.dll"
    
    if not winutils_path.exists():
        logging.info("Downloading winutils.exe for native Windows local file support...")
        url = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(winutils_path))
        except Exception as e:
            logging.warning(f"Failed to download winutils.exe: {e}")
            
    if not hadoop_dll_path.exists():
        logging.info("Downloading hadoop.dll for native Windows local file support...")
        url = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/hadoop.dll"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(hadoop_dll_path))
        except Exception as e:
            logging.warning(f"Failed to download hadoop.dll: {e}")
            
    hadoop_home = winutils_dir.parent
    os.environ["HADOOP_HOME"] = str(hadoop_home)
    os.environ["PATH"] = str(winutils_dir) + os.path.pathsep + os.environ.get("PATH", "")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    logging.info(f"Hadoop environment path configuration active: HADOOP_HOME={hadoop_home}")

def run_bootstrapping(num_agents=50, samples_per_agent=1000):
    # configure paths and directories
    # Setup filesystem reference targets for source data files and outputs
    project_root = get_project_root()
    setup_winutils(project_root)
    
    raw_dir = project_root / "data" / "raw" / "Activity recognition exp"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    acc_path = raw_dir / "Phones_accelerometer.csv"
    gyro_path = raw_dir / "Phones_gyroscope.csv"
    parquet_out_path = processed_dir / "synthetic_telemetry.parquet"
    temp_json_path = processed_dir / "temp_synthetic.json"
    
    # Terminate active SparkSession to apply new JVM configurations
    try:
        active_session = SparkSession.getActiveSession()
        if active_session is not None:
            logging.info("Stopping active Spark Session to apply new configurations...")
            active_session.stop()
    except Exception as e:
        logging.warning(f"Error stopping active session: {e}")
        
    # build adaptive Spark session
    # Retrieve active session or instantiate new one with local configuration
    logging.info("Spawning adaptive distributed Spark Session environment...")
    spark_builder = SparkSession.builder \
        .appName("HHAR-Bootstrapping-Engine") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.shuffle.partitions", "100")
        
    if os.name == 'nt':
        spark_builder = spark_builder.config("spark.driver.host", "127.0.0.1")
        
    spark = spark_builder.getOrCreate()
        
    try:
        # Define explicit CSV schema to avoid scanning the massive 1.3 GB CSV files twice
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

        # load raw sensor csv feeds
        # Read source accelerometer and gyroscope feeds concurrently
        logging.info("Ingesting raw multi-variable sensor feeds from storage layer...")
        df_acc_raw = spark.read.schema(csv_schema).csv(str(acc_path), header=True)
        df_gyro_raw = spark.read.schema(csv_schema).csv(str(gyro_path), header=True)
        
        # strip whitespace from column schemas and keep only needed columns
        # Clean target schema field headers recursively to avoid downstream resolution exceptions
        logging.info("Executing parallelized metadata schema sanitization transformations...")
        needed_cols = ["Creation_Time", "x", "y", "z", "User", "Model", "Device", "gt"]
        df_acc_clean = df_acc_raw.select(*[col(c).alias(c.strip()) for c in df_acc_raw.columns]).select(needed_cols)
        df_gyro_clean = df_gyro_raw.select(*[col(c).alias(c.strip()) for c in df_gyro_raw.columns]).select(needed_cols)
        
        valid_users = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        valid_activities = ['walk', 'bike', 'stand']
        
        df_acc_filtered = df_acc_clean.filter((col("User").isin(valid_users)) & (col("gt").isin(valid_activities)))
        df_gyro_filtered = df_gyro_clean.filter((col("User").isin(valid_users)) & (col("gt").isin(valid_activities)))
        
        # establish uniform 10ms temporal bins
        # Down-sample high-frequency readings by cast rounding creation epoch time
        logging.info("Assembling uniform 10ms temporal state bins across cluster...")
        df_acc_bin = df_acc_filtered.withColumn("time_bin", (col("Creation_Time") / 10000000).cast("long")) \
                                    .groupBy("User", "Model", "Device", "gt", "time_bin") \
                                    .agg(mean("x").alias("ax"), mean("y").alias("ay"), mean("z").alias("az"))
                                     
        df_gyro_bin = df_gyro_filtered.withColumn("time_bin", (col("Creation_Time") / 10000000).cast("long")) \
                                      .groupBy("User", "Model", "Device", "gt", "time_bin") \
                                      .agg(mean("x").alias("gx"), mean("y").alias("gy"), mean("z").alias("gz"))
                                      
        # join acceleration and gyroscope telemetry
        # Join parallel data streams on subject, infrastructure activity, and temporal bins
        logging.info("Executing 6D vector synchronization join (Accelerometer + Gyroscope streams)...")
        aligned_df = df_acc_bin.join(df_gyro_bin, ["User", "Model", "Device", "gt", "time_bin"], "inner") \
                               .withColumn("mag", expr("sqrt(ax*ax + ay*ay + az*az)"))
        
        # apply quality constraint window slice
        # Filter telemetry segments to first 5000 uniform rows per group
        window_slice = Window.partitionBy("User", "gt").orderBy("time_bin")
        sliced_df = aligned_df.withColumn("rn", expr("row_number()").over(window_slice)) \
                              .filter(col("rn") <= 5000) \
                              .drop("rn") \
                              .cache()
                              
        # frequency decomposition via moving average trend window
        # Isolate slow gravitational trends from high-frequency transit variance
        logging.info("Extracting low-frequency gravitational trends from dynamic variants...")
        trend_window = Window.partitionBy("User", "gt").orderBy("time_bin").rowsBetween(-15, 15)
        
        decomposed_df = sliced_df \
            .withColumn("t_ax", mean("ax").over(trend_window)).withColumn("t_ay", mean("ay").over(trend_window)).withColumn("t_az", mean("az").over(trend_window)) \
            .withColumn("t_gx", mean("gx").over(trend_window)).withColumn("t_gy", mean("gy").over(trend_window)).withColumn("t_gz", mean("gz").over(trend_window)) \
            .withColumn("r_ax", col("ax") - col("t_ax")).withColumn("r_ay", col("ay") - col("t_ay")).withColumn("r_az", col("az") - col("t_az")) \
            .withColumn("r_gx", col("gx") - col("t_gx")).withColumn("r_gy", col("gy") - col("t_gy")).withColumn("r_gz", col("gz") - col("t_gz"))
            
        # cache segments to driver memory
        # Collect dataframes locally into driver memory arrays for block bootstrapping
        logging.info("Caching structured bootstrap data segments down into host driver memory...")
        std_devs_df = sliced_df.groupBy("User", "gt").agg(stddev("mag").alias("std_mag"))
        
        std_by_activity = {act: [] for act in valid_activities}
        for r in std_devs_df.collect():
            if r['std_mag'] is not None:
                std_by_activity[r['gt']].append(r['std_mag'])
                
        pool_rows = decomposed_df.select(
            "User", "gt", "t_ax", "t_ay", "t_az", "t_gx", "t_gy", "t_gz",
            "r_ax", "r_ay", "r_az", "r_gx", "r_gy", "r_gz"
        ).collect()
        
        pool = {}
        for r in pool_rows:
            key = (r['User'], r['gt'])
            if key not in pool:
                pool[key] = {k: [] for k in ['t_ax', 't_ay', 't_az', 't_gx', 't_gy', 't_gz', 'r_ax', 'r_ay', 'r_az', 'r_gx', 'r_gy', 'r_gz']}
            for field in pool[key].keys():
                pool[key][field].append(r[field])
                
        for key in pool:
            for k in pool[key].keys():
                pool[key][k] = np.array(pool[key][k])
                
        # block bootstrap execution loop
        # Deploy non-parametric kinematic bootstrap engines with overlapping boundary phase matching
        logging.info(f"Deploying kinematic bootstrap engines across {num_agents} target synthetic agents...")
        synthetic_records = []
        activities_distribution = ['walk'] * 20 + ['bike'] * 15 + ['stand'] * 15
        
        B = 100
        p = 5  
        num_blocks = samples_per_agent // B
        np.random.seed(42)
        
        for agent_id in range(num_agents):
            act = activities_distribution[agent_id]
            base_user = np.random.choice(valid_users)
            key = (base_user, act)
            
            while key not in pool or len(pool[key]['t_ax']) < (B + p):
                base_user = np.random.choice(valid_users)
                key = (base_user, act)
                
            std_pool = std_by_activity[act]
            s_j = np.percentile(std_pool, np.random.uniform(0, 1) * 100) if len(std_pool) > 0 else 1.0
            sigma_base = np.std(np.sqrt(pool[key]['t_ax']**2 + pool[key]['t_ay']**2 + pool[key]['t_az']**2))
            scaling_factor = s_j / (sigma_base if sigma_base > 1e-4 else 1.0)
            
            agent_t = {k: [] for k in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']}
            agent_r = {k: [] for k in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']}
            
            max_start_pos = len(pool[key]['t_ax']) - B
            
            for block_idx in range(num_blocks):
                if block_idx == 0:
                    start = np.random.randint(0, max_start_pos)
                else:
                    # element-wise sum of tail trend and residual components
                    # Add trailing overlap samples to evaluate phase transition boundaries
                    tail_ax = np.array(agent_r['ax'][-p:]) + np.array(agent_t['ax'][-p:])
                    
                    best_start, min_distance = 0, float('inf')
                    for _ in range(30):
                        test_start = np.random.randint(0, max_start_pos)
                        head_ax = pool[key]['r_ax'][test_start:test_start+p] + pool[key]['t_ax'][test_start:test_start+p]
                        
                        dist = np.sum((tail_ax - head_ax) ** 2)
                        if dist < min_distance:
                            min_distance = dist
                            best_start = test_start
                    start = best_start
                    
                end = start + B
                for k in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
                    agent_t[k].extend(pool[key][f't_{k}'][start:end])
                    agent_r[k].extend(pool[key][f'r_{k}'][start:end] * scaling_factor)
            
            ax, ay, az = np.array(agent_t['ax']) + np.array(agent_r['ax']), np.array(agent_t['ay']) + np.array(agent_r['ay']), np.array(agent_t['az']) + np.array(agent_r['az'])
            gx, gy, gz = np.array(agent_t['gx']) + np.array(agent_r['gx']), np.array(agent_t['gy']) + np.array(agent_r['gy']), np.array(agent_t['gz']) + np.array(agent_r['gz'])
            
            timestamps = 1700000000000 + np.arange(samples_per_agent) * 10
            
            for i in range(samples_per_agent):
                synthetic_records.append({
                    "Agent_ID": int(agent_id), "Timestamp": int(timestamps[i]),
                    "ax": float(ax[i]), "ay": float(ay[i]), "az": float(az[i]),
                    "gx": float(gx[i]), "gy": float(gy[i]), "gz": float(gz[i]),
                    "Activity": str(act)
                })
                
        # write records to JSON staging area
        # Stream synthetic dict records into temporary staging area on disk
        logging.info(f"Streaming generated structures into staging file workspace: {temp_json_path}...")
        with open(temp_json_path, "w") as f:
            for record in synthetic_records:
                f.write(json.dumps(record) + "\n")
                
        # ingest JSON to spark and export parquet columnar database
        # Load JSON records into Spark cluster and compress into target Parquet format
        logging.info("Marshalling JSON structures natively into Spark engine representation...")
        synth_df = spark.read.json(str(temp_json_path))
        
        logging.info(f"Compressing data layers into target columnar format: {parquet_out_path}...")
        synth_df.write.mode("overwrite").parquet(str(parquet_out_path))
        logging.info("Bootstrap execution pipeline finished successfully.")
        
        if temp_json_path.exists():
            os.remove(temp_json_path)
            
        logging.info("Reading verification summary records from compressed output file:")
        spark.read.parquet(str(parquet_out_path)).groupBy("Activity").count().show()
        
    except Exception as e:
        logging.error(f"Execution failed inside processing engine: {e}", exc_info=True)
        if temp_json_path.exists():
            os.remove(temp_json_path)
        raise e
    finally:
        logging.info("Terminating background distributed processing engines...")
        # spark.stop()  # Keep session alive!

if __name__ == "__main__":
    run_bootstrapping()