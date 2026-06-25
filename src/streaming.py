import os
import time
import logging
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_structured_streaming_simulation(spark, project_root: Path):
    # establish directory paths for streaming buffer layers
    # Set up physical file storage paths for stream sources and checkpoint metadata on disk
    processed_dir = project_root / "data" / "processed"
    streaming_source_dir = project_root / "data" / "streaming_source"
    streaming_checkpoint_dir = project_root / "data" / "streaming_checkpoints"
    
    # create clean environments for the streaming buffers
    # Instantiate physical file path structures on the host filesystem recursively
    streaming_source_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up checkpoint directory to allow clean restart and avoid recovery mismatch errors on memory sinks
    if streaming_checkpoint_dir.exists():
        import shutil
        logging.info(f"Clearing old streaming checkpoints at: {streaming_checkpoint_dir}")
        try:
            shutil.rmtree(streaming_checkpoint_dir)
        except Exception as e:
            logging.warning(f"Could not clear checkpoint directory: {e}")
            
    streaming_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # read the master synthetic Parquet file already generated
        # Ingest the generated synthetic parquet telemetry database into Spark memory
        master_parquet_path = processed_dir / "synthetic_telemetry.parquet"
        if not master_parquet_path.exists():
            logging.error(f"Master synthetic telemetry parquet not found at '{master_parquet_path}'. Please run bootstrapping first.")
            return
            
        logging.info(f"Loading master synthetic dataset from: {master_parquet_path}")
        master_df = spark.read.parquet(str(master_parquet_path))
        
        # prepare the Simulation: Split the dataset into 10 smaller micro-batch chunks
        # Repartition and write the synthetic dataset to watch folders to simulate stream chunk arrivals
        logging.info("Splitting master data layer into physical streaming micro-batches...")
        master_df.repartition(10).write.mode("overwrite").parquet(str(streaming_source_dir))
        
        # define the Explicit Schema for the incoming Stream
        # Bind the schema metadata of the static dataframe to the read stream configurations
        stream_schema = master_df.schema
        
        # initialize the Structured Stream Reader
        # Instantiate Spark structured streaming file stream pointing to the source watch folder
        logging.info("Initializing Spark Structured Streaming File Engine...")
        telemetry_stream = spark.readStream \
            .schema(stream_schema) \
            .option("maxFilesPerTrigger", 1) \
            .parquet(str(streaming_source_dir))
            
        # apply a basic streaming transformation
        # Filter streaming rows to walking actions to simulate real-time pattern matching
        processed_stream = telemetry_stream.filter(col("Activity") == "walk")
        
        # direct the Live Stream Output to memory so we can query it in real-time
        # Configure streaming write query to output live records into an in-memory SQL catalog table
        logging.info("Activating live streaming write sink...")
        query = processed_stream.writeStream \
            .format("memory") \
            .queryName("live_citizen_telemetry") \
            .option("checkpointLocation", str(streaming_checkpoint_dir)) \
            .start()
            
        # interactive Monitoring Loop for your Jupyter Notebook cell output
        # Periodically execute SQL queries against the live in-memory table to display throughput counts
        logging.info("Stream is live and actively processing micro-batches. Monitoring output...")
        for monitor_step in range(5):
            time.sleep(4)  # Wait for a micro-batch trigger to complete
            print(f"\n--- Live Stream Inspection Update (Step {monitor_step + 1}) ---")
            spark.sql("SELECT Activity, count(*), max(Timestamp) FROM live_citizen_telemetry GROUP BY Activity").show()
            
        # gracefully arrest the streaming query loop before execution closes
        # Halt stream write threads cleanly to ensure no open lock files remain
        logging.info("Deactivating live streaming queries...")
        query.stop()
        
    except Exception as e:
        logging.error(f"Streaming environment error: {e}", exc_info=True)
