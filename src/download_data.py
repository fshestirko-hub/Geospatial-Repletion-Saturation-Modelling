import os
import zipfile
import shutil
import urllib.request
import urllib.error
import logging
from pathlib import Path

# Configure logging hierarchy and formatting.
# Configure the standard library logging output level and formatting pattern.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("hhar_downloader")


def download_and_extract_hhar():
    # Locate project root relative to script location.
    # Determine the directory path to the project root by resolving the current file context.
    script_path = Path(__file__).resolve()
    project_root = (
        script_path.parent.parent
        if script_path.parent.name in ['src', 'scripts']
        else script_path.parent
    )

    # Define target raw directory.
    # Map the filesystem target destination directory for the raw dataset.
    dest_dir = project_root / "data" / "raw"

    # Create folders if they do not exist.
    # Instantiate the raw data folder structure recursively if missing on disk.
    logger.debug(f"Creating directory structure at: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check for sentinel file to skip execution if already done.
    # Check for the sentinel accelerometer file presence to bypass duplicate downloads.
    sentinel_file = dest_dir / "Activity recognition exp" / "Phones_accelerometer.csv"
    if sentinel_file.exists():
        logger.info("Dataset already downloaded and extracted. Exiting early.")
        return

    # Define zip archive paths.
    # Configure the path targeting the primary UCI dataset zip file.
    outer_zip_name = "heterogeneity_activity_recognition.zip"
    zip_path = dest_dir / outer_zip_name

    # Define alternate backup search path in home directory.
    # Establish a default fallback path inside the user home directory for server deployments.
    home_dir = Path.home()
    alternate_dir = home_dir / "data" / "raw"
    nested_zips = ["Activity recognition exp.zip", "Still exp.zip"]

    fallback_found = False

    # Look for pre-cached files on server/local disk.
    # Check fallback caches to load dataset files locally and skip network downloads.
    logger.debug(f"Checking alternate path for cached files: {alternate_dir}")
    if alternate_dir.exists():
        files_in_alt = os.listdir(alternate_dir)
        if all(nz in files_in_alt for nz in nested_zips):
            logger.info(
                f"Detected pre-downloaded files in alternate path: {alternate_dir}"
            )
            logger.info("Copying files to workspace folder to save download time...")
            for nz in nested_zips:
                src_file = alternate_dir / nz
                dest_file = dest_dir / nz
                if not dest_file.exists():
                    logger.debug(f"Copying {nz} to {dest_file}")
                    shutil.copy2(src_file, dest_file)
            fallback_found = True
            logger.info("Cached files copied successfully.")

    # Download from UCI if no local copy was found.
    # Pull the dataset from the remote UCI web repository if local copies are absent.
    if not fallback_found:
        dataset_url = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        logger.info(f"Connecting to UCI repository: {dataset_url}")
        req = urllib.request.Request(dataset_url, headers=headers)

        try:
            # Fetch outer archive file stream.
            # Establish the HTTP connection and request the dataset stream.
            with urllib.request.urlopen(req) as response:
                total_size = int(response.info().get('Content-Length', 0))
                block_size = 1024 * 1024
                downloaded = 0

                logger.info(
                    f"File size: {total_size / (1024 * 1024):.2f} MB. Starting download..."
                )

                # Write binary data stream in chunks.
                # Stream incoming bytes from the web socket into the target binary file path.
                with open(zip_path, 'wb') as out_file:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        out_file.write(buffer)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            # Log download status periodically as debug.
                            # Output transfer rates and downloaded fractions to logs.
                            logger.debug(
                                f"Downloaded: {downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB ({percent:.2f}%)"
                            )
                logger.info("Download complete.")

            # Extract primary zip archive.
            # Extract the downloaded outer dataset zip archive into the raw directory.
            logger.info("Extracting outer zip archive...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            logger.info("Outer extraction complete.")

        except urllib.error.URLError as url_err:
            logger.error(f"Failed to connect to dataset URL: {str(url_err)}")
            if zip_path.exists():
                logger.debug("Cleaning up corrupted/partial file")
                os.remove(zip_path)
            raise url_err
        except Exception as e:
            logger.critical(f"Unexpected error during download/extraction: {str(e)}")
            if zip_path.exists():
                logger.debug("Cleaning up corrupted/partial file")
                os.remove(zip_path)
            raise e
        finally:
            # Ensure outer zip is deleted after extraction.
            # Clean up the outer zip file to preserve system storage space.
            if zip_path.exists():
                os.remove(zip_path)

    # Search and extract nested zip archives.
    # Scan for inner nested zip files and extract them sequentially.
    logger.info("Checking for nested zip archives...")
    for nested_zip_path in list(dest_dir.rglob("*.zip")):
        logger.info(f"Extracting nested zip: {nested_zip_path.name}...")
        try:
            with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            logger.debug(f"Deleting zip archive: {nested_zip_path.name}")
            os.remove(nested_zip_path)
        except Exception as e:
            logger.warning(f"Failed to extract {nested_zip_path.name}: {str(e)}")

    # List final directory contents for verification.
    # Output listing of extracted raw directory files to confirm successful completion.
    logger.info("Extraction process complete. Listing files in 'data/raw':")
    for item in dest_dir.iterdir():
        if item.is_dir():
            logger.info(f" [folder] - {item.name}/")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            logger.info(f" [file]   - {item.name} ({size_mb:.2f} MB)")