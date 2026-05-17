from pathlib import Path

import requests


# 1. Setup environment / Mocking your settings
class Settings:
    """Mock settings class to provide dataset directory."""

    def __init__(self):
        # Creates a 'data' folder in your current directory
        self.datasetdir = Path("./data")
        self.datasetdir.mkdir(parents=True, exist_ok=True)


settings = Settings()


# 2. Define the download helper
def _download(url, output_file_name, output_path, is_zip=False):
    target_path = Path(output_path) / output_file_name

    print(f"Downloading {url}...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check for HTTP errors
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Saved to: {target_path}")

    if is_zip:
        # You could add zip extraction logic here if needed
        pass


# 3. Execution Logic
output_file_name = "combosciplex.h5ad"
output_file_path = settings.datasetdir / output_file_name

if not output_file_path.exists():
    _download(
        url="https://exampledata.scverse.org/pertpy/combosciplex.h5ad",
        output_file_name=output_file_name,
        output_path=settings.datasetdir,
        is_zip=False,
    )
else:
    print(f"File already exists at {output_file_path}")
