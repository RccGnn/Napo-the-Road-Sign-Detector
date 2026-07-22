"""
dataset_zip_metrics.py

Inspects each of three dataset zip archives
(without fully extracting them to disk) and reports, per zip file:

    - Number of images found in the train/ folder
    - Image SIZE: width x height in pixels (mean and median of both
      width and height separately)
    - Image LIGHT LEVEL (exposure): mean grayscale brightness per image,
      mean and median across the dataset
"""

import io
import statistics
import zipfile
from pathlib import Path

from PIL import Image, ImageStat

from utilities.utility import get_dataset_dir

# --------------------------------------------------------------------------- #
# 1. HELPERS
# --------------------------------------------------------------------------- #

# Common image file extensions to look for inside the zip. Extend this set
# if your datasets use other formats (e.g. ".webp", ".tif").
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}


def is_inside_train_folder(entry_path: str, train_folder_name: str) -> bool:
    """
    Checks whether a zip entry lives inside a folder literally called
    `train_folder_name` (default "train"), at any depth inside the
    archive - e.g. it matches both "train/img1.jpg" and
    "Dataset/train/img1.jpg", but not "trainXYZ/img1.jpg" or
    "validation/img1.jpg".

    Zip archives always use forward slashes ("/") as the internal path
    separator (even when the archive was created on Windows), so we
    normalize on "/" here regardless of the OS running this script.
    """
    parts = entry_path.replace("\\", "/").split("/")
    return train_folder_name.lower() in [p.lower() for p in parts[:-1]]


def is_image_file(entry_path: str) -> bool:
    """Checks the file extension against the known image extensions set."""
    return Path(entry_path).suffix.lower() in IMAGE_EXTENSIONS


def get_image_brightness(img: Image.Image) -> float:
    """
    Computes the image's LIGHT LEVEL as its mean grayscale brightness.

    Steps:
      1. Convert the image to grayscale ("L" mode). PIL's RGB -> L
         conversion uses the standard luminance formula
         L = 0.299*R + 0.587*G + 0.114*B, which weights channels by how
         the human eye perceives brightness (green contributes most,
         blue least) - this is the conventional way to measure
         "brightness"/"exposure" rather than just averaging raw RGB.
      2. Use PIL.ImageStat.Stat to compute the mean pixel value of the
         grayscale image. This single number, in the 0-255 range,
         is the image's overall light level / exposure.

    Returns a float in [0, 255]: 0 = pure black, 255 = pure white.
    """
    grayscale = img.convert("L")
    return ImageStat.Stat(grayscale).mean[0]


def collect_train_image_stats(zip_path: Path, train_folder_name: str):
    """
    Opens a zip archive, finds every image file inside its train/ folder,
    and reads each one's (width, height, brightness) directly from the
    in-memory bytes - without ever writing an extracted file to disk.

    Returns a list of (width, height, brightness) tuples, one per image
    found.
    """
    stats = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        # namelist() also includes directory entries (ending in "/") and
        # any file, not just images - both are filtered out below.
        train_image_entries = [
            name for name in zf.namelist()
            if not name.endswith("/")
            and is_inside_train_folder(name, train_folder_name)
            and is_image_file(name)
        ]

        for entry_name in train_image_entries:
            with zf.open(entry_name) as file_in_zip:
                image_bytes = file_in_zip.read()
                with Image.open(io.BytesIO(image_bytes)) as img:
                    width, height = img.size
                    brightness = get_image_brightness(img)
                    stats.append((width, height, brightness))

    return stats


def compute_metrics(stats):
    """
    Given a list of (width, height, brightness) tuples, computes the
    requested metrics:
        - image count
        - mean/median width
        - mean/median height
        - mean/median LIGHT LEVEL (brightness), computed per-image via
          get_image_brightness() and aggregated across the whole dataset
    Returns a dictionary of all computed values.
    """
    widths = [w for w, h, b in stats]
    heights = [h for w, h, b in stats]
    brightness_values = [b for w, h, b in stats]

    return {
        "num_images": len(stats),
        "width_mean": statistics.mean(widths) if widths else 0,
        "width_median": statistics.median(widths) if widths else 0,
        "height_mean": statistics.mean(heights) if heights else 0,
        "height_median": statistics.median(heights) if heights else 0,
        "brightness_mean": statistics.mean(brightness_values) if brightness_values else 0,
        "brightness_median": statistics.median(brightness_values) if brightness_values else 0,
    }


def print_metrics_table(results: dict):
    """
    Prints a clean, aligned text table summarizing the metrics for every
    zip file, with one row per metric and one column per zip file - easy
    to copy straight into a report. Light Level (brightness) mean/median
    are the last two rows, as requested.
    """
    zip_names = list(results.keys())

    # Each row is (label, key_in_metrics_dict, formatting)
    rows = [
        ("Number of images",           "num_images",         "{:.0f}"),
        ("Width  - mean (px)",         "width_mean",         "{:.1f}"),
        ("Width  - median (px)",       "width_median",       "{:.1f}"),
        ("Height - mean (px)",         "height_mean",        "{:.1f}"),
        ("Height - median (px)",       "height_median",      "{:.1f}"),
        ("Light Level (0-255) - mean",   "brightness_mean",   "{:.1f}"),
        ("Light Level (0-255) - median", "brightness_median", "{:.1f}"),
    ]

    label_width = max(len(label) for label, _, _ in rows) + 2
    col_width = max(max(len(name) for name in zip_names) + 2, 18)

    # Header row
    header = f"{'Metric':<{label_width}}" + "".join(f"{name:>{col_width}}" for name in zip_names)
    print(header)
    print("-" * len(header))

    # Data rows
    for label, key, fmt in rows:
        line = f"{label:<{label_width}}"
        for name in zip_names:
            value = results[name][key]
            line += f"{fmt.format(value):>{col_width}}"
        print(line)


BASE_DIR = get_dataset_dir()

# The three zip archives, each containing a "train" folder of images.
# Update these filenames if yours differ.
ZIP_FILES = [
    BASE_DIR / "k1.xml.zip",
    BASE_DIR / "rf1.tensorflow.zip",
    BASE_DIR / "rf3.tensorflow.zip",
]

TRAIN_FOLDER_NAME = "train"       # the folder name to look for inside each zip
XML_TRAIN_FOLDER_NAME = "images"  # exception for the kaggle-style dataset


# --------------------------------------------------------------------------- #
# 3. MAIN
# --------------------------------------------------------------------------- #
def main():
    results = {}

    for zip_path in ZIP_FILES:
        print(f"Processing: {zip_path.name} ...")

        if not zip_path.exists():
            print(f"  WARNING: file not found, skipping: {zip_path}")
            continue

        if zip_path.name == "k1.xml.zip":
            stats = collect_train_image_stats(zip_path, XML_TRAIN_FOLDER_NAME)
        else:
            stats = collect_train_image_stats(zip_path, TRAIN_FOLDER_NAME)

        if not stats:
            print(f"  WARNING: no images found inside a train-equivalent folder in {zip_path.name}")

        results[zip_path.name] = compute_metrics(stats)

    print("\n=== Dataset Metrics per ZIP file (train/ folder) ===\n")
    print_metrics_table(results)

import csv
import matplotlib.pyplot as plt

def plot_class_distribution(class_counts: dict):
    """
    Takes a dictionary of class counts and plots a bar chart.
    """
    # Extract keys (classes) and values (counts)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='#4C72B0', edgecolor='black')

    # Formatting the plot
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('# immagini', fontsize=12)
    plt.title('Numero di immagini per classe', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def process_and_plot_dataset(csv_filepath: str):
    """
    Reads the CSV line by line, maps filenames to classes, counts the
    occurrences of each class, and calls the plotting function.
    """
    image_dict = {}  # Dictionary for {filename: class}
    class_counts = {}  # Dictionary for {class: count}

    try:
        # Open and read the CSV file line by line
        with open(csv_filepath, mode='r', encoding='utf-8') as file:
            # DictReader automatically uses the first row as dictionary keys
            reader = csv.DictReader(file)

            for row in reader:
                # Extract data based on the column headers in preview.webp
                filename = row.get('filename')
                image_class = row.get('class')

                # Ensure the row isn't missing required data
                if filename and image_class:
                    # 1. Populate the {nome: str, classe: str} dictionary
                    image_dict[filename] = image_class

                    # 2. Count the number of images per class
                    if image_class in class_counts:
                        class_counts[image_class] += 1
                    else:
                        class_counts[image_class] = 1

    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

    # Call the function to plot the graph using the counts
    plot_class_distribution(class_counts)

    return image_dict, class_counts

if __name__ == "__main__":
    csv_path = get_dataset_dir() / "merged.csv"
    process_and_plot_dataset(csv_path)