"""
SteamPulse — Full Project Inventory and Audit
================================================

Scans the entire SteamPulse project rather than only the directory
containing this script.

Reports:
    reports/project_inventory.csv
    reports/project_duplicate_filenames.csv
    reports/project_duplicate_contents.csv
    reports/project_audit_summary.txt

Safety:
    - Does not delete files.
    - Does not modify existing datasets or model artifacts.
    - Does not load pickle/model files.
    - Does not hash files larger than 128 MB.
    - Skips common environment/cache folders.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import csv
import hashlib


# ============================================================
# PROJECT PATHS
# ============================================================

# This script is expected to be in:
# SteamPulse/SRC/Model Features/model_audit.py
#
# parents[0] = SRC/Model Features
# parents[1] = SRC
# parents[2] = SteamPulse

ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT / "reports"

INVENTORY_PATH = REPORT_DIR / "project_inventory.csv"

DUPLICATE_NAMES_PATH = (
    REPORT_DIR / "project_duplicate_filenames.csv"
)

DUPLICATE_CONTENT_PATH = (
    REPORT_DIR / "project_duplicate_contents.csv"
)

SUMMARY_PATH = REPORT_DIR / "project_audit_summary.txt"

MAX_HASH_BYTES = 128 * 1024 * 1024

EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
}


EXPECTED_PATHS = [
    "Data/Raw/steam_app_catalog.csv",
    "Data/Cleaned/steam_games_clean.csv",
    "Data/Processed/steam_modeling.csv",
    "Data/Processed/steam_features_pricefixed.csv",
    "Data/Processed/steam_modeling_final_prices_fixed.csv",
    "Data/Processed/steam_ml_integrated.csv",
    "reports/feature_quality_audit.csv",
    "reports/feature_timing_audit.csv",
    "reports/label_distribution_audit.csv",
    "reports/logistic_regression_final_tuning.csv",
    "reports/logistic_regression_final_thresholds.csv",
    "models/steampulse_selected_model.pkl",
    "models/steampulse_tuned_model.pkl",
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    """Calculate a SHA-256 checksum without loading the whole file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def get_category(path):
    """Categorize files based on their location and extension."""

    relative = path.relative_to(ROOT)
    parts = [part.lower() for part in relative.parts]
    extension = path.suffix.lower()
    filename = path.name.lower()

    if "data" in parts:

        if "raw" in parts:
            return "raw_data"

        if "cleaned" in parts:
            return "cleaned_data"

        if "processed" in parts:
            return "processed_data"

        return "data"

    if "reports" in parts:
        return "report_or_audit"

    if "models" in parts:
        return "model_artifact"

    if extension in {
        ".pkl",
        ".joblib",
        ".onnx",
    }:
        return "model_artifact"

    if extension == ".py":

        experiment_terms = [
            "tune",
            "svm",
            "experiment",
            "benchmark",
            "validate",
            "compare",
            "analyze_model",
        ]

        if any(term in filename for term in experiment_terms):
            return "model_experiment_script"

        return "python_script"

    if extension in {
        ".md",
        ".rst",
        ".txt",
    }:
        return "documentation_or_notes"

    if extension in {
        ".csv",
        ".xlsx",
        ".xls",
        ".parquet",
        ".json",
    }:
        return "data_or_table"

    return "other"


def format_bytes(number):
    """Human-readable size."""

    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(number)

    for unit in units:

        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} TB"


def discover_files():
    """Discover files throughout the project, excluding cache folders."""

    files = []

    for path in ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.is_symlink():
            continue

        relative = path.relative_to(ROOT)

        if any(
            part in EXCLUDED_DIRECTORIES
            for part in relative.parts
        ):
            continue

        files.append(path)

    return sorted(files)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    if not ROOT.exists():
        raise FileNotFoundError(
            f"Project root does not exist: {ROOT}"
        )

    if ROOT.name.lower() != "steampulse":
        raise RuntimeError(
            "The resolved directory does not appear to be "
            f"the SteamPulse root: {ROOT}"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("STEAMPULSE FULL PROJECT AUDIT")
    print("=" * 72)

    print(f"Project root: {ROOT}")
    print(f"Reports directory: {REPORT_DIR}")

    files = discover_files()

    print(f"Files discovered: {len(files):,}")

    inventory_rows = []

    filename_groups = defaultdict(list)
    hash_groups = defaultdict(list)

    extension_counts = Counter()
    category_counts = Counter()
    folder_counts = Counter()

    total_size = 0
    hash_skipped = 0
    unreadable = 0

    for index, path in enumerate(files, start=1):

        relative_path = path.relative_to(ROOT).as_posix()

        try:
            file_size = path.stat().st_size

            modified_time = datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat(timespec="seconds")

        except OSError as exc:

            print(
                f"WARNING: Cannot inspect {relative_path}: {exc}"
            )

            unreadable += 1
            continue

        extension = path.suffix.lower() or "[no extension]"
        category = get_category(path)

        folder = path.relative_to(ROOT).parent.as_posix()

        total_size += file_size

        extension_counts[extension] += 1
        category_counts[category] += 1
        folder_counts[folder] += 1

        filename_groups[path.name.lower()].append(
            relative_path
        )

        file_hash = ""
        hash_status = ""

        if file_size <= MAX_HASH_BYTES:

            try:
                file_hash = sha256_file(path)

                hash_status = "hashed"

                # Include size as well as hash.
                hash_groups[
                    (file_size, file_hash)
                ].append(relative_path)

            except OSError as exc:

                hash_status = f"error: {exc}"
                unreadable += 1

        else:

            hash_status = "skipped_large_file"
            hash_skipped += 1

        inventory_rows.append(
            {
                "relative_path": relative_path,
                "filename": path.name,
                "extension": extension,
                "category": category,
                "folder": folder,
                "size_bytes": file_size,
                "size_mb": round(
                    file_size / (1024 ** 2),
                    3,
                ),
                "modified_time": modified_time,
                "sha256": file_hash,
                "hash_status": hash_status,
            }
        )

        if index % 500 == 0:
            print(f"Scanned {index:,} files...")

    # --------------------------------------------------------
    # SAVE INVENTORY
    # --------------------------------------------------------

    inventory_columns = [
        "relative_path",
        "filename",
        "extension",
        "category",
        "folder",
        "size_bytes",
        "size_mb",
        "modified_time",
        "sha256",
        "hash_status",
    ]

    with INVENTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=inventory_columns,
        )

        writer.writeheader()
        writer.writerows(inventory_rows)

    # --------------------------------------------------------
    # DUPLICATE FILENAME REPORT
    # --------------------------------------------------------

    duplicate_name_rows = []

    for filename, paths in sorted(filename_groups.items()):

        if len(paths) <= 1:
            continue

        for relative_path in paths:

            duplicate_name_rows.append(
                {
                    "filename": filename,
                    "relative_path": relative_path,
                    "same_name_count": len(paths),
                }
            )

    with DUPLICATE_NAMES_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "relative_path",
                "same_name_count",
            ],
        )

        writer.writeheader()
        writer.writerows(duplicate_name_rows)

    # --------------------------------------------------------
    # DUPLICATE CONTENT REPORT
    # --------------------------------------------------------

    duplicate_content_rows = []

    for (file_size, file_hash), paths in hash_groups.items():

        if len(paths) <= 1:
            continue

        for relative_path in paths:

            duplicate_content_rows.append(
                {
                    "sha256": file_hash,
                    "size_bytes": file_size,
                    "size_mb": round(
                        file_size / (1024 ** 2),
                        3,
                    ),
                    "relative_path": relative_path,
                    "group_file_count": len(paths),
                }
            )

    with DUPLICATE_CONTENT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sha256",
                "size_bytes",
                "size_mb",
                "relative_path",
                "group_file_count",
            ],
        )

        writer.writeheader()
        writer.writerows(duplicate_content_rows)

    # --------------------------------------------------------
    # EXPECTED FILE CHECKS
    # --------------------------------------------------------

    expected_results = []

    for relative_path in EXPECTED_PATHS:

        exists = (
            ROOT / relative_path
        ).is_file()

        expected_results.append(
            {
                "path": relative_path,
                "exists": exists,
            }
        )

    # --------------------------------------------------------
    # BUILD SUMMARY
    # --------------------------------------------------------

    python_files = [
        row["relative_path"]
        for row in inventory_rows
        if row["extension"] == ".py"
    ]

    model_files = [
        row["relative_path"]
        for row in inventory_rows
        if row["category"] == "model_artifact"
    ]

    data_files = [
        row["relative_path"]
        for row in inventory_rows
        if row["category"] in {
            "raw_data",
            "cleaned_data",
            "processed_data",
        }
    ]

    report_lines = []

    report_lines.append("STEAMPULSE FULL PROJECT AUDIT")
    report_lines.append("=" * 72)
    report_lines.append(f"Project root: {ROOT}")
    report_lines.append(f"Files inventoried: {len(inventory_rows):,}")
    report_lines.append(f"Total file size: {format_bytes(total_size)}")
    report_lines.append(f"Files skipped for hashing: {hash_skipped:,}")
    report_lines.append(f"Unreadable files: {unreadable:,}")

    report_lines.append("")
    report_lines.append("FILE COUNTS BY CATEGORY")

    for category, count in sorted(category_counts.items()):
        report_lines.append(f"  {category}: {count:,}")

    report_lines.append("")
    report_lines.append("FILE COUNTS BY EXTENSION")

    for extension, count in sorted(extension_counts.items()):
        report_lines.append(f"  {extension}: {count:,}")

    report_lines.append("")
    report_lines.append("FILES BY FOLDER")

    for folder, count in sorted(folder_counts.items()):
        report_lines.append(f"  {folder}: {count:,}")

    report_lines.append("")
    report_lines.append("EXPECTED FILE CHECKS")

    for result in expected_results:

        status = (
            "FOUND"
            if result["exists"]
            else "NOT FOUND"
        )

        report_lines.append(
            f"  [{status}] {result['path']}"
        )

    report_lines.append("")
    report_lines.append("PYTHON SCRIPTS")

    for path in python_files:
        report_lines.append(f"  {path}")

    report_lines.append("")
    report_lines.append("MODEL ARTIFACTS")

    if model_files:
        for path in model_files:
            report_lines.append(f"  {path}")
    else:
        report_lines.append("  No model artifacts discovered.")

    report_lines.append("")
    report_lines.append("DATA FILES")

    for path in data_files:
        report_lines.append(f"  {path}")

    report_lines.append("")
    report_lines.append("DUPLICATE CHECKS")

    report_lines.append(
        "  Duplicate filenames: "
        f"{len(duplicate_name_rows):,} file entries"
    )

    report_lines.append(
        "  Duplicate contents: "
        f"{len(duplicate_content_rows):,} file entries"
    )

    report_lines.append("")
    report_lines.append("AUDIT OUTPUTS")

    report_lines.append(
        f"  {INVENTORY_PATH.relative_to(ROOT).as_posix()}"
    )

    report_lines.append(
        f"  {DUPLICATE_NAMES_PATH.relative_to(ROOT).as_posix()}"
    )

    report_lines.append(
        f"  {DUPLICATE_CONTENT_PATH.relative_to(ROOT).as_posix()}"
    )

    report_lines.append(
        f"  {SUMMARY_PATH.relative_to(ROOT).as_posix()}"
    )

    report_lines.append("")
    report_lines.append(
        "This report does not authorize deletion of files."
    )

    SUMMARY_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("PROJECT AUDIT SUMMARY")
    print("=" * 72)

    print(f"Project root: {ROOT}")
    print(f"Files inventoried: {len(inventory_rows):,}")
    print(f"Total size: {format_bytes(total_size)}")
    print(f"Files skipped for hashing: {hash_skipped:,}")

    print("\nFiles by category:")

    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count:,}")

    print("\nExpected file checks:")

    for result in expected_results:

        status = (
            "FOUND"
            if result["exists"]
            else "NOT FOUND"
        )

        print(f"  [{status}] {result['path']}")

    print("\nPython scripts:")

    for path in python_files:
        print(f"  {path}")

    print("\nModel artifacts:")

    if model_files:
        for path in model_files:
            print(f"  {path}")
    else:
        print("  No model artifacts discovered.")

    print("\nReports saved:")
    print(f"  {INVENTORY_PATH}")
    print(f"  {DUPLICATE_NAMES_PATH}")
    print(f"  {DUPLICATE_CONTENT_PATH}")
    print(f"  {SUMMARY_PATH}")

    print("\nNo project files were deleted or modified.")
    print("Audit completed.")


if __name__ == "__main__":
    main()