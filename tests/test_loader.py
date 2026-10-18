"""Unit tests for CSV loader."""

import pytest
import pandas as pd
from pathlib import Path
from src.ingestion.loader import load_csv


def test_load_csv_success(tmp_path: Path):
    """Test successful loading of a valid CSV file."""
    csv_file = tmp_path / "test_data.csv"
    df_input = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "region": ["Bhopal", "Bhopal"],
        "cases": [10, 15]
    })
    df_input.to_csv(csv_file, index=False)

    df_loaded = load_csv(csv_file)
    assert len(df_loaded) == 2
    assert list(df_loaded.columns) == ["date", "region", "cases"]


def test_load_csv_missing_file():
    """Test loading a non-existent CSV file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file_12345.csv")


def test_load_csv_invalid_extension(tmp_path: Path):
    """Test loading a non-CSV extension raises ValueError."""
    txt_file = tmp_path / "test_data.txt"
    txt_file.write_text("date,region\n2025-01-01,Bhopal")

    with pytest.raises(ValueError, match="must have a .csv extension"):
        load_csv(txt_file)
