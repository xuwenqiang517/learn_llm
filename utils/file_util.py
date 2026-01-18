import os
from pathlib import Path

class FileUtil:
    @staticmethod
    def ensure_dirs(*paths: Path):
        for path in paths:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                pass
    
    @staticmethod
    def read_csv(file_path: Path):
        import pandas as pd
        return pd.read_csv(file_path, encoding="utf-8-sig")
    
    @staticmethod
    def write_csv(df, file_path: Path, index: bool = False):
        df.to_csv(file_path, index=index, encoding="utf-8-sig")
    
    @staticmethod
    def delete_file(file_path: Path):
        if file_path.exists():
            file_path.unlink()
