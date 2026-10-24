import pandas as pd
import numpy as np
import uuid
import os
import io
import hashlib
from typing import Any


class DatasetManager:
    def __init__(self, supabase, upload_dir: str):
        self.supabase = supabase
        self.upload_dir = upload_dir

    def load_csv(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("CSV file is empty")
        if len(df.columns) == 0:
            raise ValueError("CSV has no columns")
        return df

    def extract_schema(self, df: pd.DataFrame) -> dict:
        columns = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "nullable": bool(df[col].isna().any()),
            }
            if df[col].dtype == object:
                unique_count = df[col].nunique()
                col_info["unique_count"] = int(unique_count)
                if unique_count <= 10:
                    col_info["unique_values"] = [
                        str(v) for v in df[col].dropna().unique().tolist()
                    ]
            columns.append(col_info)
        return {"columns": columns}

    def extract_sample_rows(self, df: pd.DataFrame, n: int = 5) -> list[dict]:
        sample = df.head(n).copy()
        rows = []
        for _, row in sample.iterrows():
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val) if not isinstance(val, (list, dict)) else False:
                    row_dict[col] = None
                elif isinstance(val, (np.integer,)):
                    row_dict[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    row_dict[col] = float(val) if not np.isnan(val) else None
                else:
                    row_dict[col] = val
            rows.append(row_dict)
        return rows

    def extract_summary_stats(self, df: pd.DataFrame) -> dict:
        try:
            stats = df.describe(include="all").to_dict()
            cleaned = {}
            for col, col_stats in stats.items():
                cleaned[col] = {}
                for stat_name, val in col_stats.items():
                    if isinstance(val, float) and np.isnan(val):
                        cleaned[col][stat_name] = None
                    elif isinstance(val, (np.integer,)):
                        cleaned[col][stat_name] = int(val)
                    elif isinstance(val, (np.floating,)):
                        cleaned[col][stat_name] = float(val)
                    else:
                        cleaned[col][stat_name] = val
            cleaned["_meta"] = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            }
            return cleaned
        except Exception:
            return {
                "_meta": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "memory_usage_bytes": 0,
                }
            }

    def validate_csv(self, file_path: str) -> tuple[bool, str]:
        try:
            with open(file_path, "rb") as f:
                raw = f.read()
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                return False, "File is not UTF-8 encoded"

            if os.path.getsize(file_path) == 0:
                return False, "File is empty"

            df = pd.read_csv(file_path)
            if df.empty:
                return False, "CSV contains no data rows"
            if len(df.columns) == 0:
                return False, "CSV has no columns"
            if len(df.columns) > 500:
                return False, "CSV has too many columns (max 500)"
            return True, ""
        except pd.errors.ParserError as e:
            return False, f"CSV parse error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def save_dataset_metadata(
        self,
        chat_id: str,
        user_id: str,
        filename: str,
        storage_path: str,
        schema_info: dict,
        sample_rows: list,
        summary_stats: dict,
        chroma_collection_id: str,
        row_count: int,
        column_count: int,
        dataset_id: str | None = None,
        content_hash: str | None = None,
        profile: dict | None = None,
    ) -> dict:
        dataset_id = dataset_id or str(uuid.uuid4())
        data = {
            "id": dataset_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "filename": filename,
            "storage_path": storage_path,
            "schema_info": schema_info,
            "sample_rows": sample_rows,
            "summary_stats": summary_stats,
            "chroma_collection_id": chroma_collection_id,
            "row_count": row_count,
            "column_count": column_count,
        }
        if profile is not None:
            data.update(profile=profile, content_hash=content_hash)
        result = self.supabase.table("dataset").insert(data).execute()
        return result.data[0]

    def get_dataset_by_chat_id(self, chat_id: str, user_id: str | None = None) -> dict:
        query = self.supabase.table("dataset").select("*").eq("chat_id", chat_id)
        if user_id:
            query = query.eq("user_id", user_id)
        result = query.limit(1).execute()
        if not result.data:
            raise ValueError(f"No dataset found for chat {chat_id}")
        return result.data[0]

    def store_csv(
        self, content: bytes, user_id: str, dataset_id: str, bucket: str
    ) -> str:
        bucket_info = self.supabase.storage.get_bucket(bucket)
        if bucket_info.public:
            raise ValueError("Dataset bucket must be private")
        path = f"{user_id}/{dataset_id}.csv"
        self.supabase.storage.from_(bucket).upload(
            path=path,
            file=content,
            file_options={"content-type": "text/csv", "upsert": "false"},
        )
        return f"supabase://{bucket}/{path}"

    def remove_csv(self, storage_path: str):
        if storage_path.startswith("supabase://"):
            bucket, path = storage_path.removeprefix("supabase://").split("/", 1)
            self.supabase.storage.from_(bucket).remove([path])
        elif os.path.exists(storage_path):
            os.remove(storage_path)

    def load_dataframe_from_storage(
        self, storage_path: str, owner: str | None = None
    ) -> pd.DataFrame:
        if storage_path.startswith("supabase://"):
            bucket, path = storage_path.removeprefix("supabase://").split("/", 1)
            if owner and path.split("/")[0] != owner:
                raise ValueError("Dataset storage ownership mismatch")
            content = self.supabase.storage.from_(bucket).download(path)
            return pd.read_csv(io.BytesIO(content))
        if owner:
            expected = os.path.realpath(os.path.join(self.upload_dir, owner))
            actual = os.path.realpath(storage_path)
            if os.path.commonpath([expected, actual]) != expected:
                raise ValueError("Dataset storage ownership mismatch")
        if not os.path.exists(storage_path):
            raise FileNotFoundError(f"Dataset file not found: {storage_path}")
        return pd.read_csv(storage_path)
