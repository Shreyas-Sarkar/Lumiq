import io
import tempfile
import unittest
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient
from config import Settings
from routers.upload import router
from routers.auth_middleware import get_current_user
from services.dataset_manager import DatasetManager
from fakes import OWNER, CHAT


class Query:
    def __init__(self, db, table):
        self.db = db
        self.name = table
        self.filters = {}
        self.payload = None

    def select(self, *a):
        return self

    def eq(self, k, v):
        self.filters[k] = v
        return self

    def limit(self, *a):
        return self

    def insert(self, p):
        self.payload = p
        return self

    def execute(self):
        if self.payload:
            self.db.saved = self.payload
            return SimpleNamespace(
                data=[{**self.payload, "uploaded_at": "2026-10-02T00:00:00Z"}]
            )
        if self.name == "chat":
            return SimpleNamespace(
                data=[{"id": CHAT}]
                if self.filters.get("id") == CHAT
                and self.filters.get("user_id") == OWNER
                else []
            )
        return SimpleNamespace(data=[])


class DB:
    def __init__(self):
        self.saved = None
        self.storage = SimpleNamespace(
            from_=lambda bucket: self,
            get_bucket=lambda bucket: SimpleNamespace(public=False),
        )
        self.files = {}

    def table(self, t):
        return Query(self, t)

    def upload(self, path, file, file_options):
        self.files[path] = file

    def download(self, path):
        return self.files[path]

    def remove(self, paths):
        for p in paths:
            self.files.pop(p, None)


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = DB()
        self.manager = DatasetManager(self.db, self.temp.name)
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.state.settings = Settings(
            _env_file=None,
            investigations_enabled=True,
            upload_dir=self.temp.name,
            max_file_size_mb=1,
        )
        app.state.dataset_manager = self.manager
        app.state.rag_service = None
        app.state.supabase_client = self.db
        app.dependency_overrides[get_current_user] = lambda: {"id": OWNER}
        self.client = TestClient(app)

    def upload(self, data, chat=CHAT):
        return self.client.post(
            "/api/upload",
            data={"chat_id": chat},
            files={"file": ("revenue.csv", data, "text/csv")},
        )

    def test_upload_profile_and_recover_after_local_file_removed(self):
        response = self.upload(
            __import__("pathlib").Path("../samples/revenue.csv").read_bytes()
        )
        self.assertEqual(response.status_code, 200, response.text)
        saved = self.db.saved
        self.assertTrue(saved["storage_path"].startswith("supabase://"))
        self.assertIn(saved["id"], saved["storage_path"])
        self.assertEqual(saved["profile"]["rows"], 6)
        self.assertEqual(
            len(self.manager.load_dataframe_from_storage(saved["storage_path"])), 6
        )
        import pathlib

        self.assertEqual(list(pathlib.Path(self.temp.name).rglob("*.csv")), [])

    def test_invalid_oversized_and_foreign_upload(self):
        self.assertEqual(self.upload(b"\xff").status_code, 400)
        self.assertEqual(self.upload(b"x" * (1024 * 1024 + 1)).status_code, 413)
        self.assertEqual(
            self.upload(
                b"v\n1\n", chat="44444444-4444-4444-8444-444444444444"
            ).status_code,
            400,
        )
        self.assertEqual(self.db.files, {})

    def test_service_role_reader_rejects_foreign_storage_paths(self):
        with self.assertRaises(ValueError):
            self.manager.load_dataframe_from_storage(
                "supabase://lumiq-datasets/other-user/private.csv", owner=OWNER
            )
        with self.assertRaises(ValueError):
            self.manager.load_dataframe_from_storage(
                "/tmp/other-user/private.csv", owner=OWNER
            )
