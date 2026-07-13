import unittest

from fastapi.testclient import TestClient

import app.main as main


class DownloadStatusTests(unittest.TestCase):
    def test_download_video_returns_409_with_running_job_status_when_not_done(self):
        job_id = "job-running"
        main.JOBS[job_id] = {
            "status": "running",
            "progress": {"done": 5, "total": 10},
            "result": None,
            "error": None,
        }

        try:
            with TestClient(main.app) as client:
                response = client.get(f"/download/{job_id}/video")
        finally:
            main.JOBS.pop(job_id, None)

        self.assertEqual(response.status_code, 409)
        payload = response.json()
        self.assertEqual(payload["detail"]["job_id"], job_id)
        self.assertEqual(payload["detail"]["status"], "running")
        self.assertEqual(payload["detail"]["progress"], {"done": 5, "total": 10})

    def test_process_async_returns_clear_message_for_validation_errors(self):
        with TestClient(main.app) as client:
            response = client.post("/process_async", json={"video": "dummy", "gps_csv": "dummy"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            "Validation failed. Expected multipart/form-data with 'video' and 'gps_csv' files.",
        )
