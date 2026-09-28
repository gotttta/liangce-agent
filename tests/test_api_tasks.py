import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(root=tmp_path)) as test_client:
        yield test_client


@pytest.fixture
def store(tmp_path):
    from core.task_store import TaskStore

    return TaskStore(tmp_path / "workspace" / "tasks")


def _png_bytes(size=16, color=128) -> bytes:
    image = Image.new("L", (size, size), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _create_task(client, title="检测任务") -> dict:
    response = client.post("/api/tasks", json={"title": title})
    assert response.status_code == 200
    return response.json()


def test_task_crud_flow(client):
    task = _create_task(client)
    assert task["title"] == "检测任务"
    assert task["status"] == "draft"
    assert task["status_label"] == "草稿"
    assert task["samples"] == []
    assert task["runs"] == []
    assert task["pending_review"] is None
    assert task["best"] is None

    listed = client.get("/api/tasks").json()
    assert [item["id"] for item in listed] == [task["id"]]
    assert listed[0]["sample_count"] == 0
    assert listed[0]["running"] is False

    patched = client.patch(f"/api/tasks/{task['id']}", json={"title": "改名"}).json()
    assert patched["title"] == "改名"

    assert client.delete(f"/api/tasks/{task['id']}").json() == {"ok": True}
    response = client.get(f"/api/tasks/{task['id']}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "task_not_found"


def test_create_task_without_title_uses_default(client):
    task = client.post("/api/tasks", json={}).json()
    assert task["title"] == "新缺陷检测任务"


def test_unknown_task_returns_404(client):
    response = client.get("/api/tasks/task_20990101_000000_nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "task_not_found"


def test_upload_samples_and_read_back(client):
    task = _create_task(client)
    first, second = _png_bytes(color=10), _png_bytes(color=200)
    response = client.post(
        f"/api/tasks/{task['id']}/samples",
        files=[("files", ("a.png", first, "image/png")),
               ("files", ("样 本.jpg", second, "image/jpeg"))],
    )
    assert response.status_code == 200
    samples = response.json()["samples"]
    assert len(samples) == 2
    assert samples[0]["name"].endswith("a.png")
    assert samples[1]["name"].endswith("样 本.jpg")

    detail = client.get(f"/api/tasks/{task['id']}").json()
    assert detail["sample_count"] == 2
    assert client.get("/api/tasks").json()[0]["sample_count"] == 2

    # 返回的 URL 要能原样读回上传内容（中文与空格经过百分号编码）
    body = client.get(samples[0]["url"]).content
    assert body == first
    body = client.get(samples[1]["url"]).content
    assert body == second
    assert client.get(samples[1]["url"]).headers["content-type"].startswith("image/jpeg")


def test_upload_rejects_bad_extension(client):
    task = _create_task(client)
    response = client.post(f"/api/tasks/{task['id']}/samples",
                           files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_upload_rejects_corrupt_image(client):
    task = _create_task(client)
    response = client.post(f"/api/tasks/{task['id']}/samples",
                           files=[("files", ("fake.png", b"not an image", "image/png"))])
    assert response.status_code == 422
    assert "不是有效图片" in response.json()["error"]["message"]


def test_upload_rejects_oversize_file(client, tmp_path, monkeypatch):
    from api.routes import tasks as tasks_route

    monkeypatch.setattr(tasks_route, "MAX_SAMPLE_BYTES", 8)
    app = create_app(root=tmp_path)
    with TestClient(app) as small_client:
        task = _create_task(small_client)
        response = small_client.post(
            f"/api/tasks/{task['id']}/samples",
            files=[("files", ("big.png", _png_bytes(), "image/png"))])
        assert response.status_code == 422
        assert "50 MB" in response.json()["error"]["message"]


def test_remove_sample_by_stored_name(client):
    task = _create_task(client)
    samples = client.post(f"/api/tasks/{task['id']}/samples",
                          files=[("files", ("a.png", _png_bytes(), "image/png"))]).json()["samples"]
    name = samples[0]["name"]

    remaining = client.delete(f"/api/tasks/{task['id']}/samples/{name}").json()["samples"]
    assert remaining == []

    response = client.delete(f"/api/tasks/{task['id']}/samples/{name}")
    assert response.status_code == 404


def test_detail_includes_runs_pending_review_and_best(client, store):
    task = _create_task(client)
    overlay = store.task_dir(task["id"]).parent / "sample.png"
    overlay.write_bytes(_png_bytes())
    store.save_run_state(task["id"], {
        "task_id": task["id"], "run_id": "agent_run1", "graph_thread_id": "agent_run1",
        "run_started_at": "2026-09-28T10:00:00+00:00", "state_version": 0,
        "run_status": "awaiting_feedback",
        "pending_reference_image_id": "img_a",
        "pending_reference_overlay_path": str(overlay),
        "best_score": 0.62,
        "stop_reason": None,
        "pipeline": {"pipeline": [{"op": "normalize"}], "notes": "v1"},
        "conversation": [{"role": "assistant", "content": "请确认 img_a 的参考掩膜"}],
    })

    detail = client.get(f"/api/tasks/{task['id']}").json()
    assert detail["latest_run_id"] == "agent_run1"
    assert [run["run_id"] for run in detail["runs"]] == ["agent_run1"]
    assert detail["runs"][0]["status_label"] == "待反馈"

    review = detail["pending_review"]
    assert review["stage"] == "reference"
    assert review["message"] == "请确认 img_a 的参考掩膜"
    assert review["overlay_url"].startswith("/api/files?path=")
    assert review["stop_reason"] is None
    assert client.get(review["overlay_url"]).status_code == 200

    assert detail["best"]["score"] == 0.62
    assert detail["best"]["pipeline"] == [{"op": "normalize"}]


def test_pending_review_stage_final_when_stopped(client, store):
    task = _create_task(client)
    store.save_run_state(task["id"], {
        "task_id": task["id"], "run_id": "agent_run2", "graph_thread_id": "agent_run2",
        "run_started_at": "2026-09-28T11:00:00+00:00", "state_version": 0,
        "run_status": "awaiting_feedback",
        "stop_reason": "no_improvement", "best_score": 0.8,
        "pipeline": {}, "conversation": [{"role": "assistant", "content": "达到停止条件"}],
    })
    detail = client.get(f"/api/tasks/{task['id']}").json()
    assert detail["pending_review"]["stage"] == "final"
    assert detail["pending_review"]["stop_reason"] == "no_improvement"
    assert detail["best"]["score"] == 0.8


def test_patch_with_missing_body_is_invalid_request(client):
    task = _create_task(client)
    response = client.patch(f"/api/tasks/{task['id']}", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_health(client):
    payload = client.get("/api/health").json()
    assert payload["ok"] is True
    assert payload["model"] is None or isinstance(payload["model"], str)
