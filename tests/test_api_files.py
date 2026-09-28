import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.app import create_app
from api.files import file_roots, file_url


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(root=tmp_path)) as test_client:
        yield test_client


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path


def _write(root: Path, relative: str, data: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _png_bytes(size=(12, 8)) -> bytes:
    image = Image.new("L", size, color=99)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_serves_images_from_workspace_and_outputs(client, root):
    for relative in ("workspace/tasks/task_x/samples/a.png", "outputs/run_y/mask.jpg"):
        data = _png_bytes()
        path = _write(root, relative, data)
        response = client.get("/api/files", params={"path": relative})
        assert response.status_code == 200
        assert response.content == data
        assert response.headers["content-type"].startswith("image/")
        assert file_url(path, file_roots(root)).endswith(relative.replace(" ", "%20"))


def test_serves_json_artifact(client, root):
    _write(root, "outputs/run_y/score.json", b'{"composite_mean": 0.5}')
    response = client.get("/api/files", params={"path": "outputs/run_y/score.json"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert b"composite_mean" in response.content


def test_tiff_is_converted_to_png_in_memory(client, root):
    buffer = io.BytesIO()
    Image.new("L", (10, 6), color=7).save(buffer, format="TIFF")
    _write(root, "workspace/tasks/task_x/ref.tif", buffer.getvalue())

    response = client.get("/api/files", params={"path": "workspace/tasks/task_x/ref.tif"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as converted:
        assert converted.size == (10, 6)
    assert not (root / "workspace/tasks/task_x/ref.png").exists()  # 不落盘


def test_rejects_parent_escape(client):
    response = client.get("/api/files", params={"path": "../../etc/passwd"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden_path"


def test_rejects_absolute_path_outside_roots(client):
    response = client.get("/api/files", params={"path": "/etc/passwd"})
    assert response.status_code == 403


def test_rejects_symlink_pointing_outside_roots(client, root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.png"
    outside.write_bytes(_png_bytes())
    link_dir = root / "workspace" / "links"
    link_dir.mkdir(parents=True)
    (link_dir / "escape.png").symlink_to(outside)

    response = client.get("/api/files", params={"path": "workspace/links/escape.png"})
    assert response.status_code == 403


def test_rejects_disallowed_suffix(client, root):
    _write(root, "workspace/tasks/task_x/notes.py", b"print(1)")
    response = client.get("/api/files", params={"path": "workspace/tasks/task_x/notes.py"})
    assert response.status_code == 403


def test_missing_file_returns_404(client):
    response = client.get("/api/files", params={"path": "workspace/tasks/task_x/none.png"})
    assert response.status_code == 404


def test_file_url_quotes_unicode_and_spaces(client, root):
    relative = "workspace/tasks/task_x/样 本.png"
    _write(root, relative, _png_bytes())

    url = file_url(root / relative, file_roots(root))
    assert " " not in url
    response = client.get(url)
    assert response.status_code == 200
