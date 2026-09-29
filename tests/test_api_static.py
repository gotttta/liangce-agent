"""生产模式静态托管（web/dist + SPA fallback）。

覆盖审查要求的四类行为：静态资源、前端路由回退、未知 /api 返回 JSON 404、
dist 缺失时的构建提示页。
"""
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app


def _write_dist(root: Path) -> Path:
    dist = root / "web" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body>spa-shell</body></html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('spa');", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    return dist


def test_static_assets_and_frontend_route_served(tmp_path):
    _write_dist(tmp_path)
    client = TestClient(create_app(root=tmp_path))
    response = client.get("/assets/app.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert response.text == "console.log('spa');"

    assert client.get("/favicon.svg").status_code == 200
    # 前端路由（react-router）回退到 index.html
    for route in ("/", "/tasks/abc"):
        response = client.get(route)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "spa-shell" in response.text


def test_unknown_api_path_returns_json_404_not_index(tmp_path):
    _write_dist(tmp_path)
    client = TestClient(create_app(root=tmp_path))
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert "spa-shell" not in response.text


def test_dist_path_traversal_falls_back_to_index(tmp_path):
    dist = _write_dist(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    (dist / "link.txt").symlink_to(secret)
    client = TestClient(create_app(root=tmp_path))
    for path in ("../secret.txt", "link.txt", "..%2Fsecret.txt"):
        response = client.get(f"/{path}")
        assert response.status_code == 200
        assert "top secret" not in response.text
        assert "spa-shell" in response.text


def test_missing_dist_starts_with_build_hint(tmp_path):
    # web/dist 不存在：python -m api 仍能启动，/ 返回提示页而不是报错
    client = TestClient(create_app(root=tmp_path))
    assert client.get("/api/health").status_code == 200
    response = client.get("/")
    assert response.status_code == 200
    assert "npm run build" in response.text
    response = client.get("/tasks/abc")
    assert response.status_code == 200
    assert "npm run build" in response.text
