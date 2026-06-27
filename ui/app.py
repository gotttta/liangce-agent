import cgi
import html
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from agent import run_from_paths


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = ROOT / "uploads"
OUTPUT_ROOT = ROOT / "outputs"


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_app())
            return
        if parsed.path.startswith("/outputs/"):
            self._send_file(ROOT / unquote(parsed.path.lstrip("/")))
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/run":
            self.send_error(404)
            return

        try:
            payload = handle_run(self)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        self._send_json({"ok": True, **payload})

    def _send_html(self, content, status=200):
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        try:
            resolved = path.resolve()
            output_root = OUTPUT_ROOT.resolve()
            if output_root not in resolved.parents and resolved != output_root:
                self.send_error(403)
                return
            data = resolved.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))


def handle_run(handler):
    form = cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ={
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": handler.headers.get("Content-Type"),
        },
    )
    description = get_text(form, "description") or "面积和数量量测"
    unit = get_text(form, "unit") or "pixel"
    target_path = save_upload(form, "target")
    reference_path = save_upload(form, "reference", required=False)
    run_dir, result = run_from_paths(
        target=target_path,
        reference=reference_path,
        description=description,
        unit=unit,
        output_root=OUTPUT_ROOT,
    )

    run_rel = run_dir.relative_to(ROOT)
    files = {
        "annotated": "/" + str(run_rel / "result_annotated.png"),
        "mask": "/" + str(run_rel / "mask.png"),
        "measurements": "/" + str(run_rel / "measurements.json"),
        "strategy": "/" + str(run_rel / "strategy.json"),
        "algorithm": "/" + str(run_rel / "algorithm.py"),
    }

    return {
        "run_dir": str(run_rel),
        "files": files,
        "result": result,
    }


def get_text(form, name):
    field = form[name] if name in form else None
    if field is None or field.file:
        return ""
    return field.value.strip()


def save_upload(form, name, required=True):
    field = form[name] if name in form else None
    if field is None or not field.filename:
        if required:
            raise ValueError(f"缺少必需图片：{name}")
        return None

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    filename = Path(field.filename).name
    suffix = Path(filename).suffix or ".png"
    upload_path = UPLOAD_ROOT / f"{name}_{len(list(UPLOAD_ROOT.glob(name + '_*'))):04d}{suffix}"
    with upload_path.open("wb") as output:
        while True:
            chunk = field.file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    return upload_path


def render_app():
    return page(
        "量测 Agent",
        """
      <header class="topbar">
        <div class="brand">
          <span class="mark">量</span>
          <div>
            <strong>量测 Agent</strong>
            <small>DRAM 缺陷量测工作台</small>
          </div>
        </div>
        <div class="status"><span></span>中文对话式界面</div>
      </header>
      <main class="chat-shell">
        <aside class="context-panel">
          <section class="context-card">
            <p class="label">当前状态</p>
            <h2>把图发来，我先看一版</h2>
            <p>上传参考图和待测原图，告诉我你想量测什么。结果会直接以消息卡片的方式回你。</p>
          </section>
          <section class="context-card compact">
            <p class="label">输出文件</p>
            <ul class="output-list">
              <li><span class="dot image"></span>result_annotated.png</li>
              <li><span class="dot mask"></span>mask.png</li>
              <li><span class="dot json"></span>measurements.json</li>
              <li><span class="dot json"></span>strategy.json</li>
              <li><span class="dot code"></span>algorithm.py</li>
            </ul>
          </section>
          <section class="context-card compact">
            <p class="label">后续反馈</p>
            <p>“圈多了”“漏了左上角”“边界再贴一点”“只量测中间阵列”。</p>
          </section>
        </aside>

        <section class="conversation">
          <div id="timeline" class="timeline">
            <article class="message agent">
              <div class="avatar">A</div>
              <div class="bubble">
                <p class="speaker">量测 Agent</p>
                <p>先把参考图、待测原图和一句话描述发给我。我会生成一版自动圈选和量测结果。</p>
                <div class="chips">
                  <span>面积</span><span>数量</span><span>外接框</span><span>像素单位</span>
                </div>
              </div>
            </article>
          </div>

          <form id="composer" class="composer">
            <div class="composer-head">
              <div>
                <p class="label">发送一条新任务</p>
                <h3>像聊天一样说明量测任务</h3>
              </div>
              <div class="status-pill">等待发送</div>
            </div>
            <div class="upload-row">
              <label class="upload-tile">
                <span>参考图</span>
                <small>可选，用来说明缺陷长什么样</small>
                <input type="file" name="reference" accept="image/*">
              </label>
              <label class="upload-tile required">
                <span>待测原图</span>
                <small>必填，Agent 会在这张图上圈缺陷</small>
                <input type="file" name="target" accept="image/*" required>
              </label>
            </div>
            <label class="text-field">
              <span>你想做什么量测？</span>
              <textarea name="description" rows="4" placeholder="例如：这是桥连缺陷，帮我圈出异常区域并统计面积和数量" required></textarea>
            </label>
            <div class="composer-actions">
              <label class="unit-field">
                <span>单位</span>
                <input name="unit" value="pixel">
              </label>
              <button type="submit">发送给 Agent</button>
            </div>
          </form>
        </section>

        <aside class="right-panel">
          <section class="metric-placeholder">
            <span>状态</span>
            <strong>等待输入</strong>
          </section>
          <section class="quick-card">
            <p class="label">交互说明</p>
            <p>结果会作为 Agent 回复插入到对话里，下面会出现标注图、mask、量测表和下载入口。</p>
          </section>
        </aside>
      </main>
      <script>
        const form = document.getElementById('composer');
        const timeline = document.getElementById('timeline');
        const statusPill = form.querySelector('.status-pill');
        const statusCard = document.querySelector('.metric-placeholder strong');

        function escapeHtml(text) {
          return text
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
        }

        function fileChips(files) {
          if (!files || !files.length) return '';
          return [...files].map(file => `<span class="chip">${escapeHtml(file.name)}</span>`).join('');
        }

        function pushUserMessage(description, referenceFile, targetFile) {
          const node = document.createElement('article');
          node.className = 'message user';
          node.innerHTML = `
            <div class="avatar">你</div>
            <div class="bubble user-bubble">
              <p class="speaker">你</p>
              <p>${escapeHtml(description || '（未填写）')}</p>
              <div class="chips">
                ${targetFile ? `<span>已选待测原图：${escapeHtml(targetFile.name)}</span>` : ''}
                ${referenceFile ? `<span>已选参考图：${escapeHtml(referenceFile.name)}</span>` : '<span>未选参考图</span>'}
              </div>
            </div>
          `;
          timeline.appendChild(node);
          node.scrollIntoView({behavior: 'smooth', block: 'end'});
        }

        function pushThinking() {
          const node = document.createElement('article');
          node.className = 'message agent';
          node.dataset.thinking = '1';
          node.innerHTML = `
            <div class="avatar">A</div>
            <div class="bubble">
              <p class="speaker">量测 Agent</p>
              <p>我在看图和算量，马上给你结果。</p>
            </div>
          `;
          timeline.appendChild(node);
          node.scrollIntoView({behavior: 'smooth', block: 'end'});
          return node;
        }

        function renderResultMessage(payload) {
          const result = payload.result;
          const files = payload.files;
          const notes = (result.notes || []).map(note => `<li>${escapeHtml(note)}</li>`).join('') || '<li>没有额外说明</li>';
          const rows = (result.results || []).map(item => `
            <tr>
              <td><span class="id-chip">${item.id}</span></td>
              <td>${item.area}</td>
              <td>${item.width} x ${item.height}</td>
              <td>${item.aspect_ratio}</td>
              <td>${JSON.stringify(item.bbox)}</td>
            </tr>
          `).join('') || '<tr><td colspan="5">没有找到通过筛选的区域</td></tr>';

          const node = document.createElement('article');
          node.className = 'message agent';
          node.innerHTML = `
            <div class="avatar">A</div>
            <div class="bubble result-bubble">
              <p class="speaker">量测 Agent</p>
              <p>我已经跑完一版 baseline。当前结果是：检测到 <strong>${result.summary.count}</strong> 个区域，总面积 <strong>${result.summary.total_area} ${escapeHtml(result.summary.unit)}</strong>。</p>
              <div class="result-grid">
                <figure>
                  <img src="${files.annotated}" alt="标注结果图">
                  <figcaption>标注结果图</figcaption>
                </figure>
                <figure>
                  <img src="${files.mask}" alt="缺陷 mask">
                  <figcaption>缺陷 mask</figcaption>
                </figure>
              </div>
              <section class="table-card">
                <div class="table-head">
                  <h3>量测明细</h3>
                  <span>${result.summary.count} 个区域</span>
                </div>
                <table>
                  <thead><tr><th>ID</th><th>面积</th><th>尺寸</th><th>长宽比</th><th>外接框</th></tr></thead>
                  <tbody>${rows}</tbody>
                </table>
              </section>
              <div class="feedback-box">
                <span>下一步反馈可以这样说</span>
                <p>“圈多了”“漏了右下角”“只看中间阵列区域”“边界贴紧一点”。</p>
              </div>
              <section class="downloads">
                <a href="${files.annotated}">标注图</a>
                <a href="${files.mask}">mask</a>
                <a href="${files.measurements}">measurements.json</a>
                <a href="${files.strategy}">strategy.json</a>
                <a href="${files.algorithm}">algorithm.py</a>
              </section>
              <details class="raw-json">
                <summary>原始 JSON</summary>
                <pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre>
              </details>
              <section class="quick-card" style="margin-top:14px;">
                <p class="label">运行说明</p>
                <ul class="notes">${notes}</ul>
              </section>
            </div>
          `;
          timeline.appendChild(node);
          node.scrollIntoView({behavior: 'smooth', block: 'end'});
        }

        form.addEventListener('submit', async (event) => {
          event.preventDefault();

          const description = form.querySelector('textarea[name="description"]').value.trim();
          const referenceFile = form.querySelector('input[name="reference"]').files[0];
          const targetFile = form.querySelector('input[name="target"]').files[0];
          const unit = form.querySelector('input[name="unit"]').value.trim() || 'pixel';

          if (!targetFile) {
            alert('请先选择待测原图。');
            return;
          }

          pushUserMessage(description, referenceFile, targetFile);
          const thinking = pushThinking();
          statusPill.textContent = '处理中';
          statusCard.textContent = '处理中';

          const formData = new FormData(form);
          formData.set('unit', unit);

          try {
            const response = await fetch('/api/run', { method: 'POST', body: formData });
            const payload = await response.json();
            thinking.remove();

            if (!payload.ok) {
              throw new Error(payload.error || '运行失败');
            }

            renderResultMessage(payload);
            statusPill.textContent = '已完成';
            statusCard.textContent = `${payload.result.summary.count} 个区域`;
          } catch (error) {
            thinking.remove();
            const fail = document.createElement('article');
            fail.className = 'message agent';
            fail.innerHTML = `
              <div class="avatar">A</div>
              <div class="bubble">
                <p class="speaker">量测 Agent</p>
                <p>这次没跑起来：${escapeHtml(error.message || String(error))}</p>
              </div>
            `;
            timeline.appendChild(fail);
            statusPill.textContent = '失败';
            statusCard.textContent = '失败';
          }
        });
      </script>
    """,
    )


def render_error(exc):
    return page("运行失败", f"""
      <main class="error-shell">
        <a class="new-run" href="/">返回</a>
        <section class="error-card">
          <h1>运行失败</h1>
          <pre>{html.escape(str(exc))}</pre>
        </section>
      </main>
    """)


def page(title, body):
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #66737b;
      --line: #d8e0e1;
      --paper: #eef2f1;
      --panel: #ffffff;
      --soft: #f8faf9;
      --accent: #0f766e;
      --accent-strong: #0b5d57;
      --blue: #2563eb;
      --rose: #be123c;
      --amber: #b7791f;
      --shadow: 0 18px 44px rgba(22, 36, 42, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(255,255,255,0) 260px), var(--paper);
    }}
    .topbar {{
      width: min(1440px, calc(100vw - 28px));
      min-height: 72px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 12px 0 10px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .mark {{
      width: 40px;
      height: 40px;
      display: grid;
      place-items: center;
      border-radius: 8px;
      background: var(--ink);
      color: white;
      font-weight: 900;
      letter-spacing: 0;
    }}
    .brand strong {{
      display: block;
      font-size: 18px;
      line-height: 1.2;
    }}
    .brand small {{
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-weight: 700;
    }}
    .status, .new-run {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: white;
      color: var(--muted);
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 800;
      text-decoration: none;
    }}
    .status span {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
    }}
    .chat-shell {{
      width: min(1440px, calc(100vw - 28px));
      min-height: calc(100vh - 96px);
      margin: 0 auto 24px;
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr) 300px;
      gap: 14px;
    }}
    .context-panel, .conversation, .right-panel {{
      min-width: 0;
    }}
    .context-panel, .right-panel {{
      display: grid;
      align-content: start;
      gap: 14px;
    }}
    .context-card, .quick-card, .metric-placeholder, .raw-json, .error-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }}
    .context-card h2 {{
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.15;
      letter-spacing: 0;
    }}
    .context-card p, .quick-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }}
    .label, .speaker {{
      margin: 0 0 8px;
      color: var(--accent-strong);
      font-size: 12px;
      line-height: 1;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .output-list, .notes {{
      list-style: none;
      padding: 0;
      margin: 10px 0 0;
      display: grid;
      gap: 9px;
      color: var(--muted);
      font-size: 13px;
    }}
    .notes {{
      list-style: disc;
      padding-left: 18px;
    }}
    .output-list li {{
      display: flex;
      align-items: center;
      gap: 8px;
      overflow-wrap: anywhere;
    }}
    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      flex: 0 0 auto;
    }}
    .dot.image {{ background: var(--blue); }}
    .dot.mask {{ background: var(--rose); }}
    .dot.code {{ background: var(--amber); }}
    .conversation {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      min-height: 640px;
      overflow: hidden;
    }}
    .timeline {{
      display: grid;
      align-content: start;
      gap: 18px;
      padding: 18px;
    }}
    .message {{
      display: grid;
      grid-template-columns: 38px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }}
    .avatar {{
      width: 38px;
      height: 38px;
      display: grid;
      place-items: center;
      border-radius: 8px;
      background: var(--ink);
      color: white;
      font-size: 13px;
      font-weight: 900;
    }}
    .message.user .avatar {{
      background: var(--accent);
    }}
    .bubble {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 15px;
    }}
    .user-bubble {{
      background: #f4fbf9;
    }}
    .result-bubble {{
      background: #fff;
    }}
    .bubble p {{
      margin: 0;
      line-height: 1.55;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .chips span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      background: white;
      padding: 6px 9px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}
    .composer {{
      margin: 0 18px 18px;
      display: grid;
      gap: 14px;
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .composer-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .composer-head h3 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: white;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
      padding: 8px 11px;
    }}
    .upload-row {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .upload-tile, .text-field, .unit-field {{
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
    }}
    .upload-tile {{
      border: 1px dashed #aab7ba;
      border-radius: 8px;
      background: var(--soft);
      padding: 12px;
    }}
    .upload-tile.required {{
      border-color: var(--accent);
      background: #f4fbf9;
    }}
    .upload-tile small {{
      font-weight: 600;
      line-height: 1.35;
    }}
    input, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--ink);
      padding: 10px 11px;
      font: inherit;
    }}
    textarea {{
      resize: vertical;
    }}
    input[type="file"] {{
      padding: 8px;
      font-size: 12px;
    }}
    .composer-actions {{
      display: grid;
      grid-template-columns: 160px minmax(180px, 1fr);
      gap: 12px;
      align-items: end;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 12px 16px;
      font: inherit;
      font-weight: 900;
      cursor: pointer;
    }}
    button:hover {{
      background: var(--accent-strong);
    }}
    .metric-placeholder {{
      display: grid;
      gap: 8px;
    }}
    .metric-placeholder span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
    }}
    .metric-placeholder strong {{
      font-size: 24px;
      line-height: 1.1;
      overflow-wrap: anywhere;
    }}
    .result-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    figure {{
      margin: 0;
      overflow: hidden;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #0e1518;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      background: #0e1518;
    }}
    figcaption {{
      padding: 9px 11px;
      color: #d9e4e5;
      font-size: 13px;
      font-weight: 900;
    }}
    .table-card, .feedback-box, .downloads {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 14px;
      margin-top: 14px;
    }}
    .table-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
    }}
    h3 {{
      margin: 0;
      font-size: 14px;
      letter-spacing: 0;
    }}
    .table-head span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .id-chip {{
      min-width: 26px;
      height: 24px;
      display: inline-grid;
      place-items: center;
      border-radius: 6px;
      background: var(--ink);
      color: white;
      font-size: 12px;
      font-weight: 900;
    }}
    .feedback-box span {{
      display: block;
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 900;
      margin-bottom: 8px;
    }}
    .feedback-box p {{
      margin: 0;
      color: var(--muted);
    }}
    .download-list {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}
    .download-list a, .new-run {{
      text-decoration: none;
      color: var(--accent-strong);
      font-weight: 900;
    }}
    .download-list a {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--soft);
      padding: 10px 11px;
      overflow-wrap: anywhere;
    }}
    .raw-json {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 14px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 900;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #263238;
      font-size: 12px;
      line-height: 1.45;
    }}
    .error-shell {{
      width: min(860px, calc(100vw - 28px));
      margin: 32px auto;
    }}
    .error-card {{
      margin-top: 14px;
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }}
    .error-card h1 {{
      margin: 0 0 12px;
      color: #b42318;
    }}
    @media (max-width: 1180px) {{
      .chat-shell {{
        grid-template-columns: minmax(0, 1fr) 300px;
      }}
      .context-panel {{
        display: none;
      }}
    }}
    @media (max-width: 820px) {{
      .topbar, .chat-shell {{
        width: min(100vw - 20px, 720px);
      }}
      .topbar {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .chat-shell, .upload-row, .composer-actions, .result-grid {{
        grid-template-columns: 1fr;
      }}
      .right-panel {{
        order: -1;
      }}
      .conversation {{
        min-height: auto;
      }}
      .composer {{
        margin: 0 12px 12px;
      }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def main():
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 7860), AppHandler)
    print("Frontend running at http://127.0.0.1:7860")
    server.serve_forever()


if __name__ == "__main__":
    main()
