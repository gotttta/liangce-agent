# Docker 算法沙箱

所有 Pipeline（包括纯内置算子）都在一次性 Docker 容器中执行。模型生成的代码使用普通 Python，允许 import、循环、辅助函数和完整算法；保留 `apply(data, params)` 接口、二维图像/Mask 类型、源码大小和 Pipeline 结构检查，不再使用 Python AST 节点或 NumPy 函数白名单。主程序只做解析和契约校验，不执行生成代码。

## 本机准备（macOS）

安装并启动 Docker Desktop，完成它的首次初始化：

```sh
brew install --cask docker-desktop
open -a Docker
```

如果未安装全局命令链接，先设置 `export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"`；项目执行器会自动查找该路径。

在项目根目录构建运行镜像，然后验证：

```sh
docker info
docker build -f sandbox/Dockerfile -t liangce-sandbox:2 .
LIANGCE_REQUIRE_DOCKER_TESTS=1 .venv/bin/python -m pytest -q tests/test_sandbox.py tests/test_generated_operators.py tests/test_docker_sandbox.py
```

未安装 Docker、daemon 未启动或镜像未构建时，执行会明确失败，不会自动下载镜像或回退宿主运行。其余算法工作流测试同样需要运行镜像。代码或沙箱依赖更新后必须重新构建镜像；镜像只复制 `core/` 和沙箱依赖配置，构建上下文排除图片、项目数据、环境文件和密钥。

默认镜像 `liangce-sandbox:2`。可由部署人员设置 `LIANGCE_SANDBOX_IMAGE` 为已构建的可信镜像，生产环境建议使用不可变 digest。模型输出不能设置镜像、Docker 参数或资源上限。

## 隔离配置

- 非 root UID/GID 65534，删除全部 Linux capabilities，禁止提权；保留 Docker 默认 seccomp。
- `--network=none`，没有外网、宿主网络或内部服务访问；不传递宿主 API 密钥。
- 根目录只读，没有任何宿主目录或 Docker socket 挂载。
- `/tmp` 是 128 MiB 的临时内存文件系统，带 noexec/nosuid/nodev；容器销毁即清空。
- 默认 1 GiB 内存、禁止额外 swap、2 CPU、最多 64 个进程/线程，20 秒执行期限。
- stdout/stderr 合计最多 64 MiB；关闭 Docker 日志落盘，防止输出洪水撑满磁盘。
- 输入/输出使用有上限的 JSON 元数据与二进制 ndarray 帧；不用 pickle，不从容器取任意路径，不在宿主反序列化 Python 对象。
- 超时/错误路径强制删除整个容器，不能只杀 Docker CLI；清理失败会报告容器名供管理员处理。

`SandboxLimits` 是主程序可信配置。输入图像要求有限数值的灰度或RGB/RGBA数组，原始 float32 数组最多 8 MiB，完整 JSON 最多 64 MiB。若输出 Mask，必须为与输入空间尺寸一致的二维二值数组；中间产物、轮廓和 trace 同样校验结构。结果中的 Pipeline 以宿主原始请求为准，不信任容器回传源码。

## 模型可用环境

Python 3.12、NumPy、OpenCV、SciPy、scikit-image、Pillow，版本见 `sandbox/requirements.txt`。需要其他依赖时由维护人员更新镜像，执行期间不联网安装。仍保留可复现 Pipeline、结果展示和用户验收；接受算法不会自动发布其自定义算子。

Docker Desktop 在 Linux VM 内运行容器。此配置提高了隔离强度，但不保证抵御所有内核或虚拟化漏洞。多租户公开服务应增加专用执行主机、gVisor 或每任务虚拟机。宿主上的 `require_docker_worker` 是防误用保护，真正的安全边界是容器配置，不是 Python 标记文件。

## 镜像下载故障

如果 Docker Hub 不可达，可选择 Docker Official Images 的公共 ECR 镜像源构建（仅构建阶段联网，实验容器仍断网）：

```sh
docker build --build-arg BASE_IMAGE=public.ecr.aws/docker/library/python:3.12.14-slim -f sandbox/Dockerfile -t liangce-sandbox:2 .
```

若两者均失败，需要先解决 Docker Desktop 到镜像仓库的连接。不要关闭 TLS 校验，不要将不可信镜像代入执行环境。镜像缺失时应用会报告沙箱不可用。

若基础镜像已下载、但构建中 PyPI 连接中断，可以仅在构建时指定镜像源：

```sh
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -f sandbox/Dockerfile -t liangce-sandbox:2 .
```

此参数不改变算法容器的断网设置；默认构建源仍为 PyPI 官方源。

## 自由算法接口与预算

现已取消单份源码 20 KB 上限和前 8 个自定义算子的截断。请求/响应总字节限制仍然有效。v3 不再要求生成 Mask：用 `outputs` 将结果名称映射到节点 ID，结果可为图像、Mask、轮廓或 JSON 元数据。MetadataArtifact 自定义算子返回 JSON 对象，`boxes` 使用 `[x1,y1,x2,y2]`，`points` 使用 `[x,y]`，也可包含量测值；结构化结果保存到 `outputs.json` 和量测记录，不冒充像素准确率。

多输入自定义算子声明 `input_ports`，例如 `{"image":"ImageArtifact","stats":"MetadataArtifact"}`，并通过节点 `inputs` 连线；此时 `apply(data, params)` 中的 data 是端口名称到数组/JSON 数据的映射。未声明 input_ports 的旧算子仍接收单个数组。

外部输入由 Pipeline 的 `input_types` 声明，例如 `{"$reference":"ImageArtifact"}`，调用 `execute_pipeline_sandbox(..., inputs={"$reference": reference_array})` 提供。工作流自动提供声明为 ImageArtifact 的 `$rgb` 原始彩色图；其他输入由 previous_state.algorithm_inputs 显式提供，不读取模型指定的宿主路径。灰度 `$image` 保持兼容，ImageArtifact 支持灰度/RGB/RGBA，Mask 仍为二维。

运行预算由部署环境变量或可信调用方 SandboxLimits 配置，不能由模型自行提高：

```sh
export LIANGCE_SANDBOX_TIMEOUT_SECONDS=90
export LIANGCE_SANDBOX_MEMORY_MB=2048
export LIANGCE_SANDBOX_CPUS=2
export LIANGCE_SANDBOX_PIDS=64
export LIANGCE_SANDBOX_MAX_STEPS=256
```

默认仍为20秒、1024MiB、2CPU、64进程/线程；节点默认上限改为256且可配置。输入输出总大小与中间产物预览预算仍保留；最终显式 outputs 不受“最多8个中间预览”的筛选影响。UI画笔和像素对比仅对Mask输出适用；没有Mask时显示结果预览与结构化量测，并标记像素对比不适用。普通单输入旧 steps 格式仍以分割为目标，非Mask任务请用v3。

传输协议 v2 使用有界二进制 ndarray 帧（不使用 pickle），输入、输出分别限制 64 MiB，数组形状、dtype、长度与有限值均验证。宿主与 worker 必须同步更新；旧镜像不兼容新请求。原始科学灰度输入保留位深，显示预览独立映射。
