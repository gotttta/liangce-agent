"""python -m api：配置统一日志、加载 .env 后启动 uvicorn，只监听本机回环（计划 §6.4）。"""
import uvicorn

from api.app import create_app
from api.config import api_port
from core.runtime_logging import configure_logging


def main() -> None:
    configure_logging()
    from providers.vision import load_env_file

    load_env_file()
    uvicorn.run(create_app(), host="127.0.0.1", port=api_port(), log_level="warning")


if __name__ == "__main__":
    main()
