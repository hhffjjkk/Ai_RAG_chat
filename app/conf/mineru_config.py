# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
# 若.env不在项目根目录，可指定路径：load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()


# MinerU 支持解析的文件类型（统一转为 Markdown，小写后缀，便于匹配）
# 包含：PDF、图片（png/jpg/jpeg/jp2/webp/gif/bmp）、Office 文档（doc/docx/ppt/pptx/xls/xlsx）
MINERU_ALLOWED_EXTENSIONS = (
    ".pdf",
    # --- 图片 ---
    ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp",
    # --- Office 文档 ---
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
)
# 图片类型子集：统计页数时图片固定按 1 页处理
MINERU_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp")
# 旧版二进制 Office 类型（无法本地解析页数，页数校验交由 MinerU 服务端兜底）
MINERU_BINARY_OFFICE_EXTENSIONS = (".doc", ".ppt", ".xls")


# 定义minerU服务配置
@dataclass
class MineruConfig:
    base_url: str
    api_key : str
    max_file_size_mb: int   # 单文件大小上限（MB），默认 200
    max_pages: int          # 单文件页数上限（页），默认 200

mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL"),
    api_key=os.getenv("MINERU_API_TOKEN"),
    max_file_size_mb=int(os.getenv("MINERU_MAX_FILE_SIZE_MB", "200")),
    max_pages=int(os.getenv("MINERU_MAX_PAGES", "200")),
)