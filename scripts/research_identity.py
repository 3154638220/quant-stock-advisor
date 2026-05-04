"""轻量研究产物的统一命名与身份字段（薄层入口，实际实现在 src.cli.research_identity）。"""

# 薄层 re-export：保持 scripts/ 入口兼容现有的命令行和测试导入路径
from src.cli.research_identity import make_research_identity, slugify_token  # noqa: F401
