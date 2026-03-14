"""
Importing this package registers all tools via decorators.

Why:
- Decorators run at import time.
- So we import every tools module here once.
"""

# These imports cause @register_tool to execute and populate GLOBAL_TOOL_REGISTRY
from tools import data_loader  # noqa: F401
from tools import metrics      # noqa: F401
from tools import analysis     # noqa: F401
from tools import reporting    # noqa: F401
from tools import simulation   # noqa: F401
