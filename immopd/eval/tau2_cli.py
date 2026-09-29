"""tau2-bench CLI with the prefix-continuation agent registered (``--agent llm_agent_prefix``)."""
from __future__ import annotations

import sys

try:
    from immopd.eval.prefix import tooluse_agent  # noqa: F401
except Exception:
    pass

if __name__ == "__main__":
    from tau2.cli import main
    sys.exit(main())
