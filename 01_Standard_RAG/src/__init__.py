import os
from dotenv import load_dotenv

# 1. Load the central root .env (workspace level)
root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(root_env):
    load_dotenv(root_env)

# 2. Load the local sub-project .env if present (as a fallback)
local_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(local_env):
    load_dotenv(local_env)

