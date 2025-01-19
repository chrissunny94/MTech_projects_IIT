import os
local_rank = int(os.environ.get('LOCAL_RANK', -1))  # Default to -1 if not set
