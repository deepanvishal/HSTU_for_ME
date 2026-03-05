# Step 1: check if you can reach github
import subprocess
result = subprocess.run(['pip', 'install', 'git+https://github.com/facebookresearch/generative-recommenders.git'], 
                      capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

# Step 2: if installed, test import
from generative_recommenders.ops.jagged import jagged_self_attention
from generative_recommenders.modeling.sequential.hstu import HSTU
print("HSTU available")
