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


# correct import path
from generative_recommenders.modeling.sequential.hstu import HSTU

import subprocess
result = subprocess.run(['pip', 'show', 'generative-recommenders'], 
                      capture_output=True, text=True)
print(result.stdout)

result = subprocess.run(['pip', 'list'], 
                      capture_output=True, text=True)
# filter for generative
for line in result.stdout.split('\n'):
    if 'generative' in line.lower():
        print(line)



try:
    from generative_recommenders.modeling.sequential.hstu import HSTU
    print("HSTU import successful")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Other error: {e}")





import subprocess

# clone manually with submodules
result = subprocess.run([
    'git', 'clone', '--recurse-submodules',
    'https://github.com/facebookresearch/generative-recommenders.git'
], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)


result = subprocess.run(
    ['pip', 'install', '-e', './generative-recommenders'],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)


try:
    from generative_recommenders.modeling.sequential.hstu import HSTU
    print("HSTU import successful")
except Exception as e:
    print(f"Failed: {e}")







import subprocess

result = subprocess.run(['ls', './generative-recommenders'], 
                      capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

result = subprocess.run(['ls', './generative-recommenders/generative_recommenders'], 
                      capture_output=True, text=True)
print(result.stdout)
print(result.stderr)


import sys
sys.path.insert(0, './generative-recommenders')

try:
    from generative_recommenders.modeling.sequential.hstu import HSTU
    print("HSTU import successful")
except Exception as e:
    print(f"Failed: {e}")








import subprocess

result = subprocess.run(['ls', './generative-recommenders/generative_recommenders'], 
                      capture_output=True, text=True)
print(result.stdout)


result = subprocess.run(['ls', './generative-recommenders/generative_recommenders/ops'], 
                      capture_output=True, text=True)
print(result.stdout)






import sys
sys.path.insert(0, './generative-recommenders')

try:
    from generative_recommenders.research.modeling.sequential.hstu import HSTU
    print("HSTU import successful")
except Exception as e:
    print(f"Failed: {e}")

result = subprocess.run(['ls', './generative-recommenders/generative_recommenders/research'], 
                      capture_output=True, text=True)
print(result.stdout)







import sys
sys.path.insert(0, './generative-recommenders')

from generative_recommenders.research.modeling.sequential.hstu import HSTU



import inspect
print(inspect.signature(HSTU.__init__))
