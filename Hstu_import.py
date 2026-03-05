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


import subprocess
result = subprocess.run(['ls', './generative-recommenders/generative_recommenders/research/modeling/sequential'], 
                      capture_output=True, text=True)
print(result.stdout)


import subprocess

for f in ['embedding_modules.py', 'input_features_preprocessors.py', 'output_preprocessors.py', 'features.py']:
    result = subprocess.run(['grep', '-n', 'class ', 
                           f'./generative-recommenders/generative_recommenders/research/modeling/sequential/{f}'], 
                          capture_output=True, text=True)
    print(f"\n--- {f} ---")
    print(result.stdout)


result = subprocess.run(['grep', '-rn', 'class ', 
                       './generative-recommenders/generative_recommenders/research/rails/similarities/'], 
                      capture_output=True, text=True)
print(result.stdout)




import subprocess

# check output preprocessors
result = subprocess.run(['grep', '-n', 'class ', 
                       './generative-recommenders/generative_recommenders/research/modeling/sequential/output_preprocessors.py'], 
                      capture_output=True, text=True)
print("--- output_preprocessors ---")
print(result.stdout)

# check LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor signature
result = subprocess.run(['grep', '-n', 'def __init__', 
                       './generative-recommenders/generative_recommenders/research/modeling/sequential/input_features_preprocessors.py'], 
                      capture_output=True, text=True)
print("--- input preprocessor init ---")
print(result.stdout)





import subprocess

# full content of output_preprocessors
result = subprocess.run(['cat', 
                       './generative-recommenders/generative_recommenders/research/modeling/sequential/output_preprocessors.py'], 
                      capture_output=True, text=True)
print(result.stdout)

# get lines around each __init__ in input_features_preprocessors
result = subprocess.run(['sed', '-n', '44,90p',
                       './generative-recommenders/generative_recommenders/research/modeling/sequential/input_features_preprocessors.py'], 
                      capture_output=True, text=True)
print("--- preprocessor 1 init ---")
print(result.stdout)

result = subprocess.run(['sed', '-n', '94,150p',
                       './generative-recommenders/generative_recommenders/research/modeling/sequential/input_features_preprocessors.py'], 
                      capture_output=True, text=True)
print("--- preprocessor 2 init ---")
print(result.stdout)



import subprocess
result = subprocess.run(['cat', 
   './generative-recommenders/generative_recommenders/research/modeling/sequential/input_features_preprocessors.py'], 
   capture_output=True, text=True)
print(result.stdout)
