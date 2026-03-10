import os
import sys

# Read original
with open('colab_submission.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Strip all emojis and weird unicode that windows console hates
encoded = content.encode('ascii', 'ignore')
clean_content = encoded.decode('ascii')

with open('colab_submission_clean.py', 'w', encoding='utf-8') as f:
    f.write(clean_content)

print("Created colab_submission_clean.py")
