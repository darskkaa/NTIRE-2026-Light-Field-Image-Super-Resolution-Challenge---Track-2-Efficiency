import zipfile
import os

directories_to_zip = ['model', 'utils']
files_to_zip = [
    'option.py',
    'inference.py',
    'Generate_Validation_Data.py',
    'format_submission.py',
    'validate_submission.py',
    'eval_validation.sh',
    'colab_submission.py'
]

with zipfile.ZipFile('Colab_Submission_Pack.zip', 'w') as zipf:
    for d in directories_to_zip:
        for root, dirs, files in os.walk(d):
            # Skip large unnecessary directories
            if '__pycache__' in root or 'submission' in root or 'SEED' in root or 'model_v65' in root:
                continue
            for file in files:
                if file.endswith(('.pyc', '.zip', '.tar.gz', '.pdf', '.xls', '.md')):
                    continue
                zipf.write(os.path.join(root, file))
                
    for f in files_to_zip:
        if os.path.exists(f):
            zipf.write(f)

print("Created Colab_Submission_Pack.zip")
