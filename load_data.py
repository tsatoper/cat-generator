import kagglehub

# Download latest version
path = kagglehub.dataset_download("borhanitrash/cat-dataset")

print("Path to dataset files:", path)


"""
python -m venv ~/venvs/kagglehub-env
source ~/venvs/kagglehub-env/bin/activate
pip install kagglehub
"""
