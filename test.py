import kagglehub

# Download latest version
path = kagglehub.dataset_download("borhanitrash/cat-dataset")

print("Path to dataset files:", path)


"""
# Step 1: Load Python module (if needed)
# Step 2: Create and activate a virtual environment
python -m venv ~/venvs/kagglehub-env
source ~/venvs/kagglehub-env/bin/activate

# Step 3: Upgrade pip (optional but safe)
pip install --upgrade pip

# Step 4: Install kagglehub
pip install kagglehub
"""
