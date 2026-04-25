import json
import os

p = r'c:\Users\MITHUN\Desktop\STUDIES\Drive\Big Vision\Code\Testing.ipynb'

with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. New Colab-Specific Setup Cell
colab_setup = [
    "import os\n",
    "from google.colab import drive\n",
    "\n",
    "# 1. Mount Google Drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# 2. Define Colab Paths\n",
    "WORKSPACE_DIR = '/content/'\n",
    "RESULTS_DIR   = '/content/drive/MyDrive/Basketball_Project_Final/'\n",
    "\n",
    "os.makedirs(RESULTS_DIR, exist_ok=True)\n",
    "\n",
    "def get_yolo_paths(dataset_name, split):\n",
    "    \"\"\"Automatically find images and labels in Colab structure.\"\"\"\n",
    "    dataset_path = os.path.join(WORKSPACE_DIR, dataset_name)\n",
    "    # Check Pattern A: dataset/train/images\n",
    "    p_a_img = os.path.join(dataset_path, split, 'images')\n",
    "    p_a_lbl = os.path.join(dataset_path, split, 'labels')\n",
    "    # Check Pattern B: dataset/images/train\n",
    "    p_b_img = os.path.join(dataset_path, 'images', split)\n",
    "    p_b_lbl = os.path.join(dataset_path, 'labels', split)\n",
    "    \n",
    "    if os.path.exists(p_a_img): return p_a_img, p_a_lbl\n",
    "    return p_b_img, p_b_lbl\n",
    "\n",
    "print(f'✅ Workspace: {WORKSPACE_DIR}')\n",
    "print(f'✅ Results  : {RESULTS_DIR}')\n"
]

# Update the first code cell with pure Colab logic
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = colab_setup
        break

# 2. Update Cell 2 (Download) to use /content/ directly
for cell in nb['cells']:
    if 'rf.workspace' in ''.join(cell['source']):
        new_source = []
        for line in cell['source']:
            line = line.replace('location=os.path.join(WORKSPACE_DIR, "', 'location="/content/')
            line = line.replace('location="./workspace/', 'location="/content/')
            new_source.append(line)
        cell['source'] = new_source

# 3. Clean up any remaining local paths in all other cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace local workspace folder with Colab root
            line = line.replace('./workspace/', '/content/')
            line = line.replace("preprocess_images('dataset1')", "preprocess_images('dataset1')") # ensure consistency
            new_source.append(line)
        cell['source'] = new_source

with open(p, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Testing.ipynb is now perfectly optimized for Google Colab.")
