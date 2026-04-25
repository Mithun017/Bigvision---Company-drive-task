import json
import os

path = r'c:\Users\MITHUN\Desktop\STUDIES\Drive\Big Vision\Code\Testing_Local.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Setup Cell (Cell 1) with Absolute Paths
setup_code = [
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "# Get the absolute path to the 'Big Vision' root directory\n",
    "ROOT_DIR = Path(os.getcwd()).parent if 'Code' in os.getcwd() else Path(os.getcwd())\n",
    "WORKSPACE_DIR = os.path.join(ROOT_DIR, 'workspace')\n",
    "RESULTS_DIR = os.path.join(ROOT_DIR, 'Results')\n",
    "\n",
    "os.makedirs(WORKSPACE_DIR, exist_ok=True)\n",
    "os.makedirs(RESULTS_DIR, exist_ok=True)\n",
    "\n",
    "def get_yolo_paths(dataset_name, split):\n",
    "    \"\"\"Automatically find images and labels in any folder structure.\"\"\"\n",
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
    "print(f'✅ ROOT   : {ROOT_DIR}')\n",
    "print(f'✅ WORKSP : {WORKSPACE_DIR}')\n"
]

# Update the first code cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = setup_code
        break

# 2. Update Download Cell (Cell 2) to use WORKSPACE_DIR
for cell in nb['cells']:
    if 'rf.workspace' in ''.join(cell['source']):
        new_source = []
        for line in cell['source']:
            line = line.replace('location="./workspace/', 'location=os.path.join(WORKSPACE_DIR, "')
            line = line.replace('location=\"/content/', 'location=os.path.join(WORKSPACE_DIR, \"')
            new_source.append(line)
        cell['source'] = new_source

# 3. Update Inspection (Cell 3)
for cell in nb['cells']:
    if 'inspect_dataset' in ''.join(cell['source']) and 'def' not in ''.join(cell['source']):
        cell['source'] = [
            "inspect_dataset(os.path.join(WORKSPACE_DIR, 'dataset1'), 'Basketball 1')\n",
            "inspect_dataset(os.path.join(WORKSPACE_DIR, 'dataset2'), 'Basketball 2')\n",
            "inspect_dataset(os.path.join(WORKSPACE_DIR, 'dataset3'), 'Basketball 3')\n"
        ]

# 4. Update Filter call (Cell 4)
for cell in nb['cells']:
    if 'filter_low_quality_images' in ''.join(cell['source']) and 'def' not in ''.join(cell['source']):
        cell['source'] = [
            "filter_low_quality_images('dataset1')\n",
            "filter_low_quality_images('dataset2')\n",
            "filter_low_quality_images('dataset3')\n"
        ]
    if 'def filter_low_quality_images' in ''.join(cell['source']):
        # Update function to accept name instead of full path for easier use
        cell['source'] = [
            "def filter_low_quality_images(dataset_name, blur_threshold=80, min_brightness=25, max_brightness=235):\n",
            "    removed, kept = 0, 0\n",
            "    for split in ['train', 'valid', 'val', 'test']:\n",
            "        img_dir, lbl_dir = get_yolo_paths(dataset_name, split)\n",
            "        if not os.path.exists(img_dir): continue\n",
            "        images = []\n",
            "        for ext in ['*.jpg', '*.png', '*.jpeg']: images.extend(glob.glob(os.path.join(img_dir, ext)))\n",
            "        for img_path in images:\n",
            "            try:\n",
            "                blur = compute_laplacian_blur(img_path)\n",
            "                if blur < blur_threshold: reason = 'blur'\n",
            "                else:\n",
            "                    bright = check_brightness(img_path)\n",
            "                    if bright < min_brightness or bright > max_brightness: reason = 'bright'\n",
            "                    else: reason = None\n",
            "                if reason:\n",
            "                    os.remove(img_path)\n",
            "                    lbl_path = os.path.join(lbl_dir, Path(img_path).stem + '.txt')\n",
            "                    if os.path.exists(lbl_path): os.remove(lbl_path)\n",
            "                    removed += 1\n",
            "                else: kept += 1\n",
            "            except: pass\n",
            "    print(f'Done {dataset_name}: Kept {kept}, Removed {removed}')\n"
        ]

# 5. Update Preprocess call (Cell 5)
for cell in nb['cells']:
    if 'preprocess_images' in ''.join(cell['source']) and 'def' not in ''.join(cell['source']):
        cell['source'] = [
            "preprocess_images('dataset1')\n",
            "preprocess_images('dataset2')\n",
            "preprocess_images('dataset3')\n"
        ]
    if 'def preprocess_images' in ''.join(cell['source']):
        cell['source'] = [
            "def preprocess_images(dataset_name, target_size=(640, 640)):\n",
            "    resized, converted = 0, 0\n",
            "    for split in ['train', 'valid', 'val', 'test']:\n",
            "        img_dir, _ = get_yolo_paths(dataset_name, split)\n",
            "        if not os.path.exists(img_dir): continue\n",
            "        images = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))\n",
            "        for img_path in images:\n",
            "            img = cv2.imread(img_path)\n",
            "            if img is None: continue\n",
            "            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR); converted += 1\n",
            "            if img.shape[:2] != target_size: img = cv2.resize(img, target_size); resized += 1\n",
            "            cv2.imwrite(img_path, img)\n",
            "    print(f'Done {dataset_name}: Resized {resized}, Converted {converted}')\n"
        ]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Applied Absolute Path Fix to Testing_Local.ipynb")
