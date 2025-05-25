import os
import numpy as np

TRACT_DIR = 'tract_div'
OUTPUT_DIR = 'tract_div_npy'
LINES_TO_SKIP = 8
VALUES_PER_ROW = 16

def ensure_output_dir(directory):
    os.makedirs(directory, exist_ok=True)

def extract_tract_data(file_path, lines_to_skip=LINES_TO_SKIP, values_per_row=VALUES_PER_ROW):
    with open(file_path, 'r') as f:
        lines = f.readlines()[lines_to_skip:]
    selected_values = []
    for idx, line in enumerate(lines):
        if idx % 2 == 1:
            floats = [float(x) for x in line.strip().split()[:values_per_row]]
            selected_values.extend(floats)
    rows = len(lines) // 2
    arr = np.array(selected_values).reshape((rows, values_per_row))
    return arr

def process_tract_folder(input_folder, output_folder):
    ensure_output_dir(output_folder)
    for fname in os.listdir(input_folder):
        if not fname.endswith('.tract'):
            continue
        src_path = os.path.join(input_folder, fname)
        arr = extract_tract_data(src_path)
        out_name = os.path.splitext(fname)[0] + '.npy'
        out_path = os.path.join(output_folder, out_name)
        np.save(out_path, arr)

if __name__ == '__main__':
    process_tract_folder(TRACT_DIR, OUTPUT_DIR)
