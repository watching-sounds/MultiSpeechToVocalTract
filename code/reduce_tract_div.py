import os

SOURCE_DIR = "tract_div"

def process_and_overwrite_tracts(directory):
    for entry in os.listdir(directory):
        if not entry.endswith(".tract"):
            continue
        input_fp = os.path.join(directory, entry)
        with open(input_fp, "r") as fin:
            content = fin.readlines()
        header_part = content[:7]
        param_lines = content[8:]

        output_fp = os.path.join(directory, entry.rsplit('.', 1)[0] + ".tract")
        with open(output_fp, "w") as fout:
            fout.writelines(header_part)
            fout.write("30\n")
            for idx in range(0, len(param_lines)-1, 2):
                if idx % 16 == 0:
                    fout.write(param_lines[idx])
                    fout.write(param_lines[idx+1])
        print(f"Processed: {entry} -> {os.path.basename(output_fp)}")

if __name__ == "__main__":
    process_and_overwrite_tracts(SOURCE_DIR)
