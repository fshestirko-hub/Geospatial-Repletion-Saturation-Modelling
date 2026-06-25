import json
from pathlib import Path

notebook_path = Path(r"c:\Users\fedka\Desktop\Geospatial Repletion & Saturation Modelling - Copy\notebooks\1_part_master_notebook.ipynb")

if not notebook_path.exists():
    print(f"Error: Notebook not found at {notebook_path}")
    exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

updated = False
for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        
        # Find all matching files in this cell
        found_filenames = []
        existing_displays = set()
        
        for line in source:
            if "if not (PLOTS_DIR /" in line and ".exists():" in line:
                try:
                    parts = line.split("PLOTS_DIR /")
                    if len(parts) > 1:
                        fn = parts[1].split(")")[0].strip().replace('"', '').replace("'", "")
                        found_filenames.append(fn)
                except Exception:
                    pass
            elif "IPython.display" in line or "display(Image" in line:
                # check if there's already a display for a file
                for fn in found_filenames:
                    if fn in line:
                        existing_displays.add(fn)
        
        to_add = [fn for fn in found_filenames if fn not in existing_displays]
        
        if to_add:
            print(f"Adding displays for: {to_add}")
            if len(source) > 0 and not source[-1].endswith("\n"):
                source[-1] = source[-1] + "\n"
            
            display_lines = ["\n", "# Display inline in IPython/Jupyter notebook environments\n"]
            if not any("from IPython.display import" in line for line in source):
                display_lines.append("from IPython.display import display, Image\n")
            
            for fn in to_add:
                display_lines.append(f"display(Image(filename=str(PLOTS_DIR / \"{fn}\")))\n")
                
            source.extend(display_lines)
            cell["source"] = source
            updated = True

if updated:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("No cells needed updating.")
