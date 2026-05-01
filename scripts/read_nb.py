import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

with open("notebooks/vae_fatal_representation.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and cell.get("outputs"):
        print(f"=== Cell {i} [{cell['id']}] ===")
        for out in cell["outputs"]:
            otype = out.get("output_type", "")
            if otype == "stream":
                print("".join(out.get("text", [])))
            elif otype in ("execute_result", "display_data"):
                data = out.get("data", {})
                txt = data.get("text/plain", "")
                print("".join(txt) if isinstance(txt, list) else txt)
        print()
