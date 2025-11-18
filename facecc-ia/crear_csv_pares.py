import os
import csv

# Directorios de imágenes
cara_dcc = "CaraDCC"
cara_tui = "CaraTui"

# Archivo de salida
csv_file = "pares_DCC_TUI.csv"

# Listar archivos
dcc_imgs = os.listdir(cara_dcc)
tui_imgs = os.listdir(cara_tui)

def get_name(filename):
    """Extrae el nombre antes del primer _ o ."""
    return filename.split("_")[0].split(".")[0].lower()

rows = []

for img_dcc in dcc_imgs:
    name_dcc = get_name(img_dcc)

    for img_tui in tui_imgs:
        name_tui = get_name(img_tui)

        # Positivo si el nombre coincide
        label = 1 if name_dcc == name_tui else 0

        rows.append([
            os.path.join(cara_dcc, img_dcc),
            os.path.join(cara_tui, img_tui),
            label
        ])

# Guardar CSV
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["img1", "img2", "label"])
    writer.writerows(rows)

print(f"CSV creado con {len(rows)} pares → {csv_file}")

