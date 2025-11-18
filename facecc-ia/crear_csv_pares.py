import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

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
    return filename.split(".")[0].lower()

rows = []

for img_tui in tui_imgs:
    name_tui = get_name(img_tui)

    for img_dcc in dcc_imgs:
        name_dcc = get_name(img_dcc)

        # Positivo si el nombre coincide
        label = 1 if name_dcc == name_tui else 0

        rows.append([
            os.path.join(cara_tui, img_tui),
            os.path.join(cara_dcc, img_dcc),
            label
        ])

# Guardar CSV
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["img1", "img2", "label"])
    writer.writerows(rows)

print(f"CSV creado con {len(rows)} pares → {csv_file}")

df = pd.read_csv("pares_DCC_TUI.csv")

# Primer split: train + temp
train_df, temp_df = train_test_split(df, test_size=0.30, shuffle=True, random_state=42)

# Segundo split: temp → val + test
val_df, test_df = train_test_split(temp_df, test_size=0.50, shuffle=True, random_state=42)

# Guardar archivos
train_df.to_csv("train_pairs.csv", index=False)
val_df.to_csv("val_pairs.csv", index=False)
test_df.to_csv("test_pairs.csv", index=False)

print("Split listo:")
print(f"Train: {len(train_df)} filas")
print(f"Val:   {len(val_df)} filas")
print(f"Test:  {len(test_df)} filas")

