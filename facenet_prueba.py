import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import tarfile
from io import BytesIO

# Configura el dispositivo (GPU si está disponible, si no, CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Corriendo en el dispositivo: {device}')

# MTCNN -> Detecta caras. FaceNet -> Crear embedding
mtcnn = MTCNN(image_size=160, margin=10, keep_all=False, min_face_size=20, device=device)
facenet = InceptionResnetV1(pretrained='casia-webface', device=device).eval()


# Crear "huellas faciales" (?)
def create_gallery_from_tar(tar_path):
    print(f"\n--- Creando galería desde '{tar_path}' ---")
    known_embeddings = []
    known_labels = []

    with tarfile.open(tar_path, "r") as tar:
        # Filtra solo los archivos que son imágenes
        image_members = [m for m in tar.getmembers() if m.name.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for member in image_members:
            f = tar.extractfile(member)
            img = Image.open(BytesIO(f.read())).convert('RGB')

            # Detecta la cara en la imagen
            face_tensor = mtcnn(img)

            if face_tensor is not None:
                # Genera el embedding
                with torch.no_grad():
                    embedding = facenet(face_tensor.unsqueeze(0).to(device))
                    
                # Extrae la etiqueta del nombre del archivo (ej: 'andres_ordenez.jpg' -> 'andres_ordenez')
                filename = member.name.split('/')[-1]
                label = filename.split('.')[0]
                
                known_embeddings.append(embedding.cpu())
                known_labels.append(label)
                print(f"  - Galería: Procesada '{member.name}', Etiqueta: '{label}'")
            else:
                print(f"  - Galería: No se detectó rostro en '{member.name}'")

    
    if not known_embeddings:
        print("No se procesaron caras para la galería.")
        return None, None

    # Concatena la lista de tensores de embeddings en un solo tensor
    known_embeddings = torch.cat(known_embeddings)
    return known_embeddings, known_labels


# E
def evaluate_entries(entry_tar_path, gallery_embeddings, gallery_labels, threshold=1.0):
    if gallery_embeddings is None:
        print("La galería está vacía. No se puede evaluar.")
        return

    print(f"--- Evaluando entradas desde '{entry_tar_path}' ---")
    correct_predictions = 0
    total_entries = 0

    with tarfile.open(entry_tar_path, "r") as tar:
        image_members = [m for m in tar.getmembers() if m.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
        for member in image_members:
            f = tar.extractfile(member)
            img = Image.open(BytesIO(f.read())).convert('RGB')

            # Detecta la cara en la imagen de entrada
            entry_face_tensor = mtcnn(img)

            if entry_face_tensor is not None:
                total_entries += 1
                    
                # Genera el embedding de la cara de entrada
                with torch.no_grad():
                    entry_embedding = facenet(entry_face_tensor.unsqueeze(0).to(device))
                    
                # Extrae la etiqueta verdadera de la imagen de entrada
                true_label = member.name.split('/')[-1].split('.')[0]

                # Calcula las distancias a todas las caras de la galería
                distances = (entry_embedding.cpu() - gallery_embeddings).norm(dim=1)
                    
                # Encuentra la coincidencia más cercana
                min_dist_idx = distances.argmin()
                min_dist = distances[min_dist_idx].item()
                predicted_label = gallery_labels[min_dist_idx]

                # Compara y decide
                is_correct = (predicted_label == true_label) and (min_dist < threshold)
                
                print(f"  - Entrada: '{member.name}' (Real: '{true_label}') -> Predicción: '{predicted_label}' (Distancia: {min_dist:.4f}) -> {'CORRECTO' if is_correct else 'INCORRECTO'}")
                    
                if is_correct:
                    correct_predictions += 1
            else:
                print(f"  - Entrada: No se detectó rostro en '{member.name}'")


    # Calcula y muestra el resultado final
    if total_entries > 0:
        accuracy = (correct_predictions / total_entries) * 100
        print(f"\n--- Resultado Final ---")
        print(f"Precisión (Accuracy): {accuracy:.2f}% ({correct_predictions} de {total_entries} identificaciones correctas)")
    else:
        print("\nNo se procesó ninguna cara del archivo de entrada.")


# Ejecucion
if __name__ == '__main__':
    # Define las rutas a tus archivos .tar
    FOTOS_TUI = 'dataset_foto.tar'
    FOTOS_ENTRADA = 'dataset_entry.tar'

    # 1. Crea la galería de referencia a partir de las fotos de las tui
    galeria_embeddings, galeria_labels = create_gallery_from_tar(FOTOS_TUI)

    # 2. Evalúa las fotos de la entrada comparándolas con la galería
    evaluate_entries(FOTOS_ENTRADA, galeria_embeddings, galeria_labels)