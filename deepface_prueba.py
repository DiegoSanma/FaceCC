from deepface import DeepFace
import os
import glob
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def evaluar_con_deepface(galeria_path, entradas_path):
    # Busca todas las imágenes en la carpeta de fotos de entrada
    imagenes_de_entrada = glob.glob(os.path.join(entradas_path, '**', '*'), recursive=True)
    imagenes_de_entrada = [f for f in imagenes_de_entrada if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    correct_predictions = 0
    total_entries = len(imagenes_de_entrada)

    print(f"--- Iniciando evaluación de {total_entries} imágenes con DeepFace ---")

    for entry_image_path in imagenes_de_entrada:
        # DeepFace: encuentra la mejor coincidencia en la galería
        # model_name='Facenet' usa el mismo modelo que nuestro script anterior
        # dfs es una lista de DataFrames de pandas
        dfs = DeepFace.find(img_path=entry_image_path, 
                            db_path=galeria_path, 
                            #model_name='Facenet',
                            #model_name='VGG-Face',
                            model_name='ArcFace',
                            enforce_detection=False, # No falla si la cara no es perfecta
                            silent=True) # Evita que DeepFace imprima su propio log

        #La etiqueta real se extrae del nombre del archivo de entrada
        true_label = os.path.basename(entry_image_path).split('.')[0]

        # DeepFace devuelve una lista de DataFrames. Tomamos el primer resultado.
        if dfs and not dfs[0].empty:
            # El resultado es la ruta completa a la imagen, de ahí extraemos la etiqueta
            df = dfs[0]
            predicted_path = df['identity'].iloc[0]
            predicted_label = os.path.basename(predicted_path).split('.')[0]
            distance = df['distance'].iloc[0]

            is_correct = (predicted_label == true_label)
                
            print(f"  - Entrada: '{os.path.basename(entry_image_path)}' (Real: '{true_label}') -> Predicción: '{predicted_label}' (Dist: {distance:.4f}) -> {'CORRECTO' if is_correct else 'INCORRECTO'}")
                
            if is_correct:
                correct_predictions += 1
        else:
            print(f"  - Entrada: '{os.path.basename(entry_image_path)}' (Real: '{true_label}') -> Predicción: No se encontró coincidencia.")


    # Calcula y muestra el resultado final
    if total_entries > 0:
        accuracy = (correct_predictions / total_entries) * 100
        print(f"\n--- Resultado Final ---")
        print(f"Precisión (Accuracy): {accuracy:.2f}% ({correct_predictions} de {total_entries} identificaciones correctas)")
    else:
        print("\nNo se procesó ninguna cara.")

# ejecucion
if __name__ == '__main__':
    PATH_GALERIA = 'CaraTui'
    PATH_ENTRADAS = 'CaraDCC'

    evaluar_con_deepface(PATH_GALERIA, PATH_ENTRADAS)