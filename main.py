import entrenamiento,evaluacion
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

if __name__ == "__main__":
    # Entrenamiento y guardado de embeddings
    facenet = InceptionResnetV1(pretrained='casia-webface').to(entrenamiento.device)
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=entrenamiento.device)

    #Guarda las fotos tipo TUI en un archivo .npz, para luego compararlas
    entrenamiento.save_embeddings("dataset_foto.tar")

    #Separación del dataset de fotos de entrada al DCC en entrenamiento y validación
    train_set, val_set = entrenamiento.split_dataset(entrenamiento.dataset, train_ratio=0.8)
    
    #Entreno el modelo utilizando el dataset de entrada al DCC (en vdd aún no se hace entrenamiento,
    #solo uso facenet para extraer embeddings)
    model = entrenamiento.train_model(train_set,facenet,mtcnn)

    #Evaluo usando el modelo entrenado y el dataset de validación
    evaluacion.evaluate_model(val_set,model,mtcnn)