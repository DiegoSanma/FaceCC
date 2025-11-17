import os, json, random
from PIL import Image
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Cargar labels.json
id = 0
with open("Embeddings/labels.json", "r") as f:
    labels = json.load(f)

# Cargar embeddings y etiquetas asociadas
base_embeddings = {}

for file_name, person in labels.items():
    path = os.path.join("Embeddings", file_name)
    emb = np.load(path)
    person_name = file_name.split(".")[0]

    if person_name not in base_embeddings:
        base_embeddings[person_name] = []

    print(f"Procesando {file_name}, asignado a {person_name}")
    base_embeddings[person_name].append({"embedding": emb, "id": id})
    id += 1


dcc_embeddings = {}

for file in os.listdir("CaraDCC"):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join("CaraDCC", file)

        emb = DeepFace.represent(
            img_path=path,
            model_name='ArcFace',
            enforce_detection=False
        )[0]["embedding"]

        # Obtengo el nombre del jpg
        person_name = file.split(".")[0]
        print(f"Procesando {file}, asignado a {person_name}")

        if person_name not in dcc_embeddings:
            dcc_embeddings[person_name] = []

        dcc_embeddings[person_name].append({"embedding": emb, "id": id, "img": file})
        id += 1

gt_db_path = "gt_db"
split_ratio = 0.1  # Solo una imagen en la base de datos

def get_arcface_embedding(image_path):
    emb = DeepFace.represent(
        img_path=image_path,
        model_name="ArcFace",
        enforce_detection=False
    )
    return np.array(emb[0]["embedding"], dtype=np.float32)


for person_folder in sorted(os.listdir(gt_db_path)):
    person_path = os.path.join(gt_db_path, person_folder)
    print(f"Procesando {person_folder}...")

    if not os.path.isdir(person_path):
        continue

    # Obtener todas las imágenes de una persona
    images = [
        os.path.join(person_path, f)
        for f in os.listdir(person_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) == 0:
        continue

    random.shuffle(images)

    cutoff = max(1, int(len(images) * split_ratio))
    base_imgs = images[:cutoff]
    dcc_imgs = images[cutoff:]

    for img in base_imgs:
        add_or_not = random.random()
        if add_or_not < 0.5:
            break
        #print(f"  Agregando a base: {person_folder} - {os.path.basename(img)}")
        emb = get_arcface_embedding(img)
        if person_folder not in base_embeddings:
            base_embeddings[person_folder] = []

        base_embeddings[person_folder].append({"embedding": emb, "id": id})
        id += 1

    for img in dcc_imgs:
        emb = get_arcface_embedding(img)
        #print(type(emb))
        if person_folder not in dcc_embeddings:
            dcc_embeddings[person_folder] = []

        dcc_embeddings[person_folder].append({"embedding": emb, "id": id})
        id += 1

pairs = []
labels_pairs = []

for person, emb_list in dcc_embeddings.items():
    for emb_data in emb_list:
        emb = emb_data["embedding"]
        # Par positivo y negativo por persona
        if person in base_embeddings:
            for base_emb_data in base_embeddings[person]:
                #print(f"Generando par positivo para {person}")
                base_emb = base_emb_data["embedding"]
                pairs.append(np.concatenate([emb,base_emb]))
                #1 -> match
                labels_pairs.append('match')
                # Par negativo
        for _ in range(3):
            negative_person = random.choice(list(base_embeddings.keys()))
            while negative_person == person:
                negative_person = random.choice(list(base_embeddings.keys()))
            neg_emb_data = random.choice(base_embeddings[negative_person])
            neg_emb = neg_emb_data["embedding"] 
            pairs.append(np.concatenate([emb,neg_emb]))
            #0 -> no match
            labels_pairs.append('no match')


X = np.array(pairs)
y = np.array([1 if label == 'match' else 0 for label in labels_pairs])

print(f"Número de pares generados: {len(X)}")

extra_layer = SVC(kernel="rbf", probability=True)   # prob=True si necesitas score
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=14)

extra_layer.fit(X_train, y_train)

def evaluate_model_memory(X_val, y_val, verbose=True,threshold=0.5):
    """
    Evalúa un modelo de verificación facial entrenado sobre embeddings.

    Args:
        model: clasificador entrenado (por ejemplo, LogisticRegression)
        X_val: np.array con pares de embeddings (diferencias o concatenaciones)
        y_val: np.array con etiquetas ("match" o "no match")
        verbose: bool, si True imprime métricas detalladas

    Returns:
        accuracy: precisión del modelo (float)
    """
    # Predicciones
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    correct = 0
    total = len(X_val)
    for i in range(len(X_val)):
        best_match = 'no match'
        prob = extra_layer.predict_proba([X_val[i]])[0][1]
        distancia = 1 - prob
        #print(distancia)
        if  distancia < threshold:
            best_match = 'match'
        if best_match == 'no match' and y_val[i] == 0:
            correct += 1
            tn += 1
        elif best_match != 'no match' and y_val[i] == 1:
            correct += 1
            tp += 1
        elif best_match == 'no match' and y_val[i] == 1:
            fn += 1
        elif best_match != 'no match' and y_val[i] == 0:
            fp += 1
        
        


    # Precisión
    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if verbose:
        print(f"🔹 Accuracy: {accuracy * 100:.2f}%")
        print(f"🔹 Precision: {precision * 100:.2f}%")
        print(f"🔹 Recall: {recall * 100:.2f}%")
        print(f"🔹 F1-Score: {f1_score * 100:.2f}%")
        print("\n🔹 Matriz de Confusión:")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")


    return accuracy, f1_score, tp, tn, fp, fn

#evaluate_model_memory(X, y, verbose=True, threshold=3.0)

def plot_scores(X,y):
    thresholds = np.linspace(0, 1.0, 100)
    accuracies = []
    best_accuracy = 0
    f1_scores = []
    best_f1 = 0
    best_threshold_acc = 0
    best_threshold_f1 = 0
    for threshold in thresholds:
        accuracy, f1_score, tp,tn,_,_ = evaluate_model_memory(X, y, verbose=False, threshold=threshold)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold_acc = threshold
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold_f1 = threshold
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, accuracies, label='Accuracy', color='blue')
    plt.plot(thresholds, f1_scores, label='F1-Score', color='orange')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Model Performance vs. Threshold')
    plt.legend()
    plt.grid()
    plt.show()
    return best_threshold_acc, best_accuracy, best_threshold_f1, best_f1

best_threshold_acc, best_accuracy, best_threshold_f1, best_f1 = plot_scores(X_val,y_val)

print(f"Mejor umbral para Accuracy: {best_threshold_acc} con Accuracy: {best_accuracy}")
print(f"Mejor umbral para F1-Score: {best_threshold_f1} con F1-Score: {best_f1}")

def plot_confusion_matrix(threshold):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    _, _, tp, tn, fp, fn = evaluate_model_memory(X_val, y_val, verbose=False, threshold=threshold)

    cm = np.array([[tp, fp],
                   [fn, tn]])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Match', 'Predicted No Match'], yticklabels=['Actual Match', 'Actual No Match'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(best_threshold_acc)
plot_confusion_matrix(best_threshold_f1)

# Guardar el modelo entrenado
model_filename = "finetuned_facerecognition_classifier.joblib"
joblib.dump(extra_layer, model_filename)





