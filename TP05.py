

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine, euclidean

os.makedirs("results", exist_ok=True)

class FaceRecognitionDL:
    def __init__(self):
        
        print("Chargement des modèles (cela peut prendre quelques secondes)...")
        self.detector = MTCNN()
        self.embedder = FaceNet()
        
        
        self.database = {}
        self.labels = []
        self.embeddings = []

    def detect_face(self, image):
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        results = self.detector.detect_faces(img_rgb)
        
        if len(results) == 0:
            return None
            
        
        x, y, w, h = results[0]['box']
        
        
        x, y = abs(x), abs(y)
        
        
        face = img_rgb[y:y+h, x:x+w]
        
        
        face_resized = cv2.resize(face, (160, 160))
        
        return face_resized

    def extract_embedding(self, face):
        
        face_array = np.expand_dims(face, axis=0)
        
        
        embeddings = self.embedder.embeddings(face_array)
        
        return embeddings[0]

    def build_database(self, dataset_path):
        
        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_folder):
                continue
                
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                    
                # 1. Détecter le visage
                face = self.detect_face(image)
                if face is None:
                    print(f"Aucun visage trouvé dans {img_path}")
                    continue
                    
                
                emb = self.extract_embedding(face)
                
                
                self.embeddings.append(emb)
                self.labels.append(person_name)
                
        print(f"Base de données construite avec {len(self.embeddings)} visages.")

    def cosine_similarity(self, emb1, emb2):
    
        return 1 - cosine(emb1, emb2)

    def euclidean_distance(self, emb1, emb2):
        
        return euclidean(emb1, emb2)

    def recognize(self, image_path, threshold=0.8, method="euclidean"):
        
        image = cv2.imread(image_path)
        if image is None:
            print("Erreur : Impossible de lire l'image test.")
            return "Erreur", 0, "Erreur"

        
        face = self.detect_face(image)
        if face is None:
            return "Inconnu", 0, "No Match (Pas de visage)"

        emb_test = self.extract_embedding(face)

        
        best_distance = float('inf')
        best_similarity = -1
        best_label = "Inconnu"

        for i, emb_db in enumerate(self.embeddings):
            if method == "euclidean":
                dist = self.euclidean_distance(emb_test, emb_db)
                if dist < best_distance:
                    best_distance = dist
                    best_label = self.labels[i]
            else: 
                sim = self.cosine_similarity(emb_test, emb_db)
                if sim > best_similarity:
                    best_similarity = sim
                    best_label = self.labels[i]

        # 3. Décision
        if method == "euclidean":
            decision = "Match" if best_distance <= threshold else "No Match"
            valeur_retour = best_distance
        else:
            # Pour cosinus, le seuil typique est ~0.5. Plus c'est proche de 1, mieux c'est.
            decision = "Match" if best_similarity >= threshold else "No Match"
            valeur_retour = best_similarity

        if decision == "No Match":
            best_label = "Inconnu"

        return best_label, valeur_retour, decision


# ==========================================
# PROGRAMME PRINCIPAL ET EXPERIMENTATIONS
# ==========================================
def main():
    dataset = "dataset/"
    test_image = "test1.jpg"

    if not os.path.exists(dataset):
        print(f"Veuillez créer un dossier '{dataset}' avec des sous-dossiers par personne.")
        return
    if not os.path.exists(test_image):
        print(f"Veuillez ajouter une image '{test_image}' dans le même dossier que le script.")
        return

    
    model = FaceRecognitionDL()
    
    print("\n--- Construction de la base d'embeddings ---")
    model.build_database(dataset)
    
    print("\n--- Reconnaissance en cours (Méthode Euclidienne par défaut) ---")
    label, distance, decision = model.recognize(test_image, threshold=0.8, method="euclidean")
    
    print("\nRésultat :")
    print(f"Identité : {label}")
    print(f"Distance : {distance:.4f}")
    print(f"Décision : {decision}")

    # ==========================================
    # 5. EXPÉRIMENTATIONS OBLIGATOIRES
    # ==========================================
    print("\n" + "="*40)
    print("EXPÉRIMENTATIONS OBLIGATOIRES")
    print("="*40)

    
    print("\nA. Comparaison des distances :")
    _, dist_euc, dec_euc = model.recognize(test_image, threshold=0.8, method="euclidean")
    _, sim_cos, dec_cos = model.recognize(test_image, threshold=0.5, method="cosine")
    
    print(f"Méthode Euclidienne | Distance: {dist_euc:.4f} | Décision: {dec_euc}")
    print(f"Méthode Cosinus     | Similarité: {sim_cos:.4f} | Décision: {dec_cos}")

    
    print("\nB. Étude de l'effet du seuil (Distance Euclidienne) :")
    seuils = [0.4, 0.6, 0.8, 1.0]
    print(f"Distance mesurée : {dist_euc:.4f}")
    for s in seuils:
        dec = "Match" if dist_euc <= s else "No Match"
        print(f"Seuil: {s} -> {dec}")

if __name__ == "__main__":
    main()