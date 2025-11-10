from src.preprocess import (
    preprocess_train_data, 
    preprocess_test_data, 
    format_predictions_for_submission, 
    save_predictions_to_csv
)
from src.model_selection import find_best_depth_cv

from ia01.evaluation import partition_val_croisee
from ia01.arbre import arbre_train, arbre_pred

X_TRAIN_PATH = "data/X_train.csv"
Y_TRAIN_PATH = "data/y_train.csv"
X_TEST_PATH = "data/X_test.csv"
SUBMISSION_PATH = "results/y_test.csv" 

def main():
    
    print("Étape 1: Prétraitement des données d'entraînement...")
    X_train, y_train, median_age = preprocess_train_data(X_TRAIN_PATH, Y_TRAIN_PATH)
    print(f"Données d'entraînement chargées. Médiane d'âge calculée : {median_age}")

    print("\nÉtape 2: Sélection du meilleur hyperparamètre (profondeur)...")
    K = 5 
    X_K, y_K = partition_val_croisee(X_train, y_train, K)
    best_depth = find_best_depth_cv(X_K, y_K, K)
    print(f"Meilleure profondeur identifiée (basée sur F1-Macro) : {best_depth}")

    print("\nÉtape 3: Entraînement du modèle final sur toutes les données...")
    arbre_final = arbre_train(X_train, y_train)
    print("Modèle final entraîné.")

    print("\nÉtape 4: Prétraitement des données de test...")
    X_test = preprocess_test_data(X_TEST_PATH, median_age)
    print("Données de test nettoyées et transformées.")

    print("\nÉtape 5: Génération des prédictions sur l'ensemble de test...")
    y_test_pred_numeric = arbre_pred(X_test, arbre_final, max_prof=best_depth)
    print("Prédictions générées.")

    print("\nÉtape 6: Formatage et sauvegarde des résultats...")
    y_test_formatted = format_predictions_for_submission(y_test_pred_numeric)
    save_predictions_to_csv(y_test_formatted, SUBMISSION_PATH)
    


main()