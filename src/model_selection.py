from ia01.arbre import arbre_train, arbre_pred
from ia01.metriques import taux_erreur, f_score
from ia01.utils import argsort

def find_best_depth_cv(X_K, y_K, K=5):
    
    prof = list(range(12)) + [float("inf")]
    
    erreur_cv = [0] * len(prof)
    f_cv = [0] * len(prof)
    
    for i in range(K):
        X_val, y_val = X_K[i], y_K[i]
        X_train, y_train = [], []
        for j in range(K):
            if j != i:
                X_train += X_K[j]
                y_train += y_K[j]

        arbre = arbre_train(X_train, y_train)
        
        for j, p in enumerate(prof):
            y_pred_val = arbre_pred(X_val, arbre, max_prof=p)
            
            f1_class_0 = f_score(y_val, y_pred_val, 0)
            f1_class_1 = f_score(y_val, y_pred_val, 1)
            f1_class_2 = f_score(y_val, y_pred_val, 2)
            macro_f1 = (f1_class_0 + f1_class_1 + f1_class_2) / 3
            
            f_cv[j] += macro_f1 / K
            erreur_cv[j] += taux_erreur(y_val, y_pred_val) / K

    print("--- Résultats de la Validation Croisée ---")
    for j, p in enumerate(prof):
        print(f"Profondeur={p} : Taux Erreur={erreur_cv[j]:.4f} | F1-Macro={f_cv[j]:.4f}")

    best_f1_index = argsort(erreur_cv, True)[0]
    best_depth = prof[best_f1_index]
    
    return best_depth