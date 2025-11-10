from ia01.utils import lecture_csv, quantile

def clean_data(data, median_age):
    i = 0
    L = data.copy()
    s = 0

    for d in L:
        if data[i]["age"] == "0":
            data[i]["age"] = median_age
        
        if d["pays"] == "":
            data[i]["pays"] = "Italie"
        
        if d["sexe"] == "":
            if s == 0:
                data[i]["sexe"] = "Homme"
                s = 1
            else:
                data[i]["sexe"] = "Femme"
                s = 0
        
        if d["sommeil_qualite"] == 'Mauvaise': data[i]["sommeil_qualite"] = 0
        elif d["sommeil_qualite"] == "Passable": data[i]["sommeil_qualite"] = 1
        elif d["sommeil_qualite"] == "Bonne": data[i]["sommeil_qualite"] = 2
        elif d["sommeil_qualite"] == "Excellente": data[i]["sommeil_qualite"] = 3
        
        if d["niveau_stress"] == 'Faible': data[i]["niveau_stress"] = 0
        elif d["niveau_stress"] == "Moyen": data[i]["niveau_stress"] = 1
        elif d["niveau_stress"] == "Haut": data[i]["niveau_stress"] = 2
        
        data[i]["sexe"] = 0 if d["sexe"] == 'Homme' else 1
        data[i]["cigarette"] = 0 if d["cigarette"] == 'Non' else 1
        data[i]["alcool"] = 0 if d["alcool"] == 'Non' else 1
        
        del(d["cafe_verre"])
        del(d["profession"])
        del(d["pays"])
        del(d["sommeil_duree"])
        
        for k in d.keys():
            float(d[k])
        
        i += 1
    
    return data

def convert_to_list_of_lists(data_dict):
    X = []
    for i in range(len(data_dict)):
        y = []
        for k in data_dict[i].keys():
            y.append(float(data_dict[i][k]))
        X.append(y)
    return X

def preprocess_train_data(x_path, y_path):
    data_X = lecture_csv(x_path)
    
    age = [x["age"] for x in data_X if x["age"] != "0"] 
    age = [float(a) for a in age]
    median_age = quantile(age, 0.5)
    
    data_X_clean = clean_data(data_X, median_age)
    X_train = convert_to_list_of_lists(data_X_clean)
    
    data_y = lecture_csv(y_path)
    y_train = []
    for i in range(len(data_y)):
        label = data_y[i]["probleme_sante"]
        if label == "Aucun": y_train.append(0)
        elif label == "Faible": y_train.append(1)
        else: y_train.append(2) # Fusion de "Modere" et "Severe"
        
    return X_train, y_train, median_age

def preprocess_test_data(x_path, median_age):
    data_X_test = lecture_csv(x_path)
    
    data_X_test_clean = clean_data(data_X_test, median_age)
    
    X_test = convert_to_list_of_lists(data_X_test_clean)
    
    return X_test

def format_predictions_for_submission(y_pred):
    y_test_formatted = []
    for p in y_pred:
        if p == 0:
            y_test_formatted.append("Aucun")
        elif p == 1:
            y_test_formatted.append("Faible")
        else:
            y_test_formatted.append("Modere")
    return y_test_formatted

def save_predictions_to_csv(predictions, path="results/y_test.csv"):
    with open(path, 'w') as f:
        f.write("probleme_sante\n")
        for p in predictions:
            f.write(f"{p}\n")
    print(f"Prédictions sauvegardées dans : {path}")