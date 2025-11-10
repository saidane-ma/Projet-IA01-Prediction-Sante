from ia01.utils import *
from ia01.privacy import *
from ia01.kppv import *
from ia01.majoritaire import *
from ia01.evaluation import *
from ia01.arbre import *
from ia01.metriques import *


data = lecture_csv("data/X_train.csv")


age=[x["age"] for x in data]
imc=[x["IMC"] for x in data]
median_age = quantile(age,0.5)


X=[]
i=0
s=0
n=len(data)
L=data.copy()
for d in L:
    if data[i]["age"]=="0":
        data[i]["age"] = median_age
    if d["pays"] =="" :
        data[i]["pays"]="Italie"
    if d["sexe"]=="" :
        if s==0:
            data[i]["sexe"]="Homme"
            s=1
        else:
            data[i]["sexe"]="Femme"
            s=0
    if d["sommeil_qualite"] =='Mauvaise':
                data[i]["sommeil_qualite"]=0
    if d["sommeil_qualite"] == "Passable":
                data[i]["sommeil_qualite"]=1
    if d["sommeil_qualite"] =="Bonne":
                data[i]["sommeil_qualite"]=2
    if d["sommeil_qualite"]=="Excellente":
                data[i]["sommeil_qualite"]=3
    if d["niveau_stress"] =='Faible':
                data[i]["niveau_stress"]=0
    if d["niveau_stress"] == "Moyen":
                data[i]["niveau_stress"]=1
    if d["niveau_stress"] =="Haut":
                data[i]["niveau_stress"]=2
    if d["sexe"] =='Homme':
            data[i]["sexe"]=0
    if d["sexe"] == "Femme":
            data[i]["sexe"]=1
    if d["cigarette"] =='Non':
            data[i]["cigarette"]=0
    if d["cigarette"] == "Oui":
            data[i]["cigarette"]=1
    if d["alcool"] =='Non':
            data[i]["alcool"]=0
    if d["alcool"] == "Oui":
                data[i]["alcool"]=1
    del(d["cafe_verre"])
    del(d["profession"])
    del(d["pays"])
    del(d["sommeil_duree"])
    for k in d.keys():
        float(d[k])
    i+=1


y_train = lecture_csv("data/y_train.csv")

L=y_train.copy()
for i in range(len(L)):
    if L[i]["probleme_sante"] =="Aucun":
        y_train[i]=0
    elif L[i]["probleme_sante"] =="Faible":
        y_train[i]=1
    else :
        y_train[i]=2

X=[]
for i in range(len(data)):
    y=[]
    for k in data[i].keys():
        y.append(float(data[i][k]))
    X.append(y)
y=y_train.copy()
K = 5
X_K, y_K = partition_val_croisee(X, y, K)


prof = list(range(12)) + [float("inf")]
erreur_cv = [0] * len(prof)
prec_cv = [0] * len(prof)
rap_cv = [0] * len(prof)
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
        erreur_cv[j] += taux_erreur(y_val, y_pred_val) / K
        prec_cv[j] += (precision(y_val, y_pred_val, 1) + precision(y_val, y_pred_val, 2) + precision(y_val, y_pred_val, 0)) / (3*K)
        rap_cv[j] += (rappel(y_val, y_pred_val, 2) + rappel(y_val, y_pred_val, 1) + rappel(y_val, y_pred_val, 0)) / (3*K)
        f_cv[j] += (f_score(y_val, y_pred_val, 0) + f_score(y_val, y_pred_val, 2) + f_score(y_val, y_pred_val, 1))/ (3*K)

for j, p in enumerate(prof):
    print(
        f"Taux d'erreur pour prof={p} ; e={erreur_cv[j]} ; p={prec_cv[j]} ; r={rap_cv[j]} ; f={f_cv[j]}"
    )
bestp_e = argsort(erreur_cv)[0]
bestp_p = argsort(prec_cv, True)[0]
bestp_r = argsort(rap_cv, True)[0]
bestp_f = argsort(f_cv, True)[0]
print(f"Meilleur taux d'erreur : prof = {bestp_e}")
print(f"Meilleure précision : prof = {bestp_p}")
print(f"Meilleur rappel : prof = {bestp_r}")
print(f"Meilleur F1-score : prof = {bestp_f}")

arbre_final = arbre_train(X,y)

data = lecture_csv("data/X_test.csv")
L=[]
i=0
L=data.copy()
for d in L:
    if data[i]["age"]=="0":
        data[i]["age"] = median_age
    if d["pays"] =="" :
        data[i]["pays"]="Italie"
    if d["sexe"]=="" :
        if s==0:
            data[i]["sexe"]="Homme"
            s=1
        else:
            data[i]["sexe"]="Femme"
            s=0
    if d["sommeil_qualite"] =='Mauvaise':
                data[i]["sommeil_qualite"]=0
    if d["sommeil_qualite"] == "Passable":
                data[i]["sommeil_qualite"]=1
    if d["sommeil_qualite"] =="Bonne":
                data[i]["sommeil_qualite"]=2
    if d["sommeil_qualite"]=="Excellente":
                data[i]["sommeil_qualite"]=3
    if d["niveau_stress"] =='Faible':
                data[i]["niveau_stress"]=0
    if d["niveau_stress"] == "Moyen":
                data[i]["niveau_stress"]=1
    if d["niveau_stress"] =="Haut":
                data[i]["niveau_stress"]=2
    if d["sexe"] =='Homme':
            data[i]["sexe"]=0
    if d["sexe"] == "Femme":
            data[i]["sexe"]=1
    if d["cigarette"] =='Non':
            data[i]["cigarette"]=0
    if d["cigarette"] == "Oui":
            data[i]["cigarette"]=1
    if d["alcool"] =='Non':
            data[i]["alcool"]=0
    if d["alcool"] == "Oui":
                data[i]["alcool"]=1
    del(d["cafe_verre"])
    del(d["profession"])
    del(d["pays"])
    del(d["sommeil_duree"])
    for k in d.keys():
        float(d[k])
    i+=1


X_test=[]
for i in range(len(data)):
    y=[]
    for k in data[i].keys():
        y.append(float(data[i][k]))
    X_test.append(y)

y_test_pred = arbre_pred(X_test,arbre_final,max_prof=3)
y_test=[]
for p in y_test_pred:
    if p==0:
        y_test.append("Aucun")
    elif p==1:
        y_test.append("Faible")
    else:
        y_test.append("Modere")

ecriture_csv_projet(y_test)