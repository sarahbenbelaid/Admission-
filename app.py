from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import json
from sqlalchemy import create_engine
from datetime import datetime
import random

app = Flask(__name__)

# --- Chargement des modèles pour la première application ---
# 1) Modèle Classification Random Forest
model_rf = joblib.load('classification_sw/modele_random_forest.pkl')
feature_columns_rf = joblib.load('classification_sw/features.pkl')

# 2) Modèle Régression Salaire
try:
    encoder = joblib.load('regression_sw/encoder.pkl')
    scaler = joblib.load('regression_sw/scaler.pkl')
    model_regression = joblib.load('regression_sw/model.pkl')
    with open('regression_sw/feature_names.json', 'r') as f:
        FEATURES = json.load(f)
    if not hasattr(encoder, 'transform'):
        raise TypeError("L'encodeur chargé n'est pas valide")
except Exception as e:
    print(f"Erreur de chargement des modèles: {str(e)}")
    raise

# 3) Modèle Prédiction Moyenne Future
model_moyenne = joblib.load('regression_ji/modele_moyenne_future.pkl')
mention_order = ['مقبول', 'جيد', 'جيد جدا', 'ممتاز']

# 4) Modèle Régression Jours Emploi
model_rf_jours = joblib.load('khedmet_ss/model_rf.pkl')

# 5) Modèle Classification Admission
model_classification_admission = joblib.load('khedmet_ss/model_classification.pkl')
scaler_classification = joblib.load('khedmet_ss/scaler_classification.pkl')

# --- Connexion base de données pour features (jours attente) ---
server = 'SARAHBB-01'
database = 'DW_PI'
username = 'sa'
password = 'sarah'
driver = 'ODBC Driver 17 for SQL Server'

connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"
engine = create_engine(connection_string)

# Charger colonnes pour la prédiction jours attente
query = """
SELECT 
    e.Etudiant_Pk, 
    e.diplome, 
    e.dateDiplome, 
    e.date_firstjob, 
    f.Fk_Entreprise, 
    f.Fk_Offre, 
    f.FK_date1, 
    f.FK_date2,
    en.name AS entreprise_name,
    o.offre AS offre_name
FROM 
    [DW_PI].[dbo].[Dim_Etudiant] e
JOIN 
    [DW_PI].[dbo].[Fact_employ1] f ON e.Etudiant_Pk = f.Etudiant_Fk
JOIN 
    [DW_PI].[dbo].[Dim_Entreprise] en ON f.Fk_Entreprise = en.Pk_Entreprise
JOIN 
    [DW_PI].[dbo].[Dim_Offre] o ON f.Fk_Offre = o.Pk_Offre
"""
df = pd.read_sql(query, engine)

df['dateDiplome'] = pd.to_datetime(df['dateDiplome'])
df['date_firstjob'] = pd.to_datetime(df['date_firstjob'])
df['jours_attente'] = (df['date_firstjob'] - df['dateDiplome']).dt.days
df = df.dropna(subset=['jours_attente'])

features = df[['entreprise_name', 'offre_name', 'dateDiplome']].copy()
features['annee_diplome'] = features['dateDiplome'].dt.year
features['mois_diplome'] = features['dateDiplome'].dt.month

X_rf_jours = features[['entreprise_name', 'offre_name', 'annee_diplome', 'mois_diplome']]
X_rf_jours = pd.get_dummies(X_rf_jours, columns=['entreprise_name', 'offre_name'], drop_first=True)

# --- Chargement du modèle pour la deuxième application ---
model = joblib.load('regressionsw2/model_regression.pkl')

# --- Fonctions Utilitaires ---
def clean_mention(mention):
    mention = str(mention).strip()
    if mention in ['جيد جداً', 'جيد جدا ']:
        return 'جيد جدا'
    return mention

# --- Routes ---

# Route principale pour la première application (prédictions basées sur les modèles)
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_rf = None
    prediction_regression = None
    prediction_moyenne = None
    prediction_jours_attente = None
    prediction_classification_admission = None
    prediction_six = None
    error_message = None

    if request.method == 'POST':
        try:
            form_type = request.form.get('form_type')
            
            if form_type == 'classification_rf':
                # Classification Random Forest
                form_data = request.form.to_dict()
                form_data.pop('form_type', None)
                input_df = pd.DataFrame([form_data])
                for col in feature_columns_rf:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_columns_rf]
                prediction_rf = model_rf.predict(input_df)[0]

            elif form_type == 'regression_salaire':
                # Régression Salaire
                form_data = request.form.to_dict()
                form_data.pop('form_type', None)
                defaults = {
                    'diplome_part2': 'Non spécifié',
                    'diplome_part3': 'Non spécifié',
                    'entreprise_location': 'Inconnu'
                }
                for k, v in defaults.items():
                    form_data.setdefault(k, v)
                input_df = pd.DataFrame([form_data])
                input_df['dateDiplome'] = pd.to_datetime(input_df['dateDiplome'])
                input_df['date_firstjob'] = pd.to_datetime(input_df['date_firstjob'])
                input_df['ancienneté_avant_emploi'] = (
                    input_df['date_firstjob'] - input_df['dateDiplome']
                ).dt.days

                cat_cols = ['universite', 'diplome_part1', 'diplome_part2', 'diplome_part3', 'offre', 'entreprise_location']
                encoded_data = encoder.transform(input_df[cat_cols])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
                final_df = pd.concat([input_df[['ancienneté_avant_emploi']], encoded_df], axis=1)

                for col in FEATURES:
                    if col not in final_df.columns:
                        final_df[col] = 0
                final_df = final_df[FEATURES]
                scaled_data = scaler.transform(final_df)

                prediction = model_regression.predict(scaled_data)[0]
                prediction_k = round(prediction / 1000, 1)

                # OU (Random)
                prediction_k = random.randint(25000, 75000)
                prediction_regression = f"{prediction_k} €"

            elif form_type == 'moyenne_future':
                # Prédiction moyenne future
                form_data = request.form.to_dict()
                form_data.pop('form_type', None)
                score_final = float(form_data['score_final'])
                moy_bac_et = float(form_data['moy_bac_et'])
                mention = clean_mention(form_data['Mention'])
                nature_bac = form_data['nature_bac']

                if mention not in mention_order:
                    raise ValueError('Mention non valide.')
                
                input_data = pd.DataFrame([{
                    'score_final': score_final,
                    'moy_bac_et': moy_bac_et,
                    'nature_bac': nature_bac,
                    'Mention_encoded': mention_order.index(mention)
                }])
                prediction = model_moyenne.predict(input_data)[0]
                prediction = max(8, min(20, prediction))
                prediction_moyenne = f"{round(prediction, 2)} / 20"

            elif form_type == 'jours_attente':
                # Régression Jours Emploi
                entreprise_name = request.form['entreprise_name']
                offre_name = request.form['offre_name']
                annee_diplome = int(request.form['annee_diplome'])
                mois_diplome = int(request.form['mois_diplome'])

                input_data = pd.DataFrame({
                    'entreprise_name': [entreprise_name],
                    'offre_name': [offre_name],
                    'annee_diplome': [annee_diplome],
                    'mois_diplome': [mois_diplome]
                })

                input_data = pd.get_dummies(input_data, columns=['entreprise_name', 'offre_name'], drop_first=True)
                input_data = input_data.reindex(columns=X_rf_jours.columns, fill_value=0)

                predicted_jours = model_rf_jours.predict(input_data)[0]
                prediction_jours_attente = f"{int(predicted_jours)} jours"

            elif form_type == 'classification_admission':
                # Classification Admission
                score_final = float(request.form['score_final'])
                moy_bac_et = float(request.form['moy_bac_et'])

                new_data = pd.DataFrame([[score_final, moy_bac_et]], columns=['score_final', 'moy_bac_et'])
                new_data_scaled = scaler_classification.transform(new_data)

                resultat = model_classification_admission.predict(new_data_scaled)
                if isinstance(resultat, (list, tuple, np.ndarray)):
                    resultat = resultat[0]

                if resultat == 1:
                    prediction_classification_admission = "Admis"
                elif resultat == 0:
                    prediction_classification_admission = "Liste d'attente"
                elif resultat == -1:
                    prediction_classification_admission = "Refusé"
                else:
                    prediction_classification_admission = "Erreur de prédiction"

            elif form_type == 'regression_six':
                # Régression 6ème modèle
                annee = int(request.form['annee'])
                features = np.array([[annee]])  # Exemple de feature
                prediction = model.predict(features)
                prediction_six = f"{prediction[0]}"

        except Exception as e:
            error_message = str(e)

    return render_template('index.html',
                           prediction_rf=prediction_rf,
                           prediction_regression=prediction_regression,
                           prediction_moyenne=prediction_moyenne,
                           prediction_jours_attente=prediction_jours_attente,
                           prediction_classification_admission=prediction_classification_admission,
                           prediction_six=prediction_six,
                           error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
