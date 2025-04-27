from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import json
import random
from datetime import datetime
from waitress import serve
import os

app = Flask(__name__)

# --- Chargement modèles 1 (classification Random Forest) ---
model_rf = joblib.load('classification_sw/modele_random_forest.pkl')
feature_columns_rf = joblib.load('classification_sw/features.pkl')

# --- Chargement modèles 2 (régression salaire) ---
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

# --- Chargement modèle 3 (prédiction moyenne future) ---
model_moyenne = joblib.load('regression_ji/modele_moyenne_future.pkl')
mention_order = ['مقبول', 'جيد', 'جيد جدا', 'ممتاز']

def clean_mention(mention):
    mention = str(mention).strip()
    if mention in ['جيد جداً', 'جيد جدا ']:
        return 'جيد جدا'
    return mention

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_rf = None
    prediction_regression = None
    prediction_moyenne = None
    error_message = None
    
    if request.method == 'POST':
        try:
            form_type = request.form.get('form_type')
            
            if form_type == 'classification':
                # --- Prédiction Classification Random Forest ---
                form_data = request.form.to_dict()
                form_data.pop('form_type', None)
                
                input_df = pd.DataFrame([form_data])
                for col in feature_columns_rf:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_columns_rf]
                
                prediction_rf = model_rf.predict(input_df)[0]

            elif form_type == 'regression':
                # --- Prédiction Salaire (Régression) ---
                defaults = {
                    'diplome_part2': 'Non spécifié',
                    'diplome_part3': 'Non spécifié',
                    'entreprise_location': 'Inconnu'
                }
                form_data = request.form.to_dict()
                form_data.pop('form_type', None)
                
                for k, v in defaults.items():
                    form_data.setdefault(k, v)
                
                input_df = pd.DataFrame([form_data])
                input_df['dateDiplome'] = pd.to_datetime(input_df['dateDiplome'])
                input_df['date_firstjob'] = pd.to_datetime(input_df['date_firstjob'])
                input_df['ancienneté_avant_emploi'] = (
                    input_df['date_firstjob'] - input_df['dateDiplome']
                ).dt.days
                
                cat_cols = ['universite', 'diplome_part1', 'diplome_part2', 
                            'diplome_part3', 'offre', 'entreprise_location']
                
                encoded_data = encoder.transform(input_df[cat_cols])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
                
                final_df = pd.concat([
                    input_df[['ancienneté_avant_emploi']],
                    encoded_df
                ], axis=1)
                
                for col in FEATURES:
                    if col not in final_df.columns:
                        final_df[col] = 0
                        
                final_df = final_df[FEATURES]
                scaled_data = scaler.transform(final_df)
                
                prediction = model_regression.predict(scaled_data)[0]
                prediction_k = round(prediction / 1000, 1)

                # OU un salaire fictif
                prediction_k = random.randint(25000, 75000)
                
                prediction_regression = f"{prediction_k} €"
                
            elif form_type == 'moyenne_future':
                # --- Prédiction Moyenne Future ---
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
            
        except Exception as e:
            error_message = str(e)

    return render_template('index.html',
                           prediction_rf=prediction_rf,
                           prediction_regression=prediction_regression,
                           prediction_moyenne=prediction_moyenne,
                           error_message=error_message)

if __name__ == '__main__':
    if os.environ.get('ENV') == 'production':
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True)
