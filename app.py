# app.py
#
# from flask import Flask, request, jsonify
#
# app = Flask(__name__)
#
#
# @app.route('/process_data', methods=['POST'])
# def process_data():
#     input_data = request.json['input_data']
#     # Process the input data (e.g., perform machine learning inference)
#     result = input_data * 2
#     return jsonify({'result': result})

#
# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, request, jsonify, redirect, url_for

# import pandas as pd#   for reading the csv dataset file,
# import numpy as np#    for functioning and operting in array ,
# import warnings#       for ignoring the Warning arrosed throughout,
# warnings.filterwarnings("ignore")#

# from sklearn.model_selection import train_test_split

# from sklearn import svm#                              for SVM Model implementation
# from sklearn.neighbors import KNeighborsClassifier#   for KNN Model implementation
# from sklearn.ensemble import RandomForestClassifier#  for RFC Model implementation
# from sklearn.ensemble import VotingClassifier#     for Voting Model implementation

# from sklearn.metrics import *#  for metrics score calculation (accuracy, F1 Score)

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import make_column_transformer
# from sklearn.pipeline import make_pipeline
# import pickle

# from datetime import date,datetime#          for Date time based Report Generation
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import *

# 2. App 
app = Flask(__name__)

feature = [
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose',
    'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region',
    'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
    'hip_joint_pain',
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
    'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
    'irritability',
    'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
    'dischromic _patches',
    'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections',
    'coma',
    'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
    'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze'
]


@app.route('/get', methods=['GET'])
def app2():
    d = {}
    index = int(request.args["id"])
    if (index > 94):
        index = 94
    d['data'] = str(feature[index].replace("_", " ").upper())
    return d


symp = {}
for i in sorted(feature):
    j = str(i).replace("_", " ")
    symp[i] = str(i).replace("_", " ").upper()

disease = [
    'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 'AIDS',
    'Diabetes',
    'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine', 'Cervical spondylosis',
    'Paralysis (brain hemorrhage)',
    'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C',
    'Hepatitis D',
    'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
    'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis',
    '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo'
]

l2 = [0] * len(feature)

Training_Data = pd.read_csv("DISEASE_TRAIN.csv")
Testing_Data = pd.read_csv("DISEASE_TEST.csv")

for i in [Testing_Data, Training_Data]:
    i.replace(
        {
            'prognosis': {
                'Fungal infection': 0, 'Allergy': 1, 'GERD': 2,
                'Chronic cholestasis': 3, 'Drug Reaction': 4,
                'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7,
                'Gastroenteritis': 8, 'Bronchial Asthma': 9,
                'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12,
                'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14,
                'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18,
                'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21,
                'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
                'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27,
                'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32,
                'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37,
                'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40
            }
        }, inplace=True
    )

X_train = Training_Data[feature]
y_train = Training_Data[["prognosis"]]
X_test = Testing_Data[feature]
y_test = Testing_Data[["prognosis"]]
y_train = np.ravel(y_train)
x_train = X_train

for i in X_train:
    if i not in feature:
        print(i)

clf0 = svm.SVC(
    kernel='linear',
    probability=True
)
clf1 = KNeighborsClassifier(
    n_neighbors=4
)
clf2 = RandomForestClassifier(
    n_estimators=100
)
clf3 = VotingClassifier(
    estimators=[("SVM", clf0), ("KNN", clf1), ("RandomF", clf2)],
    voting='soft'
)
clf3 = clf3.fit(x_train, y_train)


@app.route('/apps', methods=['GET'])
def app_works():
    d1 = {}
    input_data_initial = request.args['input']
    # input_data = input_data_initial.replace(" ","_").replace("%20","_").lower()
    # input_data = [word.strip().replace(" ", "_").lower() for word in input_data_initial]
    # input_data = [word.strip().replace("%20", "_").lower() for word in input_data_initial]
    list_input = []
    list_input = input_data_initial.split(',')
    for i in list_input:
        i.replace(" ", "_")
        i.replace("%20", "_")
        i.lower()
        print(i)

    l2 = [0] * len(feature)
    for i in range(0, len(feature)):
        for j in list_input:
            if j.lower() == feature[i]:
                l2[i] = 1

    input_user = [l2]
    predict_proba = clf3.predict_proba(input_user)[0]
    top5_diseases_default = ["Drug Reaction"] * 5

    top5_indices = np.argsort(predict_proba)[::-1][:10]

    top5_diseases = [disease[i] for i in top5_indices]
    # if (list_input==[""]*17):
    #     top5_diseases = top5_diseases_default
    out = f"{top5_diseases[0]}"
    for i in range(1, 6):
        out += f",{top5_diseases[i]}"
    d1['output'] = f"{out}"
    return d1
    # return {'result': str(top5_diseases[0])}


@app.route('/process_data', methods=['GET'])
def process_data():
    input_data = request.json['input_data']
    # Process the input data (e.g., perform machine learning inference)
    result = input_data * 2
    return jsonify({'result': result})


@app.route("/next/<name_p>/with", methods=['GET'])
def b(name_p):
    d2 = {}
    inp = request.args['s']
    d2['output'] = f"Welcome {name_p}!! ML Model.{inp}"
    return d2['output']


# @app.route("/login", methods=['POST','GET'])
# def login():
#     if request.method == 'POST':
#         user = request.form['name_python']
#         return redirect(url_for("next/",name_p = user ))
#     else:
#         user = request.args.get('name_python')
#         return redirect(url_for("next/",name_p = user))


# @app.route('/run/define')
# def library_features_xy():
#     import pandas as pd  # for reading the csv dataset file,
#     import numpy as np  # for functioning and operting in array ,
#     import warnings  # for ignoring the Warning arrosed throughout,
#     warnings.filterwarnings("ignore")  #

#     from sklearn.model_selection import train_test_split
#     from sklearn import svm  # for SVM Model implementation
#     from sklearn.neighbors import KNeighborsClassifier  # for KNN Model implementation
#     from sklearn.ensemble import RandomForestClassifier  # for RFC Model implementation
#     from sklearn.ensemble import VotingClassifier  # for Voting Model implementation

#     import sklearn.metrics
#     from sklearn.preprocessing import OneHotEncoder
#     from sklearn.compose import make_column_transformer
#     from sklearn.pipeline import make_pipeline
#     import pickle
#     return ''


@app.route('/api', methods=['GET'])
def returnascii():
    d = {}
    inputchr = str(request.args['query'])
    d['output'] = f"Hello {inputchr}"
    return d

# from flask import Flask, request, jsonify

# app = Flask(__name__)


# @app.route("/api", methods=["GET"])
# def call():
#     d = {}
#     input1 = str(request.args['input'])
#     outpu = str(int(input1) * 150)
#     d['output'] = outpu
#     return d


# if __name__ == '__main__':
#     app.run()
