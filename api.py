from flask import Flask, request, jsonify, render_template
from flask_cors import cross_origin
import pickle
import nltk
from nltk.chat.util import Chat, reflections

app = Flask(__name__)
model = pickle.load(open("thyroid_model.pkl", "rb"))

# Updated and detailed cure recommendations
base_cures = {
    0: "Antithyroid medications (e.g., Methimazole, Propylthiouracil), beta-blockers for symptom control (e.g., Propranolol), and in some cases radioactive iodine therapy or thyroidectomy.",
    1: "Hormone replacement therapy with levothyroxine, monitoring TSH levels regularly to adjust the dosage as needed.",
    2: "No treatment necessary for euthyroid patients; continue with regular check-ups to monitor thyroid function.",
    3: "Supportive care addressing the underlying illness, and frequent monitoring of thyroid function to prevent complications."
}

# Function to generate detailed recommendations
def generate_recommendation(output, sick, goitre, tumor, pregnant, thyroid_surgery):
    base_cure = base_cures[output]
    additional_cures = []
    
    if sick:
        additional_cures.append("Close monitoring of overall health and addressing any underlying illnesses that could be affecting thyroid function.")
    if goitre:
        additional_cures.append("Evaluation by an endocrinologist for possible surgical intervention if the goitre is causing symptoms or cosmetic concerns.")
    if tumor:
        additional_cures.append("Consultation with an oncologist for assessment and management of the tumor, which may include surgery, radiation, or chemotherapy.")
    if pregnant:
        additional_cures.append("Regular prenatal care with adjustments to treatment as necessary to ensure both maternal and fetal health, considering the increased thyroid hormone requirements during pregnancy.")
    if thyroid_surgery:
        additional_cures.append("Postoperative monitoring of thyroid function and hormone replacement therapy if necessary, to maintain normal thyroid levels.")

    full_recommendation = f"Recommended Cure: {base_cure}"
    if additional_cures:
        full_recommendation += f" Additionally, consider: {', '.join(additional_cures)}."
    
    return full_recommendation

# Chatbot pairs
pairs = [
    [
        r"what is (thyroid|thyroid gland)?",
        ["The thyroid gland is a butterfly-shaped gland located in the front of the neck that produces hormones regulating metabolism.",]
    ],
    [
        r"what is (T3|T4|T4U|FTI)?",
        ["%1 is a hormone produced by the thyroid gland that is involved in various bodily functions.",]
    ],
    [
        r"my (age|T3|TT4|T4U|FTI|sex|sick|pregnant|thyroid_surgery|goitre|tumor) is (.*)",
        ["I got your %1 as %2. What is your next information?",]
    ],
    [
        r"predict my thyroid condition",
        ["Please provide the following details one by one: age, T3, TT4, T4U, FTI, sex, sick, pregnant, thyroid_surgery, goitre, tumor",]
    ],
    [
        r"thank you|thanks",
        ["You're welcome!",]
    ],
    [
        r"quit",
        ["Goodbye! Have a great day!",]
    ],
]

# Initialize chat
chat = Chat(pairs, reflections)

@app.route("/")
@cross_origin()
def home():
    return render_template("hom.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    try:
        data = request.get_json()
        Age = int(data["age"])
        T3 = float(data["T3"])
        TT4 = float(data["TT4"])
        T4U = float(data["T4U"])
        FTI = float(data["FTI"])
        sex = data['sex']
        sex_M = 1 if sex == "Male" else 0

        sick = data['sick'] == 'Yes'
        pregnant = data['pregnant'] == 'Yes'
        thyroid_surgery = data['thyroid_surgery'] == 'Yes'
        goitre = data['goitre'] == 'Yes'
        tumor = data['tumor'] == 'Yes'

        prediction = model.predict([[Age, T3, TT4, T4U, FTI, sex_M, int(sick), int(pregnant), int(thyroid_surgery), int(goitre), int(tumor)]])
        output = prediction[0]

        prediction_text = ["Hyperthyroid", "Hypothyroid", "Negative", "Sick"][output]
        cure_text = generate_recommendation(output, sick, goitre, tumor, pregnant, thyroid_surgery)

        return jsonify({
            'prediction': prediction_text,
            'cure': cure_text
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

@app.route("/chat", methods=["POST"])
@cross_origin()
def chatbot():
    try:
        user_input = request.get_json().get("message")
        response = chat.respond(user_input)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
