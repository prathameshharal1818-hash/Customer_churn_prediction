from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask App
app = Flask(__name__)

# Load Model and Scaler
model = joblib.load("Models/customer_churn_model.pkl")
scaler = joblib.load("Models/scaler.pkl")

# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- ABOUT PAGE ----------------
@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- PREDICT PAGE ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        # Get values from form
        gender = request.form["gender"]
        tenure = float(request.form["tenure"])
        monthly = float(request.form["monthlycharges"])
        total = float(request.form["totalcharges"])
        internet = request.form["internet"]
        security = request.form["security"]
        techsupport = request.form["techsupport"]
        contract = request.form["contract"]
        paperless = request.form["paperless"]
        payment = request.form["payment"]

        # Convert to numeric values
        gender = 1 if gender == "Male" else 0
        internet = 1 if internet == "Fiber" else 0
        security = 1 if security == "Yes" else 0
        techsupport = 1 if techsupport == "Yes" else 0
        contract = 1 if contract == "Two Year" else 0
        paperless = 1 if paperless == "Yes" else 0
        payment = 1 if payment == "Electronic" else 0

        # Create dataframe
        data = pd.DataFrame([{
            "TotalCharges": total,
            "MonthlyCharges": monthly,
            "tenure": tenure,
            "InternetService_Fiber optic": internet,
            "PaymentMethod_Electronic check": payment,
            "OnlineSecurity_Yes": security,
            "Contract_Two year": contract,
            "gender_Male": gender,
            "TechSupport_Yes": techsupport,
            "PaperlessBilling_Yes": paperless
        }])

        # Scale numerical columns
        data[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
            data[["tenure", "MonthlyCharges", "TotalCharges"]]
        )

        # Predict
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1] * 100

        if prediction == 1:
            result = "Customer is likely to Churn"
        else:
            result = "Customer is likely to Stay"

        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability, 2)
        )

    return render_template("predict.html")


# ---------------- RUN APPLICATION ----------------
if __name__ == "__main__":
    app.run(debug=True)