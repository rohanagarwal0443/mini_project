from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

model = joblib.load(os.path.join(os.path.dirname(__file__), "model.pkl"))
encoder = joblib.load(os.path.join(os.path.dirname(__file__), "encoder.pkl"))

# Prediction route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            area = float(request.form["Area_sqft"])
            bedrooms = int(request.form["Bedrooms"])
            bathroom = int(request.form["Bathroom"])
            location_type = request.form["Location_Type"]
            house_type = request.form["House_Type"]
            state = request.form["State"]
            city = request.form["City"]

            # Create dataframe same as training structure
            user_df = pd.DataFrame([{
                "Area_sqft": area,
                "Bedrooms": bedrooms,
                "Bathroom": bathroom,
                "Location_Type": location_type,
                "House_Type": house_type,
                "State": state,
                "City": city
            }])

            # Apply encoding (using already fitted encoder)
            cat_cols = ["Location_Type", "House_Type", "State", "City"]
            encoded = encoder.transform(user_df[cat_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

            # Merge numerical + encoded columns
            final_df = pd.concat([user_df.drop(columns=cat_cols), encoded_df], axis=1)

            # Ensure all model columns exist (missing = 0)
            for col in model.feature_names_in_:
                if col not in final_df.columns:
                    final_df[col] = 0
            final_df = final_df[model.feature_names_in_]

            # Predict price
            price = model.predict(final_df)[0]

            if price >= 10000000:
                formatted_price = f"{price/10000000:.2f} Crore"
            elif price >= 100000:
                formatted_price = f"{price/100000:.2f} Lakh"
            else:
                formatted_price = f"{price:,.0f}"
            
            return render_template("index.html", prediction_text=f"Estimated Price: ₹{formatted_price}")

        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {str(e)}")

    return render_template("index.html", prediction_text="")

if __name__ == "__main__":
    # Render.com requires 0.0.0.0 and port 10000
    app.run(host="0.0.0.0", port=10000, debug=True)
