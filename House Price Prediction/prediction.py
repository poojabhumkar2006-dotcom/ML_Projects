import os
import pickle
import pandas as pd


class HousePricePredictor:

    def __init__(self):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models")

        model_path = os.path.join(
            model_dir,
            "house_price_model.pkl"
        )

        vectorizer_path = os.path.join(
            model_dir,
            "vectorizer.pkl"
        )

        encoder_path = os.path.join(
            model_dir,
            "encoder.pkl"
        )

        features_path = os.path.join(
            model_dir,
            "features.pkl"
        )

        # Check files
        required_files = [
            model_path,
            vectorizer_path,
            encoder_path,
            features_path
        ]

        for file in required_files:

            if not os.path.exists(file):

                raise FileNotFoundError(
                    f"Required model file not found:\n{file}"
                )

        # Load model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load DictVectorizer
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        # Load Encoder
        with open(encoder_path, "rb") as f:
            self.encoder = pickle.load(f)

        # Load features
        with open(features_path, "rb") as f:
            self.features = pickle.load(f)

    # ======================================================
    # ENCODE CATEGORICAL FEATURES
    # ======================================================

    def encode_features(
        self,
        property_type,
        furnished,
        transport,
        facing,
        security
    ):

        ordinal_df = pd.DataFrame({

            "Property_Type": [property_type],

            "Furnished_Status": [furnished],

            "Public_Transport_Accessibility": [transport],

            "Facing": [facing],

            "Security": [security]

        })

        encoded = self.encoder.transform(
            ordinal_df
        )

        return {

            "Property_Type": encoded[0][0],

            "Furnished_Status": encoded[0][1],

            "Public_Transport_Accessibility":
                encoded[0][2],

            "Facing": encoded[0][3],

            "Security": encoded[0][4]

        }

    # ======================================================
    # PREPARE INPUT
    # ======================================================

    def prepare_input(
        self,
        state,
        city,
        property_type,
        bhk,
        size_sqft,
        price_sqft,
        year,
        furnished,
        floor,
        total_floor,
        age,
        school,
        hospital,
        transport,
        parking,
        security,
        amenities,
        facing,
        owner,
        availability
    ):

        encoded = self.encode_features(
            property_type,
            furnished,
            transport,
            facing,
            security
        )

        input_data = {

            "State": state.lower(),

            "City": city.lower(),

            "Property_Type":
                encoded["Property_Type"],

            "BHK": bhk,

            "Size_in_SqFt": size_sqft,

            "Price_per_SqFt": price_sqft,

            "Year_Built": year,

            "Furnished_Status":
                encoded["Furnished_Status"],

            "Floor_No": floor,

            "Total_Floors": total_floor,

            "Age_of_Property": age,

            "Nearby_Schools": school,

            "Nearby_Hospitals": hospital,

            "Public_Transport_Accessibility":
                encoded[
                    "Public_Transport_Accessibility"
                ],

            "Parking_Space":
                parking.lower(),

            "Security":
                encoded["Security"],

            "Amenities":
                amenities.lower(),

            "Facing":
                encoded["Facing"],

            "Owner_Type":
                owner.lower(),

            "Availability_Status":
                availability.lower()
        }

        return input_data

    # ======================================================
    # PREDICT PRICE
    # ======================================================

    def predict(self, input_data):

        X = self.vectorizer.transform(
            [input_data]
        )

        prediction = self.model.predict(X)[0]

        return float(prediction)

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    def get_feature_importance(self):

        if not hasattr(
            self.model,
            "feature_importances_"
        ):
            return None

        importance = self.model.feature_importances_

        # Handle mismatch safely
        if len(importance) != len(self.features):

            return None

        df = pd.DataFrame({

            "Feature": self.features,

            "Importance": importance

        })

        return df.sort_values(
            "Importance",
            ascending=False
        )