# ============================================================
# ManganAI - COMPLETE FRONTEND
#
# Three modules:
# 1. Manganese Location Prediction
# 2. Production Prediction
# 3. Production Improvement Simulator
#
# IMPORTANT:
# The frontend does NOT guess the model columns.
# It reads the exact feature names from the trained models.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# Purpose:
# Import everything required for the ManganAI dashboard.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# 2. PAGE CONFIGURATION
# Purpose:
# Configure the appearance of the ManganAI application.
# ============================================================

st.set_page_config(
    page_title="ManganAI",
    page_icon="⛏️",
    layout="wide"
)


# ============================================================
# 3. PROJECT DIRECTORIES
# Purpose:
# Locate the models folder automatically.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 4. MODEL FILE FINDER
# Purpose:
# Find the trained model without assuming that only one
# particular filename was used.
# ============================================================

def find_model(possible_names):

    for name in possible_names:

        path = MODEL_DIR / name

        if path.exists():
            return path

    return None


# ============================================================
# 5. FIND LOCATION MODEL
# Purpose:
# Search for the trained manganese/exploration model.
# ============================================================

LOCATION_MODEL_PATH = find_model([
    "location_model.pkl",
    "location_model.joblib",
    "exploration_model.pkl",
    "exploration_model.joblib",
    "manganese_model.pkl",
    "manganese_model.joblib",
    "manganese_location_model.pkl",
    "manganese_location_model.joblib"
])


# ============================================================
# 6. FIND PRODUCTION MODEL
# Purpose:
# Search for the trained production prediction model.
# ============================================================

PRODUCTION_MODEL_PATH = find_model([
    "production_model.pkl",
    "production_model.joblib"
])


# ============================================================
# 7. LOAD MODELS
# Purpose:
# Load the trained models only once.
# ============================================================

@st.cache_resource
def load_models():

    location_model = None
    production_model = None

    if LOCATION_MODEL_PATH is not None:

        try:
            location_model = joblib.load(
                "ManganAI/models/exploration_model.pkl"
            )

        except Exception:
            location_model = None


    if PRODUCTION_MODEL_PATH is not None:

        try:
            production_model = joblib.load(
                "ManganAI/models/production_model.pkl"
            )

        except Exception:
            production_model = None


    return location_model, production_model


location_model, production_model = load_models()


# ============================================================
# 8. GET EXACT MODEL FEATURES
# Purpose:
# Read the exact columns used during training.
#
# This is the most important part of the frontend.
# We do NOT manually guess the model input columns.
# ============================================================

def get_features(model):

    if model is None:
        return []

    if hasattr(model, "feature_names_in_"):

        return list(
            model.feature_names_in_
        )

    return []


location_features = get_features(
    location_model
)

production_features = get_features(
    production_model
)


# ============================================================
# 9. DEFAULT VALUES
# Purpose:
# Give reasonable demonstration values to input boxes.
# These are ONLY demonstration starting values.
# The model's actual feature names always come from the model.
# ============================================================

def default_value(feature):

    name = str(feature).lower().strip()

    # Location / geological
    if "latitude" in name:
        return 20.50

    if "longitude" in name:
        return 79.50

    if "elevation" in name:
        return 500.0

    if "magnetic" in name:
        return 50.0

    if "rock density" in name:
        return 2.70

    if "rock_density" in name:
        return 2.70

    if "geological score" in name:
        return 0.70

    if "geological_score" in name:
        return 0.70

    if "ndvi" in name:
        return 0.40

    if "rainfall" in name:
        return 130.0

    if "soil moisture" in name:
        return 45.0

    if "soil_moisture" in name:
        return 45.0

    if "temperature" in name:
        return 32.0

    if "band" in name:
        return 0.20

    if name == "ei" or "ei" in name:
        return 400.0

    # Production
    if "target production" in name:
        return 10000.0

    if "target_production" in name:
        return 10000.0

    if "previous production" in name:
        return 9000.0

    if "previous_production" in name:
        return 9000.0

    if "equipment downtime" in name:
        return 10.0

    if "equipment_downtime" in name:
        return 10.0

    if "equipment availability" in name:
        return 85.0

    if "equipment_availability" in name:
        return 85.0

    if "blasting delay" in name:
        return 10.0

    if "blasting_delay" in name:
        return 10.0

    if "ore grade" in name:
        return 4.0

    if "ore_grade" in name:
        return 4.0

    # Unknown numerical feature
    return 0.0
    
    
    
    


# ============================================================
# 10. CREATE INPUT BOXES
# Purpose:
# Create input boxes using EXACT model columns.
# ============================================================

def create_inputs(features, prefix):

    values = {}

    if len(features) == 0:

        return values


    columns = st.columns(2)


    for i, feature in enumerate(features):

        column = columns[i % 2]

        with column:

            values[feature] = st.number_input(
                str(feature),
                value=float(
                    default_value(feature)
                ),
                key=f"{prefix}_{feature}"
            )


    return values


# ============================================================
# 11. APPLICATION HEADER
# Purpose:
# Display the ManganAI project title.
# ============================================================

st.title("⛏️ ManganAI")

st.subheader(
    "AI-Based Manganese Mining Intelligence System"
)

st.write(
    "Explore manganese potential, predict production, "
    "identify possible shortfalls and test corrective actions."
)

st.divider()


# ============================================================
# 12. SIDEBAR
# Purpose:
# Provide navigation between the three modules.
# ============================================================

st.sidebar.title("⛏️ ManganAI")

st.sidebar.write(
    "Select a module:"
)

page = st.sidebar.radio(
    "ManganAI Modules",
    [
        "📍 Manganese Location Prediction",
        "🏭 Production Prediction",
        "🔄 Production Improvement Simulator"
    ]
)


# ============================================================
# 13. MODEL STATUS
# Purpose:
# Show whether the trained models were loaded.
# ============================================================

with st.sidebar.expander(
    "🔧 Model Status"
):

    if location_model is not None:

        st.success(
            "Location model loaded"
        )

    else:

        st.error(
            "Location model not found"
        )


    if production_model is not None:

        st.success(
            "Production model loaded"
        )

    else:

        st.error(
            "Production model not found"
        )


# ============================================================
# PAGE 1
# MANGANESE LOCATION PREDICTION
# ============================================================

if page == "📍 Manganese Location Prediction":

    st.header(
        "📍 Manganese Location Prediction"
    )

    st.write(
        "This module uses the trained location/exploration "
        "model to predict manganese potential."
    )


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if location_model is None:

        st.error(
            "Location model could not be loaded."
        )

        st.write(
            "Check the ManganAI/models folder."
        )

        st.stop()


    # --------------------------------------------------------
    # IMPORTANT:
    # Display EXACT trained model columns
    # --------------------------------------------------------

    if len(location_features) == 0:

        st.error(
            "The location model does not expose "
            "feature names."
        )

        st.write(
            "We cannot safely create the location inputs "
            "without knowing the columns used during training."
        )

        st.stop()


    st.success(
        "Using the exact columns from the trained "
        "location model."
    )

    st.caption(
        "Model features: "
        + ", ".join(
            map(str, location_features)
        )
    )


    # --------------------------------------------------------
    # Input values
    # --------------------------------------------------------

    st.subheader(
        "Enter Location / Geological Values"
    )

    location_values = create_inputs(
        location_features,
        "location"
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Manganese Location",
        type="primary",
        use_container_width=True
    ):

        location_input = pd.DataFrame(
            [location_values],
            columns=location_features
        )


        try:

            prediction = location_model.predict(
                location_input
            )[0]


            st.divider()

            st.subheader(
                "📊 Location Prediction Result"
            )


            st.success(
                f"Model Prediction: {prediction}"
            )


            st.write(
                "Values supplied to the model:"
            )

            st.dataframe(
                location_input,
                use_container_width=True
            )


        except Exception as e:

            st.error(
                "Location prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# PAGE 2
# PRODUCTION PREDICTION
# ============================================================

elif page == "🏭 Production Prediction":

    st.header(
        "🏭 Production Prediction"
    )

    st.write(
        "Predict expected manganese production from "
        "production, environmental and equipment conditions."
    )


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if production_model is None:

        st.error(
            "Production model could not be loaded."
        )

        st.write(
            "Check the ManganAI/models folder."
        )

        st.stop()


    # --------------------------------------------------------
    # Exact production model columns
    # --------------------------------------------------------

    if len(production_features) == 0:

        st.error(
            "The production model does not expose "
            "feature names."
        )

        st.stop()


    st.success(
        "Using the exact columns from the trained "
        "production model."
    )

    st.caption(
        "Model features: "
        + ", ".join(
            map(str, production_features)
        )
    )


    # --------------------------------------------------------
    # Input values
    # --------------------------------------------------------

    st.subheader(
        "Enter Current Mining Conditions"
    )

    production_values = create_inputs(
        production_features,
        "production"
    )


    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------

    if st.button(
        "🏭 Predict Production",
        type="primary",
        use_container_width=True
    ):

        production_input = pd.DataFrame(
            [production_values],
            columns=production_features
        )


        try:

            predicted_production = float(
                production_model.predict(
                    production_input
                )[0]
            )


            st.divider()

            st.subheader(
                "🎯 Production Prediction"
            )


            st.metric(
                "Predicted Production",
                f"{predicted_production:,.2f} tonnes"
            )


            # ------------------------------------------------
            # Calculate shortfall if target exists
            # ------------------------------------------------

            target_column = None

            for feature in production_features:

                feature_lower = str(
                    feature
                ).lower()

                if (
                    "target production"
                    in feature_lower
                    or
                    "target_production"
                    in feature_lower
                ):

                    target_column = feature

                    break


            if target_column is not None:

                target = float(
                    production_values[
                        target_column
                    ]
                )


                shortfall = (
                    target
                    -
                    predicted_production
                )


                if shortfall > 0:

                    st.warning(
                        f"⚠️ Possible Production Shortfall: "
                        f"{shortfall:,.2f} tonnes"
                    )

                else:

                    st.success(
                        "✅ Predicted production meets "
                        "or exceeds the target."
                    )


            # ------------------------------------------------
            # Show input
            # ------------------------------------------------

            with st.expander(
                "View Values Used for Prediction"
            ):

                st.dataframe(
                    production_input,
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                "Production prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# PAGE 3
# PRODUCTION IMPROVEMENT SIMULATOR
# ============================================================

else:

    st.header(
        "🔄 Production Improvement Simulator"
    )

    st.write(
        "Compare current mining conditions with an improved "
        "scenario and see how predicted production changes."
    )


    # --------------------------------------------------------
    # Check production model
    # --------------------------------------------------------

    if production_model is None:

        st.error(
            "Production model could not be loaded."
        )

        st.stop()


    if len(production_features) == 0:

        st.error(
            "The production model does not expose "
            "feature names."
        )

        st.stop()


    # ========================================================
    # CURRENT CONDITIONS
    # ========================================================

    st.subheader(
        "1️⃣ Current Mining Conditions"
    )

    current_values = create_inputs(
        production_features,
        "current"
    )


    st.divider()


    # ========================================================
    # IMPROVED CONDITIONS
    # ========================================================

    st.subheader(
        "2️⃣ Improved Mining Conditions"
    )

    st.write(
        "Change the values you want to improve. "
        "For example, reduce equipment downtime or "
        "blasting delay."
    )


    improved_values = {}


    columns = st.columns(2)


    for i, feature in enumerate(
        production_features
    ):

        with columns[i % 2]:

            improved_values[feature] = st.number_input(
                f"Improved {feature}",
                value=float(
                    current_values[feature]
                ),
                key=f"improved_{feature}"
            )


    st.divider()


    # ========================================================
    # COMPARE
    # ========================================================

    if st.button(
        "🔮 Compare Current vs Improved Production",
        type="primary",
        use_container_width=True
    ):

        try:

            current_input = pd.DataFrame(
                [current_values],
                columns=production_features
            )


            improved_input = pd.DataFrame(
                [improved_values],
                columns=production_features
            )


            # ------------------------------------------------
            # Current prediction
            # ------------------------------------------------

            current_prediction = float(
                production_model.predict(
                    current_input
                )[0]
            )


            # ------------------------------------------------
            # Improved prediction
            # ------------------------------------------------

            improved_prediction = float(
                production_model.predict(
                    improved_input
                )[0]
            )


            # ------------------------------------------------
            # Difference
            # ------------------------------------------------

            change = (
                improved_prediction
                -
                current_prediction
            )


            st.subheader(
                "📊 Simulation Result"
            )


            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "Current Production",
                    f"{current_prediction:,.2f} tonnes"
                )


            with c2:

                st.metric(
                    "Improved Production",
                    f"{improved_prediction:,.2f} tonnes"
                )


            with c3:

                st.metric(
                    "Production Change",
                    f"{change:+,.2f} tonnes"
                )


            # ------------------------------------------------
            # Result interpretation
            # ------------------------------------------------

            if change > 0:

                st.success(
                    "🟢 The improved scenario gives a "
                    "higher predicted production."
                )

            elif change < 0:

                st.warning(
                    "🟠 The improved scenario gives a "
                    "lower predicted production."
                )

            else:

                st.info(
                    "Production prediction is unchanged."
                )


            # =================================================
            # AUTOMATIC RECOMMENDATIONS
            # =================================================

            st.subheader(
                "💡 Recommended Corrective Actions"
            )


            recommendations = []


            for feature in production_features:

                name = str(feature).lower()


                # Equipment downtime
                if "downtime" in name:

                    old = current_values[feature]

                    new = improved_values[feature]

                    if new < old:

                        recommendations.append(
                            f"🔧 Equipment downtime reduced "
                            f"from {old} to {new}."
                        )


                # Equipment availability
                if "availability" in name:

                    old = current_values[feature]

                    new = improved_values[feature]

                    if new > old:

                        recommendations.append(
                            f"🚜 Equipment availability improved "
                            f"from {old} to {new}."
                        )


                # Blasting delay
                if "blasting" in name:

                    old = current_values[feature]

                    new = improved_values[feature]

                    if new < old:

                        recommendations.append(
                            f"💥 Blasting delay reduced "
                            f"from {old} to {new}."
                        )


            if len(recommendations) == 0:

                recommendations.append(
                    "ℹ️ No obvious operational improvement "
                    "was entered. Try reducing downtime or "
                    "blasting delay, or increasing equipment "
                    "availability."
                )


            for recommendation in recommendations:

                st.write(
                    recommendation
                )


            # ------------------------------------------------
            # Comparison table
            # ------------------------------------------------

            st.subheader(
                "📋 Current vs Improved Inputs"
            )


            comparison = pd.DataFrame({

                "Parameter": [
                    str(x)
                    for x in production_features
                ],

                "Current": [
                    current_values[x]
                    for x in production_features
                ],

                "Improved": [
                    improved_values[x]
                    for x in production_features
                ]

            })


            st.dataframe(
                comparison,
                use_container_width=True
            )


        except Exception as e:

            st.error(
                "Simulation failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# FOOTERx
# Purpose:
# Display project information.
# ============================================================

st.sidebar.divider()

st.sidebar.success(
    "ManganAI Ready"
)

st.sidebar.caption(
    "AI + Geological + Environmental + "
    "Equipment Intelligence"
)



