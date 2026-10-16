
import streamlit as st
import pandas as pd
import joblib

# ------------------------------------
# 1. Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide"
)

# ------------------------------------
# 2. Load Saved Models
# ------------------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    cluster_names = joblib.load("cluster_names.pkl")

    return kmeans, scaler, cluster_names


try:
    kmeans, scaler, cluster_names = load_models()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()

# ------------------------------------
# 3. Header
# ------------------------------------
st.title("🎯 Customer Segmentation Dashboard")

st.write(
    "Predict the customer segment using a trained "
    "K-Means clustering model."
)

st.divider()

# ------------------------------------
# 4. Sidebar
# ------------------------------------
with st.sidebar:
    st.header("Project Information")
    st.write("**Algorithm:** K-Means")
    st.write("**Number of Clusters:** 4")
    st.write("**Project:** Customer Segmentation")

    st.info(
        "Enter customer details and click the prediction "
        "button to identify the customer segment."
    )

# ------------------------------------
# 5. Customer Input Section
# ------------------------------------
st.subheader("👤 Customer Details")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:
        income = st.number_input(
            "Annual Income (k$)",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=1.0
        )

    with col3:
        spending_score = st.slider(
            "Spending Score",
            min_value=1,
            max_value=100,
            value=50
        )

    predict_button = st.form_submit_button(
        "🔍 Predict Customer Segment",
        type="primary",
        use_container_width=True
    )

# ------------------------------------
# 6. Predict Customer Segment
# ------------------------------------
if predict_button:

    # Feature names and order must match training
    customer_data = pd.DataFrame(
        [[age, income, spending_score]],
        columns=[
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    try:
        # Scale input data
        scaled_data = scaler.transform(customer_data)

        # Predict cluster
        cluster = int(kmeans.predict(scaled_data)[0])

        # Get cluster name
        if isinstance(cluster_names, dict):
            segment_name = cluster_names.get(
                cluster,
                cluster_names.get(
                    str(cluster),
                    f"Cluster {cluster}"
                )
            )
        else:
            segment_name = cluster_names[cluster]

        # ------------------------------------
        # 7. Display Prediction
        # ------------------------------------
        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Cluster",
                f"Cluster {cluster}"
            )

        with col2:
            st.metric(
                "Customer Segment",
                str(segment_name)
            )

        st.success(
            f"Customer belongs to: {segment_name}"
        )

        # ------------------------------------
        # 8. Customer Profile
        # ------------------------------------
        st.subheader("Customer Profile")

        profile = pd.DataFrame({
            "Feature": [
                "Age",
                "Annual Income (k$)",
                "Spending Score"
            ],
            "Value": [
                age,
                income,
                spending_score
            ]
        })

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------
        # 9. Cluster Distance Chart
        # ------------------------------------
        if hasattr(kmeans, "transform"):

            distances = kmeans.transform(scaled_data)[0]

            distance_data = pd.DataFrame({
                "Cluster": [
                    f"Cluster {i}"
                    for i in range(len(distances))
                ],
                "Distance": distances
            })

            st.subheader("Distance from Cluster Centers")

            st.bar_chart(
                distance_data.set_index("Cluster")
            )

            st.caption(
                "A smaller distance means the customer is "
                "closer to that cluster's centroid."
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")

        st.warning(
            "Verify that your input features and their order "
            "match the features used during model training."
        )

# ------------------------------------
# 10. Footer
# ------------------------------------
st.divider()

st.caption(
    "Customer Segmentation | K-Means Machine Learning Project"
)



