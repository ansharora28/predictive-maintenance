import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Load your dataset
@st.cache_data
def load_data(file_path='predictive_maintenance_dataset.csv'):
    """Load dataset from the specified file path."""
    return pd.read_csv(file_path)

data = load_data()

# Step 2: Preprocess the dataset
data_numeric = data.select_dtypes(include=[np.number])  # Select only numeric columns for correlation

# Display the dataset information
st.title("🔧 Predictive Maintenance Model")
st.write("This application allows you to train a model on your predictive maintenance dataset and visualize the results.")
st.write("### Columns in the dataset:")
st.write(data_numeric.columns.tolist())

# Step 3: Train the model
if st.button("Train Model"):
    with st.spinner("Training the model... Please wait."):
        # Prepare features and target
        X = data_numeric.drop('failure', axis=1)  # Features, drop the target column
        y = data_numeric['failure']  # Target variable
        
        # One-hot encode any categorical variables in the dataset if necessary
        X = pd.get_dummies(X, drop_first=True)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a RandomForest model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, 'trained_model.joblib')

        # Store the split data and feature names in session state
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X = X  # Store the feature DataFrame
        st.session_state.model = model  # Store the trained model

    st.success("Model trained successfully!")

# Step 4: Show results after training
if st.button("📊 Show Results"):
    if 'X_test' in st.session_state and 'y_test' in st.session_state:
        model = st.session_state.model  # Load the trained model

        # Make predictions
        y_pred = model.predict(st.session_state.X_test)

        # Display Confusion Matrix
        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        st.write(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix', fontsize=14)
        st.pyplot(plt)
        
        # Display Classification Report
        st.subheader("📋 Classification Report")
        st.text(classification_report(st.session_state.y_test, y_pred))

        # Display ROC Curve
        fpr, tpr, _ = roc_curve(st.session_state.y_test, model.predict_proba(st.session_state.X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic', fontsize=14)
        plt.legend(loc="lower right")
        st.pyplot(plt)

        # Display Feature Importance
        feature_importances = model.feature_importances_
        feature_names = st.session_state.X.columns  # Use the stored feature names
        indices = np.argsort(feature_importances)[::-1]

        n_features = min(10, len(feature_names))  # Display up to 10 features
        st.subheader(f"⭐ Top {n_features} Feature Importances")
        for f in range(n_features):
            st.write(f"**{feature_names[indices[f]]}**: {feature_importances[indices[f]]:.4f}")

        plt.figure(figsize=(10, 5))
        plt.title(f"Top {n_features} Feature Importances", fontsize=14)
        plt.bar(range(n_features), feature_importances[indices[:n_features]], align="center")
        plt.xticks(range(n_features), feature_names[indices[:n_features]], rotation=45)
        plt.xlim([-1, n_features])
        st.pyplot(plt)

        # Display Correlation Heatmap
        st.subheader("🌡️ Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        corr = data_numeric.corr()  # Compute correlation on the numeric data only
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap', fontsize=14)
        st.pyplot(plt)

        # Display Class Distribution
        st.subheader("📊 Class Distribution")
        st.bar_chart(data['failure'].value_counts())

        # Display Top Misclassified Instances
        misclassified = st.session_state.X_test[st.session_state.y_test != y_pred]
        st.subheader("❌ Top Misclassified Instances")
        st.dataframe(misclassified)

        # Display Sample Predictions
        st.subheader("🔍 Sample Predictions")
        sample_data = st.session_state.X_test.sample(5)
        predictions = model.predict(sample_data)
        sample_results = pd.DataFrame(sample_data)
        sample_results['Prediction'] = predictions
        st.dataframe(sample_results)

        # Display Distribution of Features
        st.subheader("📈 Distribution of Features")
        for column in st.session_state.X.columns[:5]:  # Display distribution for first 5 features
            st.write(f"### Distribution of {column}")
            plt.figure(figsize=(6, 4))
            sns.histplot(st.session_state.X[column], kde=True)
            plt.title(f'Distribution of {column}', fontsize=14)
            st.pyplot(plt)

    else:
        st.error("⚠️ Please train the model first!")

# Step 5: User input for predictions
st.subheader("🧮 Make a Prediction")
if 'model' in st.session_state:
    st.write("To increase the likelihood of a failure prediction, adjust the following inputs based on their importance.")
    
    # Dynamically generate input fields for each feature based on the trained model
    input_data = {}
    feature_importances = st.session_state.model.feature_importances_
    feature_names = st.session_state.X.columns
    indices = np.argsort(feature_importances)[::-1]

    # Display important features first
    for i in indices:
        col = feature_names[i]
        if feature_importances[i] > 0.01:  # Focus on more important features
            input_data[col] = st.slider(
                f"{col} (Higher values are more likely to result in failure)",
                min_value=float(st.session_state.X[col].min()),
                max_value=float(st.session_state.X[col].max()),
                value=float(st.session_state.X[col].max() * 0.9)  # Use 90% of the max value for higher likelihood of failure
            )
        else:
            input_data[col] = st.number_input(
                f"Enter {col}",
                min_value=float(st.session_state.X[col].min()),
                max_value=float(st.session_state.X[col].max()),
                value=float(st.session_state.X[col].mean())  # Keep this as the mean for less important features
            )
    
    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df[st.session_state.X.columns]  # Reorder columns to match training data

    # Predict based on input
    if st.button("🔍 Predict"):
        prediction = st.session_state.model.predict(input_df)
        st.subheader("📝 Prediction Result")
        
        # Highlight the prediction result
        if prediction[0] == 1:
            st.markdown('<h1 style="color: red;">FAILURE</h1>', unsafe_allow_html=True)
        else:
            st.markdown('<h1 style="color: green;">NO FAILURE</h1>', unsafe_allow_html=True)
