# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("🎓 Student Academic Performance Early Warning System")
st.markdown("""
This system predicts student performance risk based on their academic and behavioral patterns.
Enter the student's information below to get an early warning assessment.
""")

st.markdown("---")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('student_performance_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please make sure 'student_performance_model.pkl' is in the same directory.")
        return None

# Load feature names if available
@st.cache_resource
def load_feature_names():
    try:
        feature_names = joblib.load('feature_names.pkl')
        return feature_names
    except:
        return None

model = load_model()
feature_names = load_feature_names()

# Function to predict risk level based on predicted exam score
def get_risk_level(predicted_score):
    """
    Convert predicted exam score to risk level
    Based on the distribution of exam_score in your dataset (range approximately 19-100)
    """
    if predicted_score >= 70:
        return "LOW RISK", "🟢", "green", "Student is performing well. Continue current study habits."
    elif predicted_score >= 50:
        return "MEDIUM RISK", "🟡", "orange", "Student shows moderate performance. Monitor progress and provide support as needed."
    else:
        return "HIGH RISK", "🔴", "red", "Student is at significant risk. Immediate academic intervention recommended."

# Function to create one-hot encoded dataframe
def create_input_dataframe(age, gender, course, study_hours, class_attendance, 
                           internet_access, sleep_hours, sleep_quality, 
                           study_method, facility_rating, exam_difficulty):
    
    # Base data for numeric features
    data = {
        'age': [age],
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'sleep_hours': [sleep_hours]
    }
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # One-hot encode gender
    gender_values = ['male', 'female', 'other']
    for g in gender_values:
        df[f'gender_{g}'] = [1 if gender == g else 0]
    
    # One-hot encode course
    course_values = ['b.sc', 'b.tech', 'bca', 'bba', 'b.com', 'ba', 'diploma']
    for c in course_values:
        df[f'course_{c}'] = [1 if course == c else 0]
    
    # One-hot encode internet_access
    df['internet_access_yes'] = [1 if internet_access == 'yes' else 0]
    df['internet_access_no'] = [1 if internet_access == 'no' else 0]
    
    # One-hot encode sleep_quality
    sleep_quality_values = ['poor', 'average', 'good']
    for sq in sleep_quality_values:
        df[f'sleep_quality_{sq}'] = [1 if sleep_quality == sq else 0]
    
    # One-hot encode study_method
    study_method_values = ['self-study', 'group study', 'online videos', 'coaching', 'mixed']
    for sm in study_method_values:
        df[f'study_method_{sm}'] = [1 if study_method == sm else 0]
    
    # One-hot encode facility_rating
    facility_values = ['low', 'medium', 'high']
    for fr in facility_values:
        df[f'facility_rating_{fr}'] = [1 if facility_rating == fr else 0]
    
    # One-hot encode exam_difficulty
    difficulty_values = ['easy', 'moderate', 'hard']
    for ed in difficulty_values:
        df[f'exam_difficulty_{ed}'] = [1 if exam_difficulty == ed else 0]
    
    # Ensure column order matches training data if feature names are available
    if feature_names is not None:
        # Add any missing columns with 0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        # Reorder columns to match training data
        df = df[feature_names]
    
    return df

# Create input form
st.subheader("📝 Student Information Form")

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Information**")
    age = st.number_input(
        "Age",
        min_value=17,
        max_value=30,
        value=20,
        step=1,
        help="Student's age in years"
    )
    
    gender = st.selectbox(
        "Gender",
        options=["male", "female", "other"],
        help="Student's gender"
    )
    
    course = st.selectbox(
        "Course",
        options=["b.sc", "b.tech", "bca", "bba", "b.com", "ba", "diploma"],
        help="Program of study"
    )

with col2:
    st.markdown("**Academic Factors**")
    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5,
        help="Average number of hours the student studies per day"
    )
    
    class_attendance = st.slider(
        "Class Attendance (%)",
        min_value=0,
        max_value=100,
        value=75,
        step=1,
        help="Percentage of classes attended"
    )
    
    internet_access = st.selectbox(
        "Internet Access",
        options=["yes", "no"],
        help="Does the student have reliable internet access?"
    )

with col3:
    st.markdown("**Behavioral Factors**")
    sleep_hours = st.number_input(
        "Sleep Hours per Night",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5,
        help="Average hours of sleep per night"
    )
    
    sleep_quality = st.selectbox(
        "Sleep Quality",
        options=["poor", "average", "good"],
        help="Self-reported sleep quality"
    )
    
    study_method = st.selectbox(
        "Primary Study Method",
        options=["self-study", "group study", "online videos", "coaching", "mixed"],
        help="Main method used for studying"
    )

# Additional factors in expander
with st.expander("Additional Factors"):
    col4, col5 = st.columns(2)
    
    with col4:
        facility_rating = st.selectbox(
            "Facility Rating",
            options=["low", "medium", "high"],
            help="Rating of academic facilities available"
        )
    
    with col5:
        exam_difficulty = st.selectbox(
            "Exam Difficulty",
            options=["easy", "moderate", "hard"],
            help="Perceived difficulty of upcoming exams"
        )

st.markdown("---")

# Predict button
if st.button("🔮 PREDICT STUDENT RISK", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not loaded. Please make sure 'student_performance_model.pkl' is in the same directory.")
    else:
        # Create input dataframe
        input_df = create_input_dataframe(
            age, gender, course, study_hours, class_attendance,
            internet_access, sleep_hours, sleep_quality,
            study_method, facility_rating, exam_difficulty
        )
        
        # Show what the model sees (for debugging - can be removed in final version)
        with st.expander("View Model Input Data"):
            st.dataframe(input_df)
        
        # Make prediction
        try:
            predicted_score = model.predict(input_df)[0]
            risk_level, emoji, color, recommendation = get_risk_level(predicted_score)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 PREDICTION RESULTS")
            
            # Create columns for results display
            res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
            
            with res_col2:
                # Risk Level Display
                st.markdown(f"""
                <div style="
                    background-color: {color}20;
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;
                    border: 3px solid {color};
                ">
                    <h1 style="font-size: 48px; margin-bottom: 10px;">{emoji}</h1>
                    <h2 style="color: {color}; margin-bottom: 15px;">{risk_level}</h2>
                    <hr style="margin: 15px 0;">
                    <p style="font-size: 24px; margin-bottom: 10px;"><strong>Predicted Exam Score:</strong> {predicted_score:.1f} / 100</p>
                    <p style="font-size: 18px; margin-top: 15px;"><strong>Recommendation:</strong> {recommendation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show key metrics
            st.markdown("---")
            st.subheader("📈 Key Student Metrics")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Study Hours", f"{study_hours} hrs/day")
            with metric_col2:
                st.metric("Class Attendance", f"{class_attendance}%")
            with metric_col3:
                st.metric("Sleep Hours", f"{sleep_hours} hrs/night")
            with metric_col4:
                st.metric("Sleep Quality", sleep_quality.capitalize())
            
            # Intervention suggestions based on risk level
            st.markdown("---")
            st.subheader("💡 Suggested Interventions")
            
            if predicted_score < 50:
                st.warning("""
                ### ⚠️ Immediate Actions Recommended
                
                **Academic Support:**
                - Schedule meeting with academic advisor immediately
                - Refer to tutoring services for core subjects
                - Develop a personalized study plan
                
                **Monitoring:**
                - Weekly check-ins with faculty
                - Monitor attendance and engagement closely
                - Consider reducing course load if possible
                """)
            elif predicted_score < 70:
                st.info("""
                ### 📌 Proactive Support Measures
                
                **Academic Support:**
                - Encourage participation in study groups
                - Provide additional learning resources
                - Time management workshop recommendation
                
                **Monitoring:**
                - Bi-weekly progress reviews
                - Regular check-ins with academic advisor
                - Focus on improving attendance if needed
                """)
            else:
                st.success("""
                ### ✅ Recognition and Encouragement
                
                **Positive Reinforcement:**
                - Acknowledge good performance
                - Consider advanced learning opportunities
                
                **Development:**
                - Peer mentoring opportunities for other students
                - Maintain current successful study habits
                - Explore additional academic challenges
                """)
            
            # Additional insights based on key factors
            st.markdown("---")
            st.subheader("📌 Key Insights")
            
            insights = []
            if study_hours < 2:
                insights.append("⚠️ Study hours are below recommended levels")
            elif study_hours > 8:
                insights.append("ℹ️ Study hours are high - ensure proper rest")
            
            if class_attendance < 60:
                insights.append("⚠️ Attendance is low - this is a strong predictor of performance")
            
            if sleep_hours < 6:
                insights.append("⚠️ Sleep hours are low - may impact academic performance")
            elif sleep_hours > 9:
                insights.append("ℹ️ Sleep hours are high - may indicate other issues")
            
            if sleep_quality == "poor":
                insights.append("⚠️ Poor sleep quality reported - may affect concentration")
            
            if insights:
                for insight in insights:
                    st.write(insight)
            else:
                st.write("✅ All key metrics are within healthy ranges")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please check that all input values are valid.")

# Footer
st.markdown("---")
st.caption("""
**Note:** This prediction is based on historical student data analysis and serves as an early warning tool.
It should be used alongside other assessment methods and professional judgment.
The model achieved an R² score of 0.733, indicating good predictive capability.
""")