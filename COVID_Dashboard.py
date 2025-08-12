
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


# Configure page
st.set_page_config(
    page_title="COVID-19 Clinical Analytics",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .st-emotion-cache-1y4p8pa { padding: 2rem; }
    .metric-card { 
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .plot-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    .highlight { color: #e63946; font-weight: bold; }
    .positive { color: #2a9d8f; }
    .negative { color: #e76f51; }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🦠 COVID-19 Clinical Outcomes Dashboard")
st.markdown("""
This interactive dashboard analyzes clinical outcomes for COVID-19 patients, focusing on mortality risk factors, 
comorbidity impacts, and ICU resource utilization. Use the filters below to explore specific patient subgroups.
""")

# Load data 
@st.cache_data
def load_data():
    
    return pd.DataFrame()  

df = pd.read_csv("cleaned_covid_data.csv")
covid_patients = df[df['CLASIFFICATION_BINARY'] == 1].copy()

# Page Selector
page = st.sidebar.selectbox("📄 Select Page", ["📊 Analysis Page", "🤖 ML Prediction"])

if page == "📊 Analysis Page":
    # --------------------------------- Analysis Page --------------------------------
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
    
        # Age range slider
        age_range = st.slider(
            "Age Range",
            min_value=int(df['AGE'].min()),
            max_value=int(df['AGE'].max()),
            value=(30, 70),
            key="age_slider"
        )
    
        # Comorbidities selector - NOW FULLY CONTAINED IN SIDEBAR
        st.header("Comorbidities")
        all_comorbidities = ['DIABETES', 'HIPERTENSION', 'PNEUMONIA', 'OBESITY',
                            'CARDIOVASCULAR', 'RENAL_CHRONIC', 'COPD', 'ASTHMA']
        
        # Initialize session state if not exists
        if 'selected_comorbidities' not in st.session_state:
            st.session_state.selected_comorbidities = ['PNEUMONIA', 'DIABETES']
    
        # Comorbidity selection in sidebar
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.multiselect(
                "Select comorbidities",
                options=all_comorbidities,
                default=st.session_state.selected_comorbidities,
                key="comorbidity_multiselect"
            )
    
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("All", key="select_all_btn"):
                st.session_state.selected_comorbidities = all_comorbidities
                st.rerun()
    
        # Update session state
        st.session_state.selected_comorbidities = selected

    # Then in your visualization code, access the selections via st.session_state:
    selected_comorbidities = st.session_state.selected_comorbidities

    if not selected_comorbidities:
        st.info("No comorbidities selected - showing all by default")
        selected_comorbidities = all_comorbidities

    # Filter data based on selections
    filtered_data = covid_patients[
        (covid_patients['AGE'] >= age_range[0]) & 
        (covid_patients['AGE'] <= age_range[1])
        ]
    
    
    #Key Metrics - WITH UNFILTERED COMPARISONS
    st.header("📊 Key Metrics")
    
    # First Row - Filtered Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Filtered COVID Patients", 
                 f"{len(filtered_data):,}",
                 help=f"Age {age_range[0]}-{age_range[1]} only")
    
    with col2:
        mortality_count =  filtered_data[filtered_data['DIED'] == 1].shape[0]
        st.metric("Filtered Mortality", 
                 f"{(mortality_count/len(filtered_data))*100:.1f}%",
                 f"{mortality_count:,} deaths")

    with col3:
        icu_count = filtered_data[filtered_data['ICU'] == 1].shape[0]
        st.metric("Filtered ICU Admissions", 
                 f"{(icu_count/len(filtered_data))*100:.1f}%",
                 f"{icu_count:,} cases")
    
    with col4:
        st.metric("Filtered Avg Age", 
                 f"{filtered_data['AGE'].mean():.1f} years",
                 f"Range: {int(filtered_data['AGE'].min())}-{int(filtered_data['AGE'].max())}")

    # Second Row - UNFILTERED Statistics (new)
    st.markdown("---")
    st.subheader("📌 Baseline Statistics (All COVID Patients)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total COVID Patients", 
                 f"{len(covid_patients):,}",
                 "Unfiltered population")
    
    with col2:
        total_mortality = covid_patients['DIED'].sum()
        st.metric("Overall Mortality Rate", 
                 f"{(total_mortality/len(covid_patients))*100:.1f}%",
                 f"{total_mortality:,} deaths")
    
    with col3:
        total_icu = covid_patients[covid_patients['ICU'] == 1].shape[0]
        st.metric("Overall ICU Admission Rate", 
                 f"{(total_icu/len(covid_patients))*100:.1f}%",
                 f"{total_icu:,} cases")

    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Mortality Analysis", "🏥 Comorbidity Impact", "⚕️ Clinical Outcomes", "🔬 Advanced Analytics"])
    
    with tab1:
        st.header("Mortality Risk Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
            st.subheader("Age Distribution by Outcome")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                x='DIED', y='AGE', data=filtered_data, 
                showfliers=False, 
                palette={'1': '#e74c3c', '0': '#2ecc71'},
                ax=ax
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Survived', 'Died'])
            ax.set_ylabel('Age')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Calculate and annotate p-value
            _, p_value = stats.ttest_ind(
                filtered_data.loc[filtered_data['DIED'] == 0, 'AGE'],
                filtered_data.loc[filtered_data['DIED'] == 1, 'AGE'],
                equal_var=False
            )
            ax.set_title(f"Age Distribution by Mortality Status\n(p = {p_value:.3f})")
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
    
        with col2:
            
            st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
            st.subheader("Mortality Rate by Comorbidity")
            
            comorbidity_mortality = []
            for comorbidity in selected_comorbidities:
                # Calculate mortality rate for patients WITH the comorbidity
                mortality_rate = filtered_data[filtered_data[comorbidity] == 1]['DIED'].mean()
                comorbidity_mortality.append({
                    'Comorbidity': comorbidity,
                    'Mortality Rate': mortality_rate
                })
            
            plot_df = pd.DataFrame(comorbidity_mortality)
    
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                y='Comorbidity', 
                x='Mortality Rate', 
                data=plot_df.sort_values('Mortality Rate', ascending=False),
                palette='Reds_r',
                ax=ax
            )
            ax.set_xlabel('Mortality Rate')
            ax.set_ylabel('')
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.1))
            ax.grid(axis='x', linestyle='--', alpha=0.3)

            # Add value annotations
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_width():.1%}", 
                    (p.get_width(), p.get_y() + p.get_height()/2),
                    xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center'
                )
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
    
            
    with tab2:
        st.header("Comorbidity Prevalence and Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
            st.subheader("Comorbidity Prevalence")
            
            # Calculate percentages
            comorbidity_data = filtered_data[selected_comorbidities].replace({2: 0, 1: 1})
            comorbidity_stats = comorbidity_data.mean().sort_values(ascending=False) * 100
            comorbidity_counts = comorbidity_data.sum()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=comorbidity_stats.values, 
                y=comorbidity_stats.index, 
                palette='Blues_r',
                ax=ax
            )
            
            # Add value labels
            for i, (value, count) in enumerate(zip(comorbidity_stats.values, comorbidity_counts)):
                ax.text(
                    value + 2, i, 
                    f'{value:.1f}% ({count:,})', 
                    va='center',
                    fontsize=10
                )
        
            ax.set_xlabel('Percentage of Patients')
            ax.set_ylabel('')
            ax.set_xlim(0, 100)
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
    
        with col2:
            st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
            st.subheader("Comorbidity Combinations")
            
            # Create a correlation matrix
            corr_matrix = filtered_data[selected_comorbidities].replace({2: 0, 1: 1}).corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm',
                center=0,
                vmin=-1, 
                vmax=1,
                linewidths=0.5,
                square=True,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
            ax.set_title("Comorbidity Co-occurrence Patterns")
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.header("Clinical Outcomes and Resource Utilization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
            st.subheader("ICU Admission Rates")
            
            # Prepare data
            icu_analysis = filtered_data.copy()
            classification_map = {1: 'COVID', 2: 'Other'}
            icu_map = {1: 'ICU', 2: 'No ICU'}
            
            cross_tab = pd.crosstab(
                icu_analysis['CLASIFFICATION_BINARY'],
                icu_analysis['ICU'],
                normalize='index'
            ).rename(
                index=classification_map,
                columns=icu_map
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            cross_tab.plot(
                kind='bar', 
                stacked=True,
                color=['#e74c3c', '#2ecc71'],  # Red=ICU, Green=No ICU
                edgecolor='black',
                width=0.8,
                ax=ax
            )
        
            for container in ax.containers:
                ax.bar_label(
                    container, 
                    label_type='center',
                    fmt='%.1f%%',
                    color='white',
                    fontweight='bold',
                    fontsize=10
                )
            
            ax.set_xlabel('Diagnosis')
            ax.set_ylabel('Proportion')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.legend(title='ICU Status', bbox_to_anchor=(1.05, 1))
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
    
        with col2:
            st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
            st.subheader("Intubation in ICU Patients")
            
            covid_icu = filtered_data[filtered_data['ICU'] == 1].copy()
            intubed_map = {1: 'Intubated', 2: 'Not Intubated'}
            intubed_dist = covid_icu['INTUBED'].map(intubed_map).value_counts(normalize=True).mul(100)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=intubed_dist.index,
                y=intubed_dist.values,
                palette=['#e74c3c', '#3498db'],
                saturation=0.9,
                ax=ax
            )
            
            for p in ax.patches:
                ax.annotate(
                    f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    fontsize=12
                )
            
            ax.set_xlabel('Intubation Status')
            ax.set_ylabel('Percentage')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.set_title(f"Total ICU Patients: {len(covid_icu):,}")
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    # Advanced Metrics Section - 
    with tab4:
        st.header("ICU Treatment Outcomes")
        icu_data = filtered_data[filtered_data['ICU'] == 1]
            
        col1, col2 = st.columns(2)
        with col1:
            # Intubation Prevalence
            st.markdown("##### 📊 Intubation Prevalence in ICU")
            st.caption("Percentage of ICU patients who were intubated vs. not intubated")
                
            intubated = icu_data['INTUBED'].value_counts(normalize=True).mul(100)
            fig, ax = plt.subplots()
            intubated.plot(kind='bar', color=['#e63946', '#1d3557'], ax=ax)
            ax.set_xticklabels(['Intubated', 'Not Intubated'], rotation=0)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}%', 
                               (p.get_x() + p.get_width()/2, p.get_height()),
                               ha='center', va='center', xytext=(0, 5), 
                               textcoords='offset points')
            st.pyplot(fig)
    
        with col2:
            # Mortality Outcomes
            st.markdown("##### ⚠️ Mortality by Intubation Status")
            st.caption("Death rates among intubated vs. non-intubated ICU patients")
    
            outcomes = icu_data.groupby('INTUBED')['DIED'].mean().mul(100)
            fig, ax = plt.subplots()
            outcomes.plot(kind='bar', color=['#457b9d', '#a8dadc'], ax=ax)
            ax.set_xticklabels(['Intubated', 'Not Intubated'], rotation=0)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}%', 
                               (p.get_x() + p.get_width()/2, p.get_height()),
                               ha='center', va='center', xytext=(0, 5), 
                               textcoords='offset points')
            st.pyplot(fig)
        
        
elif page == "🤖 ML Prediction":
    # --------------------------------- ML Prediction Page ---------------------------------
    st.title("🤖 Mortality Risk Prediction Model")
    st.markdown("""
    This predictive model estimates mortality risk for COVID-19 patients based on their clinical characteristics.
    The model uses XGBoost with SMOTE for handling class imbalance.
    """)

    # Define the feature columns in the correct order
    feature_columns = [
        'AGE', 'SEX', 'PREGNANT', 'PNEUMONIA', 'DIABETES', 'HIPERTENSION', 
        'OBESITY', 'TOBACCO', 'CARDIOVASCULAR', 'RENAL_CHRONIC', 'COPD', 
        'ASTHMA', 'INMSUPR', 'PATIENT_TYPE', 'ICU', 'INTUBED', 'CLASIFFICATION_BINARY' ]

    try:
        pipeline = joblib.load("pipeline.pkl")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    st.subheader("📋 Patient Risk Assessment")
    
    with st.form("prediction_form"):
        st.markdown("#### Enter Patient Characteristics")
        col1, col2 = st.columns(2)
    
        with col1:
            age = st.slider("Age", min_value=0, max_value=120, value=50)
            sex = st.selectbox("Sex", options=["Male", "Female"])
            pregnant = st.selectbox("Pregnant", options=["No", "Yes"]) if sex == "Female" else "No"
            pneumonia = st.selectbox("Pneumonia", options=["No", "Yes"])
            diabetes = st.selectbox("Diabetes", options=["No", "Yes"])
            hypertension = st.selectbox("Hypertension", options=["No", "Yes"])
            obesity = st.selectbox("Obesity", options=["No", "Yes"])
            tobacco = st.selectbox("Tobacco Use", options=["No", "Yes"])
        
        with col2:
            cardiovascular = st.selectbox("Cardiovascular Disease", options=["No", "Yes"])
            renal_chronic = st.selectbox("Chronic Renal Disease", options=["No", "Yes"])
            copd = st.selectbox("COPD", options=["No", "Yes"])
            asthma = st.selectbox("Asthma", options=["No", "Yes"])
            inmsupr = st.selectbox("Immunosuppressed", options=["No", "Yes"])
            patient_type = st.selectbox("Patient Type", options=["Outpatient", "Inpatient"])
            icu = st.selectbox("ICU Admission", options=["No", "Yes"])
            intubed = st.selectbox("Intubated", options=["No", "Yes"])
            classification = st.selectbox("Covid", options=["Other", "Yes"])
            
        submitted = st.form_submit_button("Predict Mortality Risk")

    if submitted:
        # Prepare input data
        input_data = {
            'AGE': age,
            'SEX': 1 if sex == "Female" else 0,
            'PREGNANT': 1 if pregnant == "Yes" else 0,
            'PNEUMONIA': 1 if pneumonia == "Yes" else 0,
            'DIABETES': 1 if diabetes == "Yes" else 0,
            'HIPERTENSION': 1 if hypertension == "Yes" else 0,
            'OBESITY': 1 if obesity == "Yes" else 0,
            'TOBACCO': 1 if tobacco == "Yes" else 0,
            'CARDIOVASCULAR': 1 if cardiovascular == "Yes" else 0,
            'RENAL_CHRONIC': 1 if renal_chronic == "Yes" else 0,
            'COPD': 1 if copd == "Yes" else 0,
            'ASTHMA': 1 if asthma == "Yes" else 0,
            'INMSUPR': 1 if inmsupr == "Yes" else 0,
            'PATIENT_TYPE': 1 if patient_type == "Inpatient" else 0,
            'ICU': 1 if icu == "Yes" else 0,
            'INTUBED': 1 if intubed == "Yes" else 0,
            'CLASIFFICATION_BINARY': 1 if classification == "Yes" else 0
            
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            # Make prediction using the full pipeline
            proba = pipeline.predict_proba(input_df)[0][1]
            prediction = pipeline.predict(input_df)[0]
    
            # Display results
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            risk_color = "red" if prediction == 1 else "green"
            st.markdown(f"""
            <div style='border: 2px solid {risk_color}; padding: 1rem; border-radius: 0.5rem;'>
                <h3 style='color: {risk_color};'>Predicted Mortality Risk: {proba:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate summary message
            risk_factors = []
            if age >= 60:
                risk_factors.append(f"advanced age ({age} years)")
            if pneumonia == "Yes":
                risk_factors.append("pneumonia")
            if icu == "Yes":
                risk_factors.append("ICU admission")
            if intubed == "Yes":
                risk_factors.append("intubation")
            if patient_type == "Inpatient":
                risk_factors.append("inpatient status")
            if classification == "Yes":
                risk_factors.append("Covid Diagnosis")

            other_factors = []
            for condition, name in [
                (diabetes, "diabetes"),
                (hypertension, "hypertension"),
                (obesity, "obesity"),
                (cardiovascular, "cardiovascular disease"),
                (renal_chronic, "chronic renal disease"),
                (copd, "COPD"),
                (asthma, "asthma"),
                (tobacco, "tobacco use"),
                (inmsupr, "immunosuppression"),
                (pregnant, "pregnancy") if sex == "Female" else (None, None)
            ]:
                if condition == "Yes" and name is not None:
                    other_factors.append(name)
            
            st.markdown("### Risk Factor Summary")
            if risk_factors:
                st.warning(f"**Primary risk factors**: {', '.join(risk_factors)}")
            else:
                st.info("No primary risk factors identified")
        
            if other_factors:
                st.info(f"**Other comorbidities**: {', '.join(other_factors)}")
            
            if prediction == 1:
                st.error("""
                **High Risk Alert**: This patient has a high predicted mortality risk. 
                Consider intensive monitoring and early intervention strategies.
                """)
            else:
                st.success("""
                **Lower Risk Profile**: This patient has a lower predicted mortality risk, 
                but continued monitoring is recommended based on their clinical characteristics.
                """)
        
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.stop()           
          
# Footer
st.markdown("---")
st.markdown("""
**Data Source**: [COVID-19 Dataset (Mexican Government)]  
**Last Updated**: {}  
**Made by**: Reem Saad - 2025
""".format(pd.to_datetime('today').strftime('%Y-%m-%d')))
