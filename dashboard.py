# ==============================================================================
# SCRIPT FOR: FINAL HACKATHON DASHBOARD (CHAMPION XGBOOST MODEL)
# ==============================================================================

# PART 0: SETUP
# ==============================================================================
import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import joblib

# ==============================================================================
# PART 1: PAGE CONFIGURATION AND STYLING
# ==============================================================================
# Use st.set_page_config() as the first Streamlit command
st.set_page_config(
    page_title="Readmission Risk Dashboard",
    page_icon="🏥",
    layout="wide" # Use the full page width for a modern look
)

# You can inject custom CSS for styling if you want
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    .stMetric {
        border: 1px solid #2e2e2e;
        border-radius: 10px;
        padding: 15px;
        background-color: #0d1117;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# PART 2: DATA LOADING AND PROCESSING (Cached for performance)
# ==============================================================================
@st.cache_data
def load_and_process_data():
    print("--- Loading and Preprocessing Initial Data (cached) ---")
    
    code_columns = {
        'ICD9_DGNS_CD_1': str, 'ICD9_DGNS_CD_2': str, 'ICD9_DGNS_CD_3': str,
        'ICD9_DGNS_CD_4': str, 'ICD9_DGNS_CD_5': str, 'ICD9_DGNS_CD_6': str,
        'ICD9_DGNS_CD_7': str, 'ICD9_DGNS_CD_8': str, 'ICD9_DGNS_CD_9': str,
        'ICD9_DGNS_CD_10': str, 'ADMTNG_ICD9_DGNS_CD': str, 'CLM_DRG_CD': str,
        'ICD9_PRCDR_CD_1': str, 'ICD9_PRCDR_CD_2': str, 'ICD9_PRCDR_CD_3': str,
        'ICD9_PRCDR_CD_4': str, 'ICD9_PRCDR_CD_5': str, 'ICD9_PRCDR_CD_6': str
    }
    
    beneficiary_2008 = pd.read_csv("D:/Jupyter/HealthArk_data/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")
    beneficiary_2009 = pd.read_csv("D:/Jupyter/HealthArk_data/DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv")
    beneficiary_2010 = pd.read_csv("D:/Jupyter/HealthArk_data/DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv")
    
    chunk_size = 100000
    
    inpatient_agg_list, inpatient_codes_list, inpatient_readmission_list = [], [], []
    inpatient_iterator = pd.read_csv("D:/Jupyter/HealthArk_data/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv", dtype=code_columns, chunksize=chunk_size)
    for chunk in inpatient_iterator:
        inpatient_agg_list.append(chunk.groupby('DESYNPUF_ID').agg(Inpatient_Claim_Count=('CLM_ID', 'count'), Total_Inpatient_Payments=('CLM_PMT_AMT', 'sum')))
        inpatient_codes_list.append(chunk[['DESYNPUF_ID', 'ICD9_DGNS_CD_1']])
        chunk['CLM_ADMSN_DT'] = pd.to_datetime(chunk['CLM_ADMSN_DT'], format='%Y%m%d')
        chunk['CLM_THRU_DT'] = pd.to_datetime(chunk['CLM_THRU_DT'], format='%Y%m%d', errors='coerce')
        inpatient_readmission_list.append(chunk)

    inpatient_agg = pd.concat(inpatient_agg_list).groupby(level=0).sum()
    inpatient_codes = pd.concat(inpatient_codes_list)
    inpatient_claims_raw = pd.concat(inpatient_readmission_list)
    
    outpatient_agg_list, outpatient_codes_list = [], []
    outpatient_iterator = pd.read_csv("D:/Jupyter/HealthArk_data/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv", dtype=code_columns, engine='python', chunksize=chunk_size)
    for chunk in outpatient_iterator:
        outpatient_agg_list.append(chunk.groupby('DESYNPUF_ID').agg(Outpatient_Claim_Count=('CLM_ID', 'count'), Total_Outpatient_Payments=('CLM_PMT_AMT', 'sum')))
        outpatient_codes_list.append(chunk[['DESYNPUF_ID', 'ICD9_DGNS_CD_1']])
        
    outpatient_agg = pd.concat(outpatient_agg_list).groupby(level=0).sum()
    outpatient_codes = pd.concat(outpatient_codes_list)

    all_beneficiaries = pd.concat([beneficiary_2008, beneficiary_2009, beneficiary_2010], ignore_index=True)
    all_beneficiaries = all_beneficiaries.drop_duplicates(subset=['DESYNPUF_ID'], keep='last')
    
    all_beneficiaries['BENE_BIRTH_DT'] = pd.to_datetime(all_beneficiaries['BENE_BIRTH_DT'], format='%m-%d-%Y')
    all_beneficiaries['BENE_DEATH_DT'] = pd.to_datetime(all_beneficiaries['BENE_DEATH_DT'], format='%m-%d-%Y', errors='coerce')
    reference_date = datetime(2010, 12, 31)
    all_beneficiaries['Age'] = ((reference_date - all_beneficiaries['BENE_BIRTH_DT']).dt.days / 365.25).astype(int)
    all_beneficiaries['Is_Dead'] = all_beneficiaries['BENE_DEATH_DT'].notna().astype(int)
    chronic_condition_cols = [col for col in all_beneficiaries.columns if col.startswith('SP_')]
    for col in chronic_condition_cols:
        all_beneficiaries[col] = all_beneficiaries[col].replace(2, 0)
    all_beneficiaries['Chronic_Condition_Count'] = all_beneficiaries[chronic_condition_cols].sum(axis=1)
    
    master_df = all_beneficiaries.merge(inpatient_agg, on='DESYNPUF_ID', how='left')
    master_df = master_df.merge(outpatient_agg, on='DESYNPUF_ID', how='left')
    claims_cols_to_fill = ['Inpatient_Claim_Count', 'Total_Inpatient_Payments', 'Outpatient_Claim_Count', 'Total_Outpatient_Payments']
    master_df[claims_cols_to_fill] = master_df[claims_cols_to_fill].fillna(0)

    inpatient_claims_raw = inpatient_claims_raw.sort_values(by=['DESYNPUF_ID', 'CLM_ADMSN_DT'])
    inpatient_claims_raw['Next_Admission_Date'] = inpatient_claims_raw.groupby('DESYNPUF_ID')['CLM_ADMSN_DT'].shift(-1)
    days_to_next_admission = (inpatient_claims_raw['Next_Admission_Date'] - inpatient_claims_raw['CLM_THRU_DT']).dt.days
    inpatient_claims_raw['Was_Readmitted_in_30_Days'] = (days_to_next_admission <= 30).astype(int)
    readmission_summary = inpatient_claims_raw.groupby('DESYNPUF_ID')['Was_Readmitted_in_30_Days'].max().reset_index()
    readmission_summary = readmission_summary.rename(columns={'Was_Readmitted_in_30_Days': 'Had_30Day_Readmission_Ever'})
    master_df_readmission = master_df.merge(readmission_summary, on='DESYNPUF_ID', how='left')
    master_df_readmission['Had_30Day_Readmission_Ever'] = master_df_readmission['Had_30Day_Readmission_Ever'].fillna(0)
    
    all_codes = pd.concat([inpatient_codes, outpatient_codes], ignore_index=True)
    diagnosis_counts = all_codes.groupby('DESYNPUF_ID').size().reset_index(name='Total_Diagnosis_Count')
    unique_diagnosis_counts = all_codes.groupby('DESYNPUF_ID')['ICD9_DGNS_CD_1'].nunique().reset_index(name='Unique_Diagnosis_Count')
    master_df_enhanced = master_df_readmission.merge(diagnosis_counts, on='DESYNPUF_ID', how='left')
    master_df_enhanced = master_df_enhanced.merge(unique_diagnosis_counts, on='DESYNPUF_ID', how='left')
    master_df_enhanced[['Total_Diagnosis_Count', 'Unique_Diagnosis_Count']] = master_df_enhanced[['Total_Diagnosis_Count', 'Unique_Diagnosis_Count']].fillna(0)
    categorical_cols = ['BENE_SEX_IDENT_CD', 'BENE_RACE_CD']
    master_df_enhanced = pd.get_dummies(master_df_enhanced, columns=categorical_cols, drop_first=True)
    
    try:
        drug_exposure = pd.read_excel("D:/Jupyter/HealthArk_data/drug_exposure.xlsx")
        person_mapping = pd.read_excel("D:/Jupyter/HealthArk_data/person.xlsx")
        person_id_map = person_mapping[['PERSON_ID', 'PERSON_SOURCE_VALUE']].rename(columns={'PERSON_SOURCE_VALUE': 'DESYNPUF_ID'})
        drug_exposure = drug_exposure.merge(person_id_map, on='PERSON_ID', how='left')
        
        if 'DESYNPUF_ID' in drug_exposure.columns:
            drug_counts = drug_exposure.groupby('DESYNPUF_ID').size().reset_index(name='Total_Drug_Count')
            unique_drug_counts = drug_exposure.groupby('DESYNPUF_ID')['DRUG_CONCEPT_ID'].nunique().reset_index(name='Unique_Drug_Count')
            avg_days_supply = drug_exposure.groupby('DESYNPUF_ID')['DAYS_SUPPLY'].mean().reset_index(name='Avg_Days_Supply')
            master_df_final = master_df_enhanced.merge(drug_counts, on='DESYNPUF_ID', how='left')
            master_df_final = master_df_final.merge(unique_drug_counts, on='DESYNPUF_ID', how='left')
            master_df_final = master_df_final.merge(avg_days_supply, on='DESYNPUF_ID', how='left')
            drug_feature_cols = ['Total_Drug_Count', 'Unique_Drug_Count', 'Avg_Days_Supply']
            master_df_final[drug_feature_cols] = master_df_final[drug_feature_cols].fillna(0)
    except FileNotFoundError:
        print("Drug or Person files not found. Skipping drug features.")
        master_df_final = master_df_enhanced.copy()
        master_df_final[['Total_Drug_Count', 'Unique_Drug_Count', 'Avg_Days_Supply']] = 0
    
    return master_df_final

# ==============================================================================
# PART 3: MODEL TRAINING (Cached for performance)
# ==============================================================================
@st.cache_resource
def train_model(df):
    print("--- Training the Champion Model (cached) ---")
    y = df['Had_30Day_Readmission_Ever']
    features_to_drop = ['DESYNPUF_ID', 'BENE_BIRTH_DT', 'BENE_DEATH_DT', 'Had_30Day_Readmission_Ever']
    X = df.drop(columns=features_to_drop)
    X = X.select_dtypes(include=['number'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    # Using the best parameters we found during tuning
    best_params = {
        'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 7,
        'n_estimators': 300, 'subsample': 0.8
    }
    
    final_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    final_model.fit(X_train, y_train)
    
    feature_importances = pd.DataFrame(final_model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
    
    return final_model, feature_importances, X_test, y_test

# ==============================================================================
# PART 4: DASHBOARD LAYOUT
# ==============================================================================
# Load data and train model
master_df_final = load_and_process_data()
if master_df_final is not None:
    final_model, feature_importances, X_test, y_test = train_model(master_df_final)

    # --- Header ---
    st.title("🏥 Healthcare Readmission Risk Dashboard")
    st.write("An AI-driven solution to identify patients at high risk of 30-day hospital readmission.")

    # --- KPIs ---
    st.header("Model Performance Highlights")
    from sklearn.metrics import classification_report
    y_pred = final_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Accuracy", f"{report['accuracy']:.2%}")
    col2.metric("Readmission F1-Score", f"{report['1.0']['f1-score']:.2%}")
    col3.metric("Readmission Recall", f"{report['1.0']['recall']:.2%}")
    col4.metric("Readmission Precision", f"{report['1.0']['precision']:.2%}")

    st.markdown("---")

    # --- Visualizations in a two-column layout ---
    st.header("Key Insights")
    c1, c2 = st.columns((1, 1))

    with c1:
        st.subheader("Top Predictors of Readmission")
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y=feature_importances.index[:15], data=feature_importances.head(15), hue=feature_importances.index[:15], palette='viridis', legend=False, ax=ax1)
        ax1.set_xlabel('Importance Score', fontsize=12)
        ax1.set_ylabel('')
        st.pyplot(fig1)

    with c2:
        st.subheader("Readmission Rate by Chronic Condition")
        chronic_condition_cols = [
            'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN', 
            'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA'
        ]
        readmission_rates = {}
        for col in chronic_condition_cols:
            has_condition = master_df_final[col] == 1
            rate = master_df_final.loc[has_condition, 'Had_30Day_Readmission_Ever'].mean()
            clean_name = col.replace('SP_', '')
            readmission_rates[clean_name] = rate
        rates_series = pd.Series(readmission_rates).sort_values(ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.barplot(x=rates_series.values, y=rates_series.index, hue=rates_series.index, palette='plasma', legend=False, ax=ax2)
        ax2.set_xlabel('Readmission Rate', fontsize=12)
        ax2.set_ylabel('')
        ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        st.pyplot(fig2)

    # --- Actionable Recommendations ---
    st.markdown("---")
    st.header("Actionable Recommendations")
    top_feature = feature_importances.index[0]
    st.markdown(f"""
    Based on the model's insights, we can formulate targeted strategies:
    - **Focus on High-Risk Conditions**: The charts show that patients with certain chronic conditions have a significantly higher readmission rate. Implement post-discharge follow-up programs specifically for these groups.
    - **Monitor Key Predictors**: The most important feature for predicting readmission is **{top_feature}**. A patient's score on this metric should be a primary factor in their risk assessment.
    - **Develop a Risk Scorecard**: Use the top 5-10 features to create a simple risk scorecard that clinicians can use at the point of discharge to identify high-risk patients who may need additional resources.
    """)

else:
    st.error("Data could not be loaded. Dashboard cannot be displayed.")