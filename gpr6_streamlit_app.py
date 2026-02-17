"""
GPR6 Bioactivity Prediction Web Application
============================================

A Streamlit web application for predicting pIC50 values of chemical compounds
targeting GPR6 (G-protein coupled receptor 6) using machine learning models
trained on ChEMBL bioactivity data.

Author: Computational Drug Discovery Pipeline
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from io import StringIO
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="GPR6 Bioactivity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects."""
    deployment_dir = 'gpr6_deployment_models'
    
    models = {}
    
    try:
        # Load regression models
        with open(os.path.join(deployment_dir, 'model_rf_regressor.pkl'), 'rb') as f:
            models['rf_regressor'] = pickle.load(f)
        
        with open(os.path.join(deployment_dir, 'model_gb_regressor.pkl'), 'rb') as f:
            models['gb_regressor'] = pickle.load(f)
        
        with open(os.path.join(deployment_dir, 'model_svr.pkl'), 'rb') as f:
            models['svr'] = pickle.load(f)
        
        with open(os.path.join(deployment_dir, 'model_ridge.pkl'), 'rb') as f:
            models['ridge'] = pickle.load(f)
        
        # Load classification model
        with open(os.path.join(deployment_dir, 'model_rf_classifier.pkl'), 'rb') as f:
            models['rf_classifier'] = pickle.load(f)
        
        # Load preprocessing
        with open(os.path.join(deployment_dir, 'scaler.pkl'), 'rb') as f:
            models['scaler'] = pickle.load(f)
        
        # Load metadata
        with open(os.path.join(deployment_dir, 'descriptor_names.pkl'), 'rb') as f:
            models['descriptor_names'] = pickle.load(f)
        
        with open(os.path.join(deployment_dir, 'fingerprint_config.pkl'), 'rb') as f:
            models['fingerprint_config'] = pickle.load(f)
        
        with open(os.path.join(deployment_dir, 'dataset_stats.pkl'), 'rb') as f:
            models['dataset_stats'] = pickle.load(f)
        
        with open(os.path.join(deployment_dir, 'model_metrics.pkl'), 'rb') as f:
            models['model_metrics'] = pickle.load(f)
        
        return models
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: Model files not found. {str(e)}")
        st.info("Please ensure the 'gpr6_deployment_models' directory exists with all pickle files.")
        return None

# ============================================================================
# MOLECULAR DESCRIPTOR CALCULATION
# ============================================================================

def calculate_physicochemical_descriptors(smiles):
    """Calculate 17 physicochemical descriptors from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    try:
        descriptors = [
            Descriptors.MolWt(mol),                # Molecular weight
            Descriptors.MolLogP(mol),              # Lipophilicity
            Descriptors.TPSA(mol),                 # Topological polar surface area
            Lipinski.NumHDonors(mol),              # Hydrogen bond donors
            Lipinski.NumHAcceptors(mol),           # Hydrogen bond acceptors
            Lipinski.NumRotatableBonds(mol),       # Rotatable bonds
            Lipinski.RingCount(mol),               # Total ring count
            Lipinski.NumAromaticRings(mol),        # Aromatic rings
            Lipinski.NumAliphaticRings(mol),       # Aliphatic rings
            Descriptors.NumValenceElectrons(mol),  # Valence electrons
            Lipinski.HeavyAtomCount(mol),          # Heavy atom count
            Descriptors.FractionCSP3(mol),         # Fraction of sp3 carbons
            Descriptors.LabuteASA(mol),            # Labute ASA
            Descriptors.MolMR(mol),                # Molar refractivity
            Descriptors.BalabanJ(mol),             # Balaban J index
            Descriptors.BertzCT(mol),              # Bertz complexity
            Descriptors.HallKierAlpha(mol)         # Hall-Kier alpha
        ]
        return descriptors
    except:
        return None

def calculate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Calculate Morgan fingerprint from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    try:
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None

def prepare_features(smiles, descriptor_names, fingerprint_config):
    """Prepare feature vector for prediction."""
    # Calculate descriptors
    descriptors = calculate_physicochemical_descriptors(smiles)
    if descriptors is None:
        return None
    
    # Calculate fingerprint
    fp = calculate_morgan_fingerprint(
        smiles,
        radius=fingerprint_config['radius'],
        n_bits=fingerprint_config['n_bits']
    )
    if fp is None:
        return None
    
    # Combine features
    features = np.concatenate([descriptors, fp])
    
    return features

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_pic50(smiles, models, model_name='ensemble'):
    """Predict pIC50 value for a given SMILES string."""
    
    # Prepare features
    features = prepare_features(
        smiles,
        models['descriptor_names'],
        models['fingerprint_config']
    )
    
    if features is None:
        return None, None
    
    # Reshape for prediction
    X = features.reshape(1, -1)
    
    # Scale features (for models that need it)
    X_scaled = models['scaler'].transform(X)
    
    predictions = {}
    
    # Random Forest Regressor
    pred_rf = models['rf_regressor'].predict(X)[0]
    predictions['Random Forest'] = pred_rf
    
    # Gradient Boosting Regressor
    pred_gb = models['gb_regressor'].predict(X)[0]
    predictions['Gradient Boosting'] = pred_gb
    
    # SVR (needs scaled features)
    pred_svr = models['svr'].predict(X_scaled)[0]
    predictions['SVR'] = pred_svr
    
    # Ridge (needs scaled features)
    pred_ridge = models['ridge'].predict(X_scaled)[0]
    predictions['Ridge'] = pred_ridge
    
    # Ensemble (average of top 3)
    pred_ensemble = (pred_rf + pred_gb + pred_svr) / 3
    predictions['Ensemble'] = pred_ensemble
    
    # Classification
    prob_active = models['rf_classifier'].predict_proba(X)[0][1]
    
    return predictions, prob_active

def classify_bioactivity(pic50):
    """Classify compound as Active/Inactive based on pIC50."""
    if pic50 <= 6:
        return 'Inactive', '🔴'
    elif pic50 <= 8:
        return 'Intermediate', '🟡'
    else:
        return 'Active', '🟢'

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Load models
    models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check the deployment directory.")
        return
    
    # Header
    st.markdown("<div class='main-header'>🧬 GPR6 Bioactivity Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Predict pIC50 values for GPR6 inhibitors using machine learning</div>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📋 Navigation")
    app_mode = st.sidebar.radio(
        "Select Mode:",
        ["🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Information", "ℹ️ About"]
    )
    
    # ========================================================================
    # MODE 1: SINGLE PREDICTION
    # ========================================================================
    
    if app_mode == "🔍 Single Prediction":
        st.markdown("### Single Compound Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            smiles_input = st.text_input(
                "Enter SMILES String:",
                placeholder="e.g., CC(C)Cc1ccc(cc1)C(C)C(O)=O",
                help="SMILES notation for the chemical compound"
            )
        
        with col2:
            compound_id = st.text_input(
                "Compound ID (Optional):",
                placeholder="e.g., COMP_001"
            )
        
        if st.button("🔮 Predict pIC50", use_container_width=True):
            if not smiles_input:
                st.error("❌ Please enter a SMILES string")
            else:
                # Validate SMILES
                mol = Chem.MolFromSmiles(smiles_input)
                if mol is None:
                    st.error("❌ Invalid SMILES string. Please check the format.")
                else:
                    with st.spinner("Calculating molecular descriptors and making predictions..."):
                        predictions, prob_active = predict_pic50(smiles_input, models)
                        
                        if predictions is None:
                            st.error("❌ Error calculating molecular descriptors.")
                        else:
                            # Display results
                            st.markdown("---")
                            st.markdown("### 📊 Prediction Results")
                            
                            # Ensemble prediction (main result)
                            ensemble_pic50 = predictions['Ensemble']
                            bioactivity_class, emoji = classify_bioactivity(ensemble_pic50)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Predicted pIC50 (Ensemble)",
                                    f"{ensemble_pic50:.2f}",
                                    delta=None,
                                    delta_color="off"
                                )
                            
                            with col2:
                                st.metric(
                                    "Bioactivity Class",
                                    f"{emoji} {bioactivity_class}",
                                    delta=None,
                                    delta_color="off"
                                )
                            
                            with col3:
                                st.metric(
                                    "Active Probability",
                                    f"{prob_active*100:.1f}%",
                                    delta=None,
                                    delta_color="off"
                                )
                            
                            # Model comparison
                            st.markdown("### 🤖 Model Comparison")
                            
                            comparison_df = pd.DataFrame({
                                'Model': list(predictions.keys()),
                                'pIC50': list(predictions.values())
                            }).sort_values('pIC50', ascending=False)
                            
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                            
                            # Molecular properties
                            st.markdown("### 🧪 Molecular Properties")
                            
                            descriptors = calculate_physicochemical_descriptors(smiles_input)
                            descriptor_names = models['descriptor_names']
                            
                            props_df = pd.DataFrame({
                                'Property': descriptor_names,
                                'Value': [f"{d:.2f}" for d in descriptors]
                            })
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.dataframe(props_df.iloc[:9], use_container_width=True, hide_index=True)
                            with col2:
                                st.dataframe(props_df.iloc[9:], use_container_width=True, hide_index=True)
                            
                            # Lipinski's Rule of Five
                            st.markdown("### 📋 Lipinski's Rule of Five")
                            
                            mw = descriptors[0]
                            logp = descriptors[1]
                            hbd = descriptors[3]
                            hba = descriptors[4]
                            
                            lipinski_violations = 0
                            violations_text = []
                            
                            if mw > 500:
                                lipinski_violations += 1
                                violations_text.append("MW > 500")
                            if logp > 5:
                                lipinski_violations += 1
                                violations_text.append("LogP > 5")
                            if hbd > 5:
                                lipinski_violations += 1
                                violations_text.append("HBD > 5")
                            if hba > 10:
                                lipinski_violations += 1
                                violations_text.append("HBA > 10")
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("MW", f"{mw:.1f}", "✓" if mw <= 500 else "✗")
                            with col2:
                                st.metric("LogP", f"{logp:.2f}", "✓" if logp <= 5 else "✗")
                            with col3:
                                st.metric("HBD", f"{int(hbd)}", "✓" if hbd <= 5 else "✗")
                            with col4:
                                st.metric("HBA", f"{int(hba)}", "✓" if hba <= 10 else "✗")
                            with col5:
                                status = "✓ Pass" if lipinski_violations <= 1 else "✗ Fail"
                                st.metric("Violations", lipinski_violations, status)
    
    # ========================================================================
    # MODE 2: BATCH PREDICTION
    # ========================================================================
    
    elif app_mode == "📊 Batch Prediction":
        st.markdown("### Batch Compound Prediction")
        st.info("Upload a CSV file with columns: Compound_ID, SMILES")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                if 'SMILES' not in df.columns:
                    st.error("❌ CSV must contain 'SMILES' column")
                else:
                    if 'Compound_ID' not in df.columns:
                        df['Compound_ID'] = [f"COMP_{i+1}" for i in range(len(df))]
                    
                    st.markdown(f"### Processing {len(df)} compounds...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for idx, row in df.iterrows():
                        compound_id = row['Compound_ID']
                        smiles = row['SMILES']
                        
                        # Validate SMILES
                        mol = Chem.MolFromSmiles(smiles)
                        
                        if mol is None:
                            results.append({
                                'Compound_ID': compound_id,
                                'SMILES': smiles,
                                'pIC50': np.nan,
                                'Bioactivity_Class': 'Invalid',
                                'Active_Probability': np.nan,
                                'Status': 'Invalid SMILES'
                            })
                        else:
                            predictions, prob_active = predict_pic50(smiles, models)
                            
                            if predictions is None:
                                results.append({
                                    'Compound_ID': compound_id,
                                    'SMILES': smiles,
                                    'pIC50': np.nan,
                                    'Bioactivity_Class': 'Error',
                                    'Active_Probability': np.nan,
                                    'Status': 'Calculation Error'
                                })
                            else:
                                ensemble_pic50 = predictions['Ensemble']
                                bioactivity_class, _ = classify_bioactivity(ensemble_pic50)
                                
                                results.append({
                                    'Compound_ID': compound_id,
                                    'SMILES': smiles,
                                    'pIC50': f"{ensemble_pic50:.2f}",
                                    'Bioactivity_Class': bioactivity_class,
                                    'Active_Probability': f"{prob_active*100:.1f}%",
                                    'Status': 'Success'
                                })
                        
                        # Update progress
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {idx + 1}/{len(df)}")
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="gpr6_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.markdown("### 📈 Summary Statistics")
                    
                    valid_results = results_df[results_df['Status'] == 'Success']
                    
                    if len(valid_results) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Processed", len(results_df))
                        with col2:
                            st.metric("Successful", len(valid_results))
                        with col3:
                            active_count = len(valid_results[valid_results['Bioactivity_Class'] == 'Active'])
                            st.metric("Active Compounds", active_count)
                        with col4:
                            inactive_count = len(valid_results[valid_results['Bioactivity_Class'] == 'Inactive'])
                            st.metric("Inactive Compounds", inactive_count)
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # ========================================================================
    # MODE 3: MODEL INFORMATION
    # ========================================================================
    
    elif app_mode == "📈 Model Information":
        st.markdown("### Model Performance Metrics")
        
        metrics = models['model_metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Random Forest Regressor")
            st.metric("R² Score", f"{metrics['rf_r2']:.4f}")
            st.metric("RMSE", f"{metrics['rf_rmse']:.4f}")
            st.metric("MAE", f"{metrics['rf_mae']:.4f}")
        
        with col2:
            st.markdown("#### Gradient Boosting Regressor")
            st.metric("R² Score", f"{metrics['gb_r2']:.4f}")
            st.metric("RMSE", f"{metrics['gb_rmse']:.4f}")
            st.metric("MAE", f"{metrics['gb_mae']:.4f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Support Vector Regressor")
            st.metric("R² Score", f"{metrics['svr_r2']:.4f}")
            st.metric("RMSE", f"{metrics['svr_rmse']:.4f}")
            st.metric("MAE", f"{metrics['svr_mae']:.4f}")
        
        with col2:
            st.markdown("#### Ridge Regressor")
            st.metric("R² Score", f"{metrics['ridge_r2']:.4f}")
            st.metric("RMSE", f"{metrics['ridge_rmse']:.4f}")
            st.metric("MAE", f"{metrics['ridge_mae']:.4f}")
        
        st.markdown("#### Ensemble Model (Average of Top 3)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics['ensemble_r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{metrics['ensemble_rmse']:.4f}")
        with col3:
            st.metric("MAE", f"{metrics['ensemble_mae']:.4f}")
        
        # Dataset statistics
        st.markdown("---")
        st.markdown("### Dataset Statistics")
        
        stats = models['dataset_stats']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Compounds", stats['total_compounds'])
        with col2:
            st.metric("Active Compounds", stats['active_compounds'])
        with col3:
            st.metric("Inactive Compounds", stats['inactive_compounds'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("pIC50 Mean", f"{stats['pIC50_mean']:.2f}")
        with col2:
            st.metric("pIC50 Std Dev", f"{stats['pIC50_std']:.2f}")
        with col3:
            st.metric("pIC50 Min", f"{stats['pIC50_min']:.2f}")
        with col4:
            st.metric("pIC50 Max", f"{stats['pIC50_max']:.2f}")
    
    # ========================================================================
    # MODE 4: ABOUT
    # ========================================================================
    
    elif app_mode == "ℹ️ About":
        st.markdown("### About This Application")
        
        st.markdown("""
        This web application predicts the **pIC50 values** of chemical compounds targeting 
        **GPR6 (G-protein coupled receptor 6)** using machine learning models trained on 
        bioactivity data from the **ChEMBL database**.
        
        #### 🎯 Target Information
        - **Target**: GPR6 (G-protein coupled receptor 6)
        - **ChEMBL ID**: CHEMBL3714130
        - **Organism**: Homo sapiens
        - **Assay Type**: IC50 (half-maximal inhibitory concentration)
        - **Therapeutic Application**: Parkinson's disease and neurological disorders
        
        #### 🔬 Model Features
        - **Regression Models**: Random Forest, Gradient Boosting, SVR, Ridge
        - **Classification Model**: Random Forest (Active/Inactive)
        - **Molecular Descriptors**: 17 physicochemical descriptors
        - **Molecular Fingerprints**: 2048-bit Morgan fingerprints
        - **Total Features**: 2,065 features per compound
        
        #### 📊 Model Performance
        - **Ensemble R² Score**: > 0.7 (excellent predictive power)
        - **Cross-validation**: 5-fold with grid search optimization
        - **Hyperparameter Tuning**: Extensive optimization for each model
        
        #### 🧪 Molecular Descriptors
        The application calculates the following descriptors:
        1. Molecular Weight (MW)
        2. Lipophilicity (LogP)
        3. Topological Polar Surface Area (TPSA)
        4. Hydrogen Bond Donors (HBD)
        5. Hydrogen Bond Acceptors (HBA)
        6. Rotatable Bonds
        7. Ring Count
        8. Aromatic Rings
        9. Aliphatic Rings
        10. Valence Electrons
        11. Heavy Atom Count
        12. Fraction of sp3 Carbons
        13. Labute ASA
        14. Molar Refractivity
        15. Balaban J Index
        16. Bertz Complexity
        17. Hall-Kier Alpha
        
        #### 📋 Lipinski's Rule of Five
        The application checks compliance with Lipinski's Rule of Five:
        - Molecular Weight ≤ 500 Da
        - LogP ≤ 5
        - Hydrogen Bond Donors ≤ 5
        - Hydrogen Bond Acceptors ≤ 10
        
        #### 🔍 How to Use
        1. **Single Prediction**: Enter a SMILES string to predict pIC50 for one compound
        2. **Batch Prediction**: Upload a CSV file with multiple compounds
        3. **Model Information**: View detailed model performance metrics
        
        #### 📚 References
        - **RDKit**: Cheminformatics library for molecular descriptor calculation
        - **Scikit-learn**: Machine learning framework
        - **Streamlit**: Web application framework
        - **ChEMBL**: Bioactivity database
        
        #### 👨‍💻 Development
        - **Pipeline**: Computational Drug Discovery Pipeline for GPR6 Inhibitors
        - **Institution**: CBIOS - Universidade Lusofona
        - **Date**: January 2026
        
        #### ⚠️ Disclaimer
        This application is for research purposes only. Predictions should be validated 
        experimentally before use in drug discovery projects.
        """)

if __name__ == "__main__":
    main()
