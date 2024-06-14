import os, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from collections import OrderedDict
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold

from skimage.transform import rescale
from skimage.util import random_noise
from skimage.transform import rotate

#import elasticdeform

from scipy.ndimage.measurements import center_of_mass
from scipy.stats.mstats import winsorize


def affirm_dm(val):
    val = str(val)
    return np.any([v in val.lower() for v in ('no', 'distant metastasis')])

def affirm_lr(val):
    val = str(val)
    return np.any([v in val.lower() for v in ('no', 'locoregional')])

def retrieve_patients(csv_dir, dataset='HNSCC'):
    """
    csv_dir: the directory that contains all of the CSV files, namely the one with the clinical info
    """
    # feature csv locations, genomic info is stored in the clinical info csv

    if dataset == 'HNSCC':
        base_clinical_info = pd.read_csv(os.path.join(csv_dir, 'Patient_and_Treatment_Characteristics.csv'))

        updated_clinical_info = pd.read_csv(os.path.join(csv_dir, 'Radiomics_Outcome_Prediction_in_OPC_ASRM_corrected.csv'))

        base_clinical_info.set_index('TCIA code', inplace=True)
        updated_clinical_info.set_index('TCIA Radiomics dummy ID of To_Submit_Final', inplace=True)

        base_clinical_info['survival_dm'] = base_clinical_info['Disease-free interval (months)'] / 12.
        updated_clinical_info['survival_dm'] = updated_clinical_info['Freedom from distant metastasis_duration of Merged updated ASRM V2'] / 365.
        base_clinical_info['survival_lr'] = base_clinical_info['Disease-free interval (months)'] / 12.
        updated_clinical_info['survival_lr'] = updated_clinical_info['Locoregional control_duration of Merged updated ASRM V2'] / 365.
        base_clinical_info['has_dm'] = [affirm_dm(v) for v in base_clinical_info['Site of recurrence (Distal/Local/ Locoregional)']] 
        updated_clinical_info['has_dm'] = [affirm_dm(v) for v in updated_clinical_info['Freedom from distant metastasis']] 
        base_clinical_info['has_lr'] = [affirm_lr(v) for v in base_clinical_info['Site of recurrence (Distal/Local/ Locoregional)']] 
        updated_clinical_info['has_lr'] = [affirm_lr(v) for v in updated_clinical_info['Locoregional control']] 

        clinical_info = updated_clinical_info.join(base_clinical_info, how='outer', lsuffix='_updated', rsuffix='_base')

        clinical_info['survival_dm'] = clinical_info['survival_dm_updated'].combine_first(clinical_info['survival_dm_base'])
        clinical_info['has_dm'] = clinical_info['has_dm_updated'].combine_first(clinical_info['has_dm_base']) 
        clinical_info['survival_lr'] = clinical_info['survival_lr_updated'].combine_first(clinical_info['survival_lr_base'])
        clinical_info['has_lr'] = clinical_info['has_lr_updated'].combine_first(clinical_info['has_lr_base']) 
        
        patients = clinical_info[['survival_dm', 'has_dm', 'has_lr', 'survival_lr']]
   
    if dataset == 'UTSW_HNC':
        clinical_info = pd.read_excel(Path(csv_dir).joinpath('final_list_clinical.xlsx'))
        clinical_info.set_index('IDA', inplace=True)

        treatment_start = pd.to_datetime(clinical_info['Treatment Start Date'], format='%m/%d/%Y')

        recurrence_date = pd.to_datetime(clinical_info['Recurrence Date'], format='%m/%d/%Y')

        time_to_recurrence = (recurrence_date - treatment_start)/ pd.Timedelta("365 days")

        patients = time_to_recurrence.rename('survival_dm', inplace=True)

        patients.index = patients.index.astype(str)

    if dataset == 'RADCURE':
        clinical_info = pd.read_excel(Path(csv_dir).joinpath('RADCURE-DA-CLINICAL-2.xlsx'))
        clinical_info.set_index('patient_id', inplace=True)

        treatment_start = pd.to_datetime(clinical_info['RT Start'], format='%m/%d/%Y')
        recurrence_date = pd.to_datetime(clinical_info['Date Distant'], format='%m/%d/%Y')

        time_to_recurrence = (recurrence_date - treatment_start) / pd.Timedelta("365 days")

        patients = time_to_recurrence.rename('survival_dm')
        
        patients = pd.concat([patients, clinical_info['RADCURE-challenge']], axis=1)

    return patients


def split_image(seed=42):
    """
    splits and scales input dataframe# and outputs as ndarray, assumes binary categories in the first two columns of the dataframe
    """
    # separate out the inputs and lab#els 
    #y = df[['Methylated', 'Unmethylated']]

    modality_dir = os.path.join(image_dir, f"numpy_conversion_{modality}_augmented_channels")
    dsc_dir = os.path.join(image_dir, f"numpy_conversion_DSC_augmented_channels")

    patients = retrieve_patients(csv_dir, modality_dir, modality='npy', classifier='MGMT')

    if n_cat==1:
        mod_y = patients[['Methylated']]
        dsc_y = dsc_patients[['Methylated']]
    else:
        mod_y = patients
        dsc_y = dsc_patients

    y_columns = mod_y.columns.to_list()
    dsc_X = dsc_patients.index
    dsc_y = dsc_y.to_numpy()
    
    # Separate into train and test datasets.
    # train_test_split automatically shuffles and splits the data following predefined sizes can revisit if shuffling is not a good idea
    X_dsc_train, X_dsc_test, y_dsc_train, y_dsc_test = train_test_split(dsc_X, dsc_y, test_size=0.3, random_state=42, stratify=dsc_y)
    X_nok_train, X_nok_val, y_nok_train, y_nok_val = train_test_split(X_dsc_train, y_dsc_train, test_size=0.25, random_state=seed, stratify=y_dsc_train)
    
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    kfold.get_n_splits(X_kfold, y_kfold)
    y_test = pd.DataFrame(y_test, index=X_test, columns=y_columns)
    y_kfold = pd.DataFrame(y_kfold, index=X_kfold, columns=y_columns)

    return X_test, y_test, kfold, X_kfold, y_kfold
