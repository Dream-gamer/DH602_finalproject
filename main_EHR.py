import pandas as pd
import numpy as np 
import json
from icd_conversion import ICDCodeConversion

class PatientDataProcessor:
    def __init__(self, subject_id, hadm_id):
        self.subject_id = subject_id
        self.hadm_id = hadm_id
    
    def process_data(self):
        # Loading necessary dataframes here
        admissions = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/admissions.csv.gz", compression='gzip')
        patients = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/patients.csv.gz", compression='gzip')
        icustays = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/icu/icustays.csv.gz", compression='gzip')
        emar = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/emar.csv.gz", compression="gzip")
        d_items = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/icu/d_items.csv.gz", compression="gzip")
        inputevents = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/icu/inputevents.csv.gz", compression="gzip")
        procedureevents = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/icu/procedureevents.csv.gz", compression="gzip")
        poe = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/poe.csv.gz", compression="gzip")
        poe_detail = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/poe_detail.csv.gz", compression="gzip")
        procedures_icd = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/procedures_icd.csv.gz", compression="gzip")
        d_icd_procedures = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/d_icd_procedures.csv.gz", compression="gzip")
        diagnoses_icd = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz", compression="gzip")
        d_icd_diagnoses = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz", compression="gzip")
        transfers = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/transfers.csv.gz", compression="gzip")
        drgcodes = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/drgcodes.csv.gz", compression="gzip")
        hcpcsevents = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/hcpcsevents.csv.gz", compression="gzip")
        labevents = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/labevents.csv.gz", compression="gzip")
        microbiologyevents = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/microbiologyevents.csv.gz", compression="gzip")
        omr = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/omr.csv.gz", compression="gzip")
        pharmacy = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/pharmacy.csv.gz", compression="gzip")
        prescriptions = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/prescriptions.csv.gz", compression="gzip")
        services = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/hosp/services.csv.gz", compression="gzip")
        chartevents = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/icu/chartevents.csv.gz", compression="gzip")
        caregiver = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/icu/caregiver.csv.gz", compression="gzip")
        outputevents = pd.read_csv("/home/sael/EHR/mimiciv_main/physionet.org/files/mimiciv/2.2/icu/outputevents.csv.gz", compression="gzip")
        # Making a list of all the dataframe variables
        df_list = [
            icustays,
            emar,
            d_items,
            inputevents,
            procedureevents,
            poe,
            poe_detail,
            procedures_icd,
            d_icd_procedures,
            diagnoses_icd,
            d_icd_diagnoses,
            transfers,
            drgcodes,
            hcpcsevents,
            labevents,
            microbiologyevents,
            omr,
            pharmacy,
            prescriptions,
            services,
            chartevents,
            caregiver,
            outputevents
        ]
        # Converting all datetime columns into the relevant data type
        for df in df_list:
            time_columns = df.filter(like='time').columns
            dod_columns = df.filter(like='dod').columns
            df[time_columns] = df[time_columns].apply(pd.to_datetime, errors='coerce')
            df[dod_columns] = df[dod_columns].apply(pd.to_datetime, errors='coerce')

        # Creating the main identification id
        main_id = self.subject_id + "_" + self.hadm_id

        # Creating a main dataframe for the main_id consisting of all the available information from the records
        final_df = pd.merge(admissions, patients, on='subject_id', how='left')
        final_df["identifier"] = final_df["subject_id"].astype(str) + "_" + final_df["hadm_id"].astype(str)
        final_df = final_df[final_df["identifier"] == main_id]
        for df in df_list:
            if "subject_id" in df.columns and "hadm_id" in df.columns:
                df["identifier"] = df["subject_id"].astype(str) + "_" + df["hadm_id"].astype(str)
                df = df[df["identifier"] == main_id].copy()
                df_cols = set(df.columns)
                all_cols = set(final_df.columns)
                req_cols = (df_cols - all_cols) | {"identifier"}
                df = df[list(req_cols)].copy()
                final_df = final_df.merge(df, on="identifier", how="left")
                final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        
        # Loading the mapping dictionary
        with open("/home/sael/EHR/physionet.org/cat_to_column_mapping.json", "r") as file:
            mapping_cat_to_cols = json.load(file)
        
        # Making a list of the relevant self.categories for the template
        self.categories = ['Patient History', 'Patient Demographics', 'Primary Diagnoses', 'Intervention', 
        'Discharge Location (to)', 'Medications during treatment', 'Post-discharge info (follow-ups, investigations, etc)', 
        'Treatment', 'Death Status']

        # Creating the main dictionary containing information grouped under the template self.categories
        patient_info = {}
        for cat in self.categories:
            cat_cols = set(mapping_cat_to_cols[cat])
            all_cols = set(final_df.columns)
            matching_cols = all_cols.intersection(cat_cols)
            patient_info[cat] = final_df[list(matching_cols)].copy()

        patient_df = self.convert_to_dataframe(patient_info)
        patient_df.iloc[0, 1] = "Unique ID"
        indexed_dict = {}
        for row in range(len(df)):
            indexed_dict[df.iloc[row, 1]] = df.iloc[row, 2:]
        tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
        model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b")
        discharge_summary = generate_discharge_summary(main_id, indexed_dict, self.categories, model, tokenizer)
        return discharge_summary

    def convert_to_dataframe(self, patient_info):
        dataframes = []
        multiindex_tuples = []
        for cat, df in patient_info.items():
            multiindex_tuples.extend([(cat, col) for col in df.columns])
            dataframes.append(df)
        merged_df = pd.concat(dataframes, axis=1)
        merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
        merged_df.insert(0, "Patient_ID", main_id)
        icd_dict_path = "/home/sael/ICD9CMtoICD10CM/icd9to10dictionary.txt"
        icd_converter = ICDCodeConversion(icd_dict_path, merged_df)
        merged_df = icd_converter.create_icd9_to_icd10_dictionary()
        return merged_df
    
    def retrieval(self, id, indexed_dict):
        if id in indexed_dict:
            return indexed_dict[id]
        else:
            print("Entered ID doesn't exist in the records")
            return None
    
    def generate_discharge_summary(self, id, indexed_dict, categories, model, tokenizer):
        # Retrieving information for the input ID (subject_id + hadm_id)
        patient_info_dict = self.retrieval(id, indexed_dict)
        if patient_info_dict is None:
            return "Error: Patient ID not found in records."

        # tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
        # model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b")
        # Generating text for each category and then concatenating it one below the other
        discharge_summary = ""

        # Iterating through each category
        for cat in self.categories:
            # Getting information for the DST category
            category_info = patient_info_dict.get(cat)
            if category_info is not None:
            prompt = f'''You are a medical AI agent tasked with generating a discharge summary based on the provided reference information about the patient.
                        The reference text is: \n {category_info} \n This text contains information of a patient's Electronic Health records for the category - {cat}.
                        Your task is to generate a coherent passage summarizing all the available information effectively, ensuring all important details are included in sentences and not in tabular form.
                        '''
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, temperature = 0.7, max_length = 700, repetition_penalty = 1.9, pad_token_id=tokenizer.eos_token_id)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Combining category title and the generated information
            discharge_summary += f"{cat} : {generated_text}\n"

        return discharge_summary

