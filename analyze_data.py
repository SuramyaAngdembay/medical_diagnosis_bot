import pandas as pd
import json
import os

# Read the first 100 rows of the CSV file
print("Reading the CSV file...")
df = pd.read_csv('data/release_test_patients.csv', nrows=100)

# Save the subset to a new file
subset_file = 'data/test_patients_subset.csv'
print(f"Saving subset to {subset_file}...")
df.to_csv(subset_file, index=False)

# Print basic info about the dataframe
print("\nDataFrame Info:")
print(f"Number of rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Sample a few records and print them in a readable format
print("\nSample Records:")
for i in range(min(5, len(df))):
    print(f"\nRecord {i+1}:")
    record = df.iloc[i].to_dict()
    
    # Convert differential diagnosis string to actual list if it's a string
    if isinstance(record['DIFFERENTIAL_DIAGNOSIS'], str):
        try:
            record['DIFFERENTIAL_DIAGNOSIS'] = eval(record['DIFFERENTIAL_DIAGNOSIS'])
        except:
            pass
    
    # Convert evidences string to actual list if it's a string
    if isinstance(record['EVIDENCES'], str):
        try:
            record['EVIDENCES'] = eval(record['EVIDENCES'])
        except:
            pass
    
    print(json.dumps(record, indent=2, default=str))

# Analyze the evidence codes
print("\nAnalyzing Evidence Codes:")
all_evidences = []
for evidences in df['EVIDENCES']:
    if isinstance(evidences, str):
        try:
            evidences_list = eval(evidences)
            all_evidences.extend(evidences_list)
        except:
            # Handle any parsing errors
            continue
    else:
        all_evidences.extend(evidences)

unique_evidences = set(all_evidences)
print(f"Number of unique evidence codes: {len(unique_evidences)}")
print(f"Sample evidence codes: {list(unique_evidences)[:10]}")

# Read the evidence definitions if available
evidence_file = 'data/release_evidences.json'
if os.path.exists(evidence_file):
    print("\nReading evidence definitions...")
    try:
        with open(evidence_file, 'r') as f:
            evidence_data = json.load(f)
        
        # Print a few sample evidence definitions
        print("Sample evidence definitions:")
        sample_evidences = list(unique_evidences)[:5]
        for evidence_code in sample_evidences:
            base_code = evidence_code.split('_@_')[0] if '_@_' in evidence_code else evidence_code
            evidence_def = next((item for item in evidence_data if item['name'] == base_code), None)
            if evidence_def:
                print(f"\n{evidence_code}:")
                if 'question_en' in evidence_def:
                    print(f"  Question: {evidence_def['question_en']}")
                if 'data_type' in evidence_def:
                    print(f"  Type: {evidence_def['data_type']}")
                if '_@_' in evidence_code and 'value_meaning' in evidence_def:
                    value = evidence_code.split('_@_')[1]
                    if value in evidence_def.get('value_meaning', {}):
                        print(f"  Value: {evidence_def['value_meaning'][value].get('en', value)}")
    except Exception as e:
        print(f"Error processing evidence definitions: {e}")

print("\nAnalysis complete!") 