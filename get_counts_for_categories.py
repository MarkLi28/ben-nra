from collections import defaultdict
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()

client = OpenAI()

file_path = "agentclinic_medqa.jsonl"


male_pattern = re.compile(r"\b(male|man|boy)\b", re.IGNORECASE)
female_pattern = re.compile(r"\b(female|woman|girl)\b", re.IGNORECASE)
age_pattern = re.compile(r"(\d+)[-\s]*year", re.IGNORECASE)


gender_bins = defaultdict(int)
age_bins = defaultdict(int)
smoke_bins = defaultdict(int)
alcohol_bins = defaultdict(int)
drug_bins = defaultdict(int)
occupation_bins = defaultdict(int)
ses_bins = defaultdict(int)
family_support_bins = defaultdict(int)
rare_meds_bins = defaultdict(int)
comorbidity_bins = defaultdict(int)
symptom_presentation_bins = defaultdict(int)


def classify_bias_llm(demographics, social_history, past_medical_history, history_of_present_illness):
    prompt = f"""
You are a careful medical data classifier.

Given the following patient Demographics, Social_History, Past_Medical_History, and History_of_Present_Illness, classify each bias variable into one of the allowed categories below. If there is no clear mention, return "Unknown" for that variable.

---

Allowed values:

- Smoking Status: ["Smoker", "Non-smoker", "Unknown"]
- Alcohol Use: ["Drinker", "Non-drinker", "Unknown"]
- Drug Use: ["Drug User", "Non-drug User", "Unknown"]
- Occupation Type: ["Manual Labor", "Knowledge Worker", "Student", "Retired", "Unemployed", "Unknown"]
- SES Proxy: ["Lower SES Proxy", "Higher SES Proxy", "Unknown"]
- Family Support: ["Strong Support", "Limited/No Support", "Unknown"]
- Rare Medication: ["Present", "Absent", "Unknown"]
- Comorbidity Status: ["Hypertension", "Diabetes Mellitus", "Cancer", "Other", "None", "Unknown"]
- Symptom Presentation: ["Classic Textbook", "Atypical/Vague Wording", "Multi-System Complex", "Single Symptom Only", "Unknown"]

---

Example Input:

Demographics: "65-year-old female"
Social_History: "Smokes 1 pack per day, drinks socially."
Past_Medical_History: "Hypertension, Diabetes Mellitus."
History_of_Present_Illness: "Patient has crushing chest pain radiating to arm."

Example Output:

{{
  "Smoking Status": "Smoker",
  "Alcohol Use": "Drinker",
  "Drug Use": "Unknown",
  "Occupation Type": "Unknown",
  "SES Proxy": "Unknown",
  "Family Support": "Unknown",
  "Rare Medication": "Absent",
  "Comorbidity Status": "Hypertension",
  "Symptom Presentation": "Classic Textbook"
}}

---

Now classify this text:

Demographics: "{demographics}"
Social_History: "{social_history}"
Past_Medical_History: "{past_medical_history}"
History_of_Present_Illness: "{history_of_present_illness}"

Only output strict JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a careful medical data classifier."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content.strip()
    return answer


with open(file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        patient = data['OSCE_Examination']['Patient_Actor']

        demographics = patient.get('Demographics', '').strip()
        demographics_lower = demographics.lower()

        if male_pattern.search(demographics_lower):
            gender_bins["Male"] += 1
        elif female_pattern.search(demographics_lower):
            gender_bins["Female"] += 1
        else:
            gender_bins["Other/Unknown"] += 1

        age = None
        if any(x in demographics_lower for x in ["month", "newborn", "infant"]):
            age = 0
        else:
            match = age_pattern.search(demographics_lower)
            if match:
                age = int(match.group(1))

        if age is not None:
            if age <= 10:
                age_bins["0-10"] += 1
            elif age <= 20:
                age_bins["10-20"] += 1
            elif age <= 30:
                age_bins["20-30"] += 1
            elif age <= 40:
                age_bins["30-40"] += 1
            elif age <= 50:
                age_bins["40-50"] += 1
            elif age <= 60:
                age_bins["50-60"] += 1
            else:
                age_bins["60+"] += 1

        sh = patient.get('Social_History', '')
        if isinstance(sh, dict):
            social_history = " ".join(f"{k}: {v}" for k, v in sh.items()).strip()
        elif isinstance(sh, list):
            social_history = " ".join(sh).strip()
        else:
            social_history = str(sh).strip()

        pmh = patient.get('Past_Medical_History', '')
        if isinstance(pmh, list):
            past_medical_history = " ".join(pmh).strip()
        else:
            past_medical_history = str(pmh).strip()

        hpi = patient.get('History_of_Present_Illness', '')
        if isinstance(hpi, dict):
            history_of_present_illness = " ".join(f"{k}: {v}" for k, v in hpi.items()).strip()
        elif isinstance(hpi, list):
            history_of_present_illness = " ".join(hpi).strip()
        else:
            history_of_present_illness = str(hpi).strip()

        try:
            classification = classify_bias_llm(
                demographics,
                social_history,
                past_medical_history,
                history_of_present_illness
            )

            raw_output = classification.strip()
            raw_output = re.sub(r"```[a-z]*", "", raw_output).strip()
            raw_output = re.sub(r"```", "", raw_output).strip()
            result = json.loads(raw_output)
        except json.JSONDecodeError:
            print("Invalid JSON from LLM! Raw output:", classification)
            continue


        smoke_bins[result["Smoking Status"]] += 1
        alcohol_bins[result["Alcohol Use"]] += 1
        drug_bins[result["Drug Use"]] += 1
        occupation_bins[result["Occupation Type"]] += 1
        ses_bins[result["SES Proxy"]] += 1
        family_support_bins[result["Family Support"]] += 1
        rare_meds_bins[result["Rare Medication"]] += 1
        comorbidity_bins[result["Comorbidity Status"]] += 1
        symptom_presentation_bins[result["Symptom Presentation"]] += 1


print("\nGender counts:")
for group, count in gender_bins.items():
    print(f"{group}: {count}")

print("\nAge group counts:")
for group, count in age_bins.items():
    print(f"{group}: {count}")

print("\nSmoking status counts:")
for group, count in smoke_bins.items():
    print(f"{group}: {count}")

print("\nAlcohol use counts:")
for group, count in alcohol_bins.items():
    print(f"{group}: {count}")

print("\nDrug use counts:")
for group, count in drug_bins.items():
    print(f"{group}: {count}")

print("\nOccupation type counts:")
for group, count in occupation_bins.items():
    print(f"{group}: {count}")

print("\nSES proxy counts:")
for group, count in ses_bins.items():
    print(f"{group}: {count}")

print("\nFamily support counts:")
for group, count in family_support_bins.items():
    print(f"{group}: {count}")

print("\nRare medication counts:")
for group, count in rare_meds_bins.items():
    print(f"{group}: {count}")

print("\nComorbidity status counts:")
for group, count in comorbidity_bins.items():
    print(f"{group}: {count}")

print("\nSymptom presentation counts:")
for group, count in symptom_presentation_bins.items():
    print(f"{group}: {count}")
