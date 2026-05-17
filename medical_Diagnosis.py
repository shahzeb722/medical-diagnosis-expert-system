import re
import spacy
import sys
from neo4j import GraphDatabase
from pgmpy.models import DiscreteBayesianNetwork 
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# -------------------
# 1. Configuration & NLP Setup
# -------------------
uri = "bolt://127.0.0.1:7687"
username = "neo4j"
password = "luna123*" 

driver = GraphDatabase.driver(uri, auth=(username, password))
nlp = spacy.load("en_core_web_sm")

# Enhanced weights to prevent identical probabilities
SYMPTOM_WEIGHTS = {
    "Fever": [0.85, 0.25], "Stiff Neck": [0.98, 0.01], "Chest Pain": [0.88, 0.04],
    "Headache": [0.72, 0.35], "Rash": [0.65, 0.03], "Cough": [0.78, 0.18],
    "Sore Throat": [0.82, 0.12], "Shortness of Breath": [0.92, 0.04],
    "Burning Sensation": [0.95, 0.02], "Stomach Pain": [0.75, 0.15],
    "Pale Skin": [0.55, 0.08], "Red Eyes": [0.68, 0.04], "Fatigue": [0.70, 0.40],
    "Nausea": [0.65, 0.20], "Joint Pain": [0.80, 0.10], "Weight Gain": [0.90, 0.01],
    "Frequent Urination": [0.85, 0.05], "Severe Pain": [0.95, 0.02]
}
DEFAULT_WEIGHT = [0.70, 0.20]

# -------------------
# 2. Database & Persistence Functions
# -------------------
def setup_database():
    with driver.session() as session:
        # Drop legacy name constraint if it exists to avoid your previous error
        try: session.run("DROP CONSTRAINT FOR (p:Patient) REQUIRE p.name IS UNIQUE")
        except: pass
        
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) REQUIRE p.cnic IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")

def create_from_file(filename):
    try:
        with open(filename, "r") as file:
            with driver.session() as session:
                for line in file:
                    if "has symptoms" in line.lower():
                        parts = line.split("has symptoms")
                        disease = parts[0].strip()
                        symptoms = [s.strip().strip('.') for s in parts[1].split(",")]
                        session.run("MERGE (d:Disease {name: $d})", d=disease)
                        for s in symptoms:
                            session.run("""
                                MERGE (sym:Symptom {name: $s})
                                WITH sym
                                MATCH (d:Disease {name: $d})
                                MERGE (d)-[:HAS_SYMPTOM]->(sym)
                            """, d=disease, s=s)
        return True
    except FileNotFoundError: return False

def save_or_update_patient(cnic, name, age, location, symptoms):
    with driver.session() as session:
        session.run("""
            MERGE (p:Patient {cnic: $cnic}) 
            SET p.name = $name, p.age = $age, p.location = $location
        """, cnic=cnic, name=name, age=age, location=location)
        for s in symptoms:
            session.run("""
                MATCH (p:Patient {cnic: $cnic})
                MERGE (sym:Symptom {name: $s})
                MERGE (p)-[:HAS_OBSERVED]->(sym)
            """, cnic=cnic, s=s)

def get_patient_history(cnic):
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Patient {cnic: $cnic})
            OPTIONAL MATCH (p)-[:HAS_OBSERVED]->(s:Symptom)
            RETURN p.name as name, p.age as age, p.location as loc, collect(s.name) as symptoms
        """, cnic=cnic)
        record = result.single()
        if record and record["name"]:
            return record["name"], record["age"], record["loc"], record["symptoms"]
        return None, None, None, []

# -------------------
# 3. Bayesian Inference
# -------------------
def run_bayesian_inference(disease_name, matching_symptoms, total_kg_count):
    if not matching_symptoms: return 0.0
    
    edges = [(disease_name, s) for s in matching_symptoms]
    model = DiscreteBayesianNetwork(edges)
    
    # Prior probability of disease (set lower for rarer diseases)
    cpd_disease = TabularCPD(variable=disease_name, variable_card=2, values=[[0.99], [0.01]])
    
    s_cpds = []
    for s in matching_symptoms:
        w = SYMPTOM_WEIGHTS.get(s, DEFAULT_WEIGHT)
        cpd_s = TabularCPD(
            variable=s, variable_card=2,
            values=[[1-w[1], 1-w[0]], [w[1], w[0]]], 
            evidence=[disease_name], evidence_card=[2]
        )
        s_cpds.append(cpd_s)

    model.add_cpds(cpd_disease, *s_cpds)
    infer = VariableElimination(model)
    evidence = {s: 1 for s in matching_symptoms}
    result = infer.query(variables=[disease_name], evidence=evidence, show_progress=False)
    
    raw_prob = result.values[1]
    # Penalize diseases where many expected symptoms are missing
    coverage = len(matching_symptoms) / total_kg_count
    return raw_prob * coverage

# -------------------
# 4. Logic & Extraction
# -------------------
def validate_cnic(cnic):
    return bool(re.match(r'^\d{5}-\d{7}-\d{1}$', cnic))

def extract_symptoms_from_text(text):
    found = []
    with driver.session() as session:
        all_s = [r["n"] for r in session.run("MATCH (s:Symptom) RETURN s.name as n")]
        doc = nlp(text.lower())
        normalized_input = [token.lemma_ for token in doc]
        for s in all_s:
            if s.lower() in text.lower() or any(s.lower() == lemma for lemma in normalized_input):
                found.append(s)
    return found

def perform_diagnosis_calc(cnic, name, symptoms):
    with driver.session() as session:
        res = session.run("""
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            WITH d, collect(s.name) as kg_symptoms
            WITH d, kg_symptoms, [sym IN kg_symptoms WHERE sym IN $slist] as matches
            WHERE size(matches) > 0
            RETURN d.name as dname, matches, size(kg_symptoms) as total_count
        """, slist=symptoms)
        candidates = [(r["dname"], r["matches"], r["total_count"]) for r in res]

    if not candidates:
        print("\n!! No recognized symptoms linked to any disease in the Knowledge Graph.")
        return

    results = []
    for d_name, matches, total in candidates:
        prob = run_bayesian_inference(d_name, matches, total)
        results.append((d_name, prob))

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n*** TOP DIAGNOSES FOR {name.upper()} (CNIC: {cnic}) ***")
    for i, (disease, prob) in enumerate(results[:3], 1):
        print(f"{i}. {disease}: {prob * 100:.2f}%")

# -------------------
# 5. UI Menu
# -------------------
def patient_session(cnic, name, age, loc, current_symptoms):
    while True:
        print(f"\n--- SESSION: {name.upper()} ({cnic}) ---")
        print(f"Current Symptoms: {', '.join(current_symptoms) if current_symptoms else 'None'}")
        print("1. Add Symptom | 2. Run Diagnosis | 3. Finish & Save")
        
        choice = input("Option: ").strip()
        if choice == '1':
            text = input("Describe symptoms (comma separated): ").strip()
            new_s = extract_symptoms_from_text(text)
            if new_s:
                for s in new_s:
                    if s not in current_symptoms: current_symptoms.append(s)
                save_or_update_patient(cnic, name, age, loc, current_symptoms)
            else: print("!! No symptoms recognized.")
        elif choice == '2':
            if not current_symptoms: print("!! Add symptoms first.")
            else: perform_diagnosis_calc(cnic, name, current_symptoms)
        elif choice == '3': break

def main_menu():
    setup_database()
    create_from_file("knowledge.txt")
    while True:
        print("\n" + "="*40 + "\n MEDICAL KNOWLEDGE SYSTEM MAIN MENU \n" + "="*40)
        print("1. Start/Resume Patient Session")
        print("2. View All Diseases")
        print("3. Exit")
        
        choice = input("Option: ").strip()
        if choice == '1':
            cnic = input("CNIC (XXXXX-XXXXXXX-X): ").strip()
            if not validate_cnic(cnic):
                print("!! Invalid CNIC format.")
                continue
            
            ex_name, ex_age, ex_loc, ex_sym = get_patient_history(cnic)
            if ex_name:
                print(f"!! Record found for {ex_name}. Resuming...")
                patient_session(cnic, ex_name, ex_age, ex_loc, ex_sym)
            else:
                name, age, loc = input("Name: "), input("Age: "), input("City: ")
                patient_session(cnic, name, age, loc, [])
        elif choice == '2':
            with driver.session() as session:
                res = session.run("MATCH (d:Disease) RETURN d.name as n ORDER BY n")
                for r in res: print(f"- {r['n']}")
        elif choice == '3':
            driver.close()
            sys.exit()

if __name__ == "__main__":
    main_menu()