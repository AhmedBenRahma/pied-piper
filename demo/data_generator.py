"""
Pied Piper AI - Data Generator
Generates synthetic insurance claims data with embedded fraud patterns.

This module creates realistic insurance claims with two types of fraud rings:
1. Ring A (Spider Web): A corrupt doctor-pharmacy collusion network
2. Ring B (Car Crash Loop): Staged accident ring with circular collision pattern
"""

import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker for generating realistic names and data
fake = Faker()
Faker.seed(42)  # For reproducibility
random.seed(42)


def generate_normal_claims(num_claims=500):
    """
    Generate normal (non-fraudulent) insurance claims.
    
    Args:
        num_claims: Number of normal claims to generate
        
    Returns:
        tuple: (nodes_df, edges_df) containing entity nodes and claim edges
    """
    nodes = []
    edges = []
    
    # Create pools of normal entities
    num_normal_doctors = 50
    num_normal_pharmacies = 30
    num_normal_patients = 300
    num_normal_mechanics = 20
    
    doctors = [f"Dr. {fake.last_name()}" for _ in range(num_normal_doctors)]
    pharmacies = [f"{fake.company()} Pharmacy" for _ in range(num_normal_pharmacies)]
    patients = [fake.name() for _ in range(num_normal_patients)]
    mechanics = [f"{fake.last_name()} Auto Repair" for _ in range(num_normal_mechanics)]
    
    # Add all entities as nodes
    for doc in doctors:
        nodes.append({'id': doc, 'label': doc, 'type': 'Doctor', 'is_fraud': False})
    
    for pharm in pharmacies:
        nodes.append({'id': pharm, 'label': pharm, 'type': 'Pharmacy', 'is_fraud': False})
    
    for patient in patients:
        nodes.append({'id': patient, 'label': patient, 'type': 'Patient', 'is_fraud': False})
    
    for mech in mechanics:
        nodes.append({'id': mech, 'label': mech, 'type': 'Mechanic', 'is_fraud': False})
    
    # Generate random claims (edges)
    claim_id = 1
    start_date = datetime(2023, 1, 1)
    
    for _ in range(num_claims):
        claim_type = random.choice(['medical', 'auto'])
        claim_date = start_date + timedelta(days=random.randint(0, 365))
        
        if claim_type == 'medical':
            # Medical claim: Patient -> Doctor -> Pharmacy
            patient = random.choice(patients)
            doctor = random.choice(doctors)
            pharmacy = random.choice(pharmacies)
            amount = random.randint(100, 5000)
            
            # Create edges for this claim
            edges.append({
                'claim_id': f'C{claim_id}',
                'from': patient,
                'to': doctor,
                'amount': amount,
                'date': claim_date,
                'type': 'medical_visit',
                'is_fraud': False
            })
            
            edges.append({
                'claim_id': f'C{claim_id}',
                'from': doctor,
                'to': pharmacy,
                'amount': amount * 0.6,  # Prescription cost
                'date': claim_date,
                'type': 'prescription',
                'is_fraud': False
            })
            
        else:  # auto claim
            # Auto claim: Car owner -> Mechanic
            car_owner = random.choice(patients)  # Using patients as car owners
            mechanic = random.choice(mechanics)
            amount = random.randint(500, 15000)
            
            edges.append({
                'claim_id': f'C{claim_id}',
                'from': car_owner,
                'to': mechanic,
                'amount': amount,
                'date': claim_date,
                'type': 'auto_repair',
                'is_fraud': False
            })
        
        claim_id += 1
    
    return pd.DataFrame(nodes), pd.DataFrame(edges), claim_id


def inject_fraud_ring_a(nodes_df, edges_df, claim_id_start):
    """
    Inject Ring A: "The Spider Web" - Doctor-Pharmacy Collusion
    
    Pattern: 1 corrupt doctor + 1 corrupt pharmacy + 20 patients
    All patients visit the SAME doctor and pharmacy, creating a star topology.
    This is a classic collusion pattern indicating organized fraud.
    
    Args:
        nodes_df: Existing nodes DataFrame
        edges_df: Existing edges DataFrame
        claim_id_start: Starting claim ID number
        
    Returns:
        tuple: (updated_nodes_df, updated_edges_df, next_claim_id)
    """
    # Create the corrupt entities
    corrupt_doctor = "Dr. Viktor Corruption"
    corrupt_pharmacy = "Pharma Shady Inc."
    
    # Create 20 distinct patients for this fraud ring
    fraud_patients = [f"Patient_{fake.first_name()}_{i}" for i in range(1, 21)]
    
    # Add the corrupt entities to nodes (marked as fraud)
    new_nodes = [
        {'id': corrupt_doctor, 'label': corrupt_doctor, 'type': 'Doctor', 'is_fraud': True},
        {'id': corrupt_pharmacy, 'label': corrupt_pharmacy, 'type': 'Pharmacy', 'is_fraud': True}
    ]
    
    # Add fraud ring patients
    for patient in fraud_patients:
        new_nodes.append({'id': patient, 'label': patient, 'type': 'Patient', 'is_fraud': True})
    
    # Create claims: Each patient visits the corrupt doctor and pharmacy
    claim_id = claim_id_start
    new_edges = []
    start_date = datetime(2023, 6, 1)  # Mid-year surge of fraud
    
    for i, patient in enumerate(fraud_patients):
        claim_date = start_date + timedelta(days=i * 3)  # Staggered over time
        amount = random.randint(3000, 8000)  # Higher than normal claims
        
        # Patient -> Doctor
        new_edges.append({
            'claim_id': f'FRAUD_A_{claim_id}',
            'from': patient,
            'to': corrupt_doctor,
            'amount': amount,
            'date': claim_date,
            'type': 'medical_visit',
            'is_fraud': True
        })
        
        # Doctor -> Pharmacy (suspiciously high prescription costs)
        new_edges.append({
            'claim_id': f'FRAUD_A_{claim_id}',
            'from': corrupt_doctor,
            'to': corrupt_pharmacy,
            'amount': amount * 0.8,  # Much higher percentage than normal
            'date': claim_date,
            'type': 'prescription',
            'is_fraud': True
        })
        
        claim_id += 1
    
    # Append new data
    nodes_df = pd.concat([nodes_df, pd.DataFrame(new_nodes)], ignore_index=True)
    edges_df = pd.concat([edges_df, pd.DataFrame(new_edges)], ignore_index=True)
    
    return nodes_df, edges_df, claim_id


def inject_fraud_ring_b(nodes_df, edges_df, claim_id_start):
    """
    Inject Ring B: "The Car Crash Loop" - Staged Accident Ring
    
    Pattern: 5 car owners in a circular collision pattern with shared mechanic
    A -> B -> C -> D -> E -> A (closed loop)
    All use the same mechanic, indicating coordinated staging.
    
    Args:
        nodes_df: Existing nodes DataFrame
        edges_df: Existing edges DataFrame
        claim_id_start: Starting claim ID number
        
    Returns:
        tuple: (updated_nodes_df, updated_edges_df, next_claim_id)
    """
    # Create 5 car owners for the crash loop
    car_owners = [
        "Alex Crasher",
        "Bella Bumper",
        "Charlie Collide",
        "Diana Dent",
        "Eddie Impact"
    ]
    
    # The corrupt mechanic they all use
    corrupt_mechanic = "Shady Joe's Auto Scam Shop"
    
    # Add entities to nodes
    new_nodes = []
    for owner in car_owners:
        new_nodes.append({'id': owner, 'label': owner, 'type': 'Car_Owner', 'is_fraud': True})
    
    new_nodes.append({'id': corrupt_mechanic, 'label': corrupt_mechanic, 'type': 'Mechanic', 'is_fraud': True})
    
    # Create circular collision pattern: each person "hits" the next person
    claim_id = claim_id_start
    new_edges = []
    start_date = datetime(2023, 8, 1)  # Late summer "accident" spree
    
    for i in range(len(car_owners)):
        owner_a = car_owners[i]
        owner_b = car_owners[(i + 1) % len(car_owners)]  # Circular: last person hits first
        
        claim_date = start_date + timedelta(days=i * 7)  # Weekly "accidents"
        amount = random.randint(4000, 12000)
        
        # Create accident edge (for visualization, we show the collision relationship)
        new_edges.append({
            'claim_id': f'FRAUD_B_{claim_id}',
            'from': owner_a,
            'to': owner_b,
            'amount': amount,
            'date': claim_date,
            'type': 'collision',
            'is_fraud': True
        })
        
        # Both parties go to the same corrupt mechanic
        new_edges.append({
            'claim_id': f'FRAUD_B_{claim_id}_A',
            'from': owner_a,
            'to': corrupt_mechanic,
            'amount': amount,
            'date': claim_date,
            'type': 'auto_repair',
            'is_fraud': True
        })
        
        new_edges.append({
            'claim_id': f'FRAUD_B_{claim_id}_B',
            'from': owner_b,
            'to': corrupt_mechanic,
            'amount': amount * 0.9,
            'date': claim_date,
            'type': 'auto_repair',
            'is_fraud': True
        })
        
        claim_id += 1
    
    # Append new data
    nodes_df = pd.concat([nodes_df, pd.DataFrame(new_nodes)], ignore_index=True)
    edges_df = pd.concat([edges_df, pd.DataFrame(new_edges)], ignore_index=True)
    
    return nodes_df, edges_df, claim_id


def generate_all_data():
    """
    Generate complete dataset with normal claims and embedded fraud rings.
    
    Returns:
        tuple: (nodes_df, edges_df) containing all entities and claims
    """
    print("[*] Generating synthetic insurance claims data...")
    
    # Step 1: Generate normal claims
    print("  [+] Generating 500 normal claims...")
    nodes_df, edges_df, claim_id = generate_normal_claims(500)
    
    # Step 2: Inject fraud patterns
    print("  [+] Injecting Ring A: Spider Web (Doctor-Pharmacy Collusion)...")
    nodes_df, edges_df, claim_id = inject_fraud_ring_a(nodes_df, edges_df, claim_id)
    
    print("  [+] Injecting Ring B: Car Crash Loop (Staged Accidents)...")
    nodes_df, edges_df, claim_id = inject_fraud_ring_b(nodes_df, edges_df, claim_id)
    
    print(f"\n[SUCCESS] Data generation complete!")
    print(f"   Total Nodes: {len(nodes_df)}")
    print(f"   Total Edges (Claims): {len(edges_df)}")
    print(f"   Fraud Nodes: {len(nodes_df[nodes_df['is_fraud'] == True])}")
    print(f"   Fraud Edges: {len(edges_df[edges_df['is_fraud'] == True])}\n")
    
    return nodes_df, edges_df


# For testing purposes
if __name__ == "__main__":
    nodes, edges = generate_all_data()
    print("\n[DATA] Sample Nodes:")
    print(nodes.head())
    print("\n[DATA] Sample Edges:")
    print(edges.head())
