MODEL_NAME = "neural_net"
FEATURES = ["age","weight","body_condition_score","temperature","heart_rate","milk_yield"]
TARGET = ['disease']
NUM_INPUT = 6
NUM_OUTPUT = 1
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCHS = 100
MAPPING = {'Bovine Respiratory Disease': 0,
 'Bovine Viral Diarrhea': 1,
 'Foot and Mouth Disease': 2,
 'Bovine Tuberculosis': 3,
 'Ringworm': 4,
 'Healthy': 5,
 'Brucellosis': 6,
 'Anthrax': 7,
 'Leptospirosis': 8,
 "Johne's Disease": 9,
 'Salmonella': 10,
 'Mastitis': 11,
 'Infectious Bovine Rhinotracheitis': 12,
 'Blue Tongue': 13}

TEST_SIZE = 0.2