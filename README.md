![image](https://github.com/user-attachments/assets/510772c8-5a2c-43c5-a3b2-22e7750fa014)
import time
import hashlib
import numpy as np

# Blockchain class
class Blockchain:
    def __init__(self):
        self.chain = []
        self.transactions = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(time.time()),
            'proof': proof,
            'previous_hash': previous_hash,
            'transactions': self.transactions
        }
        self.transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, sender, receiver, data):
        self.transactions.append({'sender': sender, 'receiver': receiver, 'data': data})
        return self.get_previous_block()['index'] + 1

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        while not check_proof:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
        return new_proof

    def hash(self, block):
        encoded_block = str(block).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] != '0000':
                return False
            previous_block = block
            block_index += 1
        return True

# Federated Learning Node class
class FederatedLearningNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.local_model = np.random.random((10, 10))  # Random initial model
        self.blockchain = Blockchain()

    def train_local_model(self, data):
        # Simulate local training
        self.local_model += np.random.random(self.local_model.shape) * 0.01
        return self.local_model

    def aggregate_model(self, models):
        # Simulate model aggregation
        aggregated_model = np.mean(models, axis=0)
        self.local_model = aggregated_model
        return aggregated_model

# Simulation
node = FederatedLearningNode(node_id="Node_1")

# Mine a new block
def mine_block():
    previous_block = node.blockchain.get_previous_block()
    previous_proof = previous_block['proof']
    proof = node.blockchain.proof_of_work(previous_proof)
    previous_hash = node.blockchain.hash(previous_block)
    return node.blockchain.create_block(proof, previous_hash)

# Add a transaction
def add_transaction(sender, receiver, data):
    return node.blockchain.add_transaction(sender, receiver, data)

# Get the blockchain
def get_chain():
    return node.blockchain.chain

# Validate the blockchain
def is_valid():
    return node.blockchain.is_chain_valid(node.blockchain.chain)

# Train a local model
def train_local_model(data):
    return node.train_local_model(data)

# Aggregate models
def aggregate_model(models):
    return node.aggregate_model(models)

# Example Usage
# 1. Add a transaction
add_transaction("Device_A", "Device_B", "Workload data")

# 2. Mine a block
mine_block()

# 3. Train local model
data = np.random.random((10, 10))
updated_model = train_local_model(data)
print("Updated local model:", updated_model)

# 4. Aggregate models
models = [np.random.random((10, 10)) for _ in range(3)]
aggregated_model = aggregate_model(models)
print("Aggregated model:", aggregated_model)

# 5. Validate blockchain
print("Is blockchain valid?", is_valid())

# 6. Print the blockchain
print("Blockchain:", get_chain())


Updated local model: [[0.83462479 0.20371739 0.52851332 0.7975786  0.31947417 0.25448567
  0.84061816 0.65964088 0.46069467 0.02251242]
 [0.23723998 0.23145024 0.00954464 0.40760285 0.43292411 0.78276595
  0.0695515  0.91653908 0.36474862 0.126165  ]
 [0.56426403 0.00807824 0.27372883 0.16790702 0.39353294 0.37670292
  0.69834514 0.77738    0.15656371 0.74539986]
 [0.38205144 0.88427281 0.7083315  0.62136432 0.45568012 0.40872453
  0.06148926 0.07162595 0.37573247 0.07412339]
 [0.37146776 0.80881188 0.28422007 0.06874047 0.56939318 0.30709721
  0.56337517 0.97027264 0.97562673 0.01319416]
 [0.57262713 0.11410425 0.33984108 0.49802372 0.05322828 0.89626669
  0.64410705 0.88050047 0.3206441  0.2420872 ]
 [0.38352443 0.33678274 0.19863635 0.63430794 0.11016415 0.47869951
  0.42824648 0.9390114  0.74409505 0.62652859]
 [0.59424837 0.9417472  0.8040916  0.95341781 0.35254508 0.36190875
  0.75044382 0.50213503 0.6392497  0.94360029]
 [0.1839372  0.58014849 0.19523228 0.63578275 0.26259478 0.09096349
  0.0041468  0.60963727 0.24185056 0.70289569]
 [0.07235215 0.04009712 0.81626629 0.10437155 0.72189806 0.35752738
  0.87644823 0.82932812 0.71260711 1.00412889]]
Aggregated model: [[0.75801197 0.26779561 0.40652544 0.66125514 0.34327938 0.58277274
  0.55751978 0.29887371 0.27578755 0.39833373]
 [0.69190552 0.44668508 0.63575236 0.50360616 0.4546172  0.76647708
  0.70681453 0.66890403 0.49307505 0.35967107]
 [0.23779786 0.21115432 0.55486239 0.51546203 0.60384321 0.4141093
  0.310497   0.57104578 0.45484334 0.44521014]
 [0.34343949 0.80121782 0.73242289 0.60540723 0.29738987 0.63842666
  0.61245567 0.4858361  0.45431876 0.50482614]
 [0.33855509 0.60926232 0.56478004 0.40196547 0.49673238 0.55052218
  0.63423838 0.3705712  0.50306067 0.55969343]
 [0.33858631 0.54455438 0.82981298 0.63357392 0.70402267 0.60072344
  0.77086082 0.44695796 0.50173058 0.19809419]
 [0.69354332 0.46512326 0.65387435 0.06532177 0.53730498 0.62673464
  0.15077192 0.54047543 0.32936386 0.69605632]
 [0.68669524 0.59322329 0.45082403 0.3098001  0.23013756 0.65281652
  0.50335358 0.15619517 0.2067501  0.46784254]
 [0.28432605 0.58329766 0.69328344 0.49466592 0.67646984 0.4690127
  0.53393726 0.26050466 0.35743507 0.73459443]
 [0.26369682 0.29942719 0.51225868 0.34466496 0.46236315 0.50216221
  0.40127693 0.55369505 0.68023054 0.46823563]]
Is blockchain valid? True
Blockchain: [{'index': 1, 'timestamp': '1733052516.7968426', 'proof': 1, 'previous_hash': '0', 'transactions': []}, {'index': 2, 'timestamp': '1733052516.7991831', 'proof': 533, 'previous_hash': 'a6e9d0c3e5f5fa203a810424e57b943b39d896586cbd080a44827fd261100475', 'transactions': [{'sender': 'Device_A', 'receiver': 'Device_B', 'data': 'Workload data'}]}


import time
import hashlib
import numpy as np
import pandas as pd

# Blockchain class
class Blockchain:
    def __init__(self):
        self.chain = []
        self.transactions = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(time.time()),
            'proof': proof,
            'previous_hash': previous_hash,
            'transactions': self.transactions
        }
        self.transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, sender, receiver, data):
        self.transactions.append({'sender': sender, 'receiver': receiver, 'data': data})
        return self.get_previous_block()['index'] + 1

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        while not check_proof:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
        return new_proof

    def hash(self, block):
        encoded_block = str(block).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] != '0000':
                return False
            previous_block = block
            block_index += 1
        return True

# Federated Learning Node class
class FederatedLearningNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.local_model = np.random.random((10, 10))  # Random initial model
        self.blockchain = Blockchain()

    def train_local_model(self, data):
        # Simulate local training
        self.local_model += np.mean(data) * 0.01
        return self.local_model

    def aggregate_model(self, models):
        # Simulate model aggregation
        aggregated_model = np.mean(models, axis=0)
        self.local_model = aggregated_model
        return aggregated_model

# Initialize three laboratories
lab1 = FederatedLearningNode(node_id="Lab_1")
lab2 = FederatedLearningNode(node_id="Lab_2")
lab3 = FederatedLearningNode(node_id="Lab_3")

# Generate synthetic patient healthcare data
np.random.seed(42)
patient_data = pd.DataFrame({
    'patient_id': np.arange(1, 101),
    'age': np.random.randint(20, 80, size=100),
    'blood_pressure': np.random.randint(80, 180, size=100),
    'cholesterol': np.random.randint(150, 300, size=100),
    'heart_rate': np.random.randint(60, 100, size=100),
    'diagnosis': np.random.choice(['Healthy', 'At Risk', 'Critical'], size=100)
})

# Split data for three laboratories
lab1_data = patient_data.sample(frac=0.33)
lab2_data = patient_data.drop(lab1_data.index).sample(frac=0.5)
lab3_data = patient_data.drop(lab1_data.index).drop(lab2_data.index)

# Train models at each lab
lab1_model = lab1.train_local_model(lab1_data[['age', 'blood_pressure', 'cholesterol', 'heart_rate']].values)
lab2_model = lab2.train_local_model(lab2_data[['age', 'blood_pressure', 'cholesterol', 'heart_rate']].values)
lab3_model = lab3.train_local_model(lab3_data[['age', 'blood_pressure', 'cholesterol', 'heart_rate']].values)

# Aggregate models
aggregated_model = lab1.aggregate_model([lab1_model, lab2_model, lab3_model])

# Simulate adding transactions to blockchain
lab1.blockchain.add_transaction("Lab_1", "FederatedServer", "Trained model weights")
lab2.blockchain.add_transaction("Lab_2", "FederatedServer", "Trained model weights")
lab3.blockchain.add_transaction("Lab_3", "FederatedServer", "Trained model weights")

# Mine a new block
lab1.blockchain.create_block(proof=lab1.blockchain.proof_of_work(lab1.blockchain.get_previous_block()['proof']),
                             previous_hash=lab1.blockchain.hash(lab1.blockchain.get_previous_block()))

# Output Results
print("Aggregated Model:\n", aggregated_model)
print("\nBlockchain:\n", lab1.blockchain.chain)

Aggregated Model:
 [[1.59952585 1.91101602 1.85985017 1.97527509 1.68122131 1.8599282
  1.69684535 1.91083316 1.8022152  2.09901156]
 [1.92398765 1.73784747 1.88135765 1.5812221  1.99907099 1.70996772
  2.01973422 1.5523496  1.66385739 1.98494317]
 [1.82932723 1.83015947 2.02586622 1.47910514 1.92411068 1.9379837
  1.47607428 1.64486173 1.81247444 1.69764851]
 [1.72036356 1.71448823 1.76726451 1.94104983 1.67718304 1.39632957
  1.89370933 1.55565446 1.99113683 1.78098357]
 [1.83146316 1.79294978 1.74789083 2.1994262  1.96538608 1.61998418
  1.76244848 1.86501191 2.03134761 1.89840923]
 [1.81925061 2.00502712 1.76122929 1.76674257 1.58114419 1.63533226
  1.75244464 1.64963152 1.52126477 1.58627004]
 [1.99243738 1.67467657 1.86002614 1.73270538 1.69871945 1.65509166
  1.99720359 1.6124013  1.71921205 1.68581677]
 [1.57565354 1.77894869 1.42480316 2.01172239 1.66361802 1.76433246
  1.57165085 2.07099307 1.86852396 1.79164183]
 [1.60191647 1.76922803 1.892484   1.49453995 1.82755646 1.45957195
  1.86911331 1.83190299 1.70972847 1.71621062]
 [1.71340874 1.57965218 1.81984266 1.79953408 1.73354823 1.96538323
  1.84414138 1.71096492 1.71626124 1.56162314]]

Blockchain:
 [{'index': 1, 'timestamp': '1733052648.8437996', 'proof': 1, 'previous_hash': '0', 'transactions': []}, {'index': 2, 'timestamp': '1733052648.864855', 'proof': 533, 'previous_hash': '6f62a32db2965121db91fa93dd0b5281ace0b785e2a224cb5de01dd4441c6e22', 'transactions': [{'sender': 'Lab_1', 'receiver': 'FederatedServer', 'data': 'Trained model weights'}]}]

 import time
import hashlib
import numpy as np
import pandas as pd

# Blockchain class
class Blockchain:
    def __init__(self):
        self.chain = []
        self.transactions = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(time.time()),
            'proof': proof,
            'previous_hash': previous_hash,
            'transactions': self.transactions
        }
        self.transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, sender, receiver, data):
        self.transactions.append({'sender': sender, 'receiver': receiver, 'data': data})
        return self.get_previous_block()['index'] + 1

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        while not check_proof:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
        return new_proof

    def hash(self, block):
        encoded_block = str(block).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] != '0000':
                return False
            previous_block = block
            block_index += 1
        return True

# Federated Learning Node class
class FederatedLearningNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.local_model = np.random.random((10, 10))  # Random initial model
        self.blockchain = Blockchain()

    def train_local_model(self, data):
        # Simulate local training
        self.local_model += np.mean(data) * 0.01
        return self.local_model

    def aggregate_model(self, models):
        # Simulate model aggregation
        aggregated_model = np.mean(models, axis=0)
        self.local_model = aggregated_model
        return aggregated_model

# Initialize three laboratories
lab1 = FederatedLearningNode(node_id="Lab_1")
lab2 = FederatedLearningNode(node_id="Lab_2")
lab3 = FederatedLearningNode(node_id="Lab_3")

# Load synthetic healthcare data for each lab
lab1_data = pd.read_csv('lab1_patient_data.csv')
lab2_data = pd.read_csv('lab2_patient_data.csv')
lab3_data = pd.read_csv('lab3_patient_data.csv')

# Train models at each lab
lab1_model = lab1.train_local_model(lab1_data[['age', 'blood_pressure', 'cholesterol', 'heart_rate']].values)
lab2_model = lab2.train_local_model(lab2_data[['age', 'blood_pressure', 'cholesterol', 'heart_rate']].values)
lab3_model = lab3.train_local_model(lab3_data[['age', 'blood_pressure', 'cholesterol', 'heart_rate']].values)

# Aggregate models
aggregated_model = lab1.aggregate_model([lab1_model, lab2_model, lab3_model])

# Simulate adding transactions to blockchain
lab1.blockchain.add_transaction("Lab_1", "FederatedServer", "Trained model weights")
lab2.blockchain.add_transaction("Lab_2", "FederatedServer", "Trained model weights")
lab3.blockchain.add_transaction("Lab_3", "FederatedServer", "Trained model weights")

# Mine a new block
lab1.blockchain.create_block(proof=lab1.blockchain.proof_of_work(lab1.blockchain.get_previous_block()['proof']),
                             previous_hash=lab1.blockchain.hash(lab1.blockchain.get_previous_block()))

# Output Results
print("Aggregated Model:\n", aggregated_model)
print("\nBlockchain:\n", lab1.blockchain.chain)


Aggregated Model:
 [[1.69162259 1.72559275 1.48940886 1.65266013 1.71505111 1.67887335
  1.50114803 1.48669208 1.55794869 1.50189196]
 [1.7500966  1.69218287 1.8778718  1.78159723 1.91151697 1.68329635
  1.81674201 1.88971297 1.66553256 1.63486647]
 [1.83790079 1.63503153 1.89646391 1.8515695  1.75549264 1.78464201
  1.46946903 1.68127221 1.90156547 1.88933015]
 [1.77212314 1.55135348 1.81769306 1.31053996 1.63924971 1.77581675
  2.11547769 1.93681512 1.74918149 1.62397459]
 [1.96877083 1.80388121 1.64778397 1.59861871 1.50677395 1.69124529
  1.62218851 1.60821135 1.71295076 1.53567897]
 [1.92839948 1.57599171 1.84372074 1.95096163 1.84397748 1.45363856
  1.98868467 1.80112995 1.75940564 1.69483969]
 [1.6540074  1.55826591 1.52075331 1.97729364 1.87220631 1.61626174
  1.71604511 1.5425231  1.5442172  1.37427877]
 [1.64986679 1.6791981  1.70395193 1.68991377 1.6971806  1.47830956
  2.05460153 1.41373301 1.72693974 1.90232992]
 [1.90625017 1.37509921 1.89454322 1.93951904 1.800611   1.86903004
  1.80586457 1.51741286 1.66000823 1.79376947]
 [1.78076431 1.62260056 1.4822147  1.66549281 1.64105455 1.41759722
  1.76315829 1.610771   1.61215503 1.55555642]]

Blockchain:
 [{'index': 1, 'timestamp': '1733052997.7628007', 'proof': 1, 'previous_hash': '0', 'transactions': []}, {'index': 2, 'timestamp': '1733052997.7758462', 'proof': 533, 'previous_hash': 'caa78a01008ec339a77b8e0588658d04dfb8a4957150705a5c957b5140044245', 'transactions': [{'sender': 'Lab_1', 'receiver': 'FederatedServer', 'data': 'Trained model weights'}]}]


 import time
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split

# Blockchain class (unchanged)
class Blockchain:
    def __init__(self):
        self.chain = []
        self.transactions = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(time.time()),
            'proof': proof,
            'previous_hash': previous_hash,
            'transactions': self.transactions
        }
        self.transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, sender, receiver, data):
        self.transactions.append({'sender': sender, 'receiver': receiver, 'data': data})
        return self.get_previous_block()['index'] + 1

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        while not check_proof:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
        return new_proof

    def hash(self, block):
        encoded_block = str(block).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] != '0000':
                return False
            previous_block = block
            block_index += 1
        return True

# Federated Learning Node with CNN training
class FederatedLearningNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.model = self.build_cnn_model()
        self.blockchain = Blockchain()

    def build_cnn_model(self):
        model = Sequential([
            Conv1D(32, kernel_size=2, activation='relu', input_shape=(4, 1)),  # 4 features in input
            Flatten(),
            Dense(128, activation='relu'),
            Dense(3, activation='softmax')  # Assuming 3 classes for the diagnosis
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_local_model(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=3, verbose=0)
        return self.model.get_weights()

    def aggregate_model(self, models_weights):
        averaged_weights = [np.mean(weights, axis=0) for weights in zip(*models_weights)]
        self.model.set_weights(averaged_weights)
        return averaged_weights

# Load synthetic healthcare data
lab1_data = pd.read_csv('lab1_patient_data.csv')
lab2_data = pd.read_csv('lab2_patient_data.csv')
lab3_data = pd.read_csv('lab3_patient_data.csv')

# Preprocess data
def preprocess_data(data):
    x = data[['age', 'blood_pressure', 'cholesterol', 'heart_rate']].values
    y = data['diagnosis'].factorize()[0]  # Convert categorical diagnosis to numeric
    x = x.reshape(-1, 4, 1)  # Reshape for Conv1D: (samples, features, 1)
    return train_test_split(x, y, test_size=0.2, random_state=42)

x_train1, x_test1, y_train1, y_test1 = preprocess_data(lab1_data)
x_train2, x_test2, y_train2, y_test2 = preprocess_data(lab2_data)
x_train3, x_test3, y_train3, y_test3 = preprocess_data(lab3_data)

# Initialize three laboratories
lab1 = FederatedLearningNode(node_id="Lab_1")
lab2 = FederatedLearningNode(node_id="Lab_2")
lab3 = FederatedLearningNode(node_id="Lab_3")

# Train models at each lab
lab1_weights = lab1.train_local_model(x_train1, y_train1)
lab2_weights = lab2.train_local_model(x_train2, y_train2)
lab3_weights = lab3.train_local_model(x_train3, y_train3)

# Aggregate models
aggregated_weights = lab1.aggregate_model([lab1_weights, lab2_weights, lab3_weights])

# Simulate adding transactions to blockchain
lab1.blockchain.add_transaction("Lab_1", "FederatedServer", "Trained CNN model weights")
lab2.blockchain.add_transaction("Lab_2", "FederatedServer", "Trained CNN model weights")
lab3.blockchain.add_transaction("Lab_3", "FederatedServer", "Trained CNN model weights")

# Mine a new block
lab1.blockchain.create_block(proof=lab1.blockchain.proof_of_work(lab1.blockchain.get_previous_block()['proof']),
                             previous_hash=lab1.blockchain.hash(lab1.blockchain.get_previous_block()))

# Evaluate aggregated model
lab1.model.set_weights(aggregated_weights)
loss, accuracy = lab1.model.evaluate(x_test1, y_test1, verbose=0)

# Output Results
print("Aggregated Model Accuracy:", accuracy)
print("\nBlockchain:\n", lab1.blockchain.chain)


/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Aggregated Model Accuracy: 0.3030303120613098

Blockchain:
 [{'index': 1, 'timestamp': '1733053534.032722', 'proof': 1, 'previous_hash': '0', 'transactions': []}, {'index': 2, 'timestamp': '1733053538.701576', 'proof': 533, 'previous_hash': '06f35509521ee18fb85bf628c2c41f8cf4dde0a1d64ad4f67acd7316465f4dc2', 'transactions': [{'sender': 'Lab_1', 'receiver': 'FederatedServer', 'data': 'Trained CNN model weights'}]}]

 


