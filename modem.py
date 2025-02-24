import numpy as np
import json
from dec import Qbit

def encode_bb84(message):
    num_bits = len(message)
    alice_bits = np.array([int(bit) for bit in message])
    
    # Alice elige bases aleatorias (0 = base Z, 1 = base X)
    alice_bases = np.random.randint(0, 2, num_bits)
    
    # Alice prepara los qubits
    qubits = []
    for bit, base in zip(alice_bits, alice_bases):
        q = Qbit(1, 0) if bit == 0 else Qbit(0, 1)  # |0⟩ o |1⟩
        if base == 1:
            q.h()  # Aplicar Hadamard si está en la base X
        qubits.append(q)
    
    return qubits, alice_bits.tolist(), alice_bases.tolist()

def decode_bb84(qubits, alice_bits, alice_bases):
    num_bits = len(qubits)
    bob_bases = np.random.randint(0, 2, num_bits)
    
    # Bob mide los qubits
    bob_results = []
    for q, base in zip(qubits, bob_bases):
        if base == 1:
            q.h()  # Cambiar a la base X si Bob usa base X
        result = q.medir_qubit(shots=1)
        bob_results.append(0 if '0' in result else 1)
    
    # Comparación de bases
    matching_indices = np.where(np.array(alice_bases) == np.array(bob_bases))[0]
    alice_key = np.array([alice_bits[i] for i in matching_indices])
    bob_key = np.array([bob_results[i] for i in matching_indices])
    
    return bob_results, bob_bases.tolist(), alice_key.tolist(), bob_key.tolist()

def detect_eavesdropper(alice_key, bob_key):
    if len(alice_key) == 0 or len(bob_key) == 0:
        return "No hay suficientes datos para detectar un espía."
    
    discrepancies = np.sum(np.array(alice_key) != np.array(bob_key))
    
    if discrepancies > 0:
        return f"Posible espía detectado. Discrepancias encontradas: {discrepancies}"
    else:
        return "No se detectó ningún espía."

def save_alice_data(filename, alice_bases, alice_key):
    data = {
        "bases_alice": alice_bases,
        "clave_alice": alice_key
    }
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

# Ejemplo de uso
message = "00000111"  # Mensaje a codificar en BB84
qubits, alice_message, alice_bases = encode_bb84(message)
bob_message, bob_bases, alice_key, bob_key = decode_bb84(qubits, alice_message, alice_bases)

eavesdropper_status = detect_eavesdropper(alice_key, bob_key)

# Guardar datos de Alice en un archivo
save_alice_data("alice_data.json", alice_bases, alice_key)

print("Mensaje enviado por Alice:", alice_message)
print("Bases de Alice:", alice_bases)
print("Mensaje decodificado por Bob:", bob_message)
print("Bases de Bob:", bob_bases)
print("Clave de Alice:", alice_key)
print("Clave de Bob:  ", bob_key)
print(eavesdropper_status)
