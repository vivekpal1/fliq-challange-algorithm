#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator

print("=" * 80)
print("VQE Simulation using Qiskit Aer")
print("=" * 80)
print("This script runs a Variational Quantum Eigensolver (VQE) algorithm using the ")
print("Qiskit Aer simulator instead of real quantum hardware.")
print()

# Define the problem Hamiltonian
H = SparsePauliOp.from_list([
    ("ZZ", -1.0),
    ("XI", -0.2),
    ("IX", -0.2)
])

print("Problem Hamiltonian:")
print(f"H = {H}")
print()

# Define individual Hamiltonian terms for later analysis
H_ZZ = SparsePauliOp.from_list([("ZZ", 1.0)])
H_XI = SparsePauliOp.from_list([("XI", 1.0)])
H_IX = SparsePauliOp.from_list([("IX", 1.0)])

# Create the ansatz circuit
ansatz = EfficientSU2(H.num_qubits)
print(f"Ansatz: EfficientSU2 with {H.num_qubits} qubits")
print(f"Number of parameters: {ansatz.num_parameters}")
print()

# Define the cost function
def cost_function(params, ansatz, H, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        H (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    job = estimator.run([ansatz], [H], [params])
    result = job.result()
    return result.values[0]

# Initialize random parameters
rng = np.random.default_rng(42)  # for reproducibility
num_params = ansatz.num_parameters
initial_params = 2 * np.pi * rng.random(num_params)

# Create the Aer simulator
simulator = AerSimulator()
print(f"Using simulator: {simulator.name}")

# Transpile the circuit for the simulator
transpiled_ansatz = transpile(ansatz, simulator)
print(f"Transpiled circuit depth: {transpiled_ansatz.depth()}")
print(f"Transpiled circuit size: {transpiled_ansatz.size()}")
print()

# Create the Aer estimator
shots = 10000
estimator = AerEstimator(
    run_options={"shots": shots},
    approximation=True  # Use shot-based sampling
)
print(f"Using Aer Estimator with {shots} shots per circuit evaluation")
print()

# Define the optimization module
from scipy.optimize import minimize

# Keep track of intermediate results
cost_history = []
param_history = []
term_values_history = []

# Function to compute expectation values of individual terms
def compute_term_values(params, ansatz, estimator):
    job = estimator.run(
        [ansatz, ansatz, ansatz], 
        [H_ZZ, H_XI, H_IX], 
        [params, params, params]
    )
    result = job.result()
    return {
        'ZZ': result.values[0],
        'XI': result.values[1],
        'IX': result.values[2]
    }

# Callback function to save intermediate results
def callback(params):
    energy = cost_function(params, transpiled_ansatz, H, estimator)
    cost_history.append(energy)
    param_history.append(params.copy())
    
    # Compute individual term expectation values
    term_values = compute_term_values(params, transpiled_ansatz, estimator)
    term_values_history.append(term_values)
    
    print(f"Iteration {len(cost_history)}: Energy = {energy:.6f}, " +
          f"<ZZ> = {term_values['ZZ']:.4f}, " +
          f"<XI> = {term_values['XI']:.4f}, " +
          f"<IX> = {term_values['IX']:.4f}")

# Run the VQE optimization with the COBYLA method
print("Starting VQE optimization with COBYLA optimizer...")
result = minimize(
    cost_function,
    initial_params,
    args=(transpiled_ansatz, H, estimator),
    method='COBYLA',
    callback=callback,
    options={'maxiter': 100}
)

# Print the final results
print("\nOptimization Results:")
print(f"Final energy: {result.fun:.6f}")
print(f"Optimization success: {result.success}")
print(f"Number of iterations: {len(cost_history)}")
print(f"Optimal parameters: {result.x}")

# Calculate theoretical ground state energy
from numpy import sqrt
exact_gs_energy = -sqrt(1 + 0.2**2 + 0.2**2)
print(f"\nTheoretical ground state energy: {exact_gs_energy:.6f}")
print(f"Energy difference from theoretical: {abs(result.fun - exact_gs_energy):.6f}")
print(f"Accuracy: {(1 - abs(result.fun - exact_gs_energy)/abs(exact_gs_energy)) * 100:.2f}%")

# Run the optimized circuit
optimized_params = result.x
best_result = estimator.run([transpiled_ansatz], [H], [optimized_params]).result()
min_energy = best_result.values[0]

# Calculate final expectation values for individual terms
final_term_values = compute_term_values(optimized_params, transpiled_ansatz, estimator)
print("\nFinal expectation values:")
print(f"<ZZ>: {final_term_values['ZZ']:.6f}")
print(f"<XI>: {final_term_values['XI']:.6f}")
print(f"<IX>: {final_term_values['IX']:.6f}")
print(f"Total energy: {min_energy:.6f}")

# Plot the energy convergence
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(cost_history, 'o-', color='blue', label='Total Energy')
plt.axhline(y=exact_gs_energy, color='r', linestyle='--', label=f'Exact GS Energy: {exact_gs_energy:.4f}')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('VQE Energy Convergence')
plt.legend()

# Plot the term values
plt.subplot(2, 1, 2)
zz_values = [term['ZZ'] for term in term_values_history]
xi_values = [term['XI'] for term in term_values_history]
ix_values = [term['IX'] for term in term_values_history]

plt.plot(zz_values, 'o-', label='<ZZ>')
plt.plot(xi_values, 'o-', label='<XI>')
plt.plot(ix_values, 'o-', label='<IX>')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Expectation Value')
plt.title('Individual Term Expectation Values')
plt.legend()

plt.tight_layout()
plt.savefig('vqe_convergence.png')
plt.close()

# Plot a histogram showing the distribution of measurements
# First, we need to prepare and measure the optimized state
measure_circuit = QuantumCircuit(H.num_qubits)
measure_circuit.compose(ansatz, inplace=True)
measure_circuit.measure_all()

# Bind the optimized parameters to the ansatz
bound_circuit = ansatz.assign_parameters(optimized_params)
bound_measure_circuit = measure_circuit.assign_parameters(optimized_params)

# Transpile the measurement circuit for the simulator
transpiled_measure_circuit = transpile(bound_measure_circuit, simulator)

# Run the circuit
simulator = AerSimulator()
counts_job = simulator.run(transpiled_measure_circuit, shots=10000)
counts = counts_job.result().get_counts()

# Calculate state probabilities
state_labels = sorted(counts.keys())
probabilities = [counts[label]/sum(counts.values()) for label in state_labels]

# Print measurement statistics
print("\nMeasurement results:")
for state, prob in zip(state_labels, probabilities):
    print(f"|{state}>: {prob:.4f}")

# Create a combined visualization
plt.figure(figsize=(12, 10))

# Plot counts histogram
plt.subplot(2, 1, 1)
plot_histogram(counts, title="Measurement Counts for Optimized State")

# Plot comparison with theoretical results
plt.subplot(2, 1, 2)
x = np.arange(len(state_labels))
width = 0.35

# Theoretical state for this simple Hamiltonian is a superposition of |00⟩ and |11⟩
theoretical_state = {
    "00": 0.5,
    "11": 0.5,
    "01": 0.0,
    "10": 0.0
}
theoretical_probs = [theoretical_state.get(label, 0) for label in state_labels]

plt.bar(x - width/2, probabilities, width, label='Measured')
plt.bar(x + width/2, theoretical_probs, width, label='Theoretical')
plt.xticks(x, state_labels)
plt.ylabel('Probability')
plt.title('Comparison with Theoretical State')
plt.legend()

plt.tight_layout()
plt.savefig('vqe_results.png')
plt.close()

print("\nVisualization files generated:")
print("1. vqe_convergence.png - Shows the energy convergence and term expectation values")
print("2. vqe_results.png - Shows measurement distributions and comparison with theory")
print("\nVQE Simulation complete!") 