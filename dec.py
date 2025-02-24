import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

class Qbit:
    def __init__(self, alpha=0, beta=1):
        self.alpha = complex(alpha)
        self.beta = complex(beta)
        self.sqrt_2 = np.sqrt(2)
        
        # Normalización del estado
        norm = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        if norm != 1:
            print("Los valores de alpha y beta no son adecuados, se normalizarán.")
            self.alpha /= norm
            self.beta /= norm

        # Define el ket como un vector columna
        self.ket_normal = np.array([[self.alpha], [self.beta]], dtype=complex)
        
        # Transpuesto de ket_normal (bra)
        self.ket_traspuesto = self.ket_normal.T
        
        # Conjugado transpuesto (ket_daga)
        self.ket_daga = np.conjugate(self.ket_traspuesto)

        # Matriz de densidad
        self.matriz_densidad = np.outer(self.ket_normal, self.ket_daga)
    
    def medir_qubit(self, shots):
        """
        Simula la medición del qubit y devuelve un diccionario con las frecuencias de medición.
        Además, actualiza el estado del qubit (alpha y beta) al resultado medido.
        """
        sv = Statevector([self.alpha, self.beta])
        counts = sv.sample_counts(shots=shots)

        # Tomamos el resultado más frecuente para simular la medición
        resultado_medido = max(counts, key=counts.get)

        # Actualizar el estado del qubit al estado medido (colapso del estado)
        if resultado_medido == '0':
            self.alpha, self.beta = 1, 0
        elif resultado_medido == '1':
            self.alpha, self.beta = 0, 1

        # Actualizamos también el ket_normal
        self.ket_normal = np.array([[self.alpha], [self.beta]], dtype=complex)

        return counts
    

    @staticmethod
    def producto_tensorial(matriz1, matriz2):
        """ Calcula el producto tensorial de dos matrices. """
        return np.kron(matriz1, matriz2)
    
    ####################### COMPUERTAS LÓGICAS #######################

    def aplicar_compuerta(self, compuerta):
        """ Aplica una compuerta cuántica al qubit. """
        self.ket_normal = compuerta @ self.ket_normal
        self.alpha, self.beta = self.ket_normal.flatten()
        return self.ket_normal
    
    def i(self):
        return self.aplicar_compuerta(np.eye(2, dtype=complex))

    def x(self):
        return self.aplicar_compuerta(np.array([[0, 1], [1, 0]], dtype=complex))
    
    def h(self):
        return self.aplicar_compuerta(np.array([[1/self.sqrt_2, 1/self.sqrt_2], 
                                                [1/self.sqrt_2, -1/self.sqrt_2]], dtype=complex))
    
    def y(self):
        return self.aplicar_compuerta(np.array([[0, -1j], [1j, 0]], dtype=complex))
    
    def z(self):
        return self.aplicar_compuerta(np.array([[1, 0], [0, -1]], dtype=complex))
    
    def r(self, theta):
        return self.aplicar_compuerta(np.array([[np.exp(-1j * theta / 2), 0], 
                                                [0, np.exp(1j * theta / 2)]], dtype=complex))
    
    def s(self):
        return self.aplicar_compuerta(np.array([[1, 0], [0, 1j]], dtype=complex))
    
    def t(self):
        return self.aplicar_compuerta(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex))
    
    ####################### FIN COMPUERTAS LÓGICAS #######################
    
    def bloch(self):
        """ Devuelve los ángulos theta y phi en la esfera de Bloch. """
        theta = 2 * np.arccos(abs(self.alpha))
        phi = np.angle(self.beta)
        return [theta, phi]
    
    def plot_bloch_sphere(self):
        """ Dibuja la esfera de Bloch con el estado del qubit. """
        theta, phi = self.bloch()
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Dibujar la esfera de Bloch
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color='c', alpha=0.1, edgecolor='k', linewidth=0.2)

        # Dibujar los ejes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=2, label='|+⟩')
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=2, label='|i⟩')
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=2, label='|0⟩')
        ax.quiver(0, 0, 0, 0, 0, -1, color='b', linewidth=2)

        # Coordenadas del vector en la esfera de Bloch
        x_q = np.sin(theta) * np.cos(phi)
        y_q = np.sin(theta) * np.sin(phi)
        z_q = np.cos(theta)

        # Dibujar el vector del estado del qubit
        ax.quiver(0, 0, 0, x_q, y_q, z_q, color='k', linewidth=3, label='Qubit')

        # Ajustes de la gráfica
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=30)
        ax.legend()
        plt.show()