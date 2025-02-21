import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from time import sleep

class Qbit():
    def __init__(self):
        self.alpha = complex(0)
        self.beta = complex(1)
        self.sqrt_2 = np.sqrt(2)

        # Verificación de normalización
        if abs(self.alpha)**2 + abs(self.beta)**2 != 1:
            print("Los valores de alpha y beta no son adecuados, se establecerán valores por defecto.")
            self.alpha = complex(0)
            self.beta = complex(1)

        # Define el ket como un vector columna
        self.ket_normal = np.array([[self.alpha], [self.beta]], dtype=complex)
        
        # Transpuesto de ket_normal, es decir, el bra
        self.ket_traspuesto = self.ket_normal.T
        
        # El conjugado transpuesto (ket_daga)
        self.ket_daga = np.conjugate(self.ket_traspuesto)

        # La matriz de densidad...
        self.matriz_densidad = np.array([[abs(self.alpha), self.alpha*(self.beta.conjugate())], 
                                        [self.alpha.conjugate()*self.beta, abs(self.beta)]], 
                                        dtype=complex)
    


    def medir_qubit(self, ket : list, shots : int):
        """
        Inicializa un qubit en el estado |ψ⟩ = α|0⟩ + β|1⟩, simula la medición
        y devuelve los conteos de resultados.

        Parámetros:
            alpha (array:complex): Coeficiente para el estado |0⟩.
            beta (array:complex): Coeficiente para el estado |1⟩.
            shots (int): Número de simulaciones de la medición.

        Retorna:
            dict: Un diccionario con el conteo de las mediciones {'0': num, '1': num}.
        """

        from qiskit.quantum_info import Statevector
        alpha = ket[0] 
        beta =  ket[1]
        # Normalización de los coeficientes
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        alpha /= norm
        beta  /= norm

        # Definimos el estado del qubit
        estado = [alpha, beta]

        # Creamos el Statevector
        sv = Statevector(estado)

        # Simulamos la medición y obtenemos los conteos
        counts = sv.sample_counts(shots=shots)
        return counts




    def producto_tensorial(matriz1, matriz2):
        """__summary__

        Args:
            matriz1 (array): Pues una matriz de cualquier tipo
            matriz2 (array): Lo mismo que la matriz 1 pero es otra distinta(bueno, no tiene pq serlo)

        Returns:
            array: Devuelve el producto tensorial de las 2 matrices
        """
        return matriz1 @ np.kron(np.eye(matriz1.shape[0]), matriz2)







    ####################### INCIO COMPUERTAS LÓGICAS




    def i(self, ket):
        # De identidad, no cambia el estado de qbit(por que mierda querrías utilizar esto)
        compuerta_logica = np.array([[1, 0], [0, 1]], dtype=complex)
        return compuerta_logica @ ket



    def x(self, ket):
        # Pauli-X también llamada NOT
        compuerta_logica = np.array([[0, 1], [1, 0]], dtype=complex)
        return compuerta_logica @ ket



    def h(self, ket):
        # hadammard



        compuerta_logica = np.array([[1/self.sqrt_2, 1/self.sqrt_2], 
                                    [1/self.sqrt_2, -1/self.sqrt_2]], dtype=complex)
        return compuerta_logica @ ket



    def y(self, ket):
        # Pauli-Y rota 180º en Y en Bloch
        compuerta_logica = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return compuerta_logica @ ket



    def z(self, ket):
        # Pauli-Z rota 180º en Z en Bloch
        compuerta_logica = np.array([[1, 0], [0, -1]], dtype=complex)
        return compuerta_logica @ ket



    def cnot(self, control, target):
        # Representación de CNOT en un sistema de 2 qubits (4x4)
        compuerta_logica = np.array([[1, 0, 0, 0], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 0, 1], 
                                    [0, 0, 1, 0]], dtype=complex)
        # Crear el estado combinado de los dos qubits
        ket_2q = np.kron(control, target)  # Producto de Kronecker
        return compuerta_logica @ ket_2q



    def ccnot(self, control1, control2, target):
        # Compuerta CCNOT


        # Representación de CCNOT (Toffoli) en un sistema de 3 qubits (8x8)
        compuerta_logica = np.array([[1, 0, 0, 0, 0, 0, 0, 0], 
                                        [0, 1, 0, 0, 0, 0, 0, 0], 
                                        [0, 0, 1, 0, 0, 0, 0, 0], 
                                        [0, 0, 0, 1, 0, 0, 0, 0], 
                                        [0, 0, 0, 0, 1, 0, 0, 0], 
                                        [0, 0, 0, 0, 0, 1, 0, 0], 
                                        [0, 0, 0, 0, 0, 0, 0, 1], 
                                        [0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex)
        # Crear el estado combinado de los tres qubits
        ket_3q = np.kron(np.kron(control1, control2), target)  # Producto de Kronecker
        return compuerta_logica @ ket_3q
    


    def r(self, ket, theta):
        # pauli de Rotación, rota theta sobre cualquier eje
        compuerta_logica = np.array([[np.e**(-complex(0, 1)*((theta)/(2))), 0], 
                                    [0, np.e**(complex(0, 1)*((theta)/(2)))]], 
                                    dtype=complex)
        return compuerta_logica @ ket
    

    def s(self, ket):
        # Introduce una fase de pi/2 (90º) al qbit |1>
        compuerta_logica = np.array([[1, 0], 
                                    [0, complex(0, 1)]], 
                                    dtype=complex)
        return compuerta_logica @ ket
    


    def t(self, ket):
        # Introduce una fase de pi/4 (45º)
        compuerta_logica = np.array([[1, 0], 
                                    [0, np.e**(complex(0, 1)*((np.pi)/(4)))]], 
                                    dtype=complex)
        return compuerta_logica @ ket


    
    ####################### FIN COMPUERTAS LÓGICAS




    def bloch(self, ket):
        
        """_summary_

        Args:
            ket (array_complex:float): [[alpha], [beta]]

        Returns:
            array_real:float: theta and phi in the bloch sphere 
        """
        alpha = ket[0][0]
        beta = ket[0][0]
        theta = 2*np.arccos(abs(alpha))
        phi = np.angle(beta) #arg()
        return [theta, phi]
        


    
    def plot_bloch_sphere(self, ket):
        """_summary_

        Args:
            list (array:real_float): de entrada toma un ket de toda la vida 

        Return:
            ...: Una matplotlib window representando la esfera de bloch 
        """
        theta = ket[0]
        phi = ket[1]
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

        # Etiquetas y ajustes
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


"""
# Pruebas
qbit1 = Qbit()
# Estado inicial de los qubits
ket1 = qbit1.ket_normal
ket1 = qbit1.h(ket1)

print("Resultado:\n", ket1)
shots = 2**10
while True:
    resultado = qbit1.medir_qubit(ket1, shots)
    print(resultado)
    print("Delta:\n", abs(resultado['0']-resultado['1']))
    sleep(0.5)
"""
