# MIMO-DL
Detección MIMO con Deep Learning

******************************************************************************
Puede descargar y ejecutar localmente los archivos en un entorno de python como anaconda,
aunque se aconseja la ejecución de los notebooks en google colab. 

Se incluye un environment.

******************************************************************************
CONTENIDO
******************************************************************************
enviroment.yml 	   -> Entorno de programación

MIMO_numpy.py      -> Sistema MIMO que evalua los métodos de detección ZF, LMMSE y ML para
		         3 matrices del canal diferentes, generadas manualmente.
Comparison.py      -> Simulación del sistema en varios canales con matrices de coeficientes
			 aleatorios y con distribución normal
mimo_pytorch.py    -> Versión .py de 'MIMO_pytorch.ipynb'
mimo_nn.py	   -> Versión .py de 'MIMO_pytorch.ipynb'

labcomgid.py 	   -> Archivo en el que vienen definidas funciones para las prácticas de laboratorio
		         de comunicaciones digitales
functions.py  	   -> Archivo con nuevas funciones creadas para este proyecto
network.py	   -> Archivo en el que se definen varias clases usadas en 'MIMO_nn.ipynb'

MIMO_pytorch.ipynb -> Notebook con la versión del sistema basada en torch que simula un
		         sistema MIMO con detección los 3 métodos de detección
MIMO_nn.ipynb	   -> Programa en el que se usa una red neural usada para detectar los símbolos recibidos
			 y se compara con los métodos ZF, LMMSE y ML.
