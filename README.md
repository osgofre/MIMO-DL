# MIMO-DL

## Información general

Este repositorio recoge los archivos utilizados en la realización del Trabajo de Fin de Grado de Ingeniería de Telecomunicaciones ***Detección MIMO con Deep Learning***, realizado por Óscar González Fresno, con la tutela de Juan José Murillo Fuentes, Catedrático de la Universidad de Sevilla.

El proyecto se divide en 3 partes:

1. Simulación de Sistema MIMO en Python
2. Simulación de Sistema MIMO en PyTorch
3. Diseño y simulación de detección MIMO basada en aprendizaje profundo

Puede descargar y ejecutar localmente los archivos en un entorno de python como anaconda, aunque se aconseja la ejecución de los notebooks en google colab, aprovechando la acelaración por hardware y cargando los archivos _labcomgid.py_, _functions.py_ y _network.py_.   

Se incluye un environment.


## Contenido

* _enviroment.yml_ - Entorno de programación
* _MIMO_numpy.py_ - Sistema MIMO que evalua los métodos de detección ZF, LMMSE y ML para
		         3 matrices del canal diferentes, generadas manualmente.
* _Comparison.py_ - Simulación del sistema en varios canales con matrices de coeficientes
			 aleatorios y con distribución normal
* _mimo_pytorch.py_   	- Versión _.py_ de _MIMO_pytorch.ipynb_
* _mimo_nn.py_	   	- Versión _.py_ de _MIMO_pytorch.ipynb_

* _labcomgid.py_ 	- Archivo en el que vienen definidas funciones para las prácticas de laboratorio
		         de comunicaciones digitales
* _functions.py_  	- Archivo con nuevas funciones creadas para este proyecto
* _network.py_	   	- Archivo en el que se definen varias clases usadas en _MIMO_nn.ipynb_

* _MIMO_pytorch.ipynb_ 	- Notebook con la versión del sistema basada en torch que simula un
		         sistema MIMO con detección los 3 métodos de detección
* _MIMO_nn.ipynb_	- Programa en el que se usa una red neural usada para detectar los símbolos recibidos
			 y se compara con los métodos ZF, LMMSE y ML.
