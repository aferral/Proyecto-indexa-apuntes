from pyx import *
import random
# Para usar instalar con sudo apt-get install python-pyx

"""
Generacion de datos

Dado dimX,dimY,simbols,k

Generar imagenes binarias de dimX,dimY con cantidades variables de simbolos (1-k).
Los simbolos aparecen en posiciones aleatorias dentro del canvas de dimX,dimY.
Tienen diversos tamanos y sufren rotaciones,escalamientos,op-morfologicas.


"""

# Selecciona lista de simbolos latex
simbols = ['\lambda','\sum','\prod']

# Set properties of the defaulttexrunner, e.g. switch to LaTeX.
text.set(text.LatexRunner)

c = canvas.canvas()

to_g = random.randint(0,10)

dim = 200

c.stroke(path.rect(0, 0,10, 10))


for i in range(to_g):
    s = random.randint(1,len(simbols)-1)
    x,y = random.random() * dim,random.random() * dim

    # Aplicar transformaciones

    print(simbols[s])
    c.text_pt(x, y, "$ {0} $".format(simbols[s]))

# c.writePDFfile("texrunner")
# c.writeSVGfile("texrunner")

# Resolucion en dots per inch

c.writeGSfile("out.png","png256",resolution=100)

