from pyx import *
import random
import numpy as np
# Para usar instalar con sudo apt-get install python-pyx

"""
Generacion de datos

Dado dimX,dimY,simbols,k

Generar imagenes binarias de dimX,dimY con cantidades variables de simbolos (1-k).
Los simbolos aparecen en posiciones aleatorias dentro del canvas de dimX,dimY.
Tienen diversos tamanos y sufren rotaciones,escalamientos,op-morfologicas.


"""

def notebook_pos_distribution(elements,n_lines,width,height):
    dx = width*1.0/elements * n_lines

    outcords = []

    y_fixed = np.linspace(height*0.1 ,height*0.9,n_lines)
    x_last = [0 for i in range(n_lines)]
    x_random = (dx*0.3) +dx*np.random.rand(elements)

    for i in range(elements):

        line_index = random.randint(0,n_lines-1)
        x_cord =  x_last[line_index] + x_random[i]
        x_last[line_index]=x_cord
        y_cord = y_fixed[line_index]

        outcords.append((x_cord,y_cord))

    return outcords


    pass

alfabeta = [r"\alpha",r"\theta",r"o",r"\tau",r"\beta",r"\vartheta","\pi",r"\upsilon",
"\gamma",r"\varpi","\phi","\delta","\kappa",r"\rho",r"\varphi",
"\epsilon","\lambda",r"\varrho","\chi",
r"\varepsilon","\mu","\sigma","\psi","\zeta",r"\nu",r"\varsigma","\omega","\eta",r"\xi","\Gamma","\Lambda","\Sigma","\Psi","\Delta","\Xi",r"\Upsilon","\Omega","\Theta","\Pi","\Phi"]

# Selecciona lista de simbolos latex
simbols =  ["<",">","\leq","\geq","=","\int","\partial",'\sum','\prod']


all_simbols = alfabeta + [str(i) for i in range(10)] + simbols
# Set properties of the defaulttexrunner, e.g. switch to LaTeX.
#text.set(text.LatexRunner)



to_g = 50 #random.randint(0,1000)

dim = 200

# Set a box in the corners to assure fixed image size
c = canvas.canvas([canvas.clip(path.rect_pt(0, 0,dim, dim))])
# it also need a drawed element there otherwise it get croped.
c.stroke(path.rect_pt(0, 0,dim, dim))

all_positions = notebook_pos_distribution(to_g,6,dim,dim)

for i in range(to_g):
    s = random.randint(1,len(all_simbols)-1)
    chosen = all_simbols[s]

    x,y = all_positions[i]#random.random() * dim,random.random() * dim

    # Aplicar transformaciones lineales
    #http://pyx.sourceforge.net/manual/trafo.html?highlight=trafo#module-trafo
    print(chosen)
    size = random.randint(1,5)

    # deformar suavemente
    t1=trafo.scale(random.normalvariate(1, 0.3), random.normalvariate(1, 0.3))

    # rotar x grados
    t2=trafo.rotate(random.randint(-20, 20))

    transform = t1 * t2
    a=c.text_pt(x, y, "$ {0} $".format(chosen),[text.size(size),transform])

    # SAVE BOUND BOX
    c.stroke(a.bbox().path())

    # Realizar operaciones morfologicas

    # tirar lineas de ruido


# c.writePDFfile("texrunner")
# c.writeSVGfile("texrunner")

# Resolucion en dots per inch
c.writeGSfile("out.png","png256",resolution=100)

