import math
import numpy as np
import matplotlib.pyplot as plt




# On considere -nu u''(x) + b u'(x) + c(x)u(x) = f(x) pour x dans [0,1]
# avec les conditions initiales u(0) = 0 et u(1) = 1

# On defini ci-apres nu, b, c et f
# Pour pouvoir appeler c et f les vecteurs discretises on note

nu = 1

b = 10

def c(x):
    return 1

def f(x):
    return 0


# On defini J et donc le pas de discretisation

J = 100
h = 1/(J+1)


# On cherche V avec les indices 1 a J que l'on numerote V[0] a V[J-1]

V = np.zeros(J, float)


# Sur la base de notre etude on va choisir le schema le plus adapte

lbda = b*h/nu

if lbda < -2 : # lambda < -2
    
    print ("lambda = %e, on choisit le schema aval"%lbda)

    Ah_aval = np.zeros((J, J), float)

    for j in range(J):
    
        # ATTENTION : j varie entre 0 et J-1 et
        # la composante a la ligne 1, colonne 1 est A[0,0]
        
        # diagonale
        Ah_aval[j,j] = 2*nu/h**2 - b/h + c((j+1)*h)
        
        # sur-digonale
        if j<J-1:
            Ah_aval[j,j+1] = -nu/h**2+b/h
            
        # sous-diagonale
        if j>0:
            Ah_aval[j,j-1] = -nu/h**2
            
            
    Fh_aval = np.zeros(J, float)
            
    Fh_aval[0] = f(h) + nu/h**2  # F[0] : ligne 1 
            
    for j in range(1,J): # F[1] a F[J-1] : lignes 2 a J 
        Fh_aval[j] = f((j+1)*h)


    V = np.linalg.solve(Ah_aval,Fh_aval)
                    

                    
elif lbda <2 : # lambda entre -2 et 2
                
    print ("lambda = %e, on choisit le schema centre"%lbda)

    Ah_centre = np.zeros((J, J), float)

    for j in range(J):

        # ATTENTION : j varie entre 0 et J-1 et
        # la composante a la ligne 1, colonne 1 est A[0,0]

        # diagonale
        Ah_centre[j,j] = 2*nu/h**2 + c((j+1)*h)

        # sur-digonale
        if j<J-1:
            Ah_centre[j,j+1] = -nu/h**2+b/(2*h)

        # sous-diagonale
        if j>0:
            Ah_centre[j,j-1] = -nu/h**2-b/(2*h)


    Fh_centre = np.zeros(J, float)

    Fh_centre[0] = f(h) + nu/h**2 +b/(2*h) # F[0] : ligne 1 

    for j in range(1,J): # F[1] a F[J-1] : lignes 2 a J 
        Fh_centre[j] = f((j+1)*h)


    V = np.linalg.solve(Ah_centre,Fh_centre)

    
else: # lambda > 2
    
    print ("lambda = %e, on choisit le schema amont"%lbda)
    
    Ah_amont = np.zeros((J, J), float)

    for j in range(J):

        # ATTENTION : j varie entre 0 et J-1 et
        # la composante a la ligne 1, colonne 1 est A[0,0]

        # diagonale
        Ah_amont[j,j] = 2*nu/h**2 + b/h + c((j+1)*h)

        # sur-digonale
        if j<J-1:
            Ah_amont[j,j+1] = -nu/h**2

        # sous-diagonale
        if j>0:
            Ah_amont[j,j-1] = -nu/h**2-b/h


    Fh_amont = np.zeros(J, float)

    Fh_amont[0] = f(h) + nu/h**2 +b/h # F[0] : ligne 1 

    for j in range(1,J): # F[1] a F[J-1] : lignes 2 a J 
        Fh_amont[j] = f((j+1)*h)


    V = np.linalg.solve(Ah_amont,Fh_amont)


# On ajoute a chaque extremite les valeurs connues de V
# Cela a pour effet d'avoir desormais V[j] qui est la j-eme composante de V

V = np.concatenate((np.array([1]),V,np.array([0])))


# Travons la fonction trouvee en interpolant lineairement entre les noeuds

x = np.arange(J+2,dtype=float)

for j in range(J+2):
    x[j] = j*h

plt.title("Solution de -%.2fu''(x) + %.2fu'(x) + c(x)u(x) = f(x)\nu(0)=1, u(1)=1\nc et f sont deux fonctions indiquees dans le programme."%(nu,b))
plt.plot(x,V,'r')
plt .show()
