# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:38:33 2020

@author: 766810
"""

elements = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Uut Fl Uup Lv Uus Uuo".split()

def decompose(word):
    """Express given word as chemical compound. If there are multiple solutions, return one of minimal weight."""
    progress = [False for x in range(len(word)+1)] # solution for word[:i]
    progress[0] = []

    for i in range(1, len(word)+1):
        possibles = list()
        for j in range(max(i-3,0), i):
            if progress[j] == False:
                continue
            alchemical = word[j:i].title()
            if alchemical in elements:
                possibles.append(progress[j] + [alchemical])

        if possibles:
            # choose minimal solution
            progress[i] = min(possibles, key=len)

    if progress[-1] == False:
        return False
    return "".join(progress[-1])

assert decompose("sine") == "SiNe" # rather than S-I-Ne
assert decompose("bismuth") == "BiSmUTh"
assert decompose("jam") == False

if __name__ == "__main__":
    import fileinput
    words = [line.strip() for line in fileinput.input()]
    words.sort(key=len)

    for word in words:
        result = decompose(word)
        if result:
            print(result)