import math
from random import *
import random

import numpy as np



# function HammingG
# input: a number r
# output: G, the generator matrix of the (2^r-1,2^r-r-1) Hamming code
def hammingGeneratorMatrix(r):
    n = 2 ** r - 1

    # construct permutation pi
    pi = []
    for i in range(r):
        pi.append(2 ** (r - i - 1))
    for j in range(1, r):
        for k in range(2 ** j + 1, 2 ** (j + 1)):
            pi.append(k)

    # construct rho = pi^(-1)
    rho = []
    for i in range(n):
        rho.append(pi.index(i + 1))

    # construct H'
    H = []
    for i in range(r, n):
        H.append(decimalToVector(pi[i], r))

    # construct G'
    GG = [list(i) for i in zip(*H)]
    for i in range(n - r):
        GG.append(decimalToVector(2 ** (n - r - i - 1), n - r))

    # apply rho to get Gtranpose
    G = []
    for i in range(n):
        G.append(GG[rho[i]])

    # transpose
    G = [list(i) for i in zip(*G)]

    return G


def decimalToVector(n, r):
    v = []
    for s in range(r):
        v.insert(0, n % 2)
        n //= 2
    return v


def simulation(r, N, p):


    # creates a random message
    def randomMessage(r):
        x = randint(0, (2 ** r - r - 1) ** 2)
        return decimalToVector(x, (2 ** r - r - 1))


    # encodes the message
    def encoder(m):
        generator_matrix = hammingGeneratorMatrix(r)
        generator_matrix = np.array(generator_matrix)
        identity_matrix = np.identity(r ** 2 - 1)
        m_matrix = np.c_[identity_matrix, np.ones(r ** 2 - 1)]
        g_prime = np.dot(generator_matrix, m_matrix)
        codeword = np.dot(m, g_prime)
        codeword = [int(item % 2) for item in codeword]
        return codeword


    # passes message through channel which could cause an error
    def BSC(c, p):
        for i in range(len(c)):
            if random.random() < p:
                c[i] = (c[i] + 1) % 2

        return c


    # corrects any errors found and returns codeword estimate
    def syndrome(v):
        parity_check_matrix = []

        for i in range(1, 2 ** r):
            parity_check_matrix.append(decimalToVector(i, r))
        parity_check_matrix = np.array(parity_check_matrix)
        parity_check_matrix = parity_check_matrix.transpose()

        parity_check_matrix = np.c_[parity_check_matrix, np.zeros(r)]
        parity_check_matrix = np.vstack([parity_check_matrix, np.ones(r ** 2)])

        parity_check_matrix = parity_check_matrix.transpose()
        #print(parity_check_matrix)
        parity_check_matrix = np.array(parity_check_matrix, dtype=bool)
        v = np.array(v, dtype=bool)
        vh = 1 * np.dot(v, parity_check_matrix)
        # vh = np.asarray(vh)
        vh = vh.tolist()

        #print(parity_check_matrix)
        #print(vh)
        final_char = vh[-1]
        #print(final_char)
        del vh[-1]
        if final_char%2 == 1:
            s_f_e = 3
        vh_string = ''
        for item in vh:
            vh_string = vh_string + str(item)
        #print(vh_string)
        char_to_change = int(vh_string, 2)
        #print(char_to_change)
        if v[char_to_change - 1] == 0:
            v[char_to_change - 1] = 1
        if v[char_to_change - 1] == 1:
            v[char_to_change - 1] = 0
        print("Codeword estimate: " + str(1*v))
        return 1*v


    def retrieve_message(c):
        indexes = []
        c = c.tolist()
        highest_power_two = math.floor(math.log(len(c), 2))
        for i in range(highest_power_two+1):
            indexes.append(2**i)
        indexes.reverse()
        print(c)
        print(indexes)
        for index in indexes:
            del c[index - 1]
        return c

    """
    # setup variables
    s_f_e = 0
    successes = 0
    failures = 0
    errors = 0
    r = 4
    N = 1
    p = 0.2
    message = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    received_vector = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    retrieve_string = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    
    #run functions
    syndrome(received_vector)
    print(retrieve_message(retrieve_string))
    #print(encoder(message))
    #print(BSC(encoder(message), 0.2))
    """
    successes = 0
    failures = 0
    errors = 0
    for i in range(N):
        s_f_e = 0
        # run functions final
        message = randomMessage(r)
        codeword = encoder(message)
        received_vector = BSC(codeword, p)
        codeword_estimate = syndrome(received_vector)
        message_estimate = retrieve_message(codeword_estimate)
        if(s_f_e == 3):
            errors += 1
            print("Error")
        elif message_estimate == message:
            successes += 1
            print("Success")
        elif message_estimate != message:
            failures += 1
            print("Failure")
        else:
            print("Should never ever go here")


simulation(4, 3, 0.2)


