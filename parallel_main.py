from scipy.stats import *
from scipy.integrate import quad
import numpy as np
import math
import random
import concurrent.futures

F = 14 #Number of sub-channels
Tslt = 0.005 #Time slot duration
Tth = 0.08 #Time threshold
SINRTh = 10 #SINR threshold
Omega = 2 #Rayleigh Fading parameter
c = 3 * 1e8
d0 = 10 #Reference distance
f = 2.4 * 1e9 #Operating frequency
k = 1.38 * 1e-23 #Boltzmann constant
T = 290 #Temperature
BW = 1e8 #Bandwidth
sen = 30 #Sensitivity
normalizedBuffer = 50 #Normalized buffer size
GaNoise = k * T * BW #Noise power
xMin = -50
xMax = 50
yMin = -50
yMax = 50
zMin = 0
zMax = 2
xDelta = xMax - xMin
yDelta = yMax - yMin
zDelta = zMax - zMin
areaTotal = xDelta * yDelta * zDelta
lambda0 = 0.001 #Newtork density
size = np.random.poisson(lambda0 * areaTotal) #Number of nodes
if size % 2 == 1: #Even number of nodes
    size += 1
x = xDelta * np.random.uniform(0, 1, size) + xMin
y = yDelta * np.random.uniform(0, 1, size) + yMin
z = zDelta * np.random.uniform(0, 1, size - 2) + zMin
z = np.insert(z, 0, 60) #Interferer UAV altitude
z = np.insert(z, 0, 50) #Streamer UAV altitude
Ln = [3.04 for i in range(size)] #Average video packet length
Beta = [2 for i in range(size)] #Channel fading threshold
Lambdan = [100 for i in range(size)] #Incoming packet rate
Pt = [0.2 for i in range(size)] #Transmit power
mp = [size - i - 1 for i in range(size)] #All nodes' indices


def PLoS(s, n, zet=20, v=3 * 1e-4, mi=0.5): #Line of sight probability
    if z[s] == z[n]:
        print("Same Height")
        return (1 - np.exp(-(z[s] ** 2) / (2 * (zet ** 2)))) ** (
                math.dist((x[s], y[s], z[s]), (x[n], y[n], z[n])) * np.sqrt(v * mi))
    else:
        return (1 - (((np.sqrt(2 * np.pi) * zet) / math.dist((0, 0, z[s]), (0, 0, z[n]))) * np.abs(
            (1 - norm.cdf(z[s] / zet)) - (1 - norm.cdf(z[n] / zet))))) ** (
                math.dist((x[s], y[s], 0), (x[n], y[n], 0)) * np.sqrt(v * mi))


def b(s, n):
    return np.sqrt(2 * np.exp(2.708 * PLoS(s, n) ** 2))


def RorR(s, n):
    if PLoS(s, n) < 0.5:
        return 0
    else:
        return 1


def Miun(betan, s, n):
    if RorR(s, n) == 1:
        return 1 - ((rice.cdf(betan, b(s, n))) ** F)
    else:
        return 1 - ((rayleigh.cdf(betan)) ** F)


def Pdly(s, n): #Time threshold probability
    return np.exp(-Tth * ((Miun(Beta[s], s, n) / Tslt) - Lambdan[s]))


def rho(s, n):
    return Lambdan[s] * Tslt / (Miun(Beta[s], s, n))


def Pov(s, n): #Buffer overflow probability
    return ((1 - rho(s, n)) * np.exp(-normalizedBuffer * (1 - rho(s, n)))) / (
            1 - rho(s, n) * np.exp(-normalizedBuffer * (1 - rho(s, n))))


def hn(s, n): # Path loss model
    return (c / (4 * np.pi * f)) * np.sqrt(
        (d0 ** ((-1.5 * PLoS(s, n) + 3.5) - 2)) / ((math.dist((x[s], y[s], z[s]), (x[n], y[n], z[n]))) ** (-1.5 * PLoS(s, n) + 3.5)))


def EIfunc(x, RorR, b):
    if RorR == 0:
        return (x ** 2) * (rayleigh.pdf(x))
    else:
        return (x ** 2) * (rice.pdf(x, b))


def EI(m, s, n):
    sum = 0
    for i in range(len(m)):
        sum += Pt[m[i]] * (hn(m[i], n) ** 2) * (
            (quad(lambda x: EIfunc(x, RorR(m[i], n), b(m[i], n)), Beta[m[i]], 10))[
                0]) * (Miun(Beta[m[i]], s, n) / F)
    return sum


def DIfunc(x, RorR, b):
    if RorR == 0:
        return (x ** 4) * (rayleigh.pdf(x))
    else:
        return (x ** 4) * (rice.pdf(x, b))


def DI(m, s, n):
    sum = 0
    sum1 = 0
    for i in range(len(m)):
        j = 0
        sum += (Pt[m[i]] ** 2) * (hn(m[i], n) ** 4) * (
            (quad(lambda x: DIfunc(x, RorR(m[i], n), b(m[i], n)), Beta[m[i]], 10))[
                0]) * ((Miun(Beta[m[i]], s, n) / F) ** 2)
        while j < i:
            sum1 += 2 * Pt[m[i]] * (hn(m[i], n) ** 2) * (
                (quad(lambda x: EIfunc(x, RorR(m[i], n), b(m[i], n)), Beta[m[i]], 10))[
                    0]) * (
                            Miun(Beta[m[j]], s, n) / F) * Pt[m[j]] * (hn(m[j], n) ** 2) * (
                        (quad(lambda x: EIfunc(x, RorR(m[j], n), b(m[j], n)), Beta[m[j]], 10))[0]) * (
                            Miun(Beta[m[j]], s, n) / F)
            j += 1
    return sum + sum1 - (EI(m, s, n) ** 2)


def locmu(m, s, n):
    return np.log(EI(m, s, n)) - np.log(1 + (DI(m, s, n) / (EI(m, s, n) ** 2))) / 2


def sclsigma(m, s, n):
    return np.sqrt(np.log(1 + (DI(m, s, n) / (EI(m, s, n) ** 2))))


def Perrfunc(x, m, s, n):
    if RorR(s, n) == 0:
        return rayleigh.pdf(x) * (0.5 - (0.5 * math.erf((np.log(
            (Pt[s] * (hn(s, n) ** 2) * (x ** 2) / SINRTh) - GaNoise) - locmu(m, s, n)) / (
                                                                np.sqrt(2) * sclsigma(m, s, n)))))
    else:
        return rice.pdf(x, b(s, n)) * (0.5 - (0.5 * math.erf((np.log(
            (Pt[s] * (hn(s, n) ** 2) * (x ** 2) / SINRTh) - GaNoise) - locmu(m, s, n)) / (
                                                                     np.sqrt(2) * sclsigma(m, s, n)))))


def Perr(m, s, n):
    return quad(lambda x: Perrfunc(x, m, s, n), Beta[s], 10)[0]


def BetaUpRayleigh(s):
    return np.sqrt(- Omega * np.log(1 - (1 - Tslt * Lambdan[s]) ** (1 / F)))


def BetaUpRician(s, n, betan):
    if rice.cdf(betan, b(s, n)) <= ((1 - Lambdan[s] * Tslt) ** (1 / F)):
        return betan
    else:
        return 0


def BetamUpRician(s, n):
    ini_betam = 2
    while BetaUpRician(s, n, ini_betam) != 0:
        ini_betam += 0.01
    return ini_betam


def Ploss(m, s, n): #Overall packet loss probability
    plss = Pov(s, n) + ((1 - Pov(s, n)) * Pdly(s, n)) + ((1 - Pov(s, n)) * (1 - Pdly(s, n)) * Perr(m, s, n))
    if plss < 1:
        return plss
    else:
        return 1


def Rn(m, s, n): #Expected throughput
    return Lambdan[s] * (1 - Ploss(m, s, n))


def DisLoss(m, s, n): #Packet loss distortion
    return sen * Ploss(m, s, n)


def DisCMP(s, D0=1.18, E0=0.67, Theta0=858): #Compression distortion
    return D0 + (Theta0 / ((Lambdan[s] * Ln[s]) - E0))


def PSNR(m, s, n): #PSNR
    return 10 * np.log10(255 ** 2 / (DisLoss(m, s, n) + DisCMP(s)))


def adj_dis(s, n, add_dis): #Adjust distance with the same angle
    old_dis = math.dist((x[s], y[s], z[s]), (x[n], y[n], z[n]))
    old_angle = math.atan(math.dist((0, 0, z[s]), (0, 0, z[n])) / math.dist((x[s], y[s], 0), (x[n], y[n], 0)))
    print('Streamer UAV Old Location:', x[s], y[s], z[s])
    ratio = (old_dis + add_dis) / old_dis
    if xMax > (x[n] - ratio * (x[n] - x[s])) > xMin and yMax > (y[n] - ratio * (y[n] - y[s])) > yMin and (
            z[n] - ratio * (z[n] - z[s])) > 0:
        x[s] = x[n] - ratio * (x[n] - x[s])
        y[s] = y[n] - ratio * (y[n] - y[s])
        z[s] = z[n] - ratio * (z[n] - z[s])
    print('Streamer UAV New Location:', x[s], y[s], z[s])
    new_dis = math.dist((x[s], y[s], z[s]), (x[n], y[n], z[n]))
    new_angle = math.atan(math.dist((0, 0, z[s]), (0, 0, z[n])) / math.dist((x[s], y[s], 0), (x[n], y[n], 0)))
    return old_angle * 180 / np.pi, new_angle * 180 / np.pi, old_dis, new_dis


def adj_angle(s, n, add_angle): #Adjust elevation angle with the same distance
    old_dis = math.dist((x[s], y[s], z[s]), (x[n], y[n], z[n]))
    old_angle = math.atan(math.dist((0, 0, z[s]), (0, 0, z[n])) / math.dist((x[s], y[s], 0), (x[n], y[n], 0)))
    print('UAV Old Location:', x[s], y[s], z[s])
    new_angle = old_angle + (add_angle * np.pi / 180)
    if yMax > y[n] > yMin and (z[n] + (old_dis / np.sqrt((1 / (np.tan(new_angle) ** 2)) + 1))) > 0:
        z[s] = z[n] + (old_dis / np.sqrt((1 / (np.tan(new_angle) ** 2)) + 1))
        if xMax > (x[n] + np.sqrt(abs(old_dis ** 2 - (y[s] - y[n]) ** 2 - (z[s] - z[n]) ** 2))) > xMin:
            x[s] = x[n] + np.sqrt(abs(old_dis ** 2 - (y[s] - y[n]) ** 2 - (z[s] - z[n]) ** 2))
    print('UAV New Location:', x[s], y[s], z[s])
    new_dis = math.dist((x[s], y[s], z[s]), (x[n], y[n], z[n]))
    new_angle = math.atan(math.dist((0, 0, z[s]), (0, 0, z[n])) / math.dist((x[s], y[s], 0), (x[n], y[n], 0)))
    return old_angle * 180 / np.pi, new_angle * 180 / np.pi, old_dis, new_dis


def BLCS(bst_betan, bst_rn, j, m, n, stpdiv=0.5, stpini=0.01, stpth=0.01): #Beta local coordinate search
    stp = stpini * stpdiv
    flag = 0
    k = 0
    while flag != 4:
        while flag == 0:
            stp /= stpdiv
            if (RorR(mp[j], n) == 1 and BetaUpRician(mp[j], n,
                                                                      bst_betan + stp) != 0) or (
                    RorR(mp[j], n) == 0 and bst_betan + stp < BetaUpRayleigh(mp[j])):
                Beta[mp[j]] = bst_betan + stp
            rn_can = PSNR(m, mp[j], n)
            if rn_can > bst_rn:
                bst_rn = rn_can
                bst_betan = Beta[mp[j]]
                k += 1
            elif k != 0 and stpdiv != 1:
                flag = 1
            elif k != 0 and stpdiv == 1:
                flag = 4
            elif k == 0:
                flag = 2
                stp *= stpdiv
        while flag == 1:
            stp *= stpdiv
            if stp < stpth:
                flag = 4
                break
            if (RorR(mp[j], n) == 1 and BetaUpRician(mp[j], n,
                                                                      bst_betan + stp) != 0) or (
                    RorR(mp[j], n) == 0 and bst_betan + stp < BetaUpRayleigh(mp[j])):
                Beta[mp[j]] = bst_betan + stp
            rn_can = PSNR(m, mp[j], n)
            if rn_can > bst_rn:
                bst_rn = rn_can
                bst_betan = Beta[mp[j]]
                if stp == stpth:
                    flag = 4
            elif stp == stpth:
                flag = 2
                stp *= stpdiv
        while flag == 2:
            stp /= stpdiv
            if bst_betan - stp > 0:
                Beta[mp[j]] = bst_betan - stp
            rn_can = PSNR(m, mp[j], n)
            if rn_can > bst_rn:
                bst_rn = rn_can
                bst_betan = Beta[mp[j]]
            elif stpdiv != 1:
                flag = 3
            elif stpdiv == 1:
                flag = 4
        while flag == 3:
            stp *= stpdiv
            if stp < stpth:
                flag = 4
                break
            if bst_betan - stp > 0:
                Beta[mp[j]] = bst_betan - stp
            rn_can = PSNR(m, mp[j], n)
            if rn_can > bst_rn:
                bst_rn = rn_can
                bst_betan = Beta[mp[j]]
                if stp == stpth:
                    flag = 4
            elif stp == stpth:
                flag = 0
                stp *= stpdiv
    return bst_betan, bst_rn


def parallel_DVTC(j, mp, n):
    m_local = [x for x in mp if x not in [mp[j], n]]
    bst_rn = PSNR(m_local, mp[j], n)
    print('Interferers:', m_local, 'Transmitter:', mp[j], 'Receiver:', n, 'PSNR:', bst_rn)
    beta_star_j, rn_bst_j = BLCS(Beta[mp[j]], bst_rn, j, m_local, n)
    return mp[j], beta_star_j, rn_bst_j


def DVTC(mp, itr=100): #Channel fading threshold optimizer
    global Beta
    prv_Betan = Beta.copy()
    rn_bst = Lambdan.copy()
    for i in range(itr):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(parallel_DVTC, j, mp, size - (mp[j] + 1)) for j in range(len(mp))]
            for future in concurrent.futures.as_completed(futures):
                idx, beta_star_j, rn_bst_j = future.result()
                prv_Betan[idx] = beta_star_j
                rn_bst[idx] = rn_bst_j
        print("Optimal Beta:", prv_Betan)
        print("Optimal PSNR:", rn_bst)
        if prv_Betan == Beta:
            break
        Beta = prv_Betan.copy()
    return Beta


def ELCS(bst_lambdan, bst_psnr, j, m, n, stpdiv=0.5, stpini=1, stpth=1): #Encoder local coordinate search
    global Lambdan, mp
    stp = stpini * stpdiv
    flag = 0
    k = 0
    while flag != 4:
        while flag == 0:
            stp /= stpdiv
            Lambdan[mp[j]] = bst_lambdan + stp
            psnr_can = PSNR(m, mp[j], n)
            if psnr_can > bst_psnr:
                bst_psnr = psnr_can
                bst_lambdan = Lambdan[mp[j]]
                k += 1
            elif k != 0 and stpdiv != 1:
                flag = 1
            elif k != 0 and stpdiv == 1:
                flag = 4
            elif k == 0:
                flag = 2
                stp *= stpdiv
                Lambdan[mp[j]] = bst_lambdan
        while flag == 1:
            stp *= stpdiv
            if stp < stpth:
                flag = 4
                break
            Lambdan[mp[j]] = bst_lambdan + stp
            psnr_can = PSNR(m, mp[j], n)
            if psnr_can > bst_psnr:
                bst_psnr = psnr_can
                bst_lambdan = Lambdan[mp[j]]
                if stp == stpth:
                    flag = 4
            elif stp == stpth:
                flag = 2
                stp *= stpdiv
                Lambdan[mp[j]] = bst_lambdan
        while flag == 2:
            stp /= stpdiv
            if bst_lambdan - stp > 0:
                Lambdan[mp[j]] = bst_lambdan - stp
            else:
                Lambdan[mp[j]] = 0
            psnr_can = PSNR(m, mp[j], n)
            if psnr_can > bst_psnr:
                bst_psnr = psnr_can
                bst_lambdan = Lambdan[mp[j]]
            elif stpdiv != 1:
                flag = 3
            elif stpdiv == 1:
                flag = 4
            Lambdan[mp[j]] = bst_lambdan
        while flag == 3:
            stp *= stpdiv
            if stp < stpth:
                flag = 4
                break
            if bst_lambdan - stp > 0:
                Lambdan[mp[j]] = bst_lambdan - stp
            else:
                Lambdan[mp[j]] = 0
            psnr_can = PSNR(m, mp[j], n)
            if psnr_can > bst_psnr:
                bst_psnr = psnr_can
                bst_lambdan = Lambdan[mp[j]]
                if stp == stpth:
                    flag = 4
            elif stp == stpth:
                flag = 0
                stp *= stpdiv
            Lambdan[mp[j]] = bst_lambdan
    return bst_lambdan, bst_psnr


def parallel_DVEC(j, mp, n):
    m_local = [x for x in mp if x not in [mp[j], n]]
    bst_psnr = PSNR(m_local, mp[j], n)
    print('Interferers:', m_local, 'Transmitter:', mp[j], 'Receiver:', n, 'PSNR:', bst_psnr)
    lambda_star_j, psnr_bst_j = ELCS(Lambdan[mp[j]], bst_psnr, j, m_local, n)
    return mp[j], lambda_star_j, psnr_bst_j


def DVEC(mp):  # Video encoding rate optimizer
    global Lambdan
    prv_lambda = Lambdan.copy()
    psnr_bst = Lambdan.copy()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(parallel_DVEC, j, mp, size - (mp[j] + 1)) for j in range(int(len(mp)/2))]
        for future in concurrent.futures.as_completed(futures):
            idx, lambda_star_j, psnr_bst_j = future.result()
            prv_lambda[idx] = lambda_star_j
            psnr_bst[idx] = psnr_bst_j
    print("Optimal Lambda:", prv_lambda)
    print("Optimal PSNR:", psnr_bst)
    Lambdan = prv_lambda.copy()
    return Lambdan
    

def JDVT_EC(mp, itr=100): #Joint channel fading threshold and video encoding rate optimizer
    global Beta, Lambdan
    betan_star = [BetamUpRician(mp[i], size - (mp[i] + 1)) - 0.01 if RorR(mp[i], size - (mp[i] + 1)) == 1 else BetaUpRayleigh(mp[i]) - 0.01 for i in
                  range(size)]
    Beta = betan_star.copy()
    for i in range(itr):
        print('Counter:', i + 1)
        beta_prv = Beta.copy()
        lambda_prv = Lambdan.copy()
        print('Current Beta:', Beta)
        print('Current Lambda:', Lambdan)
        Beta = DVTC(mp).copy()
        Lambdan = DVEC(mp).copy()
        if Beta == beta_prv and Lambdan == lambda_prv:
            break
    return Beta, Lambdan


print(JDVT_EC(mp))
