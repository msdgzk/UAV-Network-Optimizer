import math
from scipy.stats import *
from scipy.integrate import quad
import numpy as np

F = 14 #Number of sub-channels
Tslt = 0.005 #Time slot duration
Tth = 0.08 #Time threshold
SINRTh = 10 #SINR threshold
Omega = 2 #Rayleigh Fading parameter
c = 3 * 1e8 #Light speed
d0 = 10 #Reference distance
f = 2.4 * 1e9 #Operating frequency
k = 1.38 * 1e-23 #Boltzmann constant
T = 290 #Temperature
BW = 1e8 #Bandwidth
sen = 30 #Sensitivity
normalizedBuffer = 50 #Normalized buffer size
Pt = 0.2 #Transmit power
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
x = xDelta * np.random.uniform(0, 1, size) + xMin
y = yDelta * np.random.uniform(0, 1, size) + yMin
z = zDelta * np.random.uniform(0, 1, size - 2) + zMin
z = np.insert(z, 0, 60) #Interferer UAV altitude
z = np.insert(z, 0, 50) #Streamer UAV altitude
Ln = [3.04 for i in range(size)] #Average video packet length
Beta = [2 for i in range(size)] #Channel fading threshold
Lambdan = [100 for i in range(size)] #Incoming packet rate
mp = [i for i in range(int(size/2))] #Streamer nodes' indices


def PLoS(s, n, zet=20, v=3 * 1e-4, mi=0.5): #Line of sight probability
    if z[s] == z[n]:
        return (1 - np.exp(-(z[s] ** 2) / (2 * (zet ** 2)))) ** (
                math.dist((x[s], y[s], z[s]), (x[n], y[n], z[n])) * np.sqrt(v * mi))
    else:
        return (1 - (((np.sqrt(2 * np.pi) * zet) / math.dist((0, 0, z[s]), (0, 0, z[n]))) * abs(
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
        sum += Pt * (hn(m[i], n) ** 2) * (
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
        sum += (Pt ** 2) * (hn(m[i], n) ** 4) * (
            (quad(lambda x: DIfunc(x, RorR(m[i], n), b(m[i], n)), Beta[m[i]], 10))[
                0]) * ((Miun(Beta[m[i]], s, n) / F) ** 2)
        while j < i:
            sum1 += 2 * Pt * (hn(m[i], n) ** 2) * (
                (quad(lambda x: EIfunc(x, RorR(m[i], n), b(m[i], n)), Beta[m[i]], 10))[
                    0]) * (
                            Miun(Beta[m[j]], s, n) / F) * Pt * (hn(m[j], n) ** 2) * (
                        (quad(lambda x: EIfunc(x, RorR(m[j], n), b(m[j], n)), Beta[m[j]], 10))[0]) * (
                            Miun(Beta[m[j]], s, n) / F)
            j += 1
    return sum + sum1 - (EI(m, s, n) ** 2)


def locmu(m, s, n): #Interference log-normal model parameters
    return np.log(EI(m, s, n)) - np.log(1 + (DI(m, s, n) / (EI(m, s, n) ** 2))) / 2


def sclsigma(m, s, n): #Interference log-normal model parameters
    return np.sqrt(np.log(1 + (DI(m, s, n) / (EI(m, s, n) ** 2))))


def Perrfunc(x, m, s, n): 
    if RorR(s, n) == 0:
        return rayleigh.pdf(x) * (0.5 - (0.5 * math.erf((np.log(
            (Pt * (hn(s, n) ** 2) * (x ** 2) / SINRTh) - (GaNoise)) - locmu(m, s, n)) / (
                                                                np.sqrt(2) * sclsigma(m, s, n)))))
    else:
        return rice.pdf(x, b(s, n)) * (0.5 - (0.5 * math.erf((np.log(
            (Pt * (hn(s, n) ** 2) * (x ** 2) / SINRTh) - (GaNoise)) - locmu(m, s, n)) / (
                                                                     np.sqrt(2) * sclsigma(m, s, n)))))


def Perr(m, s, n): #Transmission error probability
    return quad(lambda x: Perrfunc(x, m, s, n), Beta[s], 10)[0]


def BetaUpRayleigh(s):
    return np.sqrt(- Omega * np.log(1 - (1 - Tslt * Lambdan[s]) ** (1 / F)))


def BetaUpRician(s, n, betan):
    if rice.cdf(betan, b(s, n)) <= ((1 - Lambdan[s] * Tslt) ** (1 / F)):
        return betan
    else:
        return 0


def BetamUpRician(s):
    ini_betam = 2
    while BetaUpRician(s, size - (s + 1), ini_betam) != 0:
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


def DVTC(mp, stpdiv, itr=100, stpini=0.01, stpth=0.01): #Channel fading threshold optimizer
    global Beta, Lambdan, psnr_bst
    betan_star = Beta.copy()
    m = mp.copy()
    for i in range(itr):
        prv_Betan = betan_star.copy()
        for j in range(len(mp)):
            Beta = prv_Betan.copy()
            m.remove(mp[j])
            m.remove(size - (mp[j] + 1))
            bst_psnr = PSNR(m, mp[j], size - (mp[j] + 1))
            print('Interferers:', m, 'Transmitter:', mp[j], 'Receiver:', size - (mp[j] + 1), 'PSNR:', bst_psnr)
            bst_betan = Beta[mp[j]]
            stp = stpini * stpdiv
            flag = 0
            k = 0
            while flag != 4:
                while flag == 0:
                    stp /= stpdiv
                    if (RorR(mp[j], size - (mp[j] + 1)) == 1 and BetaUpRician(mp[j], size - (mp[j] + 1),
                                                                              bst_betan + stp) != 0) or (
                            RorR(mp[j], size - (mp[j] + 1)) == 0 and bst_betan + stp < BetaUpRayleigh(mp[j])):
                        Beta[mp[j]] = bst_betan + stp
                    psnr_can = PSNR(m, mp[j], size - (mp[j] + 1))
                    print('Current Beta0+:', Beta[mp[j]], 'Current Step0+:', stp, 'PSNR Candidate0+:', psnr_can)
                    if psnr_can > bst_psnr:
                        bst_psnr = psnr_can
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
                    if (RorR(mp[j], size - (mp[j] + 1)) == 1 and BetaUpRician(mp[j], size - (mp[j] + 1),
                                                                              bst_betan + stp) != 0) or (
                            RorR(mp[j], size - (mp[j] + 1)) == 0 and bst_betan + stp < BetaUpRayleigh(mp[j])):
                        Beta[mp[j]] = bst_betan + stp
                    psnr_can = PSNR(m, mp[j], size - (mp[j] + 1))
                    print('Current Beta1+:', Beta[mp[j]], 'Current Step1+:', stp, 'PSNR Candidate1+:', psnr_can)
                    if psnr_can > bst_psnr:
                        bst_psnr = psnr_can
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
                    psnr_can = PSNR(m, mp[j], size - (mp[j] + 1))
                    print('Current Beta2-:', Beta[mp[j]], 'Current Step2-:', stp, 'PSNR Candidate2-:', psnr_can)
                    if psnr_can > bst_psnr:
                        bst_psnr = psnr_can
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
                    psnr_can = PSNR(m, mp[j], size - (mp[j] + 1))
                    print('Current Beta3-:', Beta[mp[j]], 'Current Step3-:', stp, 'PSNR Candidate3-:', psnr_can)
                    if psnr_can > bst_psnr:
                        bst_psnr = psnr_can
                        bst_betan = Beta[mp[j]]
                        if stp == stpth:
                            flag = 4
                    elif stp == stpth:
                        flag = 0
                        stp *= stpdiv
            m = mp.copy()
            betan_star[mp[j]] = bst_betan
            psnr_bst[mp[j]] = bst_psnr
        print("Optimal Fading Threshold:", betan_star)
        print("Optimal PSNR:", psnr_bst)
        if betan_star == prv_Betan:
            break
    return betan_star


def DVEC(mp, stpdiv, stpini=1, stpth=1): #Video encoding rate optimizer
    global Beta, Lambdan, psnr_bst
    lambda_star = Lambdan.copy()
    m = mp.copy()
    for i in range(int(len(mp) / 2)):
        m.remove(mp[i])
        m.remove(size - (mp[i] + 1))
        bst_psnr = PSNR(m, size - (mp[i] + 1), mp[i])
        print('Interferers:', m, 'Transmitter:', size - (mp[i] + 1), 'Receiver:', mp[i], 'PSNR:', bst_psnr)
        stp = stpini * stpdiv
        flag = 0
        j = 0
        while flag != 4:
            while flag == 0:
                stp /= stpdiv
                Lambdan[size - (mp[i] + 1)] += stp
                psnr_can = PSNR(m, size - (mp[i] + 1), mp[i])
                print('Current Lambda0+:', Lambdan[size - (mp[i] + 1)], 'Current Step0+:', stp, 'PSNR Candidate0+:', psnr_can)
                if psnr_can > bst_psnr:
                    bst_psnr = psnr_can
                    j += 1
                elif j != 0 and stpdiv != 1:
                    flag = 1
                    Lambdan[size - (mp[i] + 1)] -= stp
                elif j != 0 and stpdiv == 1:
                    flag = 4
                    Lambdan[size - (mp[i] + 1)] -= stp
                elif j == 0:
                    flag = 2
                    Lambdan[size - (mp[i] + 1)] -= stp
                    stp *= stpdiv
            while flag == 1:
                stp *= stpdiv
                if stp < stpth:
                    flag = 4
                    break
                Lambdan[size - (mp[i] + 1)] += stp
                psnr_can = PSNR(m, size - (mp[i] + 1), mp[i])
                print('Current Lambda1+:', Lambdan[size - (mp[i] + 1)], 'Current Step1+:', stp, 'PSNR Candidate1+:', psnr_can)
                if psnr_can > bst_psnr:
                    bst_psnr = psnr_can
                else:
                    Lambdan[size - (mp[i] + 1)] -= stp
                    if stp == stpth:
                        flag = 2
                        stp *= stpdiv
            while flag == 2:
                stp /= stpdiv
                if Lambdan[size - (mp[i] + 1)] - stp > 0:
                    Lambdan[size - (mp[i] + 1)] -= stp
                psnr_can = PSNR(m, size - (mp[i] + 1), mp[i])
                print('Current Lambda2-:', Lambdan[size - (mp[i] + 1)], 'Current Step2-:', stp, 'PSNR Candidate2-:', psnr_can)
                if psnr_can > bst_psnr:
                    bst_psnr = psnr_can
                elif stpdiv != 1:
                    flag = 3
                    Lambdan[size - (mp[i] + 1)] += stp
                elif stpdiv == 1:
                    flag = 4
                    Lambdan[size - (mp[i] + 1)] += stp
            while flag == 3:
                stp *= stpdiv
                if stp < stpth:
                    flag = 4
                    break
                if Lambdan[size - (mp[i] + 1)] - stp > 0:
                    Lambdan[size - (mp[i] + 1)] -= stp
                psnr_can = PSNR(m, size - (mp[i] + 1), mp[i])
                print('Current Lambda3-:', Lambdan[size - (mp[i] + 1)], 'Current Step3-:', stp, 'PSNR Candidate3-:', psnr_can)
                if psnr_can > bst_psnr:
                    bst_psnr = psnr_can
                else:
                    Lambdan[size - (mp[i] + 1)] += stp
                    if stp == stpth:
                        flag = 0
                        stp *= stpdiv
        m = mp.copy()
        print('Transmitter:', size - (mp[i] + 1), 'Receiver:', mp[i], 'Lambda:', Lambdan[size - (mp[i] + 1)])
        lambda_star[size - (mp[i] + 1)] = Lambdan[size - (mp[i] + 1)]
        psnr_bst[size - (mp[i] + 1)] = bst_psnr
    print("Optimal Lambda:", lambda_star)
    print("Optimal PSNR:", psnr_bst)
    return lambda_star


def JDVT_EC(mp, itr=100): #Joint channel fading threshold and video encoding rate optimizer
    global Beta, Lambdan, psnr_bst
    psnr_bst = Lambdan.copy()
    for i in range(itr):
        if i == 0:
            stpdiv = 0.5
        else:
            stpdiv = 1
        print('Counter:', i + 1)
        beta_prv = Beta.copy()
        lambda_prv = Lambdan.copy()
        print('Current Beta:', Beta)
        print('Current Lambda:', Lambdan)
        Beta = DVTC(mp, stpdiv).copy()
        Lambdan = DVEC(mp, stpdiv).copy()
        if Beta == beta_prv and Lambdan == lambda_prv:
            break
    return Beta, Lambdan
