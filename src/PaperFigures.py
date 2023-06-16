import numpy as np
import matplotlib.pyplot as plt

''' I used this code to replicate figure 8 and figure S4.
n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# B3067/B3067, cross2
B3067E = [21.3587081, 15.17970714, 10.62110174, 5.83347194, 3.59740196, 2.63089558, 2.4470285, 2.18840372, 2.13009046, 1.73676738]
εB3067E = [1.44257363, 1.93729254, 1.77537533, 1.90916378, 1.48526883, 0.60964792, 0.50944835, 0.27585647, 0.2970057, 0.39076581]
B3067σ = [107.92597028, 29.58839829, 11.33450252, 8.07809034, 5.70168262, 5.07757324, 4.14280765, 3.39958904, 2.84703485, 1.34625944]
εB3067σ = [13.63118035, 18.54100985, 7.24847445, 3.51824078, 1.72730742, 1.24935494, 1.01473675, 0.8416667, 0.43855084, 0.29775513]
# B6067/B3067, cross2
B6067E = [21.86513095, 15.97902402, 11.33452419, 6.29951899, 4.4087073, 3.16120926, 3.0836971, 2.64530076, 2.71041657, 2.06586028]
εB6067E = [1.81986102, 2.03211454, 1.8591468, 1.99921365, 1.43528778, 0.77954084, 0.6638711, 0.4093299, 0.4680657, 0.4921772]
B6067σ = [111.60741854, 40.53392859, 15.20257648, 12.67832207, 9.71118466, 7.67745787, 6.06400137, 5.05725624, 4.76658447, 3.00835796]
εB6067σ = [16.79943322, 13.6095181, 7.87590965, 6.16570392, 3.345566, 2.93356203, 2.1635525, 1.2292518, 1.02953349, 0.50625259]
# S6067/B3067, cross2
S6067E = [20.82023652, 13.4583958, 8.23690522, 4.5921995, 2.67258566, 2.35725327, 2.4355712, 2.02472747, 2.17061739, 1.83605998]
εS6067E = [1.23432011, 0.95440984, 1.72845628, 1.95613838, 0.56935845, 0.34257265, 0.25177797, 0.42057363, 0.23071537, 0.39856154]
S6067σ = [114.58549574, 32.89365019, 13.11705236, 9.11924106, 7.41115995, 6.24127302, 5.84637484, 5.67386665, 5.57657808, 5.71970148]
εS6067σ = [9.57891162, 14.53988492, 9.59867651, 3.62367311, 2.7857931, 1.50900763, 1.54266482, 1.10836452, 0.90272851, 0.39887549]
# Second round of figures. Include B3067 first in figures.
# B3090/B3067, cross2
B3090E = [18.40507986, 12.81159777, 7.97587591, 5.48326667, 3.4110104, 2.5976216, 2.3027464, 2.09862443, 2.09303332, 1.8776037]
εB3090E = [1.58800388, 0.93433756, 1.57910684, 2.37402331, 1.54619928, 0.58068648, 0.25335815, 0.26541917, 0.19477368, 0.37367781]
B3090σ = [94.30001823, 28.79962952, 10.21452489, 8.23864583, 6.37878271, 5.35067482, 5.56139654, 4.85588632, 4.39152337, 4.0394429]
εB3090σ = [9.95664891, 12.11649881, 6.03205157, 3.75291747, 1.83745355, 0.7730129, 1.81483202, 0.60761518, 0.76550689, 0.30855093]
# B6090/B3067, cross2
B6090E = [24.7293195, 18.35147545, 13.07307203, 8.76646468, 6.40371496, 5.48787024, 5.169384, 4.73354612, 4.92136463, 4.02956289]
εB6090E = [0.98238716, 1.36122021, 2.00444049, 2.17092822, 1.37857735, 1.19949635, 1.14281607, 0.49742864, 0.65398379, 0.43942064]
B6090σ = [120.79600702, 38.56529645, 21.65162809, 15.57287961, 12.54348168, 10.6505493, 9.19924291, 7.08953313, 6.7596994, 4.65640856]
εB6090σ = [9.69974086, 15.34733139, 13.33774907, 5.31425509, 4.02025819, 3.83862651, 3.30922242, 1.59115733, 1.22065426, 0.53751798]
# S3067/B3067, cross2
S3067E = [18.77958216, 13.52654776, 8.99808355, 4.20804689, 3.31900112, 2.28273024, 2.15719216, 2.02536767, 2.13356616, 1.83792307]
εS3067E = [2.10004989, 1.89485143, 1.68680205, 1.79280777, 1.65212379, 0.35646241, 0.32786598, 0.50457573, 0.36894116, 0.27895091]
S3067σ = [106.27089425, 39.83501063, 10.07343989, 7.77294687, 5.59024581, 4.61104463, 4.03373157, 3.20475078, 3.13865279, 1.54156128]
εS3067σ = [9.04948078, 10.89933402, 6.11076593, 4.2858199, 1.7758109, 1.30623827, 0.97480197, 0.70326613, 0.7673939, 0.432201]

fig, ax = plt.subplots()
ax.errorbar(n, B3067E, yerr = εB3067E, label = "B3067")
ax.errorbar(n, B6067E, yerr = εB6067E, label = "B6067")
ax.errorbar(n, S6067E, yerr = εS6067E, label = "S6067")
ax.set_yscale('log')
ax.set_ylim([1, 20])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 30])
ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 30])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$E^{\star}$: Cross2 method for 3D printed Ti alloy (B3067)", y=0.05, fontsize=16)
plt.savefig("figure1.png")
plt.show()

fig, ax = plt.subplots()
ax.errorbar(n, B3067σ, yerr = εB3067σ, label = "B3067")
ax.errorbar(n, B6067σ, yerr = εB6067σ, label = "B6067")
ax.errorbar(n, S6067σ, yerr = εS6067σ, label = "S6067")
ax.set_yscale('log')
ax.set_ylim([1, 20])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$\sigma_{y}$: Cross2 method for 3D printed Ti alloy (B3067)", y=0.05, fontsize=16)
plt.savefig("figure2.png")
plt.show()

fig, ax = plt.subplots()
ax.errorbar(n, B3067E, yerr = εB3067E, label = "B3067")
ax.errorbar(n, B3090E, yerr = εB3090E, label = "B3090")
ax.errorbar(n, B6090E, yerr = εB6090E, label = "B6090")
ax.errorbar(n, S3067E, yerr = εS3067E, label = "S3067")
ax.set_yscale('log')
ax.set_ylim([1, 20])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 30])
ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 30])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$E^{\star}$: Cross2 method for 3D printed Ti alloy (B3067)", y=0.05, fontsize=16)
plt.savefig("figure3.png")
plt.show()

fig, ax = plt.subplots()
ax.errorbar(n, B3067σ, yerr = εB3067σ, label = "B3067")
ax.errorbar(n, B3090σ, yerr = εB3090σ, label = "B3090")
ax.errorbar(n, B6090σ, yerr = εB6090σ, label = "B6090")
ax.errorbar(n, S3067σ, yerr = εS3067σ, label = "S3067")
ax.set_yscale('log')
ax.set_ylim([1, 20])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$\sigma_{y}$: Cross2 method for 3D printed Ti alloy (B3067)", y=0.05, fontsize=16)
plt.savefig("figure4.png")
plt.show()
'''

''' I used this code to replicate figure 9A.
labels = ["$\sigma_{y}$", "$\sigma_{0.8\%}$", "$\sigma_{1.5\%}$", "$\sigma_{3.3\%}$"]
x = np.arange(4)
# B3090 (self)
μB3090 = [3.7342872, 5.13522576, 5.9153901, 8.49087502]
σB3090 = [1.14439324, 1.15316418, 1.23122, 1.50619123]
# B3067 (peer)
μB3067 = [5.44653162, 7.24383922, 8.47500916, 10.40762874]
σB3067 = [1.00719594, 1.14083534, 2.01489362, 1.87494608]

# Create bar plot with error bars for self
fig, ax = plt.subplots()
ax.bar(x, μB3090, yerr=σB3090, align='center', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yticks([0, 5, 10, 15])
ax.set_yticklabels([0, 5, 10, 15])
ax.set_xlabel('Self')
ax.set_ylabel('MAPE (%)')
plt.subplots_adjust(bottom=0.18)
plt.suptitle('Error for 3D printed B3090 Ti alloy (self data)', y=0.05, fontsize=16)
plt.savefig("figure5.png")
plt.show()

# Create bar plot with error bars for peer
fig, ax = plt.subplots()
ax.bar(x, μB3067, yerr=σB3067, align='center', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yticks([0, 5, 10, 15])
ax.set_yticklabels([0, 5, 10, 15])
ax.set_xlabel('Peer')
ax.set_ylabel('MAPE (%)')
plt.subplots_adjust(bottom=0.18)
plt.suptitle('Error for 3D printed B3090 Ti alloy (peer B3067 data)', y=0.05, fontsize=16)
plt.savefig("figure6.png")
plt.show()
'''

''' I used this code to replicate figure 9A.
# B3090 (self)
εB3090s = [0, 0.010812718777945549, 0.018812718777945547, 0.025812718777945547, 0.04381271877794555]
μB3090s = [0, 1.1893990655740103, 1.261440849221415, 1.3095634858641358, 1.3541651956737042]
σB3090s = [0, 0.06263559539294837, 0.08554026324183601, 0.099457473279119, 0.12865278228554924]
# B3090 (peer)
εB3090p = [0, 0.010517707991630139, 0.01851770799163014, 0.02551770799163014, 0.04351770799163014]
μB3090p = [0, 1.1569478790793153, 1.2285127836383052, 1.269093168195751, 1.3423976590236029]
σB3090p = [0, 0.07368485059492222, 0.11021887086879645, 0.12448425179108871, 0.17113237307809065]
# B3067 (self)
εB3067s = [0.010518512618210579, 0.01851851261821058, 0.02551851261821058, 0.04351851261821058]
μB3067s = [1.1570363880031638, 1.2068389612767432, 1.2543196709619628, 1.3031249678797192]
σB3067s = [0.0915608906318954, 0.11213870726982947, 0.13221502976271032, 0.15441696867509516]

# Create two plots with error bars for self
fig, ax = plt.subplots()
ax.errorbar(εB3090s, μB3090s, yerr = σB3090s, label = "B3090, $n \\approx 0.093 \\pm 0.009$")
ax.errorbar(εB3090s, μB3090s, linestyle = "dashed", yerr = σB3090s, label = "B3067, $n \\approx 0.085 \\pm 0.006$")
ax.set_yscale('linear')
ax.set_xlim([0, 0.045])
ax.set_ylim([0, 1.7])
ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
ax.set_xticklabels([0, 0.01, 0.02, 0.03, 0.04])
ax.set_yticklabels([0, 0.4, 0.8, 1.2, 1.6])
ax.legend()
ax.set_ylabel("$\sigma$ (GPa)")
ax.set_xlabel("$\epsilon$")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("Self training data (n=5)", y=0.05, fontsize=16)
plt.savefig("figure7.png")
plt.show()

# Create a plot with error bars for peer
fig, ax = plt.subplots()
ax.errorbar(εB3090p, μB3090p, yerr = σB3090p, label = "B3090, $n \\approx 0.105 \\pm 0.001$")
ax.set_yscale('linear')
ax.set_xlim([0, 0.045])
ax.set_ylim([0, 1.7])
ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
ax.set_xticklabels([0, 0.01, 0.02, 0.03, 0.04])
ax.set_yticklabels([0, 0.4, 0.8, 1.2, 1.6])
ax.legend()
ax.set_ylabel("$\sigma$ (GPa)")
ax.set_xlabel("$\epsilon$")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("Peer training data", y=0.05, fontsize=16)
plt.savefig("figure8.png")
plt.show()
'''

'''This segment reproduces figure 5.
# Al7075 (2D+3D+exp)
ε7075e = [0, 0.007048165782598786, 0.04004816578259879, 0.07304816578259879, 0.10704816578259879]
μ7075e = [0, 0.4940764213601748, 0.6144114792346954, 0.6642303397258122, 0.6964227795600891]
σ7075e = [0, 0.012607406272022907, 0.017675353937183322, 0.01843075993004304, 0.019873686573573017]
# Al6061 (2D+3D+exp)
ε6061e = [0, 0.00405522551045665, 0.03705522551045665, 0.07005522551045665, 0.10405522551045665]
μ6061e = [0, 0.2842713082830111, 0.33685481250286103, 0.35236377964417137, 0.3604452187816302]
σ6061e = [0, 0.009398300285191211, 0.01061580789351233, 0.009844318794656338, 0.010466897238489327]
# Al7075 (2D+3D)
ε7075f = [0, 0.004453865789086038, 0.03745386578908604, 0.07045386578908604, 0.10445386578908604]
μ7075f = [0, 0.31221599181493126, 0.5319813231627146, 0.588776042064031, 0.6527868231137594]
σ7075f = [0, 0.02317692120585135, 0.02969466544021186, 0.02548728884527546, 0.02976832549702644]
# Al6061 (2D+3D)
ε6061f = [0.004515378975761151, 0.037515378975761154, 0.07051537897576116, 0.10451537897576116]
μ6061f = [0.3016273155808449, 0.316699339201053, 0.3300789381066958, 0.33758289366960526]
σ6061f = [0.03691037131722045, 0.03298896340550387, 0.026911666661394298, 0.030724686883124804]

# Create plot with error bars for self, FEM+Exp
fig, ax = plt.subplots()
ax.errorbar(ε7075e, μ7075e, yerr = σ7075e, label = "Al7075-T651, $n \\approx 0.127 \\pm 0.001$")
ax.errorbar(ε6061e, μ6061e, linestyle = "dashed", yerr = σ6061e, label = "Al6061-T6511, $n \\approx 0.074 \\pm 0.002$")
ax.set_yscale('linear')
ax.set_xlim([0, 0.11])
ax.set_ylim([0, 0.8])
ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_xticklabels([0, 0.02, 0.04, 0.06, 0.08, 0.1])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8])
ax.legend()
ax.set_ylabel("$\sigma$ (GPa)")
ax.set_xlabel("$\epsilon$")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("2D+3D FEM+Exp (n=3)", y=0.05, fontsize=16)
plt.savefig("figure9.png")
plt.show()

# Create plot with error bars for self, FEM+Exp
fig, ax = plt.subplots()
ax.errorbar(ε7075f, μ7075f, yerr = σ7075f, label = "Al7075-T651, $n \\approx 0.228 \\pm 0.014$")
ax.errorbar(ε6061f, μ6061f, linestyle = "dashed", yerr = σ6061f, label = "Al6061-T6511, $n \\approx 0.035 \\pm 0.007$")
ax.set_yscale('linear')
ax.set_xlim([0, 0.11])
ax.set_ylim([0, 0.8])
ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_xticklabels([0, 0.02, 0.04, 0.06, 0.08, 0.1])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8])
ax.legend()
ax.set_ylabel("$\sigma$ (GPa)")
ax.set_xlabel("$\epsilon$")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("2D+3D FEM", y=0.05, fontsize=16)
plt.savefig("figure10.png")
plt.show()
'''

'''This segment reproduces figure 7.
n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# B3090, self (raw)
Esr = [20.85738611, 14.93193336, 10.62470933, 6.17081165, 3.53980273, 2.83338057, 2.39439426, 2.23912841, 2.14931529, 1.52928879]
dEsr = [1.15774057, 2.76461151, 2.76123732, 2.28674632, 1.65000347, 1.14926682, 0.35276605, 0.4731503, 0.36532754, 0.36058943]
σsr = [92.90621932, 31.1172741, 7.86850271, 5.42416156, 3.9266917, 3.91898551, 3.62685968, 2.97950902, 2.42648245, 1.47304528]
dσsr = [5.72839659, 19.77325765, 6.55123033, 2.5604357, 0.74303447, 0.74347853, 0.48021291, 0.43923216, 0.46950993, 0.25023615]
# B3090, peer (raw)
Epr = [20.0316801, 15.30074303, 9.98728221, 6.99652784, 3.77915319, 2.39346351, 2.45927424, 2.32525492, 2.23636191, 2.0069952]
dEpr = [0.91949316, 1.3166407, 1.9172967, 1.91926169, 1.77190192, 0.37770116, 0.4530649, 0.35565338, 0.29393664, 0.43860418]
σpr = [99.6364055, 31.69011169, 12.93410265, 8.84905106, 7.0674062, 5.66981487, 5.6574326, 4.55043188, 4.73105714, 3.98191056]
dσpr = [8.74225627, 11.67643018, 9.39229757, 5.01972869, 2.86092334, 1.08581043, 1.09132802, 0.70862583, 0.78805065, 0.50179615]
# B3090, self (tip)
Est = [7.69691729, 3.51176923, 3.38300571, 3.21087096, 2.79969631, 2.59377006, 2.43205044, 2.32385096, 2.06323201, 1.71391161]
dEst = [1.40011309, 0.69368449, 0.83880401, 0.7035978, 0.4457362, 0.41749276, 0.35852926, 0.32584997, 0.30787117, 0.16913883]
σst = [35.38345948, 14.27849484, 7.19610592, 4.75816201, 3.93128569, 3.41866129, 4.06738206, 2.93299789, 2.20067251, 1.28286056]
dσst = [6.76921265, 11.30269792, 5.40612749, 2.23143059, 1.13466907, 1.4055051, 1.27503581, 1.00971715, 0.70018148, 0.27910456]
# B3090, peer (tip)
Ept = [7.53550653, 3.69536815, 4.13335628, 3.78174001, 3.75054876, 3.77565691, 3.94921767, 3.84544807, 3.63185784, 3.18179324]
dEpt = [1.00549554, 0.86157193, 1.16533158, 0.82219966, 1.04377179, 0.70888312, 0.88801928, 0.75955593, 0.52060274, 0.44731189]
σpt = [30.15499917, 14.59552122, 8.4559425, 7.03875584, 4.12011616, 3.37037623, 4.01957046, 2.75383188, 3.08886804, 3.26111124]
dσpt = [5.07634106, 8.1286953, 2.4777729, 4.79712265, 1.3122539, 0.43977898, 1.33841325, 0.54669229, 0.46025783, 2.33967126]

fig, ax = plt.subplots()
ax.errorbar(n, Esr, yerr = dEsr, label = "Self (raw)")
ax.errorbar(n, Est, yerr = dEst, label = "Self (tip)")
ax.errorbar(n, Epr, yerr = dEpr, label = "Peer (raw)")
ax.errorbar(n, Ept, yerr = dEpt, label = "Peer (tip)")
ax.set_yscale('log')
ax.set_ylim([1, 25])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([1, 2, 3, 4, 5, 10, 20])
ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$E^{\star}$: Cross2 method for 3D printed Ti alloy (B3090)", y=0.05, fontsize=16)
plt.savefig("figure11.png")
plt.show()

fig, ax = plt.subplots()
ax.errorbar(n, σsr, yerr = dσsr, label = "Self (raw)")
ax.errorbar(n, σst, yerr = dσst, label = "Self (tip)")
ax.errorbar(n, σpr, yerr = dσpr, label = "Peer (raw)")
ax.errorbar(n, σpt, yerr = dσpt, label = "Peer (tip)")
ax.set_yscale('log')
ax.set_ylim([1, 120])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 50, 100])
ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 50, 100])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$\sigma_{y}$: Cross2 method for 3D printed Ti alloy (B3090)", y=0.05, fontsize=16)
plt.savefig("figure12.png")
plt.show()
'''

'''These results reproduce figure 6.
categories = ['NN (raw)', 'NN (tip)', 'NN self (raw, 5)', 'NN self (tip, 5)']
EB3067 = [21.3587081, 5.67446185, 3.26508468, 2.40736801]
εEB3067 = [1.44257363, 1.24167231, 1.30605511, 0.36433321]
σB3067 = [107.92597028, 35.9870154, 4.82834841, 4.32661172]
εσB3067 = [13.63118035, 5.45248244, 0.96458256, 0.98667419]
EB3090 = [18.40507986, 7.69691729, 2.83338057, 2.59377006]
εEB3090 = [1.58800388, 1.40011309, 1.14926682, 0.41749276]
σB3090 = [94.30001823, 35.38345948, 3.91898551, 3.41866129]
εσB3090 = [9.95664891, 6.76921265, 0.74347853, 1.4055051]

# Bar graph for Estar
bar_width = 0.35
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width
fig, ax = plt.subplots()
ax.bar(bar_positions1, EB3067, yerr=εEB3067, width=bar_width, label='B3067')
ax.bar(bar_positions2, EB3090, yerr=εEB3090, width=bar_width, label='B3090')
ax.set_xticks(bar_positions1 + bar_width/2)
ax.set_xticklabels(categories)
ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax.set_yticklabels([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel('Test')
ax.set_ylabel('MAPE (%)')
ax.legend()
plt.subplots_adjust(bottom=0.18)
plt.suptitle('Result of adding training data for $E^{\star}$', y=0.05, fontsize=16)
plt.savefig("figure13.png")
plt.show()

# Bar graph for sigma_y
bar_width = 0.35
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width
fig, ax = plt.subplots()
ax.bar(bar_positions1, σB3067, yerr=εσB3067, width=bar_width, label='B3067')
ax.bar(bar_positions2, σB3090, yerr=εσB3090, width=bar_width, label='B3090')
ax.set_xticks(bar_positions1 + bar_width/2)
ax.set_xticklabels(categories)
ax.set_yticks([0, 25, 50, 75, 100, 125])
ax.set_yticklabels([0, 25, 50, 75, 100, 125])
ax.set_xlabel('Test')
ax.set_ylabel('MAPE (%)')
ax.legend()
plt.subplots_adjust(bottom=0.18)
plt.suptitle('Result of adding training data for $\sigma_{y}$', y=0.05, fontsize=16)
plt.savefig("figure14.png")
plt.show()'''


'''Copy of figure 10
categories = ['2D+3D FEM', 'Transfer learning']
EAl6061 = [2.69722282, 1.77801087]
εEAl6061 = [0.56832727, 0.22860735]
σAl6061 = [8.96220736, 2.43543746]
εσAl6061 = [2.17231215, 0.75463634]
EAl7075 = [2.39101247, 1.74734526]
εEAl7075 = [0.48842031, 0.15124274]
σAl7075 = [35.37985226, 1.91336423]
εσAl7075 = [6.10472297, 0.60501141]
EB3067 = [21.3587081, 3.26508468]
εEB3067 = [1.44257363, 1.30605511]
σB3067 = [107.92597028, 4.82834841]
εσB3067 = [13.63118035, 0.96458256]
EB3090 = [18.40507986, 2.83338057]
εEB3090 = [1.58800388, 1.14926682]
σB3090 = [94.30001823, 3.91898551]
εσB3090 = [9.95664891, 0.74347853]

# Bar graph for Estar
bar_width = 0.15
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width
bar_positions3 = bar_positions1 + bar_width * 2
bar_positions4 = bar_positions1 + bar_width * 3
fig, ax = plt.subplots()
ax.bar(bar_positions1, EAl7075, yerr=εEAl7075, width=bar_width, label='Al7075-T651')
ax.bar(bar_positions2, EAl6061, yerr=εEAl6061, width=bar_width, label='Al6067-T6511')
ax.bar(bar_positions3, EB3067, yerr=εEB3067, width=bar_width, label='B3067')
ax.bar(bar_positions4, EB3090, yerr=εEB3090, width=bar_width, label='B3090')
ax.set_xticks(bar_positions1 + bar_width*3/2)
ax.set_xticklabels(categories)
ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax.set_yticklabels([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel('Test')
ax.set_ylabel('MAPE (%)')
ax.legend()
plt.subplots_adjust(bottom=0.18)
plt.suptitle('Result of adding training data for $E^{\star}$', y=0.05, fontsize=16)
plt.savefig("figure15.png")
plt.show()

# Bar graph for sigma_y
bar_width = 0.15
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width
bar_positions3 = bar_positions1 + bar_width * 2
bar_positions4 = bar_positions1 + bar_width * 3
fig, ax = plt.subplots()
ax.bar(bar_positions1, σAl7075, yerr=εσAl7075, width=bar_width, label='Al7075-T651')
ax.bar(bar_positions2, σAl6061, yerr=εσAl6061, width=bar_width, label='Al6067-T6511')
ax.bar(bar_positions3, σB3067, yerr=εσB3067, width=bar_width, label='B3067')
ax.bar(bar_positions4, σB3090, yerr=εσB3090, width=bar_width, label='B3090')
ax.set_xticks(bar_positions1 + bar_width*3/2)
ax.set_xticklabels(categories)
ax.set_yticks([0, 25, 50, 75, 100, 125])
ax.set_yticklabels([0, 25, 50, 75, 100, 125])
ax.set_xlabel('Test')
ax.set_ylabel('MAPE (%)')
ax.legend()
plt.subplots_adjust(bottom=0.18)
plt.suptitle('Result of adding training data for $\sigma_{y}$', y=0.05, fontsize=16)
plt.savefig("figure16.png")
plt.show()
'''
'''
n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# Second round of figures. Include B3067 first in figures.
# 25˚, cross2
Ti25E = [2.08916258, 1.73458529, 1.76113493, 1.66044661, 1.57735542, 1.49558173, 1.44553545, 1.25069252, 1.20741493, 0.9812278]
εTi25E = [0.84352267, 0.37753768, 0.23317704, 0.29399772, 0.27203281, 0.2960974, 0.23108403, 0.19055048, 0.15260457, 0.04176756]
# 250˚, cross2
Ti250E = [35.24559953, 13.55870661, 11.8248246, 10.923051, 9.54057722, 9.21058842, 8.54009045, 8.15317396, 7.56763099, 5.78381901]
εTi250E = [3.1661163, 4.73711992, 2.33398521, 1.85770404, 0.8655968, 0.71457763, 0.30699462, 0.62376367, 0.43500931, 0.28294596]
# 500˚, cross2
Ti500E = [11.66735235, 10.29306847, 9.47812185, 8.33647048, 7.58027689, 7.51398051, 7.40959369, 6.99470173, 6.64627248, 5.97185385]
εTi500E = [0.60209906, 1.20103306, 1.37526918, 1.40743936, 0.41836403, 0.47261672, 0.55111534, 0.41686947, 0.41297669, 0.12461302]
# 750˚, cross2
Ti750E = [22.74219275, 20.62261021, 20.63547436, 19.4810463, 18.91002247, 18.19225275, 17.66865073, 17.40025689, 16.67137397, 14.09023671]
εTi750E = [1.09400474, 2.42701076, 1.67360611, 1.40042242, 1.14474779, 1.18580074, 1.15735988, 1.24283096, 0.9555188, 0.69974678]
# B3090 (self)
B3090E = [7.69691729, 3.51176923, 3.38300571, 3.21087096, 2.79969631, 2.59377006, 2.43205044, 2.32385096, 2.06323201, 1.71391161]
εB3090E = [1.40011309, 0.69368449, 0.83880401, 0.7035978, 0.4457362, 0.41749276, 0.35852926, 0.32584997, 0.30787117, 0.16913883]


fig, ax = plt.subplots()
ax.errorbar(n, Ti25E, yerr = εTi25E, label = "Ti33: 25˚C")
ax.errorbar(n, Ti250E, yerr = εTi250E, label = "Ti33: 250˚C")
ax.errorbar(n, Ti500E, yerr = εTi500E, label = "Ti33: 500˚C")
ax.errorbar(n, Ti750E, yerr = εTi750E, label = "Ti33: 750˚C")
ax.errorbar(n, B3090E, yerr = εB3090E, label = "B3090 (from Lu et al.)")
ax.set_yscale('log')
ax.set_ylim([0.5, 40])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 40])
ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 40])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$E^{\star}$: Cross2 method for Ti33 and B3090", y=0.05, fontsize=16)
plt.savefig("/Users/Joe/Desktop/figure17.png")
plt.show()
'''
'''
n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# Second round of figures. Include B3067 first in figures.
# 25˚, cross2
Ti25σ = [367.02558874, 101.13129713, 42.56915805, 27.87445748, 21.04812475, 16.92794671, 14.21538944, 7.98641605, 5.30872148, 2.36812937]
εTi25σ = [23.11441495, 64.43529896, 17.61905282, 9.35230254, 12.86762465, 6.23069119, 4.93143527, 1.75701109, 1.78478156, 0.59860558]
# B3090, cross2
B3090σ = [94.30001823, 28.79962952, 10.21452489, 8.23864583, 6.37878271, 5.35067482, 5.56139654, 4.85588632, 4.39152337, 4.0394429]
εB3090σ = [9.95664891, 12.11649881, 6.03205157, 3.75291747, 1.83745355, 0.7730129, 1.81483202, 0.60761518, 0.76550689, 0.30855093]



fig, ax = plt.subplots()
ax.errorbar(n, Ti25σ, yerr = εTi25σ, label = "Ti33: 25˚C")
ax.errorbar(n, B3090σ, yerr = εB3090σ, label = "B3090 (from Lu et al.)")
ax.set_yscale('log')
ax.set_ylim([1, 500])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([1, 2, 4, 10, 20, 50, 100, 200, 500])
ax.set_yticklabels([1, 2, 4, 10, 20, 50, 100, 200, 500])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$\sigma_{y}$: Cross2 method for Ti33 and B3090", y=0.05, fontsize=16)
plt.savefig("/Users/Joe/Desktop/figure18.png")
plt.show()
'''
'''
n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# Second round of figures. Include B3067 first in figures.
# 750˚, cross2
Ti750E = [22.74219275, 20.62261021, 20.63547436, 19.4810463, 18.91002247, 18.19225275, 17.66865073, 17.40025689, 16.67137397, 14.09023671]
εTi750E = [1.09400474, 2.42701076, 1.67360611, 1.40042242, 1.14474779, 1.18580074, 1.15735988, 1.24283096, 0.9555188, 0.69974678]
# 750˚, cross2 (other data)
Ti750E2 = [131240.65157525, 602.97822105, 168.87176337, 148.80100279, 103.11250767, 66.44952443, 72.37295145, 51.3428061, 43.92186545, 43.32837547]
εTi750E2 = [34888.34923082, 578.25803895, 131.42889324, 179.51778821, 80.38856433, 50.42210657, 63.81577009, 21.47969897, 4.37473982, 1.83507536]


fig, ax = plt.subplots()
ax.errorbar(n, Ti750E, yerr = εTi750E, label = "Ti33: 750˚C, inputs same as paper")
ax.errorbar(n, Ti750E2, yerr = εTi750E2, label = "Ti33: 750˚C, other inputs")
ax.set_yscale('log')
ax.set_ylim([0.5, 200000])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
ax.set_yticklabels(["$10^{0}$", "10", "$10^{2}$", "$10^{3}$", "$10^{4}$", "$10^{5}$"])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$E^{\star}$: Cross2 method comparison for Ti33", y=0.05, fontsize=16)
plt.savefig("/Users/Joe/Desktop/figure19.png")
plt.show()
'''

n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# 750˚, cross2
Ti750E = [22.74219275, 20.62261021, 20.63547436, 19.4810463, 18.91002247, 18.19225275, 17.66865073, 17.40025689, 16.67137397, 14.09023671]
εTi750E = [1.09400474, 2.42701076, 1.67360611, 1.40042242, 1.14474779, 1.18580074, 1.15735988, 1.24283096, 0.9555188, 0.69974678]
# 750˚, cross2 (1 Estar)
Ti750E2 = [44.56400654, 43.67288301, 43.20008668, 41.82067908, 40.78293067, 40.2062695, 40.06553969, 39.03624527, 37.38997642, 33.02307318]
εTi750E2 = [0.34273797, 0.53787038, 0.79239417, 2.48630407, 2.48558065, 2.55202247, 2.26708601, 2.60285348, 2.71574786, 2.55183609]
# 500˚, cross2
Ti500E = [11.66735235, 10.29306847, 9.47812185, 8.33647048, 7.58027689, 7.51398051, 7.40959369, 6.99470173, 6.64627248, 5.97185385]
εTi500E = [0.60209906, 1.20103306, 1.37526918, 1.40743936, 0.41836403, 0.47261672, 0.55111534, 0.41686947, 0.41297669, 0.12461302]
# 500˚, cross2 (1 Estar)
Ti500E2 = [13.83313093, 12.83991465, 12.01739788, 11.62618593, 10.31228624, 9.28984869, 8.90663149, 7.79179508, 6.94111119, 4.81097428]
εTi500E2 = [0.38075165, 0.61804777, 1.14054096, 1.02286922, 1.01814732, 1.15182234, 1.13429586, 0.71245511, 0.55410064, 0.35023459]


fig, ax = plt.subplots()
ax.errorbar(n, Ti750E, yerr = εTi750E, label = "Ti33: 750˚C, different $\sigma_{y}$")
ax.errorbar(n, Ti750E2, yerr = εTi750E2, label = "Ti33: 750˚C, single $\sigma_{y}$")
ax.errorbar(n, Ti500E, yerr = εTi500E, label = "Ti33: 500˚C, different $\sigma_{y}$")
ax.errorbar(n, Ti500E2, yerr = εTi500E2, label = "Ti33: 500˚C, single $\sigma_{y}$")
ax.set_yscale('log')
ax.set_ylim([4, 50])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([4, 10, 20, 50])
ax.set_yticklabels([4, 10, 20, 50])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
plt.subplots_adjust(bottom=0.18)
plt.suptitle("$E^{\star}$: Average and distribution comparison for Ti33", y=0.05, fontsize=16)
plt.savefig("/Users/Joe/Desktop/figure20.png")
plt.show()