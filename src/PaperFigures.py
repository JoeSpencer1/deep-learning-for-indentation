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
'''
'''
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
# plt.suptitle("$E^{\star}$: Cross2 method for 3D printed Ti alloy (B3067)", y=0.05, fontsize=16)
plt.savefig("figure1.png", dpi=800, bbox_inches="tight")
plt.show()
'''
'''
B3067l = [109.8089341, 37.36280053, 11.96022579, 9.243638237, 6.154097234, 5.159452671, 4.953769264, 3.854641378, 3.481884273, 1.629349696]
B6067l = [117.5125112, 37.36280053, 13.06230256, 8.521326837, 5.634871197, 5.337364324, 4.660515817, 4.629023047, 4.01468835, 2.746324325]

fig, ax = plt.subplots()
ax.plot(n, B3067l, linestyle = '--', label = "Trained by B3067 (Lu et al.)", color = 'blue')
ax.plot(n, B6067l, linestyle = '--', label = "B6067 (Lu et al.)", color = 'blue')
ax.errorbar(n, B3067σ, yerr = εB3067σ, label = "B3067", color = 'orange')
ax.errorbar(n, B6067σ, yerr = εB6067σ, label = "B6067", color = 'orange')
# ax.errorbar(n, S6067σ, yerr = εS6067σ, label = "S6067")
ax.set_yscale('log')
ax.set_ylim([1, 20])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
ax.legend()
ax.set_ylabel("MAPE (%) finding $\sigma_{y}$ for B3067")
ax.set_xlabel("Randomly selected experimental training data set size")
plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$\sigma_{y}$: Cross2 method for 3D printed Ti alloy (B3067)", y=0.05, fontsize=16)
plt.savefig("figure2.png", dpi=800, bbox_inches="tight")
plt.show()
'''
# fig, ax = plt.subplots()
# ax.errorbar(n, B3067E, yerr = εB3067E, label = "B3067")
# ax.errorbar(n, B3090E, yerr = εB3090E, label = "B3090")
# ax.errorbar(n, B6090E, yerr = εB6090E, label = "B6090")
# ax.errorbar(n, S3067E, yerr = εS3067E, label = "S3067")
# ax.set_yscale('log')
# ax.set_ylim([1, 20])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 30])
# ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 30])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Cross2 method for 3D printed Ti alloy (B3067)", y=0.05, fontsize=16)
# plt.savefig("figure3.png", dpi=800, bbox_inches="tight")
# plt.show()
'''
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
# plt.suptitle("$\sigma_{y}$: Cross2 method for 3D printed Ti alloy (B3067)", y=0.05, fontsize=16)
plt.savefig("figure4.png", dpi=800, bbox_inches="tight")
plt.show()
'''

#  I used this code to replicate figure 9A.
'''
labels = ["$\sigma_{y}$", "$\sigma_{0.8\%}$", "$\sigma_{1.5\%}$", "$\sigma_{3.3\%}$"]
x = np.arange(4)
# B3090 (self)
μB3090 = [3.7342872, 5.13522576, 5.9153901, 8.49087502]
σB3090 = [1.14439324, 1.15316418, 1.23122, 1.50619123]
# B3067 (peer)
μB3067 = [5.44653162, 7.24383922, 8.47500916, 10.40762874]
σB3067 = [1.00719594, 1.14083534, 2.01489362, 1.87494608]
p3067 = [5.44653162 + 1.00719594, 7.24383922 + 1.14083534, 8.47500916 + 2.01489362, 10.40762874 + 1.87494608]
m3067 = [5.44653162 - 1.00719594, 7.24383922 - 1.14083534, 8.47500916 - 2.01489362, 10.40762874 - 1.87494608]

# Create bar plot with error bars for self
fig, ax = plt.subplots()
ax.bar(x, μB3090, yerr=σB3090, align='center', alpha=0.7, color = 'darkblue', label = 'B3090 (self) trainig data')
ax.scatter(x, μB3067, marker = 'x', color = 'black', label = 'B3067 (peer) training data')
ax.scatter(x, p3067, marker = '^', color = 'black')
ax.scatter(x, m3067, marker = 'v', color = 'black')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yticks([0, 5, 10, 15])
ax.set_yticklabels([0, 5, 10, 15])
ax.set_xlabel('Stress, B3090 Titanium')
ax.set_ylabel('MAPE (%)')
ax.legend()
plt.subplots_adjust(bottom=0.18)
# plt.suptitle('Error for 3D printed B3090 Ti alloy (self data)', y=0.05, fontsize=16)
plt.savefig("figure5.png", dpi=800, bbox_inches="tight")
plt.show()
'''
# # Create bar plot with error bars for peer
# fig, ax = plt.subplots()
# ax.bar(x, μB3067, yerr=σB3067, align='center', alpha=0.7)
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.set_yticks([0, 5, 10, 15])
# ax.set_yticklabels([0, 5, 10, 15])
# ax.set_xlabel('Peer')
# ax.set_ylabel('MAPE (%)')
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle('Error for 3D printed B3090 Ti alloy (peer B3067 data)', y=0.05, fontsize=16)
# plt.savefig("figure6.png", dpi=800, bbox_inches="tight")
# plt.show()


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
# plt.suptitle("Self training data (n=5)", y=0.05, fontsize=16)
plt.savefig("figure7.png", dpi=800, bbox_inches="tight")
plt.show()
'''
# # Create a plot with error bars for peer
# fig, ax = plt.subplots()
# ax.errorbar(εB3090p, μB3090p, yerr = σB3090p, label = "B3090, $n \\approx 0.105 \\pm 0.001$")
# ax.set_yscale('linear')
# ax.set_xlim([0, 0.045])
# ax.set_ylim([0, 1.7])
# ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
# ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
# ax.set_xticklabels([0, 0.01, 0.02, 0.03, 0.04])
# ax.set_yticklabels([0, 0.4, 0.8, 1.2, 1.6])
# ax.legend()
# ax.set_ylabel("$\sigma$ (GPa)")
# ax.set_xlabel("$\epsilon$")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("Peer training data", y=0.05, fontsize=16)
# plt.savefig("figure8.png", dpi=800, bbox_inches="tight")
# plt.show()


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
ε6061f = [0, 0.004515378975761151, 0.037515378975761154, 0.07051537897576116, 0.10451537897576116]
μ6061f = [0, 0.3016273155808449, 0.316699339201053, 0.3300789381066958, 0.33758289366960526]
σ6061f = [0, 0.03691037131722045, 0.03298896340550387, 0.026911666661394298, 0.030724686883124804]

# Al7075 (2D+3D+exp)
lε7075e = [0, 0.007104317, 0.040017986, 0.073111511, 0.107104317] 
lμ7075e = [0, 0.061756374 * 8, 0.076487252 * 8, 0.082577904 * 8, 0.087252125 * 8]
# Al6061 (2D+3D+exp)
lε6061e = [0, 0.004316547, 0.037230216, 0.070323741, 0.104226619]
lμ6061e = [0, 0.035835694 * 8, 0.042209632 * 8, 0.043909348 * 8, 0.04490085 * 8]
# Al7075 (2D+3D)
lε7075f = [0, 0.004340124, 0.037378211, 0.070327724, 0.104340124]
lμ7075f = [0, 0.037796374 * 8, 0.064156206 * 8, 0.073221757 * 8, 0.080055788 * 8]
# Al6061 (2D+3D)
lε6061f = [0, 0.004517272, 0.037466785, 0.070504872, 0.104517272]
lμ6061f = [0, 0.037099024 * 8, 0.039051604 * 8, 0.041143654 * 8, 0.042538354 * 8]

# Create plot with error bars for self, FEM+Exp
fig, ax = plt.subplots()
ax.errorbar(ε7075e, μ7075e, yerr = σ7075e, color = 'blue', label = "Al7075, 2D + 3D FEM + Exp")
ax.errorbar(ε7075f, μ7075f, yerr = σ7075f, linestyle = '--', color = 'blue', label = "Al7075, 2D + 3D FEM")
ax.errorbar(ε6061e, μ6061e, yerr = σ6061e, color = 'orange', label = "Al6061, 2D + 3D FEM + Exp")
ax.errorbar(ε6061f, μ6061f, yerr = σ6061f, linestyle = "--", color = 'orange', label = "Al6061, 2D + 3D FEM")
ax.plot(lε7075e, lμ7075e, color = 'red', marker = 'o', linestyle = '', label = 'Lu et al. results')
ax.plot(lε6061e, lμ6061e, color = 'red', marker = 'o', linestyle = '')
ax.plot(lε6061f, lμ6061f, color = 'red', marker = 'o', linestyle = '')
ax.plot(lε7075f, lμ7075f, color = 'red', marker = 'o', linestyle = '')
ax.set_yscale('linear')
ax.set_xlim([0, 0.11])
ax.set_ylim([0, 0.8])
ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_xticklabels([0, 0.02, 0.04, 0.06, 0.08, 0.1])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8])
ax.legend()
ax.set_ylabel("$\sigma$ (GPa)")
ax.set_xlabel("$\epsilon$ (First point is $\epsilon_{y}$)")
plt.subplots_adjust(bottom=0.18)
# plt.suptitle("2D+3D FEM+Exp (n=3)", y=0.05, fontsize=16)
plt.savefig("figure9.png", dpi=800, bbox_inches="tight")
plt.show()
'''
# # Create plot with error bars for self, FEM+Exp
# fig, ax = plt.subplots()
# ax.errorbar(ε7075f, μ7075f, yerr = σ7075f, label = "Al7075-T651, $n \\approx 0.228 \\pm 0.014$")
# ax.errorbar(ε6061f, μ6061f, linestyle = "dashed", yerr = σ6061f, label = "Al6061-T6511, $n \\approx 0.035 \\pm 0.007$")
# ax.set_yscale('linear')
# ax.set_xlim([0, 0.11])
# ax.set_ylim([0, 0.8])
# ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
# ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
# ax.set_xticklabels([0, 0.02, 0.04, 0.06, 0.08, 0.1])
# ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8])
# ax.legend()
# ax.set_ylabel("$\sigma$ (GPa)")
# ax.set_xlabel("$\epsilon$")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("2D+3D FEM", y=0.05, fontsize=16)
# plt.savefig("figure10.png", dpi=800, bbox_inches="tight")
# plt.show()


# '''This segment reproduces figure 7.'''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # B3090, self (raw)
# Esr = [20.85738611, 14.93193336, 10.62470933, 6.17081165, 3.53980273, 2.83338057, 2.39439426, 2.23912841, 2.14931529, 1.52928879]
# dEsr = [1.15774057, 2.76461151, 2.76123732, 2.28674632, 1.65000347, 1.14926682, 0.35276605, 0.4731503, 0.36532754, 0.36058943]
# σsr = [92.90621932, 31.1172741, 7.86850271, 5.42416156, 3.9266917, 3.91898551, 3.62685968, 2.97950902, 2.42648245, 1.47304528]
# dσsr = [5.72839659, 19.77325765, 6.55123033, 2.5604357, 0.74303447, 0.74347853, 0.48021291, 0.43923216, 0.46950993, 0.25023615]
# # B3090, peer (raw)
# Epr = [20.0316801, 15.30074303, 9.98728221, 6.99652784, 3.77915319, 2.39346351, 2.45927424, 2.32525492, 2.23636191, 2.0069952]
# dEpr = [0.91949316, 1.3166407, 1.9172967, 1.91926169, 1.77190192, 0.37770116, 0.4530649, 0.35565338, 0.29393664, 0.43860418]
# σpr = [99.6364055, 31.69011169, 12.93410265, 8.84905106, 7.0674062, 5.66981487, 5.6574326, 4.55043188, 4.73105714, 3.98191056]
# dσpr = [8.74225627, 11.67643018, 9.39229757, 5.01972869, 2.86092334, 1.08581043, 1.09132802, 0.70862583, 0.78805065, 0.50179615]
# # B3090, self (tip)
# Est = [7.69691729, 3.51176923, 3.38300571, 3.21087096, 2.79969631, 2.59377006, 2.43205044, 2.32385096, 2.06323201, 1.71391161]
# dEst = [1.40011309, 0.69368449, 0.83880401, 0.7035978, 0.4457362, 0.41749276, 0.35852926, 0.32584997, 0.30787117, 0.16913883]
# σst = [35.38345948, 14.27849484, 7.19610592, 4.75816201, 3.93128569, 3.41866129, 4.06738206, 2.93299789, 2.20067251, 1.28286056]
# dσst = [6.76921265, 11.30269792, 5.40612749, 2.23143059, 1.13466907, 1.4055051, 1.27503581, 1.00971715, 0.70018148, 0.27910456]
# # B3090, peer (tip)
# Ept = [7.53550653, 3.69536815, 4.13335628, 3.78174001, 3.75054876, 3.77565691, 3.94921767, 3.84544807, 3.63185784, 3.18179324]
# dEpt = [1.00549554, 0.86157193, 1.16533158, 0.82219966, 1.04377179, 0.70888312, 0.88801928, 0.75955593, 0.52060274, 0.44731189]
# σpt = [30.15499917, 14.59552122, 8.4559425, 7.03875584, 4.12011616, 3.37037623, 4.01957046, 2.75383188, 3.08886804, 3.26111124]
# dσpt = [5.07634106, 8.1286953, 2.4777729, 4.79712265, 1.3122539, 0.43977898, 1.33841325, 0.54669229, 0.46025783, 2.33967126]

# fig, ax = plt.subplots()
# ax.errorbar(n, Esr, yerr = dEsr, label = "Self (raw)")
# ax.errorbar(n, Est, yerr = dEst, label = "Self (tip)")
# ax.errorbar(n, Epr, yerr = dEpr, label = "Peer (raw)")
# ax.errorbar(n, Ept, yerr = dEpt, label = "Peer (tip)")
# ax.set_yscale('log')
# ax.set_ylim([1, 25])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([1, 2, 3, 4, 5, 10, 20])
# ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Cross2 method for 3D printed Ti alloy (B3090)", y=0.05, fontsize=16)
# plt.savefig("figure11.png", dpi=800, bbox_inches="tight")
# plt.show()

# fig, ax = plt.subplots()
# ax.errorbar(n, σsr, yerr = dσsr, label = "Self (raw)")
# ax.errorbar(n, σst, yerr = dσst, label = "Self (tip)")
# ax.errorbar(n, σpr, yerr = dσpr, label = "Peer (raw)")
# ax.errorbar(n, σpt, yerr = dσpt, label = "Peer (tip)")
# ax.set_yscale('log')
# ax.set_ylim([1, 120])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 50, 100])
# ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 50, 100])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$\sigma_{y}$: Cross2 method for 3D printed Ti alloy (B3090)", y=0.05, fontsize=16)
# plt.savefig("figure12.png", dpi=800, bbox_inches="tight")
# plt.show()


# '''These results reproduce figure 6.'''
# categories = ['NN (raw)', 'NN (tip)', 'NN self (raw, 5)', 'NN self (tip, 5)']
# EB3067 = [21.3587081, 5.67446185, 3.26508468, 2.40736801]
# εEB3067 = [1.44257363, 1.24167231, 1.30605511, 0.36433321]
# σB3067 = [107.92597028, 35.9870154, 4.82834841, 4.32661172]
# εσB3067 = [13.63118035, 5.45248244, 0.96458256, 0.98667419]
# EB3090 = [18.40507986, 7.69691729, 2.83338057, 2.59377006]
# εEB3090 = [1.58800388, 1.40011309, 1.14926682, 0.41749276]
# σB3090 = [94.30001823, 35.38345948, 3.91898551, 3.41866129]
# εσB3090 = [9.95664891, 6.76921265, 0.74347853, 1.4055051]

# # Bar graph for Estar
# bar_width = 0.35
# bar_positions1 = np.arange(len(categories))
# bar_positions2 = bar_positions1 + bar_width
# fig, ax = plt.subplots()
# ax.bar(bar_positions1, EB3067, yerr=εEB3067, width=bar_width, label='B3067')
# ax.bar(bar_positions2, EB3090, yerr=εEB3090, width=bar_width, label='B3090')
# ax.set_xticks(bar_positions1 + bar_width/2)
# ax.set_xticklabels(categories)
# ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
# ax.set_yticklabels([0, 5, 10, 15, 20, 25, 30])
# ax.set_xlabel('Test')
# ax.set_ylabel('MAPE (%)')
# ax.legend()
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle('Result of adding training data for $E^{\star}$', y=0.05, fontsize=16)
# plt.savefig("figure13.png", dpi=800, bbox_inches="tight")
# plt.show()

# # Bar graph for sigma_y
# bar_width = 0.35
# bar_positions1 = np.arange(len(categories))
# bar_positions2 = bar_positions1 + bar_width
# fig, ax = plt.subplots()
# ax.bar(bar_positions1, σB3067, yerr=εσB3067, width=bar_width, label='B3067')
# ax.bar(bar_positions2, σB3090, yerr=εσB3090, width=bar_width, label='B3090')
# ax.set_xticks(bar_positions1 + bar_width/2)
# ax.set_xticklabels(categories)
# ax.set_yticks([0, 25, 50, 75, 100, 125])
# ax.set_yticklabels([0, 25, 50, 75, 100, 125])
# ax.set_xlabel('Test')
# ax.set_ylabel('MAPE (%)')
# ax.legend()
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle('Result of adding training data for $\sigma_{y}$', y=0.05, fontsize=16)
# plt.savefig("figure14.png", dpi=800, bbox_inches="tight")
# plt.show()


'''Copy of figure 10
categories = ['2D+3D FEM', '2D+3D FEM+Experiment']
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

lEAl6061 = [2.046589018, 2.645590682]
lεEAl6061 = [0.898502496, 0.449251248]
lEAl7075 = [2.046589018, 2.396006656]
lεEAl7075 = [1.647254576, 0.549084859]
lEB3067 = [23.01164725, 3.144758735]
lεEB3067 = [4.442595674, 2.895174709]
lEB3090 = [20.26622296, 3.394342762]
lεEB3090 = [4.492512479, 3.594009983]

# Bar graph for Estar
bar_width = 0.09
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width
bar_positions3 = bar_positions1 + bar_width * 2
bar_positions4 = bar_positions1 + bar_width * 3
bar_positions5 = bar_positions1 + bar_width * 4
bar_positions6 = bar_positions1 + bar_width * 5
bar_positions7 = bar_positions1 + bar_width * 6
bar_positions8 = bar_positions1 + bar_width * 7
fig, ax = plt.subplots()
ax.bar(bar_positions1, EAl7075, yerr=εEAl7075, width=bar_width, color='darkred', label='Al7075-T651')
ax.bar(bar_positions3, EAl6061, yerr=εEAl6061, width=bar_width, color='darkgreen', label='Al6067-T6511')
ax.bar(bar_positions5, EB3067, yerr=εEB3067, width=bar_width, color='darkorange', label='B3067')
ax.bar(bar_positions7, EB3090, yerr=εEB3090, width=bar_width, color='darkblue', label='B3090')
ax.bar(bar_positions2, lEAl7075, yerr=lεEAl7075, width=bar_width, color='darkred', edgecolor='black', linewidth=2.2, label = 'Lu et al. results outlined')
ax.bar(bar_positions4, lEAl6061, yerr=lεEAl6061, width=bar_width, color='darkgreen', edgecolor='black', linewidth=2.2)
ax.bar(bar_positions6, lEB3067, yerr=lεEB3067, width=bar_width, color='darkorange', edgecolor='black', linewidth=2.2)
ax.bar(bar_positions8, lEB3090, yerr=lεEB3090, width=bar_width, color='darkblue', edgecolor='black', linewidth=2.2)
ax.set_xticks(bar_positions1 + bar_width*7/2)
ax.set_xticklabels(categories)
ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax.set_yticklabels([0, 5, 10, 15, 20, 25, 30])
#ax.set_xlabel('Test')
ax.set_ylabel('MAPE (%)')
ax.legend()
plt.subplots_adjust(bottom=0.18)
# plt.suptitle('Result of adding training data for $E^{\star}$', y=0.05, fontsize=16)
plt.savefig("figure15.png", dpi=800, bbox_inches="tight")
plt.show()
'''
# # Bar graph for sigma_y
# bar_width = 0.15
# bar_positions1 = np.arange(len(categories))
# bar_positions2 = bar_positions1 + bar_width
# bar_positions3 = bar_positions1 + bar_width * 2
# bar_positions4 = bar_positions1 + bar_width * 3
# fig, ax = plt.subplots()
# ax.bar(bar_positions1, σAl7075, yerr=εσAl7075, width=bar_width, label='Al7075-T651')
# ax.bar(bar_positions2, σAl6061, yerr=εσAl6061, width=bar_width, label='Al6067-T6511')
# ax.bar(bar_positions3, σB3067, yerr=εσB3067, width=bar_width, label='B3067')
# ax.bar(bar_positions4, σB3090, yerr=εσB3090, width=bar_width, label='B3090')
# ax.set_xticks(bar_positions1 + bar_width*3/2)
# ax.set_xticklabels(categories)
# ax.set_yticks([0, 25, 50, 75, 100, 125])
# ax.set_yticklabels([0, 25, 50, 75, 100, 125])
# ax.set_xlabel('Test')
# ax.set_ylabel('MAPE (%)')
# ax.legend()
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle('Result of adding training data for $\sigma_{y}$', y=0.05, fontsize=16)
# plt.savefig("figure16.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # Second round of figures. Include B3067 first in figures.
# # 25˚, cross2
# Ti25E = [2.08916258, 1.73458529, 1.76113493, 1.66044661, 1.57735542, 1.49558173, 1.44553545, 1.25069252, 1.20741493, 0.9812278]
# εTi25E = [0.84352267, 0.37753768, 0.23317704, 0.29399772, 0.27203281, 0.2960974, 0.23108403, 0.19055048, 0.15260457, 0.04176756]
# # 250˚, cross2
# Ti250E = [35.24559953, 13.55870661, 11.8248246, 10.923051, 9.54057722, 9.21058842, 8.54009045, 8.15317396, 7.56763099, 5.78381901]
# εTi250E = [3.1661163, 4.73711992, 2.33398521, 1.85770404, 0.8655968, 0.71457763, 0.30699462, 0.62376367, 0.43500931, 0.28294596]
# # 500˚, cross2
# Ti500E = [11.66735235, 10.29306847, 9.47812185, 8.33647048, 7.58027689, 7.51398051, 7.40959369, 6.99470173, 6.64627248, 5.97185385]
# εTi500E = [0.60209906, 1.20103306, 1.37526918, 1.40743936, 0.41836403, 0.47261672, 0.55111534, 0.41686947, 0.41297669, 0.12461302]
# # 750˚, cross2
# Ti750E = [22.74219275, 20.62261021, 20.63547436, 19.4810463, 18.91002247, 18.19225275, 17.66865073, 17.40025689, 16.67137397, 14.09023671]
# εTi750E = [1.09400474, 2.42701076, 1.67360611, 1.40042242, 1.14474779, 1.18580074, 1.15735988, 1.24283096, 0.9555188, 0.69974678]
# # B3090 (self)
# B3090E = [7.69691729, 3.51176923, 3.38300571, 3.21087096, 2.79969631, 2.59377006, 2.43205044, 2.32385096, 2.06323201, 1.71391161]
# εB3090E = [1.40011309, 0.69368449, 0.83880401, 0.7035978, 0.4457362, 0.41749276, 0.35852926, 0.32584997, 0.30787117, 0.16913883]


# fig, ax = plt.subplots()
# ax.errorbar(n, Ti25E, yerr = εTi25E, label = "Ti33: 25˚C")
# ax.errorbar(n, Ti250E, yerr = εTi250E, label = "Ti33: 250˚C")
# ax.errorbar(n, Ti500E, yerr = εTi500E, label = "Ti33: 500˚C")
# ax.errorbar(n, Ti750E, yerr = εTi750E, label = "Ti33: 750˚C")
# ax.errorbar(n, B3090E, yerr = εB3090E, label = "B3090 (from Lu et al.)")
# ax.set_yscale('log')
# ax.set_ylim([0.5, 40])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 40])
# ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 40])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Cross2 method for Ti33 and B3090", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure17.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # Second round of figures. Include B3067 first in figures.
# # 25˚, cross2
# Ti25σ = [367.02558874, 101.13129713, 42.56915805, 27.87445748, 21.04812475, 16.92794671, 14.21538944, 7.98641605, 5.30872148, 2.36812937]
# εTi25σ = [23.11441495, 64.43529896, 17.61905282, 9.35230254, 12.86762465, 6.23069119, 4.93143527, 1.75701109, 1.78478156, 0.59860558]
# # B3090, cross2
# B3090σ = [94.30001823, 28.79962952, 10.21452489, 8.23864583, 6.37878271, 5.35067482, 5.56139654, 4.85588632, 4.39152337, 4.0394429]
# εB3090σ = [9.95664891, 12.11649881, 6.03205157, 3.75291747, 1.83745355, 0.7730129, 1.81483202, 0.60761518, 0.76550689, 0.30855093]



# fig, ax = plt.subplots()
# ax.errorbar(n, Ti25σ, yerr = εTi25σ, label = "Ti33: 25˚C")
# ax.errorbar(n, B3090σ, yerr = εB3090σ, label = "B3090 (from Lu et al.)")
# ax.set_yscale('log')
# ax.set_ylim([1, 500])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([1, 2, 4, 10, 20, 50, 100, 200, 500])
# ax.set_yticklabels([1, 2, 4, 10, 20, 50, 100, 200, 500])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$\sigma_{y}$: Cross2 method for Ti33 and B3090", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure18.png", dpi=800, bbox_inches="tight")
# plt.show()
# '''
# '''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # Second round of figures. Include B3067 first in figures.
# # 750˚, cross2
# Ti750E = [22.74219275, 20.62261021, 20.63547436, 19.4810463, 18.91002247, 18.19225275, 17.66865073, 17.40025689, 16.67137397, 14.09023671]
# εTi750E = [1.09400474, 2.42701076, 1.67360611, 1.40042242, 1.14474779, 1.18580074, 1.15735988, 1.24283096, 0.9555188, 0.69974678]
# # 750˚, cross2 (other data)
# Ti750E2 = [131240.65157525, 602.97822105, 168.87176337, 148.80100279, 103.11250767, 66.44952443, 72.37295145, 51.3428061, 43.92186545, 43.32837547]
# εTi750E2 = [34888.34923082, 578.25803895, 131.42889324, 179.51778821, 80.38856433, 50.42210657, 63.81577009, 21.47969897, 4.37473982, 1.83507536]


# fig, ax = plt.subplots()
# ax.errorbar(n, Ti750E, yerr = εTi750E, label = "Ti33: 750˚C, inputs same as paper")
# ax.errorbar(n, Ti750E2, yerr = εTi750E2, label = "Ti33: 750˚C, other inputs")
# ax.set_yscale('log')
# ax.set_ylim([0.5, 200000])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
# ax.set_yticklabels(["$10^{0}$", "10", "$10^{2}$", "$10^{3}$", "$10^{4}$", "$10^{5}$"])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Cross2 method comparison for Ti33", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure19.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # 750˚, cross2
# Ti750E = [22.74219275, 20.62261021, 20.63547436, 19.4810463, 18.91002247, 18.19225275, 17.66865073, 17.40025689, 16.67137397, 14.09023671]
# εTi750E = [1.09400474, 2.42701076, 1.67360611, 1.40042242, 1.14474779, 1.18580074, 1.15735988, 1.24283096, 0.9555188, 0.69974678]
# # 750˚, cross2 (1 Estar)
# Ti750E2 = [44.56400654, 43.67288301, 43.20008668, 41.82067908, 40.78293067, 40.2062695, 40.06553969, 39.03624527, 37.38997642, 33.02307318]
# εTi750E2 = [0.34273797, 0.53787038, 0.79239417, 2.48630407, 2.48558065, 2.55202247, 2.26708601, 2.60285348, 2.71574786, 2.55183609]
# # 500˚, cross2
# Ti500E = [11.66735235, 10.29306847, 9.47812185, 8.33647048, 7.58027689, 7.51398051, 7.40959369, 6.99470173, 6.64627248, 5.97185385]
# εTi500E = [0.60209906, 1.20103306, 1.37526918, 1.40743936, 0.41836403, 0.47261672, 0.55111534, 0.41686947, 0.41297669, 0.12461302]
# # 500˚, cross2 (1 Estar)
# Ti500E2 = [13.83313093, 12.83991465, 12.01739788, 11.62618593, 10.31228624, 9.28984869, 8.90663149, 7.79179508, 6.94111119, 4.81097428]
# εTi500E2 = [0.38075165, 0.61804777, 1.14054096, 1.02286922, 1.01814732, 1.15182234, 1.13429586, 0.71245511, 0.55410064, 0.35023459]


# fig, ax = plt.subplots()
# ax.errorbar(n, Ti750E, yerr = εTi750E, label = "Ti33: 750˚C, different $E^{\star}$")
# ax.errorbar(n, Ti750E2, yerr = εTi750E2, label = "Ti33: 750˚C, single $E^{\star}$")
# ax.errorbar(n, Ti500E, yerr = εTi500E, label = "Ti33: 500˚C, different $E^{\star}$")
# ax.errorbar(n, Ti500E2, yerr = εTi500E2, label = "Ti33: 500˚C, single $E^{\star}$")
# ax.set_yscale('log')
# ax.set_ylim([4, 50])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([4, 10, 20, 50])
# ax.set_yticklabels([4, 10, 20, 50])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Average and distribution comparison for Ti33", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure20.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # 750˚, cross2 (self)
# Ti750sE = [22.74219275, 20.62261021, 20.63547436, 19.4810463, 18.91002247, 18.19225275, 17.66865073, 17.40025689, 16.67137397, 14.09023671]
# εTi750sE = [1.09400474, 2.42701076, 1.67360611, 1.40042242, 1.14474779, 1.18580074, 1.15735988, 1.24283096, 0.9555188, 0.69974678]
# # 750˚, cross2 (1 Estar, self)
# Ti750sE2 = [44.56400654, 43.67288301, 43.20008668, 41.82067908, 40.78293067, 40.2062695, 40.06553969, 39.03624527, 37.38997642, 33.02307318]
# εTi750sE2 = [0.34273797, 0.53787038, 0.79239417, 2.48630407, 2.48558065, 2.55202247, 2.26708601, 2.60285348, 2.71574786, 2.55183609]
# # 750˚, cross2 (250˚, uncleaned)
# Ti750pE = [22.91273365, 22.47692978, 23.06164819, 22.51761829, 23.88522265, 24.65573764, 24.24359862, 24.35779437, 24.96326324, 24.74265974]
# εTi750pE = [1.16481071, 0.85124713, 1.79050013, 1.36539445, 1.34524191, 2.14761082, 0.99580356, 1.32409663, 0.84506931, 0.98756287]
# # 750˚, cross2 (250˚)
# Ti750pE2 = [22.7432109, 20.30678536, 20.38437269, 20.2808522, 19.92654862, 20.71763083, 20.6963388, 20.39988734, 20.55628761, 20.27633542]
# εTi750pE2 = [1.58256565, 0.4674164, 0.95313367, 0.52655137, 0.45183505, 0.72136933, 0.40416965, 0.6261124, 0.64190076, 0.78686324]

# fig, ax = plt.subplots()
# ax.errorbar(n, Ti750sE, yerr = εTi750sE, label = "Ti33: 750˚C, different $E^{\star}$")
# ax.errorbar(n, Ti750sE2, yerr = εTi750sE2, label = "Ti33: 750˚C, single same $E^{\star}$")
# ax.errorbar(n, Ti750pE, yerr = εTi750pE, label = "Ti33: 750˚C, trained from 25˚C")
# ax.errorbar(n, Ti750pE2, yerr = εTi750pE2, label = "Ti33: 750˚C, trained from 25˚C, $h_{max} \\leq 400 nm$")
# ax.set_yscale('log')
# ax.set_ylim([4, 50])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([4, 10, 20, 50])
# ax.set_yticklabels([4, 10, 20, 50])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Average and distribution comparison for Ti33, 750˚C", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure21.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # 250˚, cross2 (self)
# Ti250E = [35.24559953, 13.55870661, 11.8248246, 10.923051, 9.54057722, 9.21058842, 8.54009045, 8.15317396, 7.56763099, 5.78381901]
# εTi250E = [3.1661163, 4.73711992, 2.33398521, 1.85770404, 0.8655968, 0.71457763, 0.30699462, 0.62376367, 0.43500931, 0.28294596]
# # 250˚, cross2 (25˚)
# Ti25E = [34.33835857, 34.93023696, 34.00767512, 34.63857446, 34.13918443, 34.95016874, 36.04625686, 36.48948703, 38.87897481, 36.99936865]
# εTi25E = [3.07228799, 3.09623858, 3.38829581, 2.59930911, 1.92108184, 2.68523295, 2.24788623, 2.90450047, 1.91752321, 3.00116909]
# # 250˚, cross2 (500˚)
# Ti500E = [33.38612255, 35.74112078, 36.4283278, 36.20695545, 38.7743991, 37.25338799, 39.02619327, 36.79238692, 35.63924399, 35.97073564]
# εTi500E = [2.34985368, 4.21925637, 3.93393791, 4.9898504, 4.21978169, 5.33950463, 4.53798639, 5.63944341, 6.16649966, 3.55755834]

# fig, ax = plt.subplots()
# ax.errorbar(n, Ti250E, yerr = εTi250E, label = "Ti33: 250˚C, trained by 250˚C")
# ax.errorbar(n, Ti25E, yerr = εTi25E, label = "Ti33: 250˚C, trained by 25˚C")
# ax.errorbar(n, Ti500E, yerr = εTi500E, label = "Ti33: 250˚C, trained by 500˚C")
# ax.set_yscale('log')
# ax.set_ylim([4, 50])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([4, 10, 20, 50])
# ax.set_yticklabels([4, 10, 20, 50])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Average and distribution comparison for Ti33, 250˚C", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure22.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # 250˚, edited
# Ti250Ee = [32.23345345, 32.83613913, 31.95558785, 34.926363, 34.34764688, 33.56840004, 35.65507285, 34.56202343, 35.02562089, 36.9571808]
# εTi250Ee = [2.3581232, 2.90034917, 2.58022542, 2.82644808, 2.74713307, 2.41362589, 3.41212201, 2.91091963, 2.14901452, 2.00551627]
# # 250˚, unedited
# Ti250Eu = [35.24559953, 13.55870661, 11.8248246, 10.923051, 9.54057722, 9.21058842, 8.54009045, 8.15317396, 7.56763099, 5.78381901]
# εTi250Eu = [3.1661163, 4.73711992, 2.33398521, 1.85770404, 0.8655968, 0.71457763, 0.30699462, 0.62376367, 0.43500931, 0.28294596]

# fig, ax = plt.subplots()
# ax.errorbar(n, Ti250Ee, yerr = εTi250Ee, label = "Ti33: 250˚C, large changes in depth excluded")
# ax.errorbar(n, Ti250Eu, yerr = εTi250Eu, label = "Ti33: 250˚C, trained by all 250˚C data")
# ax.set_yscale('log')
# ax.set_ylim([4, 50])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([4, 10, 20, 50])
# ax.set_yticklabels([4, 10, 20, 50])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Average and distribution comparison for Ti33, 250˚C", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure23.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # 750˚, 25˚ data there, 750˚ data added
# Ti750E1 = [17.10079961, 16.99942028, 17.44123487, 17.42660323, 17.33004997, 17.67587651, 18.22724009, 17.44525551, 18.11950215, 18.26867337]
# εTi750E1 = [5.97674614, 5.63917552, 5.64814358, 6.29537589, 6.26110722, 6.02255403, 6.73762308, 6.73125919, 6.82417437, 7.08645536]
# # 750˚, 25˚ data added
# Ti750E2 = [23.67731907, 23.61807471, 22.2258443, 22.19685189, 21.64394846, 21.27171676, 21.24659084, 19.87388692, 19.78350748, 18.27975746]
# εTi750E2 = [1.81490396, 2.99965854, 3.43967608, 3.42143266, 3.81803575, 4.12182853, 4.32363957, 4.67874097, 5.44705205, 7.03637642]
# # 750˚, cross2, 25˚ training data
# Ti750E3 = [22.76704549, 23.58014485, 23.04126101, 23.0897574, 23.39195654, 23.60942335, 23.95367643, 24.51037973, 24.64922514, 24.40430102]
# εTi750E3 = [1.69016798, 1.39470559, 0.96538455, 1.11105919, 1.27402408, 1.54786652, 1.21917963, 0.90542558, 1.11773548, 0.82672601]
# # 750˚, cross2, 750˚ training data
# Ti750E4 = [23.37565133, 20.93590694, 19.54231555, 19.22836006, 18.63762583, 18.17080232, 17.84399624, 17.04697986, 16.6985118, 14.14832345]
# εTi750E4 = [1.33550844, 2.47647777, 1.11288338, 1.67852863, 1.02832587, 1.18450794, 1.24313337, 0.94244277, 0.62762372, 0.53368235]

# fig, ax = plt.subplots()
# ax.errorbar(n, Ti750E1, yerr = εTi750E1, label = "Ti33: 750˚C, trained previously by 20 25˚C points.")
# ax.errorbar(n, Ti750E2, yerr = εTi750E2, label = "Ti33: 750˚C, trained after by 20 750˚C points.")
# ax.errorbar(n, Ti750E3, yerr = εTi750E3, label = "Ti33: 750˚C, trained only by 25˚C points.")
# ax.errorbar(n, Ti750E4, yerr = εTi750E4, label = "Ti33: 750˚C, trained only by 750˚C points.")
# ax.set_yscale('log')
# ax.set_ylim([4, 50])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([4, 10, 20, 50])
# ax.set_yticklabels([4, 10, 20, 50])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Average and distribution comparison for Ti33, 750˚C", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure24.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# n2 = [0, 1, 2, 3, 4, 5, 6, 8, 10]
# # 500, 25˚ data there, 500˚ data added
# Ti500E1 = [9.06844964, 8.95897659, 8.74157255, 8.88099513, 8.96389273, 8.86759218, 8.84647914, 8.62334759, 8.65142586, 8.79556547]
# εTi500E1 = [2.38190357, 2.37549759, 2.56228789, 2.27952376, 2.27980669, 2.24994576, 2.23420322, 2.43671908, 2.47324881, 2.18491273]
# # 500, 25˚ data added
# Ti500E2 = [10.95020934, 10.95486886, 9.71612238, 9.5857391, 9.56613242, 9.16158257, 9.39940477, 9.2141574, 9.00350518, 8.57545337]
# εTi500E2 = [0.97045022, 0.92872509, 1.38839237, 1.63530712, 1.46739893, 1.76158313, 1.52044744,1.70269884, 1.95306613, 2.30776601]
# # 500, points with large jumps in depth removed
# Ti500E3 = [12.97952732, 11.33137112, 11.0732476, 9.97881354, 9.96821186, 9.74260725, 9.41073104, 8.86642822, 8.67789414]
# εTi500E3 = [1.07508901, 1.79608641, 2.46895544, 2.49647574, 2.30221106, 2.53031004, 2.78672357, 3.1900416, 3.31327466]
# # 500, points with depth above 400nm removed
# Ti500E4 = [12.57313217, 9.66946921, 9.99841411, 9.42267512, 9.61091743, 9.33324485, 9.14848195, 9.10865716, 8.90918735]
# εTi500E4 = [2.28695071, 1.29059535, 1.08894692, 1.60213386, 1.51607007, 1.66444883, 1.70230611, 1.7526536, 2.13574849]
# # 500, cross2, 25˚ training data
# Ti500E5 = [11.68147749, 11.02821955, 11.09554448, 10.90373067, 11.23115768, 11.06519976, 11.08157888, 11.10420958, 10.97754063, 10.91304784]
# εTi500E5 = [0.55719927, 0.50194689, 0.60334575, 0.28988114, 0.16126524, 0.20246318, 0.20679666, 0.32117734, 0.30247793, 0.27745809]
# # 500, cross2, 500˚ training data
# Ti500E6 = [11.69849773, 10.33584506, 9.15315264, 8.32445867, 7.69065487, 7.63522217, 7.46217101, 6.85775915, 6.71061151, 5.96008199]
# εTi500E6 = [0.7359779, 1.06572191, 1.00514533, 1.6434037, 0.66580158, 0.48297202, 0.52732167, 0.35164309, 0.30710129, 0.18202609]

# fig, ax = plt.subplots()
# ax.errorbar(n, Ti500E2, yerr = εTi500E2, label = "Ti33: 500˚C, trained previously by 20 25˚C points.")
# ax.errorbar(n, Ti500E1, yerr = εTi500E1, label = "Ti33: 500˚C, trained after by 20 250˚C points.")
# ax.errorbar(n2, Ti500E3, yerr = εTi500E3, label = "Ti33: 500˚C, training points with large jumps removed.")
# ax.errorbar(n2, Ti500E4, yerr = εTi500E4, label = "Ti33: 500˚C, training points above 400$nm$ removed.")
# ax.errorbar(n, Ti500E5, yerr = εTi500E5, label = "Ti33: 500˚C, trained only by 25˚C points.")
# ax.errorbar(n, Ti500E6, yerr = εTi500E6, label = "Ti33: 500˚C, trained only by 500˚C points.")
# ax.set_yscale('log')
# ax.set_ylim([4, 50])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([4, 10, 20, 50])
# ax.set_yticklabels([4, 10, 20, 50])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Average and distribution comparison for Ti33, 750˚C", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure25.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # B3090, σy, n = 1
# n1_0 = [98.49592034, 28.3315571, 9.13279462, 4.75395415, 3.84127982, 3.7892252, 3.31617797, 2.9334443, 2.57516411, 1.57088288]
# εn1_0 = [11.24499941, 19.30820605, 8.08683741, 2.17333073, 0.8701756, 0.75429947, 0.69527508, 0.61149343, 0.61207465, 0.47639986]
# # B3090, σy, n = 0.9
# n0_9 = [114.72239649, 38.53480713, 7.96026169, 5.63138492, 4.43061994, 4.38413313, 4.08039391, 3.46620677, 3.20172058, 2.01742421]
# εn0_9 = [9.28745359, 28.83949786, 5.52702985, 1.63956769, 0.90186818, 0.71692988, 0.78514276, 0.48432413, 0.77302467, 0.58174929]
# # B3090, σy, n = 2
# n2_0 = [7.08088132, 8.45037008, 6.97305585, 5.25959933, 4.54565971, 4.45630638, 4.44548592, 3.80771263, 3.05205506, 1.40631264]
# εn2_0 = [2.320879, 3.07690361, 3.71589515, 1.2148858, 0.86722283, 0.81242867, 1.06853352, 0.87643804, 0.73767514, 0.80456655]

# fig, ax = plt.subplots()
# ax.errorbar(n, n1_0, yerr = εn1_0, label = "B3090: $\sigma_{y}$ kept same")
# ax.errorbar(n, n0_9, yerr = εn0_9, label = "B3090: $\sigma_{y}$ scaled by 0.9")
# ax.errorbar(n, n2_0, yerr = εn2_0, label = "B3090: $\sigma_{y}$ scaled by 2.0")
# ax.set_yscale('log')
# ax.set_ylim([1, 150])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
# ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$\sigma_{y}$: Average and distribution comparison for B3090", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure26.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # B3090, σy, n = 1
# n1_0 = [98.49592034, 28.3315571, 9.13279462, 4.75395415, 3.84127982, 3.7892252, 3.31617797, 2.9334443, 2.57516411, 1.57088288]
# εn1_0 = [11.24499941, 19.30820605, 8.08683741, 2.17333073, 0.8701756, 0.75429947, 0.69527508, 0.61149343, 0.61207465, 0.47639986]
# # B3090, σy, n = 1.2
# n1_2 = [62.18746881, 20.77006246, 6.07386145, 4.67094792, 3.80993917, 3.21559106, 2.90994435, 2.2756584, 1.89705863, 0.94360063]
# εn1_2 = [6.36293821, 15.08719327, 5.03784865, 2.10963245, 0.83552138, 0.65420633, 0.60701194, 0.67162512, 0.5162857, 0.27499685]
# # B3090, σy, n = 1.4
# n1_4 = [39.87091245, 12.00560813, 7.19394071, 4.84015491, 3.43191733, 3.00892746, 2.92389963, 2.45194226, 1.60882474, 0.70353023]
# εn1_4 = [5.95626358, 11.85366255, 5.04170983, 2.48065939, 0.90542564, 0.98852857, 0.85372437, 0.50520964, 0.56505485, 0.24393745]

# fig, ax = plt.subplots()
# ax.errorbar(n, n1_0, yerr = εn1_0, label = "B3090: $\sigma_{y}$ kept same")
# ax.errorbar(n, n1_2, yerr = εn1_2, label = "B3090: $\sigma_{y}$ scaled by 1.2")
# ax.errorbar(n, n1_4, yerr = εn1_4, label = "B3090: $\sigma_{y}$ scaled by 1.4")
# ax.set_yscale('log')
# ax.set_ylim([0.4, 150])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
# ax.set_yticklabels([0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$\sigma_{y}$: Average and distribution comparison for B3090", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure27.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # B3090, σy, n = 1
# n1_0 = [98.49592034, 28.3315571, 9.13279462, 4.75395415, 3.84127982, 3.7892252, 3.31617797, 2.9334443, 2.57516411, 1.57088288]
# εn1_0 = [11.24499941, 19.30820605, 8.08683741, 2.17333073, 0.8701756, 0.75429947, 0.69527508, 0.61149343, 0.61207465, 0.47639986]
# # B3090, σy, n = 100
# n100_0 = [98.04639147, 98.00439382, 98.07543755, 98.12216364, 98.14245471, 98.05236912, 98.13138538, 98.12003924, 98.16982926, 98.0218759]
# εn100_0 = [0.06621925, 0.09920123, 0.11724964, 0.0875019, 0.071634, 0.11097822, 0.06326675, 0.07954329, 0.09023578, 0.08980225]

# fig, ax = plt.subplots()
# ax.errorbar(n, n1_0, yerr = εn1_0, label = "B3090: $\sigma_{y}$ kept same")
# ax.errorbar(n, n100_0, yerr = εn100_0, marker = '.', label = "B3090: $\sigma_{y}$ scaled by 100")
# ax.set_yscale('log')
# ax.set_ylim([1, 150])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
# ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 50, 100, 150])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$\sigma_{y}$: Average and distribution comparison for B3090", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure28.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # B3090, E*, n = 1
# n1_0 = [20.16392164, 14.80426556, 10.46854229, 4.88294269, 3.85449327, 2.52047819, 2.44898775, 2.3127348, 2.09152695, 1.47288942]
# εn1_0 = [2.10357489, 3.16661384, 2.75502941, 2.73140086, 1.55048376, 0.68777228, 0.46392257, 0.70412065, 0.46266797, 0.1687927]
# # B3090, E*, n = 0.9
# n0_9 = [34.23329871, 26.54762807, 21.13117563, 15.16722394, 5.88164963, 2.9613689, 2.34424501, 2.33933122, 2.14942776, 1.3599664]
# εn0_9 = [0.98378458, 2.29682186, 2.72428294, 2.97363746, 4.08360106, 1.24194156, 0.48046822, 0.4793543, 0.44819542, 0.27097482]
# # B3090, E*, n = 2.0
# n2_0 = [39.63453442, 38.1203831, 36.42018219, 34.92167888, 33.34318091, 31.96219133, 30.63494942, 24.88528291, 15.38010812, 3.49460808]
# εn2_0 = [0.68663658, 0.3144766, 0.94900132, 0.84514082, 0.6721743, 1.2078501, 1.17746678, 3.79028052, 4.61559651, 0.46473715]

# fig, ax = plt.subplots()
# ax.errorbar(n, n1_0, yerr = εn1_0, label = "B3090: $\sigma_{y}$ kept same")
# ax.errorbar(n, n0_9, yerr = εn0_9, label = "B3090: $\sigma_{y}$ scaled by 0.9")
# ax.errorbar(n, n2_0, yerr = εn2_0, label = "B3090: $\sigma_{y}$ scaled by 2.0")
# ax.set_yscale('log')
# ax.set_ylim([1, 45])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 40])
# ax.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 40])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: Average and distribution comparison for B3090", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure29.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # TiAlTa, σy, n = 1
# n1_0 = [1759.85911657, 593.16904768, 215.12340113, 141.02275578, 99.50019637, 64.25943463, 36.97041575, 28.77422742, 19.10477151, 7.27430088]
# εn1_0 = [87.09786518, 394.64346437, 71.04115847, 65.97763099, 70.31349131, 39.49880751, 12.8745031, 7.5893146, 6.5958577, 4.78252914]
# # TiAlTa, σy, n = 2
# n2_0 = [835.58111989, 262.66396771, 96.47755345, 63.75761493, 45.49796753, 32.21981716, 20.54848219, 14.75753135, 9.65252048, 4.6405925]
# εn2_0 = [57.90181842, 177.96028916, 35.93773782, 27.55124647, 32.81857051, 16.96113523, 8.10328142, 4.16914335, 2.80308026, 1.60786508]

# fig, ax = plt.subplots()
# ax.errorbar(n, n1_0, yerr = εn1_0, label = "TiAlTa: $\sigma_{y}$ kept same")
# ax.errorbar(n, n2_0, yerr = εn2_0, label = "TiAlTa: $\sigma_{y}$ scaled by 2")
# ax.set_yscale('log')
# ax.set_ylim([2, 2600])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([2, 4, 10, 40, 100, 500, 2000])
# ax.set_yticklabels([2, 4, 10, 40, 100, 500, 2000])
# ax.legend()
# ax.set_ylabel("MAPE (%)")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$\sigma_{y}$: Average and distribution comparison for 33%TiAlTa", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure30.png", dpi=800, bbox_inches="tight")
# plt.show()

# ''''''
# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # TiAlTa, σy, n = 1
# n1_0 = [3.864059, 1.365728, 0.535192, 0.37235823, 0.34945855, 0.29876357, 0.25746316, 0.25134218, 0.23387516, 0.21406484]
# εn1_0 = [1.3333834, 1.3333834, 0.5511996, 0.4215711, 0.31248736, 0.2278596, 0.13587719, 0.108948134, 0.08867363, 0.047126405]
# n1_0 = [x * 1000 for x in n1_0]
# εn1_0 = [x * 1000 for x in εn1_0]
# # Actual value
# value = [209, 209]
# x = [-1, 21]

# fig, ax = plt.subplots()
# ax.plot(x, value, label = '$\sigma_{y} = 209 MPa$')
# ax.errorbar(n, n1_0, yerr = εn1_0, label = "TiAlTa: 25˚C")
# ax.set_yscale('linear')
# ax.set_ylim([0, 5300])
# ax.set_xlim([-1, 21])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([0, 209, 500, 1000, 2000, 5000])
# ax.set_yticklabels([0, 209, 500, 1000, 2000, 5000])
# ax.legend()
# ax.set_ylabel("$\sigma_{y}$, MPa")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$\sigma_{y}$: program findings with added training data", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure31.png", dpi=800, bbox_inches="tight")
# plt.show()


# n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# # TiAlTa, E*, T = 100˚
# n100 = [107.98086, 128.77345, 129.83005, 123.704216, 119.43117, 119.78103, 119.8425, 120.29419, 121.057365, 121.02052]
# εn100 = [52.252518, 44.448933, 35.05853, 35.810158, 41.508427, 42.551655, 42.826797, 43.019463, 43.65513, 44.754738]
# # TiAlTa, E*, T = 200˚
# n200 = [108.73268, 128.22731, 129.9833, 124.54512, 118.53557, 119.37834, 119.91793, 119.599785, 121.14458, 121.09966]
# εn200 = [50.40294, 45.410503, 35.427982, 35.49514, 41.152905, 42.339294, 42.827354, 43.35869, 43.716194, 44.876877]
# # TiAlTa, T = 250˚
# n250 = [67.35, 79.05415, 81.262184, 81.65729, 83.02792, 83.415115, 84.349846, 84.3668, 84.86247, 86.60203]
# εn250 = [47.370438, 39.15883, 37.265903, 37.256607, 37.454945, 37.081303, 37.238724, 37.96447, 38.57067, 39.733036]

# fig, ax = plt.subplots()
# ax.errorbar(n, n100, yerr = εn100, label = "TiAlTa: 100˚C")
# ax.errorbar(n, n200, yerr = εn200, label = "TiAlTa: 200˚C")
# ax.errorbar(n, n250, yerr = εn250, label = "TiAlTa: 250˚C (cross2)")
# ax.set_yscale('linear')
# ax.set_ylim([0, 250])
# ax.set_xlim([-1, 21])
# ax.set_xticks([0, 5, 10, 15, 20])
# ax.set_yticks([0, 100, 200, 250])
# ax.set_yticklabels([0, 100, 200, 250])
# ax.legend()
# ax.set_ylabel("$E^{\star}$, GPa")
# ax.set_xlabel("Randomly selected training data set size ($n_{exp}$)")
# plt.subplots_adjust(bottom=0.18)
# plt.suptitle("$E^{\star}$: trained by 25˚C and 250˚C data", y=0.05, fontsize=16)
# plt.savefig("/Users/Joe/Desktop/figure32.png", dpi=800, bbox_inches="tight")
# plt.show()
'''
n = [10, 20, 30, 40, 50, 60, 70]
# 
HF = [14.27148194, 7.098381071, 5.803237858, 5.056039851, 4.707347447, 3.586550436, 3.387297634]
εHF = [19.6014944 - 14.27148194, 9.01618929 - 7.098381071, 8.343711083 - 5.803237858, 6.650062267 - 5.056039851, 6.575342466 - 4.707347447, 4.508094645 - 3.586550436, 4.408468244 - 3.387297634]
# 
MF = [5.056039851, 4.508094645, 4.333748443, 4.159402242, 3.860523039, 3.337484433, 3.105022958]
εMF = [5.678704857 - 5.056039851, 5.080946451 - 4.508094645, 4.856787049 - 4.333748443, 4.806973848 - 4.159402242, 4.358655044 - 3.860523039, 3.810709838 - 3.337484433, 4.30053977 - 3.105022958]
# 
FITx = [0, 80]
FITy = [8.546285065, 8.546285065]

fig, ax = plt.subplots()
ax.errorbar(n, HF, yerr = εHF, color = 'red', label = "High-fidelity only (2D FEM)")
ax.errorbar(n, MF, linestyle='--', color = 'blue', yerr = εMF, label = "Mulit-fidelity (equations + 2D FEM)")
ax.plot(FITx, FITy, linestyle=':', color = 'gray', label = 'Fitting function only')
ax.set_yscale('linear')
ax.set_ylim([0, 20])
ax.set_xlim([0, 80])
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_yticklabels([0, 5, 10, 15, 20])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size")
plt.subplots_adjust(bottom=0.180)
# # plt.suptitle("$E^{\star}$: Al7075", y=0.05, fontsize=16)
plt.savefig("/Users/Joe/Desktop/figure33.png", dpi=800, bbox_inches="tight")
plt.show()
'''
'''
n = [10, 20, 30, 40, 50, 60, 70]
# 
eE = [5.8935575, 7.6295156, 8.989728, 7.5137663, 7.7682357, 9.665442, 13.890027]
εeE = [1.9547685, 3.0702984, 3.438875, 3.1124375, 3.218682, 4.165104, 7.478386]
# 
eσ = [343.7671, 642.84216, 454.07074, 731.2926, 556.8475, 694.9691, 435.2622]
εeσ = [82.25972, 323.35184, 347.1608, 920.5331, 339.55984, 386.63605, 205.11884]

fig, ax = plt.subplots()
ax.errorbar(n, eσ, yerr = εeσ, color = 'red', label = "B3067 Titanium")
ax.set_yscale('linear')
ax.set_ylim([0, 1700])
ax.set_xlim([0, 80])
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
ax.set_yticks([0, 500, 1000, 1500])
ax.set_yticklabels([0, 500, 1000, 1500])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size")
plt.subplots_adjust(bottom=0.180)
# # plt.suptitle("$E^{\star}$: Al7075", y=0.05, fontsize=16)
plt.savefig("/Users/Joe/Desktop/figure34.png", dpi=800, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.errorbar(n, eE, yerr = εeE, color = 'red', label = "B3067 Titanium")
ax.set_yscale('linear')
ax.set_ylim([0, 20])
ax.set_xlim([0, 80])
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_yticklabels([0, 5, 10, 15, 20])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size")
plt.subplots_adjust(bottom=0.180)
# # plt.suptitle("$E^{\star}$: Al7075", y=0.05, fontsize=16)
plt.savefig("/Users/Joe/Desktop/figure35.png", dpi=800, bbox_inches="tight")
plt.show()


n = [1, 5, 10, 15, 20]
#
Ti250σ = [11561.413, 664.87213, 639.9049, 846.042, 438.77017]
εTi250σ = [11732.797, 353.99075, 328.27576, 646.9536, 361.80612]
Ti250E = [17.021046, 10.106288, 9.25302, 6.7808356, 6.323163]
εTi250E = [8.124523, 4.1910205, 4.886315, 2.8530917, 2.6351047]
#
Ti25σ = [5681.1235, 5464.918, 4425.865, 3855.78, 4540.3574]
εTi25σ = [4709.873, 2883.6755, 2425.2305, 1209.8816, 3478.931]
Ti25E = [8.817591, 8.074902, 6.2651687, 6.9462066, 7.5449853]
εTi25E = [5.034724, 2.330901, 3.2355785, 4.1040573, 3.2267334]

fig, ax = plt.subplots()
ax.errorbar(n, Ti25E, yerr = εTi25E, color = 'blue', label = "25% TiAlTa (25˚)")
ax.errorbar(n, Ti250E, yerr = εTi250E, color = 'red', label = "25% TiAlTa (250˚)")
ax.set_yscale('linear')
ax.set_ylim([0, 25])
ax.set_xlim([0, 22])
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_yticklabels([0, 5, 10, 15, 20])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size")
plt.subplots_adjust(bottom=0.180)
# # plt.suptitle("$E^{\star}$: Al7075", y=0.05, fontsize=16)
plt.savefig("/Users/Joe/Desktop/figure36.png", dpi=800, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.errorbar(n, Ti25σ, yerr = εTi25σ, color = 'blue', label = "25% TiAlTa (25˚)")
ax.errorbar(n, Ti250σ, yerr = εTi250σ, color = 'red', label = "25% TiAlTa (250˚)")
ax.set_yscale('log')
ax.set_ylim([100, 25000])
ax.set_xlim([0, 22])
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_yticks([100, 500, 1000, 5000, 10000])
ax.set_yticklabels([100, 500, 1000, 5000, 10000])
ax.legend()
ax.set_ylabel("MAPE (%)")
ax.set_xlabel("Randomly selected training data set size")
plt.subplots_adjust(bottom=0.180)
# # plt.suptitle("$E^{\star}$: Al7075", y=0.05, fontsize=16)
plt.savefig("/Users/Joe/Desktop/figure37.png", dpi=800, bbox_inches="tight")
plt.show()
'''

n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
n2 = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
# B3067/B3067, cross2
B3067E = [21.3587081, 15.17970714, 10.62110174, 5.83347194, 3.59740196, 2.63089558, 2.4470285, 2.18840372, 2.13009046, 1.73676738]
εB3067E = [1.44257363, 1.93729254, 1.77537533, 1.90916378, 1.48526883, 0.60964792, 0.50944835, 0.27585647, 0.2970057, 0.39076581]
B3067σ = [107.92597028, 29.58839829, 11.33450252, 8.07809034, 5.70168262, 5.07757324, 4.14280765, 3.39958904, 2.84703485, 1.34625944]
εB3067σ = [13.63118035, 18.54100985, 7.24847445, 3.51824078, 1.72730742, 1.24935494, 1.01473675, 0.8416667, 0.43855084, 0.29775513]
# B3090/B3067, cross2
B3090E = [18.40507986, 12.81159777, 7.97587591, 5.48326667, 3.4110104, 2.5976216, 2.3027464, 2.09862443, 2.09303332, 1.8776037]
εB3090E = [1.58800388, 0.93433756, 1.57910684, 2.37402331, 1.54619928, 0.58068648, 0.25335815, 0.26541917, 0.19477368, 0.37367781]
B3090σ = [94.30001823, 28.79962952, 10.21452489, 8.23864583, 6.37878271, 5.35067482, 5.56139654, 4.85588632, 4.39152337, 4.0394429]
εB3090σ = [9.95664891, 12.11649881, 6.03205157, 3.75291747, 1.83745355, 0.7730129, 1.81483202, 0.60761518, 0.76550689, 0.30855093]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.errorbar(n2, B3067E, yerr = εB3067E, color = 'blue', label = "B3067 (self) trainined")
ax1.errorbar(n2, B3090E, yerr = εB3090E, color = 'red', label = "B3090 (peer) trainined˚)")
ax1.set_yscale('log')
ax1.set_ylim([1, 25])
ax1.set_xlim([-0.5, 21])
ax1.set_xticks([0, 5, 10, 15, 20])
ax1.set_yticks([1, 2, 3, 4, 5, 10, 20])
ax1.set_yticklabels([1, 2, 3, 4, 5, 10, 20])
ax1.legend()
ax1.set_ylabel("MAPE (%)")
ax1.set_xlabel("Experimental training data size")
ax1.annotate("A: $E^{\star}$", xy=(0.15, 0.95), xycoords="axes fraction",
              fontsize=12, ha="center",
              bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

ax2.errorbar(n, B3067σ, yerr = εB3067σ, color = 'blue', label = "B3067 (self) trainined")
ax2.errorbar(n, B3090σ, yerr = εB3090σ, color = 'red', label = "B3090 (peer) trainined")
ax2.set_yscale('log')
ax2.set_ylim([1, 110])
ax2.set_xlim([-0.5, 21])
ax2.set_xticks([0, 5, 10, 15, 20])
ax2.set_yticks([1, 2, 3, 4, 5, 10, 20, 50, 100])
ax2.set_yticklabels([1, 2, 3, 4, 5, 10, 20, 50, 100])
ax2.legend()
ax2.set_ylabel("MAPE (%)")
ax2.set_xlabel("Experimental training data size")
plt.subplots_adjust(bottom=0.180)
fig.tight_layout()
ax2.annotate("B: $\sigma_{y}$", xy=(0.15, 0.95), xycoords="axes fraction",
              fontsize=12, ha="center",
              bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))
plt.savefig("/Users/Joe/Desktop/figure38.jpeg", dpi=800, bbox_inches="tight")
plt.show()