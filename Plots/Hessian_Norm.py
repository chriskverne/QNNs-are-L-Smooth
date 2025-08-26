import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

# """Two Gates"""
# # q = 4, g = 2, l = [1,2,3,4,5,6,7,8,9,10]
# fourQ_twoG = [(1, 4, 2, 0.8879927230939715), (2, 4, 2, 0.7149009407554339), (3, 4, 2, 1.118382167660306), (4, 4, 2, 1.4537154108324442), (5, 4, 2, 1.432937049675312), (6, 4, 2, 1.7363925738398756), (7, 4, 2, 2.184481617369816), (8, 4, 2, 2.1005096791101936), (9, 4, 2, 2.519314170603223), (10, 4, 2, 2.8205321708960467)]
#
# # q = 8, g = 2, l = [1,2,3,4, 5]
# eightQ_twoG = [(1, 8, 2, 0.4380778457243751), (2, 8, 2, 0.26729427076705997), (3, 8, 2, 0.32466597511889955), (4, 8, 2, 0.4134197059027252), (5, 8, 2, 0.3455305946654719)]
#
# # q = 10, g = 2, l = [1,2,3,4]
# tenQ_twoG = [(1, 10, 2, 0.40978141209777946), (2, 10, 2, 0.163454521219675), (3, 10, 2, 0.32221696271360833), (4, 10, 2, 0.2017337963842811)]


"""3 Gates WITH ENTANGLEMENT"""
# q = 1, g = 3, l = [1,5,10, 15, 20, 25, 30, 35, 40]
oneQ_threeG = [(1, 1, 3, 1.965525726370158), (5, 1, 3, 6.436282116066916), (10, 1, 3, 11.776384364142976), (15, 1, 3, 16.783976234411604), (20, 1, 3, 22.177215728608942), (25, 1, 3, 27.513140120674816), (30, 1, 3, 32.29561382430927), (35, 1, 3, 39.08914586864514), (40, 1, 3, 44.63155788758197)]

# q = 2, g = 3, l = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
twoQ_threeG = [(1, 2, 3, 1.9182122556740817), (2, 2, 3, 2.4071741737372507), (3, 2, 3, 3.5236568840002587), (4, 2, 3, 4.811272362899584), (5, 2, 3, 4.808879133191292), (6, 2, 3, 6.080643970822507), (7, 2, 3, 7.861919779111797), (8, 2, 3, 7.275262304341641), (9, 2, 3, 8.493986456653259), (10, 2, 3, 9.421856492003737), (11, 2, 3, 9.75924005523155), (12, 2, 3, 11.694962176840404), (13, 2, 3, 11.357062829698963), (14, 2, 3, 13.150317977607731), (15, 2, 3, 11.88016958288265), (16, 2, 3, 13.597651814764738), (17, 2, 3, 14.184934646092023), (18, 2, 3, 16.267916204165356), (19, 2, 3, 17.336605361833524), (20, 2, 3, 16.710640450856296)]

# q = 4, g = 3, l = [1,2,3,4,5,6,7,8,9,10]
fourQ_threeG= [(1, 4, 3, 1.1836489427456423), (2, 4, 3, 1.3624049124275992), (3, 4, 3, 1.7315430795542082), (4, 4, 3, 2.371395867254742), (5, 4, 3, 2.561128199093816), (6, 4, 3, 3.2193956584675707), (7, 4, 3, 2.869838176284573), (8, 4, 3, 3.6540822054632667), (9, 4, 3, 3.9578410371561397), (10, 4, 3, 3.7212311229688693)]

# q = 8, g = 3, l =[1,2,3,4,5]
eightQ_threeG = [(1, 8, 3, 0.7682947642956021), (2, 8, 3, 0.5068252216608563), (3, 8, 3, 0.6432356774512358), (4, 8, 3, 0.46605237711884384), (5, 8, 3, 0.5931924859826063)]

# q = 10, g = 3, l = [1,2,3,4]
tenQ_threeG = [(1, 10, 3, 0.6015859484660139), (2, 10, 3, 0.31655758281558377), (3, 10, 3, 0.3018106946248048), (4, 10, 3, 0.262804728536154)]

"""3 Gates NO ENTANGLEMENT"""
no_twoQ_threeG = [(1, 2, 3, 0.9902355133415315), (2, 2, 3, 1.4818437030837714), (3, 2, 3, 2.153426077099075), (4, 2, 3, 2.8551045049470862), (5, 2, 3, 3.223809002946435), (6, 2, 3, 3.8730663851775966), (7, 2, 3, 4.5329139332533845), (8, 2, 3, 4.958577680421357), (9, 2, 3, 5.443345644449277), (10, 2, 3, 5.946322380201764), (11, 2, 3, 7.091564604608551), (12, 2, 3, 6.868824899442566), (13, 2, 3, 7.373758022139593), (14, 2, 3, 8.237350838681651), (15, 2, 3, 8.968637669240085), (16, 2, 3, 9.596989818977747), (17, 2, 3, 10.025201102425342), (18, 2, 3, 10.475304985201685), (19, 2, 3, 11.088152551149335), (20, 2, 3, 11.442693900868045)]

no_fourQ_threeG = [(1, 4, 3, 0.49806386817873116), (2, 4, 3, 0.7444517789590265), (3, 4, 3, 1.1045676552885146), (4, 4, 3, 1.4077797951282935), (5, 4, 3, 1.7289122060684776), (6, 4, 3, 1.9702909790517709), (7, 4, 3, 2.222742359965731), (8, 4, 3, 2.57487787250329), (9, 4, 3, 2.821396209557449), (10, 4, 3, 3.0824107749363328)]

no_eightQ_threeG = [(1, 8, 3, 0.24913336241455764), (2, 8, 3, 0.3731099860619447), (3, 8, 3, 0.5813920938301481), (4, 8, 3, 0.732694961298253), (5, 8, 3, 0.8534865904199935)]

no_tenQ_threeG = [(1, 10, 3, 0.19979157602750372), (2, 10, 3, 0.2985465788024339), (3, 10, 3, 0.47265486585354294), (4, 10, 3, 0.5806415854686175)]

entanglement_data = twoQ_threeG
no_entanglement_data = no_twoQ_threeG
n_qubits = 2

# Extract data for plotting
n_layers_ent = [(item[0] * item[1] * item[2]) for item in entanglement_data]
tightness_ratios_ent = [item[3]*100 / (item[0] * item[1] * item[2]) for item in entanglement_data]

n_layers_no_ent = [(item[0] * item[1] * item[2]) for item in no_entanglement_data]
tightness_ratios_no_ent = [item[3]*100 / (item[0] * item[1] * item[2]) for item in no_entanglement_data]

# Create the comparison plot
plt.figure(figsize=(12, 7))
plt.plot(n_layers_ent, tightness_ratios_ent, color='blue', linewidth=3,
         marker='o', markersize=6, label='With Entanglement')
plt.plot(n_layers_no_ent, tightness_ratios_no_ent, color='red', linewidth=3,
         marker='s', markersize=6, label='No Entanglement')

# Add titles and labels
plt.xlabel('Number of Parameters', fontsize=12)
plt.ylabel('Measured Norm / Theoretical Bound (%)', fontsize=12)
plt.title(f'Bound Tightness Comparison: {n_qubits}-Qubit QNN with 3 Gates', fontsize=14)
plt.legend(fontsize=11, loc='best')
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.gca().set_facecolor("#f5f5f5")

# Set y-axis limit
max_y = max(max(tightness_ratios_ent), max(tightness_ratios_no_ent))
plt.ylim(0, max_y + 5)

plt.tight_layout()
plt.show()





# active = no_fourQ_threeG
#
# # Extract the number of layers for the x-axis labels
# n_layers_list = [(item[0] * item[1] * item[2]) for item in active]
#
# # Ratio = max_hessian_norm / (n_layers * n_qubits * n_gates)
# tightness_ratios = [item[3]*100 / (item[0] * item[1] * item[2]) for item in active]
#
# # Create the bar chart
# plt.figure(figsize=(10, 6))
# plt.plot(n_layers_list, tightness_ratios, color='blue', linewidth=4)
#
# # Add titles and labels for clarity
# plt.xlabel('Number of Parameters')
# plt.ylabel('Measured Norm / Theoretical Bound')
# plt.title('Bound Tightness vs. Number of Layers for 4-Qubit QNN')
# plt.xticks(n_layers_list) # Ensure all layer numbers are shown as ticks
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.ylim(0, max(tightness_ratios) + 10)
# plt.gca().set_facecolor("#e2dddd")
# plt.show()