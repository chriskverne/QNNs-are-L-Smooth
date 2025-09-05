import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


from sympy.printing.pretty.pretty_symbology import line_width

# ==================================================================
# 1. DATA
# ==================================================================
# Structure: {n_qubits: [(params_1, L_max), (params_2, L_max), ...]}
# Strucutre [(Norm M, L_max, L_bound), (Norm M2, L_max2, L_bound) ...]
data_with_entanglement = {
    2: [(1, 2, 3, 1.9182122556740817), (2, 2, 3, 2.4071741737372507), (3, 2, 3, 3.5236568840002587), (4, 2, 3, 4.811272362899584), (5, 2, 3, 4.808879133191292), (6, 2, 3, 6.080643970822507), (7, 2, 3, 7.861919779111797), (8, 2, 3, 7.275262304341641), (9, 2, 3, 8.493986456653259), (10, 2, 3, 9.421856492003737), (11, 2, 3, 9.75924005523155), (12, 2, 3, 11.694962176840404), (13, 2, 3, 11.357062829698963), (14, 2, 3, 13.150317977607731), (15, 2, 3, 11.88016958288265), (16, 2, 3, 13.597651814764738), (17, 2, 3, 14.184934646092023), (18, 2, 3, 16.267916204165356), (19, 2, 3, 17.336605361833524), (20, 2, 3, 16.710640450856296)],
    4: [(1, 4, 3, 1.1836489427456423), (2, 4, 3, 1.3624049124275992), (3, 4, 3, 1.7315430795542082),(4, 4, 3, 2.371395867254742), (5, 4, 3, 2.561128199093816), (6, 4, 3, 3.2193956584675707),(7, 4, 3, 2.869838176284573), (8, 4, 3, 3.6540822054632667), (9, 4, 3, 3.9578410371561397),(10, 4, 3, 3.7212311229688693)],
    8: [(1, 8, 3, 0.7682947642956021), (2, 8, 3, 0.5068252216608563), (3, 8, 3, 0.6432356774512358), (4, 8, 3, 0.46605237711884384), (5, 8, 3, 0.5931924859826063)],
    10: [(1, 10, 3, 0.6015859484660139), (2, 10, 3, 0.31655758281558377), (3, 10, 3, 0.3018106946248048), (4, 10, 3, 0.262804728536154)]
}

data_no_entanglement = {
    1: [(1, 1, 3, 1.965525726370158), (5, 1, 3, 6.436282116066916), (10, 1, 3, 11.776384364142976),(15, 1, 3, 16.783976234411604), (20, 1, 3, 22.177215728608942), (25, 1, 3, 27.513140120674816),(30, 1, 3, 32.29561382430927), (35, 1, 3, 39.08914586864514), (40, 1, 3, 44.63155788758197)],
    2: [(1, 2, 3, 0.9902355133415315), (2, 2, 3, 1.4818437030837714), (3, 2, 3, 2.153426077099075),(4, 2, 3, 2.8551045049470862), (5, 2, 3, 3.223809002946435), (6, 2, 3, 3.8730663851775966),(7, 2, 3, 4.5329139332533845), (8, 2, 3, 4.958577680421357), (9, 2, 3, 5.443345644449277),(10, 2, 3, 5.946322380201764), (11, 2, 3, 7.091564604608551), (12, 2, 3, 6.868824899442566),(13, 2, 3, 7.373758022139593), (14, 2, 3, 8.237350838681651), (15, 2, 3, 8.968637669240085),(16, 2, 3, 9.596989818977747), (17, 2, 3, 10.025201102425342), (18, 2, 3, 10.475304985201685),(19, 2, 3, 11.088152551149335), (20, 2, 3, 11.442693900868045)],
    4: [(1, 4, 3, 0.49806386817873116), (2, 4, 3, 0.7444517789590265), (3, 4, 3, 1.1045676552885146),(4, 4, 3, 1.4077797951282935), (5, 4, 3, 1.7289122060684776), (6, 4, 3, 1.9702909790517709),(7, 4, 3, 2.222742359965731), (8, 4, 3, 2.57487787250329), (9, 4, 3, 2.821396209557449),(10, 4, 3, 3.0824107749363328)],
    8: [(1, 8, 3, 0.24913336241455764), (2, 8, 3, 0.3731099860619447), (3, 8, 3, 0.5813920938301481),(4, 8, 3, 0.732694961298253), (5, 8, 3, 0.8534865904199935)],
    10: [(1, 10, 3, 0.19979157602750372), (2, 10, 3, 0.2985465788024339), (3, 10, 3, 0.47265486585354294), (4, 10, 3, 0.5806415854686175)]
}


# NORM SCALING DATA 2L, 2,4,8Q 3G W in [0.1, 5]
data_M_scaling = {
    2: [(1.0198039027185564, 2.809453408072925, 12.237646832622676), (1.6313290801984848, 3.818025887991097, 19.575948962381815), (2.579501339502237, 5.124609239641493, 30.954016074026846), (3.60801576739595, 6.503050008736427, 43.2961892087514), (4.664020413736746, 7.937873946163769, 55.96824496484095), (5.732342722344832, 9.389242209358097, 68.78811266813798), (6.8071857457967, 10.854920093546278, 81.6862289495604), (7.885883621625874, 12.332948126673504, 94.6306034595105), (8.96704529009238, 13.821600487568363, 107.60454348110855), (10.049875621120892, 15.319370398839641, 120.59850745345071)],
    4: [(1.0198, 3.1412, 24.4753), (1.6313, 4.4154, 39.1519), (2.5795, 5.7150, 61.9080), (3.6080, 7.0263, 86.5924), (4.6640, 8.3969, 111.9365), (5.7323, 9.8629, 137.5762), (6.8072, 11.3506, 163.3725), (7.8859, 12.8675, 189.2612), (8.9670, 14.3893, 215.2091), (10.0499, 15.9149, 241.1970)],
    8: [(1.0198039027185564, 2.837724799451568, 48.950587330490706), (1.6313290801984848, 3.8067383397450416, 78.30379584952726), (2.579501339502237, 5.166990862510972, 123.81606429610738), (3.60801576739595, 6.5821347488822015, 173.1847568350056), (4.664020413736746, 8.023954413404837, 223.8729798593638), (5.732342722344832, 9.488314808160979, 275.15245067255194), (6.8071857457967, 10.971713319421015, 326.7449157982416), (7.885883621625874, 12.471127979286742, 378.522413838042), (8.96704529009238, 13.983974359395393, 430.4181739244342), (10.049875621120892, 15.508071523687942, 482.39402981380283)]
}

# GENERATOR SCALING 2L 2,4,8Q 3G Gk = 0.5 or 2. Ratio = [0, 25, 50, 75, 100]
data_Gk_scaling = {
    2: [(0, 1.4613, 12.00, 12.00), (25, 9.5675, 57.00, 192.00), (50, 9.5675, 102.00, 192.00), (75, 23.6868, 147.00, 192.00), (100, 23.6868, 192.00, 192.00)],
    4: [(0, 0.7464, 24.00, 24.00),(25, 4.7764, 114.00, 384.00),(50, 8.0193, 204.00, 384.00),(75, 11.8971, 294.00, 384.00),(100, 11.8971, 384.00, 384.00)],
    8: [(0, 0.3715, 48.00, 48.00), (25, 5.8104, 228.00, 768.00), (50, 5.8104, 408.00, 768.00), (75, 5.9469, 588.00, 768.00), (100, 5.9816, 768.00, 768.00)]
}

# KL vs Parameter Scaling
# 2,4,8 Qubits , Entangelemnt, 3 Gates [1-10 Layers]
[{'layers': 1, 'P': 6, 'L_max_ratio': 0.3318576374824204, 'D_KL': 0.05440392634943307}, {'layers': 2, 'P': 12, 'L_max_ratio': 0.2139310344182754, 'D_KL': 0.021502161291199706}, {'layers': 3, 'P': 18, 'L_max_ratio': 0.2060053026190467, 'D_KL': 0.013409746400984131}, {'layers': 4, 'P': 24, 'L_max_ratio': 0.19748358164679916, 'D_KL': 0.018522306085856397}, {'layers': 5, 'P': 30, 'L_max_ratio': 0.21597246148738694, 'D_KL': 0.019334704267582138}, {'layers': 6, 'P': 36, 'L_max_ratio': 0.1903739996457256, 'D_KL': 0.019669463078819466}, {'layers': 7, 'P': 42, 'L_max_ratio': 0.18844286699178733, 'D_KL': 0.022385657166621265}, {'layers': 8, 'P': 48, 'L_max_ratio': 0.17782857037372382, 'D_KL': 0.015387487759360413}, {'layers': 9, 'P': 54, 'L_max_ratio': 0.18390644177901233, 'D_KL': 0.01431003557335625}, {'layers': 10, 'P': 60, 'L_max_ratio': 0.1758583634701901, 'D_KL': 0.019777026410283175}]
[{'layers': 1, 'P': 12, 'L_max_ratio': 0.1637990427601808, 'D_KL': 0.2338772466150277}, {'layers': 2, 'P': 24, 'L_max_ratio': 0.12098582775582319, 'D_KL': 0.02135498594226106}, {'layers': 3, 'P': 36, 'L_max_ratio': 0.08482524477416811, 'D_KL': 0.011179034709099913}, {'layers': 4, 'P': 48, 'L_max_ratio': 0.08342671107985349, 'D_KL': 0.009338308403378431}, {'layers': 5, 'P': 60, 'L_max_ratio': 0.061855452689485405, 'D_KL': 0.004920165050547805}, {'layers': 6, 'P': 72, 'L_max_ratio': 0.05827594153186555, 'D_KL': 0.006612911155774292}, {'layers': 7, 'P': 84, 'L_max_ratio': 0.05705149794073222, 'D_KL': 0.005822226327926969}, {'layers': 8, 'P': 96, 'L_max_ratio': 0.05299504397811419, 'D_KL': 0.008144022005582448}, {'layers': 9, 'P': 108, 'L_max_ratio': 0.06801924170547302, 'D_KL': 0.008122507761685365}, {'layers': 10, 'P': 120, 'L_max_ratio': 0.05822983034661855, 'D_KL': 0.005908770055948315}]
[{'layers': 1, 'P': 24, 'L_max_ratio': 0.08255271408457514, 'D_KL': 0.1765455907601364}, {'layers': 2, 'P': 48, 'L_max_ratio': 0.057895523236510045, 'D_KL': 0.060746621994808105}, {'layers': 3, 'P': 72, 'L_max_ratio': 0.05120352937959477, 'D_KL': 0.004552686304385713}, {'layers': 4, 'P': 96, 'L_max_ratio': 0.03524512995691607, 'D_KL': 0.0017037048881130452}, {'layers': 5, 'P': 120, 'L_max_ratio': 0.02850950701371178, 'D_KL': 0.0005379930594233187}]


# ==================================================================
# 2. PLOTTING LOGIC P SCALING
# ==================================================================
def plot_P():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('gainsboro')

    colors = {1: 'orange', 2: 'orangered', 4: 'royalblue', 8: 'forestgreen', 10: 'purple'}
    markers = {1: 'o', 2: 'o', 4: 'o', 8: 'o', 10: 'o'}

    # Plot data... (omitted for brevity, same as before)
    # Plot data WITH entanglement (solid lines)
    for n_qubits, data_points in data_with_entanglement.items():
        params = [p[0] * p[1] * p[2] for p in data_points]
        ratios = [(p[3] / (p[0] * p[1] * p[2])) * 100 for p in data_points]
        ax.plot(params, ratios,
                marker=markers.get(n_qubits),
                linestyle='-',
                color=colors.get(n_qubits),
                linewidth=3)

    # Plot data NO entanglement (dashed lines)
    for n_qubits, data_points in data_no_entanglement.items():
        params = [p[0] * p[1] * p[2] for p in data_points]
        ratios = [(p[3] / (p[0] * p[1] * p[2])) * 100 for p in data_points]
        ax.plot(params, ratios,
                linestyle='--',
                color=colors.get(n_qubits),
                linewidth=3)

    # ==================================================================
    # 3. STYLING THE PLOT
    # ==================================================================
    ax.set_xlabel('Number of Parameters (P)', fontsize=14)
    ax.set_ylabel('$L_{max}$ / $L_{upper}$ (%)', fontsize=16)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=2.0, alpha=1)
    ax.grid(True, which='minor', linestyle='-', linewidth=1, alpha=0.4)


    # --- Create Legend Handles ---
    handles_qubits = [
        Line2D([0], [0], marker='s', color='orange', markersize=8, lw=0, label='n=1'),
        Line2D([0], [0], marker='s', color='orangered', markersize=8, lw=0, label='n=2'),
        Line2D([0], [0], marker='s', color='royalblue', markersize=8, lw=0, label='n=4'),
        Line2D([0], [0], marker='s', color='forestgreen', markersize=8, lw=0, label='n=8'),
        Line2D([0], [0], marker='s', color='purple', markersize=8, lw=0, label='n=10'),
    ]
    handles_type = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='With Entanglement'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='No Entanglement')
    ]

    # --- Create Two Separate, Styled, and Aligned Legends ---

    # 1. Create the first legend for the qubits and save it to a variable.
    leg1 = ax.legend(handles=handles_qubits,
                     title='VQA Architecture',
                     bbox_to_anchor=(0.98, 0.98), loc='upper right', ncol=5,
                     fancybox=True, facecolor='white', edgecolor='black',
                     fontsize=14, framealpha=1.0, columnspacing=1.0,
                     handletextpad=0.5, title_fontsize=14)
    # Manually add the first legend so it isn't overwritten by the second call.
    ax.add_artist(leg1)

    # 2. Create the second legend for the entanglement types.
    #    Adjusted the vertical position from 0.88 to 0.85 to add spacing.
    ax.legend(handles=handles_type,
              bbox_to_anchor=(0.98, 0.85), loc='upper right', ncol=2, # KEY CHANGE HERE
              fancybox=True,
              edgecolor='black',
              facecolor='white',
              framealpha=1.0,
              fontsize=14,
              columnspacing=1.0,
              handletextpad=0.5)

    ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.show()


def plot_M():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('gainsboro')

    # Extract data for element 2
    m_norms_2 = [point[0] for point in data_M_scaling[2]]
    ratios_2 = [100 * point[1] / point[2] for point in data_M_scaling[2]]  # L_max / L_upper as percentage

    # Extract data for element 4
    m_norms_4 = [point[0] for point in data_M_scaling[4]]
    ratios_4 = [100 * point[1] / point[2] for point in data_M_scaling[4]]  # L_max / L_upper as percentage

    # Extract data for element 8
    m_norms_8 = [point[0] for point in data_M_scaling[8]]
    ratios_8 = [100 * point[1] / point[2] for point in data_M_scaling[8]]  # L_max / L_upper as percentage

    ax.plot(m_norms_2, ratios_2, color='red', linewidth=4, markersize=6, label='n=2')
    ax.plot(m_norms_4, ratios_4, 'o-', color='blue', linewidth=4, markersize=8, label='n=4')
    ax.plot(m_norms_8, ratios_8, 's-', color='green', linewidth=4, markersize=8, label='n=8')
    ax.set_xlabel('$||M||_2$ Norm', fontsize=14)
    ax.set_ylabel('$L_{max}$ / $L_{upper}$ (%)', fontsize=16)
    ax.set_ylim(0, 100)  # Changed from (0, 1) to (0, 100)
    ax.legend(title='VQA Architecture',
              ncol=3,
              fancybox=True,
              edgecolor='black',
              facecolor='white',
              framealpha=1.0,
              fontsize=14,
              title_fontsize=14)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=2.0, alpha=1)
    ax.grid(True, which='minor', linestyle='-', linewidth=1, alpha=0.4)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.show()


def plot_G():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data from data_Gk_scaling
    for n_qubits, data_points in data_Gk_scaling.items():
        ratios = [point[0] for point in data_points]
        l_max = [point[1] for point in data_points]
        l_bound1 = [point[2] for point in data_points]
        l_bound2 = [point[3] for point in data_points]

        # Calculate the normalized values
        norm1 = [l_max[i] / l_bound1[i] for i in range(len(ratios))]
        norm2 = [l_max[i] / l_bound2[i] for i in range(len(ratios))]

        # Set up bar positions
        x = np.arange(len(ratios))
        width = 0.35

        # Create bars
        bars1 = ax.bar(x - width / 2, norm1, width, label=f'L_max / Our Bound (n={n_qubits})', alpha=0.8)
        bars2 = ax.bar(x + width / 2, norm2, width, label=f'L_max / Liu Bound (n={n_qubits})', alpha=0.8)
        # ax.plot(x - width / 2, norm1, width, label=f'L_max / Our Bound (n={n_qubits})', alpha=0.8)
        # ax.plot(x + width / 2, norm2, width, label=f'L_max / Liu Bound (n={n_qubits})', alpha=0.8)

        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels([f'{r}%' for r in ratios])

    ax.set_xlabel('Ratio (%)')
    ax.set_ylabel('L_max / L_bound')
    # ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.show()


# plot_P()
# plot_M()
# plot_G()