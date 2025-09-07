# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
#
# def analyze_learning_rate_heuristic(data, n_qubits, calibration_depth, target_depths):
#     print(f"\n--- Analysis for {n_qubits}-Qubit Circuit ---")
#     print(f"Calibrating kappa(n) using a shallow {calibration_depth}-layer circuit...")
#
#     # Find the data entry for the specified number of qubits
#     circuit_data = {entry[0]: entry for entry in data.get(n_qubits, [])}
#     if not circuit_data:
#         print(f"No data found for {n_qubits} qubits.")
#         return None
#
#     # --- Step 1: Calibrate kappa(n) using the shallow circuit ---
#     cal_layers, _, cal_gates, l_max_cal = circuit_data[calibration_depth]
#
#     # The upper bound L_upper is the total number of parameters P.
#     # P = n_layers * n_qubits * n_gates_per_layer
#     l_upper_cal = cal_layers * n_qubits * cal_gates
#
#     # Estimate the scaling factor kappa(n)
#     kappa_est = l_max_cal / l_upper_cal
#     print(f"  > Measured L_max at {cal_layers} layers: {l_max_cal:.4f}")
#     print(f"  > Calculated L_upper: {l_upper_cal}")
#     print(f"  > Estimated kappa({n_qubits}) = {kappa_est:.4f}")
#     print("-" * 35)
#
#     # --- Step 2: Predict and Compare for deeper target circuits ---
#     results = []
#     for target_l in target_depths:
#         if target_l not in circuit_data:
#             print(f"Target depth {target_l} not in data, skipping.")
#             continue
#
#         # Get the ground truth values for the target circuit
#         _, _, target_gates, l_max_true = circuit_data[target_l]
#         eta_ideal = 1.0 / l_max_true
#
#         # Predict the L_max and learning rate using the calibrated kappa
#         l_upper_target = target_l * n_qubits * target_gates
#         l_max_pred = kappa_est * l_upper_target
#         eta_pred = 1.0 / l_max_pred
#
#         # Calculate the error
#         error_percent = abs(eta_pred - eta_ideal) / eta_ideal * 100
#
#         results.append({
#             "Target Layers": target_l,
#             "Ideal η (1/L_max_true)": eta_ideal,
#             "Predicted η (1/L_max_pred)": eta_pred,
#             "Error (%)": error_percent,
#         })
#
#     return pd.DataFrame(results)
#
#
# def analyze_and_plot_heuristic(data, n_qubits, calibration_depth, target_depths, ax):
#     circuit_data = {entry[0]: entry for entry in data.get(n_qubits, [])}
#     if not circuit_data:
#         return 0
#
#     # Calibrate kappa(n)
#     cal_layers, _, cal_gates, l_max_cal = circuit_data[calibration_depth]
#     l_upper_cal = cal_layers * n_qubits * cal_gates
#     kappa_est = l_max_cal / l_upper_cal
#
#     results = []
#     for target_l in target_depths:
#         if target_l not in circuit_data:
#             continue
#
#         _, _, target_gates, l_max_true = circuit_data[target_l]
#         eta_ideal = 1.0 / l_max_true
#
#         l_upper_target = target_l * n_qubits * target_gates
#         l_max_pred = kappa_est * l_upper_target
#         eta_pred = 1.0 / l_max_pred
#
#         error_percent = abs(eta_pred - eta_ideal) / eta_ideal * 100
#
#         results.append({
#             "Target Layers": target_l,
#             "Ideal η": eta_ideal,
#             "Predicted η": eta_pred,
#             "Error (%)": error_percent,
#         })
#
#     df = pd.DataFrame(results)
#     avg_error = df['Error (%)'].mean()
#
#     # Plotting
#     sns.lineplot(data=df, x='Target Layers', y='Ideal η', marker='o', ax=ax, label='Ideal η (Ground Truth)')
#     sns.lineplot(data=df, x='Target Layers', y='Predicted η', marker='x', linestyle='--', ax=ax,
#                  label='Predicted η (Heuristic)')
#
#     ax.set_title(f'{n_qubits}-Qubit Circuit (Avg. Error: {avg_error:.1f}%)')
#     ax.set_xlabel('Target Circuit Depth (Layers)')
#     ax.set_ylabel('Learning Rate (η)')
#     ax.legend()
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#
#     # Add annotation for calibration depth
#     ax.text(0.95, 0.95, f'Calibrated at L={calibration_depth}',
#             transform=ax.transAxes, fontsize=9,
#             verticalalignment='top', horizontalalignment='right',
#             bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', ec='grey', lw=0.5))
#
#     return avg_error
#
#
# if __name__ == "__main__":
#     # The experimental data from the paper's authors
#     data_with_entanglement = {
#         2: [(1, 2, 3, 1.9182122556740817), (2, 2, 3, 2.4071741737372507), (3, 2, 3, 3.5236568840002587),
#             (4, 2, 3, 4.811272362899584), (5, 2, 3, 4.808879133191292), (6, 2, 3, 6.080643970822507),
#             (7, 2, 3, 7.861919779111797), (8, 2, 3, 7.275262304341641), (9, 2, 3, 8.493986456653259),
#             (10, 2, 3, 9.421856492003737), (11, 2, 3, 9.75924005523155), (12, 2, 3, 11.694962176840404),
#             (13, 2, 3, 11.357062829698963), (14, 2, 3, 13.150317977607731), (15, 2, 3, 11.88016958288265),
#             (16, 2, 3, 13.597651814764738), (17, 2, 3, 14.184934646092023), (18, 2, 3, 16.267916204165356),
#             (19, 2, 3, 17.336605361833524), (20, 2, 3, 16.710640450856296)],
#         4: [(1, 4, 3, 1.1836489427456423), (2, 4, 3, 1.3624049124275992), (3, 4, 3, 1.7315430795542082),
#             (4, 4, 3, 2.371395867254742), (5, 4, 3, 2.561128199093816), (6, 4, 3, 3.2193956584675707),
#             (7, 4, 3, 2.869838176284573), (8, 4, 3, 3.6540822054632667), (9, 4, 3, 3.9578410371561397),
#             (10, 4, 3, 3.7212311229688693)],
#         8: [(1, 8, 3, 0.7682947642956021), (2, 8, 3, 0.5068252216608563), (3, 8, 3, 0.6432356774512358),
#             (4, 8, 3, 0.46605237711884384), (5, 8, 3, 0.5931924859826063)],
#         10: [(1, 10, 3, 0.6015859484660139), (2, 10, 3, 0.31655758281558377), (3, 10, 3, 0.3018106946248048),
#              (4, 10, 3, 0.262804728536154)]
#     }
#
#     data_no_entanglement = {
#         1: [(1, 1, 3, 1.965525726370158), (5, 1, 3, 6.436282116066916), (10, 1, 3, 11.776384364142976),
#             (15, 1, 3, 16.783976234411604), (20, 1, 3, 22.177215728608942), (25, 1, 3, 27.513140120674816),
#             (30, 1, 3, 32.29561382430927), (35, 1, 3, 39.08914586864514), (40, 1, 3, 44.63155788758197)],
#         2: [(1, 2, 3, 0.9902355133415315), (2, 2, 3, 1.4818437030837714), (3, 2, 3, 2.153426077099075),
#             (4, 2, 3, 2.8551045049470862), (5, 2, 3, 3.223809002946435), (6, 2, 3, 3.8730663851775966),
#             (7, 2, 3, 4.5329139332533845), (8, 2, 3, 4.958577680421357), (9, 2, 3, 5.443345644449277),
#             (10, 2, 3, 5.946322380201764), (11, 2, 3, 7.091564604608551), (12, 2, 3, 6.868824899442566),
#             (13, 2, 3, 7.373758022139593), (14, 2, 3, 8.237350838681651), (15, 2, 3, 8.968637669240085),
#             (16, 2, 3, 9.596989818977747), (17, 2, 3, 10.025201102425342), (18, 2, 3, 10.475304985201685),
#             (19, 2, 3, 11.088152551149335), (20, 2, 3, 11.442693900868045)],
#         4: [(1, 4, 3, 0.49806386817873116), (2, 4, 3, 0.7444517789590265), (3, 4, 3, 1.1045676552885146),
#             (4, 4, 3, 1.4077797951282935), (5, 4, 3, 1.7289122060684776), (6, 4, 3, 1.9702909790517709),
#             (7, 4, 3, 2.222742359965731), (8, 4, 3, 2.57487787250329), (9, 4, 3, 2.821396209557449),
#             (10, 4, 3, 3.0824107749363328)],
#         8: [(1, 8, 3, 0.24913336241455764), (2, 8, 3, 0.3731099860619447), (3, 8, 3, 0.5813920938301481),
#             (4, 8, 3, 0.732694961298253), (5, 8, 3, 0.8534865904199935)],
#         10: [(1, 10, 3, 0.19979157602750372), (2, 10, 3, 0.2985465788024339), (3, 10, 3, 0.47265486585354294),
#              (4, 10, 3, 0.5806415854686175)]
#     }
#
#     # # --- Run Case Study 1: 4-Qubit Circuit ---
#     # # We calibrate on a 5-layer circuit and predict for a 10-layer one.
#     # results_4_qubit = analyze_learning_rate_heuristic(
#     #     data=data_no_entanglement,
#     #     n_qubits=4,
#     #     calibration_depth=5,
#     #     target_depths=[6,7,8,9,10]
#     # )
#     # print(results_4_qubit.to_string(index=False))
#     #
#     # # --- Run Case Study 2: 2-Qubit Circuit ---
#     # # We calibrate on a 5-layer circuit and predict for deeper 10, 15, and 20 layer circuits.
#     # results_2_qubit = analyze_learning_rate_heuristic(
#     #     data=data_no_entanglement,
#     #     n_qubits=2,
#     #     calibration_depth=5,
#     #     target_depths=[10, 15, 20]
#     # )
#     # print(results_2_qubit.to_string(index=False))
#     #
#     # results_1_qubit = analyze_learning_rate_heuristic(
#     #     data=data_no_entanglement,
#     #     n_qubits=1,
#     #     calibration_depth=10,
#     #     target_depths=[15, 20,25,30,35,40]
#     # )
#     # print(results_1_qubit.to_string(index=False))
#
#     # Create the figure
#     sns.set_theme(style="whitegrid")
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     #fig.suptitle('Predictive Power of the κ(n) Heuristic for Setting Learning Rates', fontsize=16, y=1.02)
#
#     # 4-Qubit Analysis
#     analyze_and_plot_heuristic(data_no_entanglement, 4, 5, [6, 7, 8, 9, 10], axes[0])
#
#     # 2-Qubit Analysis
#     analyze_and_plot_heuristic(data_no_entanglement, 2, 5, [10, 15, 20], axes[1])
#
#     # 1-Qubit Analysis
#     analyze_and_plot_heuristic(data_no_entanglement, 1, 10, [15, 20, 25, 30, 35, 40], axes[2])
#
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()
#


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def analyze_and_plot_heuristic(data, n_qubits, calibration_depth, target_depths, ax):
    circuit_data = {entry[0]: entry for entry in data.get(n_qubits, [])}
    if not circuit_data:
        return 0

    # Calibrate kappa(n)
    cal_layers, _, cal_gates, l_max_cal = circuit_data[calibration_depth]
    l_upper_cal = cal_layers * n_qubits * cal_gates
    kappa_est = l_max_cal / l_upper_cal

    results = []
    for target_l in target_depths:
        if target_l not in circuit_data:
            continue

        _, _, target_gates, l_max_true = circuit_data[target_l]
        eta_ideal = 1.0 / l_max_true

        l_upper_target = target_l * n_qubits * target_gates
        l_max_pred = kappa_est * l_upper_target
        eta_pred = 1.0 / l_max_pred

        error_percent = abs(eta_pred - eta_ideal) / eta_ideal * 100

        results.append({
            "Target Layers": target_l,
            "Ideal η": eta_ideal,
            "Predicted η": eta_pred,
            "Error (%)": error_percent,
        })

    df = pd.DataFrame(results)
    avg_error = df['Error (%)'].mean()

    # Plotting
    sns.lineplot(data=df, x='Target Layers', y='Ideal η', marker='o', ax=ax, label='Ideal η (Ground Truth)', linewidth=3, markersize=8)
    ax.tick_params(axis='both', which='major', labelsize=12)
    sns.lineplot(data=df, x='Target Layers', y='Predicted η', linestyle='--', ax=ax,
                 label='Predicted η (Heuristic)', linewidth=3)

    ax.set_title(f'{n_qubits}-Qubit Circuit (Avg. Error: {avg_error:.1f}%)', fontsize=14)
    ax.set_xlabel('Target Circuit Depth (Layers)', fontsize=14)
    ax.set_ylabel('')  # Remove individual y-axis labels
    ax.legend(framealpha=1.0,fontsize=12)
    ax.set_facecolor('gainsboro')
    ax.grid(True, which='major', linestyle='-', linewidth=2.0, alpha=1)
    ax.grid(True, which='minor', linestyle='-', linewidth=1, alpha=0.4)

    # --- STYLING CHANGES ---
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # 2. Make annotation text larger
    ax.text(0.95, 0.95, f'Calibrated at L={calibration_depth}',
            transform=ax.transAxes, fontsize=12,  # Increased font size
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', ec='grey', lw=0.5))

    return avg_error


if __name__ == "__main__":
    data_with_entanglement = {
            2: [(1, 2, 3, 1.9182122556740817), (2, 2, 3, 2.4071741737372507), (3, 2, 3, 3.5236568840002587),
                (4, 2, 3, 4.811272362899584), (5, 2, 3, 4.808879133191292), (6, 2, 3, 6.080643970822507),
                (7, 2, 3, 7.861919779111797), (8, 2, 3, 7.275262304341641), (9, 2, 3, 8.493986456653259),
                (10, 2, 3, 9.421856492003737), (11, 2, 3, 9.75924005523155), (12, 2, 3, 11.694962176840404),
                (13, 2, 3, 11.357062829698963), (14, 2, 3, 13.150317977607731), (15, 2, 3, 11.88016958288265),
                (16, 2, 3, 13.597651814764738), (17, 2, 3, 14.184934646092023), (18, 2, 3, 16.267916204165356),
                (19, 2, 3, 17.336605361833524), (20, 2, 3, 16.710640450856296)],
            4: [(1, 4, 3, 1.1836489427456423), (2, 4, 3, 1.3624049124275992), (3, 4, 3, 1.7315430795542082),
                (4, 4, 3, 2.371395867254742), (5, 4, 3, 2.561128199093816), (6, 4, 3, 3.2193956584675707),
                (7, 4, 3, 2.869838176284573), (8, 4, 3, 3.6540822054632667), (9, 4, 3, 3.9578410371561397),
                (10, 4, 3, 3.7212311229688693)],
            8: [(1, 8, 3, 0.7682947642956021), (2, 8, 3, 0.5068252216608563), (3, 8, 3, 0.6432356774512358),
                (4, 8, 3, 0.46605237711884384), (5, 8, 3, 0.5931924859826063)],
            10: [(1, 10, 3, 0.6015859484660139), (2, 10, 3, 0.31655758281558377), (3, 10, 3, 0.3018106946248048),
                 (4, 10, 3, 0.262804728536154)]
    }

    data_no_entanglement = {
            1: [(1, 1, 3, 1.965525726370158), (5, 1, 3, 6.436282116066916), (10, 1, 3, 11.776384364142976),
                (15, 1, 3, 16.783976234411604), (20, 1, 3, 22.177215728608942), (25, 1, 3, 27.513140120674816),
                (30, 1, 3, 32.29561382430927), (35, 1, 3, 39.08914586864514), (40, 1, 3, 44.63155788758197)],
            2: [(1, 2, 3, 0.9902355133415315), (2, 2, 3, 1.4818437030837714), (3, 2, 3, 2.153426077099075),
                (4, 2, 3, 2.8551045049470862), (5, 2, 3, 3.223809002946435), (6, 2, 3, 3.8730663851775966),
                (7, 2, 3, 4.5329139332533845), (8, 2, 3, 4.958577680421357), (9, 2, 3, 5.443345644449277),
                (10, 2, 3, 5.946322380201764), (11, 2, 3, 7.091564604608551), (12, 2, 3, 6.868824899442566),
                (13, 2, 3, 7.373758022139593), (14, 2, 3, 8.237350838681651), (15, 2, 3, 8.968637669240085),
                (16, 2, 3, 9.596989818977747), (17, 2, 3, 10.025201102425342), (18, 2, 3, 10.475304985201685),
                (19, 2, 3, 11.088152551149335), (20, 2, 3, 11.442693900868045)],
            4: [(1, 4, 3, 0.49806386817873116), (2, 4, 3, 0.7444517789590265), (3, 4, 3, 1.1045676552885146),
                (4, 4, 3, 1.4077797951282935), (5, 4, 3, 1.7289122060684776), (6, 4, 3, 1.9702909790517709),
                (7, 4, 3, 2.222742359965731), (8, 4, 3, 2.57487787250329), (9, 4, 3, 2.821396209557449),
                (10, 4, 3, 3.0824107749363328)],
            8: [(1, 8, 3, 0.24913336241455764), (2, 8, 3, 0.3731099860619447), (3, 8, 3, 0.5813920938301481),
                (4, 8, 3, 0.732694961298253), (5, 8, 3, 0.8534865904199935)],
            10: [(1, 10, 3, 0.19979157602750372), (2, 10, 3, 0.2985465788024339), (3, 10, 3, 0.47265486585354294),
                 (4, 10, 3, 0.5806415854686175)]
    }

    # --- STYLING CHANGES ---
    # Create the figure
    sns.set_theme(style="whitegrid", rc={"axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # Made figure wider

    # 3. Add a single, shared y-axis label and a main title
    fig.supylabel('Learning Rate (η)', fontsize=14)
    # fig.suptitle('Heuristic Performance: Predicting Optimal Learning Rates from Shallow Circuits', fontsize=18, y=1.0)

    # 4-Qubit Analysis
    analyze_and_plot_heuristic(data_no_entanglement, 4, 5, [6, 7, 8, 9, 10], axes[0])

    # 2-Qubit Analysis
    analyze_and_plot_heuristic(data_no_entanglement, 2, 5, [10, 15, 20], axes[1])

    # 1-Qubit Analysis
    analyze_and_plot_heuristic(data_no_entanglement, 1, 10, [15, 20, 25, 30, 35, 40], axes[2])

    plt.tight_layout(rect=[0.03, 0, 1, 0.95])  # Adjust rect to make space for titles
    plt.subplots_adjust(wspace=0.3)  # Increase horizontal spacing between subplots
    plt.show()

