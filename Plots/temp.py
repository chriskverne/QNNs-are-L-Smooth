import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Data from previous turns and the new 10-qubit data
# data_2q = [{'layers': 1, 'P': 6, 'L_max_ratio': 0.3318576374824204, 'D_KL': 0.05440392634943307},
#            {'layers': 2, 'P': 12, 'L_max_ratio': 0.2139310344182754, 'D_KL': 0.021502161291199706},
#            {'layers': 3, 'P': 18, 'L_max_ratio': 0.2060053026190467, 'D_KL': 0.013409746400984131},
#            {'layers': 4, 'P': 24, 'L_max_ratio': 0.19748358164679916, 'D_KL': 0.018522306085856397},
#            {'layers': 5, 'P': 30, 'L_max_ratio': 0.21597246148738694, 'D_KL': 0.019334704267582138},
#            {'layers': 6, 'P': 36, 'L_max_ratio': 0.1903739996457256, 'D_KL': 0.019669463078819466},
#            {'layers': 7, 'P': 42, 'L_max_ratio': 0.18844286699178733, 'D_KL': 0.022385657166621265},
#            {'layers': 8, 'P': 48, 'L_max_ratio': 0.17782857037372382, 'D_KL': 0.015387487759360413},
#            {'layers': 9, 'P': 54, 'L_max_ratio': 0.18390644177901233, 'D_KL': 0.01431003557335625},
#            {'layers': 10, 'P': 60, 'L_max_ratio': 0.1758583634701901, 'D_KL': 0.019777026410283175}]

data_2q = [{'layers': 1, 'P': 6, 'L_max_ratio': 0.3318576374824204, 'D_KL': 0.05440392634943307}, {'layers': 2, 'P': 12, 'L_max_ratio': 0.21393103441827543, 'D_KL': 0.021502161291199706}, {'layers': 3, 'P': 18, 'L_max_ratio': 0.2060053026190467, 'D_KL': 0.013409746400984131}, {'layers': 4, 'P': 24, 'L_max_ratio': 0.19748358164679913, 'D_KL': 0.018522306085856397}, {'layers': 5, 'P': 30, 'L_max_ratio': 0.21597246148738694, 'D_KL': 0.019334704267582138}, {'layers': 6, 'P': 36, 'L_max_ratio': 0.1903739996457256, 'D_KL': 0.019669463078819466}, {'layers': 7, 'P': 42, 'L_max_ratio': 0.18844286699178736, 'D_KL': 0.022385657166621265}, {'layers': 8, 'P': 48, 'L_max_ratio': 0.1778285703737239, 'D_KL': 0.015387487759360413}, {'layers': 9, 'P': 54, 'L_max_ratio': 0.18390644177901233, 'D_KL': 0.01431003557335625}, {'layers': 10, 'P': 60, 'L_max_ratio': 0.17585836347019015, 'D_KL': 0.019777026410283175}, {'layers': 11, 'P': 66, 'L_max_ratio': 0.18081925632122264, 'D_KL': 0.0186050677882523}, {'layers': 12, 'P': 72, 'L_max_ratio': 0.1837072277230795, 'D_KL': 0.018785662720846707}, {'layers': 13, 'P': 78, 'L_max_ratio': 0.1641933448403345, 'D_KL': 0.0182511424550208}, {'layers': 14, 'P': 84, 'L_max_ratio': 0.16988643824707977, 'D_KL': 0.023631184618993197}, {'layers': 15, 'P': 90, 'L_max_ratio': 0.17167422613109862, 'D_KL': 0.02146304946386004}, {'layers': 16, 'P': 96, 'L_max_ratio': 0.1565302189614245, 'D_KL': 0.017904738521898004}, {'layers': 17, 'P': 102, 'L_max_ratio': 0.15343214116509368, 'D_KL': 0.021301030343846467}, {'layers': 18, 'P': 108, 'L_max_ratio': 0.17109024762075856, 'D_KL': 0.02005727304296698}, {'layers': 19, 'P': 114, 'L_max_ratio': 0.16721626249136085, 'D_KL': 0.013793412494053732}, {'layers': 20, 'P': 120, 'L_max_ratio': 0.16051339419964933, 'D_KL': 0.017375320376486526}]

data_4q = [{'layers': 1, 'P': 12, 'L_max_ratio': 0.1637990427601808, 'D_KL': 0.2338772466150277},
           {'layers': 2, 'P': 24, 'L_max_ratio': 0.12098582775582319, 'D_KL': 0.02135498594226106},
           {'layers': 3, 'P': 36, 'L_max_ratio': 0.08482524477416811, 'D_KL': 0.011179034709099913},
           {'layers': 4, 'P': 48, 'L_max_ratio': 0.08342671107985349, 'D_KL': 0.009338308403378431},
           {'layers': 5, 'P': 60, 'L_max_ratio': 0.061855452689485405, 'D_KL': 0.004920165050547805},
           {'layers': 6, 'P': 72, 'L_max_ratio': 0.05827594153186555, 'D_KL': 0.006612911155774292},
           {'layers': 7, 'P': 84, 'L_max_ratio': 0.05705149794073222, 'D_KL': 0.005822226327926969},
           {'layers': 8, 'P': 96, 'L_max_ratio': 0.05299504397811419, 'D_KL': 0.008144022005582448},
           {'layers': 9, 'P': 108, 'L_max_ratio': 0.06801924170547302, 'D_KL': 0.008122507761685365},
           {'layers': 10, 'P': 120, 'L_max_ratio': 0.05822983034661855, 'D_KL': 0.005908770055948315}]
data_8q = [{'layers': 1, 'P': 24, 'L_max_ratio': 0.08255271408457514, 'D_KL': 0.1765455907601364},
           {'layers': 2, 'P': 48, 'L_max_ratio': 0.057895523236510045, 'D_KL': 0.060746621994808105},
           {'layers': 3, 'P': 72, 'L_max_ratio': 0.05120352937959477, 'D_KL': 0.004552686304385713},
           {'layers': 4, 'P': 96, 'L_max_ratio': 0.03524512995691607, 'D_KL': 0.0017037048881130452},
           {'layers': 5, 'P': 120, 'L_max_ratio': 0.02850950701371178, 'D_KL': 0.0005379930594233187}]
data_10q = [{'layers': 1, 'P': 30, 'L_max_ratio': 0.06559991761296041, 'D_KL': 0.13987422164198374},
            {'layers': 2, 'P': 60, 'L_max_ratio': 0.04554965335377666, 'D_KL': 0.03226308970123189},
            {'layers': 3, 'P': 90, 'L_max_ratio': 0.03823965175381332, 'D_KL': 0.009465083859765298},
            {'layers': 4, 'P': 120, 'L_max_ratio': 0.029316445382331344, 'D_KL': 1.0113630363008302e-06}]

df_2q = pd.DataFrame(data_2q)
df_4q = pd.DataFrame(data_4q)
df_8q = pd.DataFrame(data_8q)
df_10q = pd.DataFrame(data_10q)

datasets = [(df_2q, 2), (df_4q, 4), (df_8q, 8), (df_10q, 10)]

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
ax2_top = None # Initialize a variable to hold the top plot's second axis

for i, (df, n_qubits) in enumerate(datasets):
    ax1 = axes[i]
    ax1.set_facecolor('gainsboro')

    # Plot Hessian Norm Ratio
    color = 'tab:blue'
    ax1.set_ylabel(r'($L_{\max}/L_{\mathrm{upper}}$)', color=color, fontsize=12)
    ax1.plot(df['P'], df['L_max_ratio'], marker='o', linestyle='-', color=color, label='Hessian Norm Ratio', linewidth=3)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.7)

    # Create a second y-axis for Expressibility
    ax2 = ax1.twinx()
    color = 'tab:red'
    #ax2.set_ylabel(r'Expressibility ($D_{\mathrm{KL}}$)', color=color, fontsize=12)
    ax2.set_ylabel(r'Expres. $D_{\mathrm{KL}}$', color=color, fontsize=12)

    ax2.plot(df['P'], df['D_KL'], marker='x', linestyle='--', color=color, label='Expressibility', linewidth=3, markersize=10)
    ax2.tick_params(axis='y', labelcolor=color)

    #ax1.set_xlabel('Number of Parameters (P)', fontsize=12)
    ax1.set_title(f'{n_qubits}-Qubit System', fontsize=12)

    # Save the second axis of the top plot to use for the legend later
    if i == 0:
        ax2_top = ax2
        # Fix the D_KL axis scaling
        d_kl_max = df['D_KL'].max()
        upper_limit = d_kl_max * 1.15
        ax2.set_ylim(0, upper_limit)

    # --- APPLIED CHANGES START HERE ---

    # Format ALL y-axes to have two decimal places for consistency
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

axes[-1].set_xlabel('Number of Parameters (P)', fontsize=12)


# Create a single legend for the entire figure
handles1, labels1 = axes[0].get_legend_handles_labels()
handles2, labels2 = ax2_top.get_legend_handles_labels()
fig.legend(handles1 + handles2, labels1 + labels2,ncols=2 ,loc='upper right', bbox_to_anchor=(0.9, 0.99))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.savefig('qnn_saturation_final_formatted.png', dpi=300)
plt.show()






