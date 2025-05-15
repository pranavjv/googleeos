import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from os import environ
import argparse

# --- Font Size Configuration ---
TITLE_FONT_SIZE = 25
LEGEND_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 18
AXIS_TICK_FONT_SIZE = 15
# --- End Font Size Configuration ---

# def plot_adam_vadam_comparison(vadam_eta=0.001, vadam_beta1=0.9, vadam_beta2=0.999, vadam_beta3=1.0, 
#                               vadam_power=2.0, vadam_normgrad=False, vadam_lr_cutoff=19.0, vadam_epsilon=1e-7,
#                               adam_lr=0.001, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-7,
#                               dataset="cifar10-5k", arch="fc-tanh", loss="mse", seed=0):
#     """
#     Plot comparison between Adam and VADAM results
#     """



vadam_eta=0.002
vadam_beta1=0.9
vadam_beta2=0.999
vadam_beta3=1.0
vadam_power=2.0
vadam_normgrad=False
vadam_lr_cutoff=19.0
vadam_epsilon=1e-7

adam_lr=0.001
adam_beta1=0.9
adam_beta2=0.999
adam_epsilon=1e-7


vadam_dir = f"{environ['RESULTS']}/cifar10-5k/resnet32/seed_0/mse/vadam/lr_0.002_beta1_0.9_beta2_0.999_eps_1e-07_vbeta3_1.0_vpower_2_vnormgrad_True_vlrcutoff_19.0"

adam_dir = f"{environ['RESULTS']}/cifar10-5k/resnet32/seed_0/mse/adam/lr_0.001_beta1_0.9_beta2_0.999_eps_1e-07"

try:
    # Load VADAM data
    vadam_train_loss = torch.load(f"{vadam_dir}/train_loss_final")
    vadam_train_acc = torch.load(f"{vadam_dir}/train_acc_final")
    
    try:
        vadam_eigs = torch.load(f"{vadam_dir}/eigs_final")
        vadam_has_eigs = True
        vadam_eig_freq = 5  # Default value, can be detected from data
    except FileNotFoundError:
        vadam_has_eigs = False
        print("No VADAM eigenvalue data found")
    
    try:
        vadam_learning_rates = torch.load(f"{vadam_dir}/learning_rates_final")
        vadam_has_lr = True
    except FileNotFoundError:
        vadam_has_lr = False
        print("No VADAM learning rate data found")
    
    # Load Adam data
    adam_train_loss = torch.load(f"{adam_dir}/train_loss_final")
    adam_train_acc = torch.load(f"{adam_dir}/train_acc_final")
    
    try:
        adam_eigs = torch.load(f"{adam_dir}/eigs_final")
        adam_has_eigs = True
        adam_eig_freq = 5  # Default value, can be detected from data
    except FileNotFoundError:
        adam_has_eigs = False
        print("No Adam eigenvalue data found")



    ## Parameters
    xlima = -5
    xlimb = 195
    
    # Create figure with subplots
    plt.figure(figsize=(15, 12), dpi=100)
    
    # 1. Plot training losses
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(vadam_train_loss)), vadam_train_loss, label="VRAdam")
    plt.plot(np.arange(len(adam_train_loss)), adam_train_loss, label="Adam")
    plt.title("Training Loss Comparison", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Loss", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.yticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlim(xlima, xlimb)
    plt.grid(True)
    
    # 2. Plot training accuracy
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(len(vadam_train_acc)), vadam_train_acc, label="VRAdam")
    plt.plot(np.arange(len(adam_train_acc)), adam_train_acc, label="Adam")
    plt.title("Training Accuracy Comparison", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Accuracy", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.yticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlim(xlima, xlimb)
    plt.grid(True)
    

    # 3. Plot eigenvalues if available
    if vadam_has_eigs and adam_has_eigs:
        plt.subplot(2, 2, 3)
        vadam_eig_iterations = torch.arange(len(vadam_eigs)) * vadam_eig_freq
        adam_eig_iterations = torch.arange(len(adam_eigs)) * adam_eig_freq
        
        # Plot leading eigenvalues
        plt.scatter(vadam_eig_iterations, vadam_eigs[:, 0], s=10, label="VRAdam max eigenvalue")
        plt.scatter(adam_eig_iterations, adam_eigs[:, 0], s=10, label="Adam max eigenvalue")
        
        # # Plot thresholds
        # if vadam_has_lr:
        #     # For VADAM: dynamic threshold based on current learning rate
        #     sampled_lr = vadam_learning_rates[vadam_eig_iterations]
        #     vadam_thresholds = 2.0 / sampled_lr
        #     plt.plot(vadam_eig_iterations, vadam_thresholds, 'r--', 
        #             label="VADAM dynamic threshold (2/lr)")
            
        # plt.axhline(38/0.002, linestyle='dotted', color='b', 
        #             label=f"38/0.002")

        # plt.axhline(38/0.001, linestyle='dotted', color='orange', 
        #             label=f"38/0.001")
        
        # For Adam: fixed threshold based on learning rate and beta1
        #adam_threshold = (2 + 2*adam_beta1)/((1 - adam_beta1)*adam_lr)
        #plt.axhline(adam_threshold, linestyle='dotted', color='b', 
        #            label=f"Adam threshold: {adam_threshold:.1f}")
        plt.xlim(xlima, xlimb)
        plt.title("Sharpness Comparison", fontsize=TITLE_FONT_SIZE)
        plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
        plt.ylabel("Eigenvalue Magnitude", fontsize=AXIS_LABEL_FONT_SIZE)
        plt.xticks(fontsize=AXIS_TICK_FONT_SIZE)
        plt.yticks(fontsize=AXIS_TICK_FONT_SIZE)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax = plt.gca() # Get current axes
        ax.yaxis.get_offset_text().set_fontsize(AXIS_LABEL_FONT_SIZE) # Increase exponent font size
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.grid(True)
    
    # 4. Plot learning rates (dynamic for VADAM, fixed for Adam)
    plt.subplot(2, 2, 4)
    if vadam_has_lr:
        plt.plot(vadam_learning_rates[:-1], label="VRAdam dynamic lr")
        
        # Show min/max rates for VADAM
        plt.axhline(vadam_eta, linestyle='dotted', color='b',
                    label=f"VRAdam max lr")
        min_lr = vadam_eta / (1 + vadam_lr_cutoff)
        plt.axhline(min_lr, linestyle='dotted', color='b',
                    label=f"VRAdam min lr")
    
    # Show Adam fixed learning rate
    plt.axhline(adam_lr, linestyle='-', color='orange', 
                label=f"Adam fixed lr")
    
    plt.xlim(xlima, xlimb)
    plt.title("Learning Rate Comparison", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Learning Rate", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.yticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = plt.gca() # Get current axes
    ax.yaxis.get_offset_text().set_fontsize(AXIS_LABEL_FONT_SIZE) # Increase exponent font size
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True)
    
    plt.tight_layout()
    output_file = "adam_vadam_comparison.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Comparison plot saved to {output_file}")
    
    # If both have eigenvalues and VADAM has learning rates, create normalized sharpness plot
    # if vadam_has_eigs and adam_has_eigs and vadam_has_lr:
    #     plt.figure(figsize=(10, 6), dpi=100)
        
    #     # Compute normalized eigenvalues (eig / threshold)
    #     # For VADAM: dynamic normalization
    #     vadam_norm_eigs = vadam_eigs[:, 0] / vadam_thresholds
        
    #     # For Adam: fixed normalization
    #     adam_norm_eigs = adam_eigs[:, 0] / adam_threshold
        
    #     # Plot normalized eigenvalues
    #     plt.scatter(vadam_eig_iterations, vadam_norm_eigs, s=10, label="VADAM normalized eigenvalue")
    #     plt.scatter(adam_eig_iterations, adam_norm_eigs, s=10, label="Adam normalized eigenvalue")
        
    #     # Stability threshold is always 1.0 after normalization
    #     plt.axhline(1.0, linestyle='dotted', color='k', label="Stability threshold")
        
    #     plt.title("Normalized Sharpness Comparison")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Eigenvalue / Threshold")
    #     plt.legend()
    #     plt.grid(True)
        
    #     normalized_file = "adam_vadam_normalized_sharpness.png"
    #     plt.savefig(normalized_file)
    #     plt.close()
    #     print(f"Normalized sharpness plot saved to {normalized_file}")
    
except FileNotFoundError as e:
    print(f"Error: {e} - Required data files not found.")

