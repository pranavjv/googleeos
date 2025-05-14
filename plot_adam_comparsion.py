import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from os import environ
import argparse




def plot_adam_vadam_comparison(vadam_eta=0.001, vadam_beta1=0.9, vadam_beta2=0.999, vadam_beta3=1.0, 
                              vadam_power=2.0, vadam_normgrad=False, vadam_lr_cutoff=19.0, vadam_epsilon=1e-7,
                              adam_lr=0.001, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-7,
                              dataset="cifar10-5k", arch="fc-tanh", loss="mse", seed=0):
    """
    Plot comparison between Adam and VADAM results
    """
    # Set environment variables from the command line if not already set
    if "RESULTS" not in os.environ:
        os.environ["RESULTS"] = os.path.join(os.getcwd(), "../results")
    
    # Construct directories
    vadam_dir = f"{environ['RESULTS']}/{dataset}/{arch}/seed_{seed}/{loss}/vadam/eta_{vadam_eta}_beta1_{vadam_beta1}_beta2_{vadam_beta2}_beta3_{vadam_beta3}_power_{vadam_power}_normgrad_{vadam_normgrad}_lr_cutoff_{vadam_lr_cutoff}_eps_{vadam_epsilon}"
    adam_dir = f"{environ['RESULTS']}/{dataset}/{arch}/seed_{seed}/{loss}/adam/lr_{adam_lr}_beta1_{adam_beta1}_beta2_{adam_beta2}_eps_{adam_epsilon}"
    
    print(f"Loading VADAM data from: {vadam_dir}")
    print(f"Loading Adam data from: {adam_dir}")




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
        
        # Create figure with subplots
        plt.figure(figsize=(15, 12), dpi=100)
        
        # 1. Plot training losses
        plt.subplot(2, 2, 1)
        iterations = np.arange(min(len(vadam_train_loss), len(adam_train_loss)))
        plt.plot(iterations, vadam_train_loss[:len(iterations)], label="VADAM")
        plt.plot(iterations, adam_train_loss[:len(iterations)], label="Adam")
        plt.title("Training Loss Comparison")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # 2. Plot training accuracy
        plt.subplot(2, 2, 2)
        plt.plot(iterations, vadam_train_acc[:len(iterations)], label="VADAM")
        plt.plot(iterations, adam_train_acc[:len(iterations)], label="Adam")
        plt.title("Training Accuracy Comparison")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        
        # 3. Plot eigenvalues if available
        if vadam_has_eigs and adam_has_eigs:
            plt.subplot(2, 2, 3)
            vadam_eig_iterations = torch.arange(len(vadam_eigs)) * vadam_eig_freq
            adam_eig_iterations = torch.arange(len(adam_eigs)) * adam_eig_freq
            
            # Plot leading eigenvalues
            plt.scatter(vadam_eig_iterations, vadam_eigs[:, 0], s=10, label="VADAM Eigenvalue 1")
            plt.scatter(adam_eig_iterations, adam_eigs[:, 0], s=10, label="Adam Eigenvalue 1")
            
            # Plot thresholds
            if vadam_has_lr:
                # For VADAM: dynamic threshold based on current learning rate
                sampled_lr = vadam_learning_rates[vadam_eig_iterations]
                vadam_thresholds = 2.0 / sampled_lr
                plt.plot(vadam_eig_iterations, vadam_thresholds, 'r--', 
                        label="VADAM dynamic threshold (2/lr)")
            
            # For Adam: fixed threshold based on learning rate and beta1
            adam_threshold = (2 + 2*adam_beta1)/((1 - adam_beta1)*adam_lr)
            plt.axhline(adam_threshold, linestyle='dotted', color='b', 
                       label=f"Adam threshold: {adam_threshold:.1f}")
            
            plt.title("Sharpness Comparison")
            plt.xlabel("Iteration")
            plt.ylabel("Eigenvalue Magnitude")
            plt.legend()
            plt.grid(True)
        
        # 4. Plot learning rates (dynamic for VADAM, fixed for Adam)
        plt.subplot(2, 2, 4)
        if vadam_has_lr:
            plt.plot(vadam_learning_rates, label="VADAM dynamic lr")
            
            # Show min/max rates for VADAM
            plt.axhline(vadam_eta, linestyle='dotted', color='g',
                      label=f"VADAM max lr (eta): {vadam_eta}")
            min_lr = vadam_eta / (1 + vadam_lr_cutoff)
            plt.axhline(min_lr, linestyle='dotted', color='r',
                      label=f"VADAM min lr: {min_lr:.6f}")
        
        # Show Adam fixed learning rate
        plt.axhline(adam_lr, linestyle='-', color='b', 
                   label=f"Adam fixed lr: {adam_lr}")
        
        plt.title("Learning Rate Comparison")
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        output_file = "adam_vadam_comparison.png"
        plt.savefig(output_file)
        plt.close()
        print(f"Comparison plot saved to {output_file}")
        
        # If both have eigenvalues and VADAM has learning rates, create normalized sharpness plot
        if vadam_has_eigs and adam_has_eigs and vadam_has_lr:
            plt.figure(figsize=(10, 6), dpi=100)
            
            # Compute normalized eigenvalues (eig / threshold)
            # For VADAM: dynamic normalization
            vadam_norm_eigs = vadam_eigs[:, 0] / vadam_thresholds
            
            # For Adam: fixed normalization
            adam_norm_eigs = adam_eigs[:, 0] / adam_threshold
            
            # Plot normalized eigenvalues
            plt.scatter(vadam_eig_iterations, vadam_norm_eigs, s=10, label="VADAM normalized eigenvalue")
            plt.scatter(adam_eig_iterations, adam_norm_eigs, s=10, label="Adam normalized eigenvalue")
            
            # Stability threshold is always 1.0 after normalization
            plt.axhline(1.0, linestyle='dotted', color='k', label="Stability threshold")
            
            plt.title("Normalized Sharpness Comparison")
            plt.xlabel("Iteration")
            plt.ylabel("Eigenvalue / Threshold")
            plt.legend()
            plt.grid(True)
            
            normalized_file = "adam_vadam_normalized_sharpness.png"
            plt.savefig(normalized_file)
            plt.close()
            print(f"Normalized sharpness plot saved to {normalized_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e} - Required data files not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison between Adam and VADAM.")
    # VADAM parameters
    parser.add_argument("--vadam_eta", type=float, default=0.001, help="VADAM eta parameter")
    parser.add_argument("--vadam_beta1", type=float, default=0.9, help="VADAM beta1 parameter")
    parser.add_argument("--vadam_beta2", type=float, default=0.999, help="VADAM beta2 parameter")
    parser.add_argument("--vadam_beta3", type=float, default=1.0, help="VADAM beta3 parameter")
    parser.add_argument("--vadam_power", type=float, default=2.0, help="VADAM power parameter")
    parser.add_argument("--vadam_normgrad", action="store_true", help="VADAM normgrad flag")
    parser.add_argument("--vadam_lr_cutoff", type=float, default=19.0, help="VADAM lr_cutoff parameter")
    parser.add_argument("--vadam_epsilon", type=float, default=1e-7, help="VADAM epsilon parameter")
    
    # Adam parameters
    parser.add_argument("--adam_lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1 parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 parameter")
    parser.add_argument("--adam_epsilon", type=float, default=1e-7, help="Adam epsilon parameter")
    
    # Common parameters
    parser.add_argument("--dataset", type=str, default="cifar10-5k", help="Dataset name")
    parser.add_argument("--arch", type=str, default="fc-tanh", help="Architecture name")
    parser.add_argument("--loss", type=str, default="mse", help="Loss function")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    plot_adam_vadam_comparison(
        vadam_eta=args.vadam_eta,
        vadam_beta1=args.vadam_beta1,
        vadam_beta2=args.vadam_beta2,
        vadam_beta3=args.vadam_beta3,
        vadam_power=args.vadam_power,
        vadam_normgrad=args.vadam_normgrad,
        vadam_lr_cutoff=args.vadam_lr_cutoff,
        vadam_epsilon=args.vadam_epsilon,
        adam_lr=args.adam_lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        dataset=args.dataset,
        arch=args.arch,
        loss=args.loss,
        seed=args.seed
    )