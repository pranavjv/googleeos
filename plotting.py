import torch
import matplotlib.pyplot as plt
from os import environ



def plot_1():
    dataset = "cifar10-5k"
    arch = "fc-tanh"
    loss = "mse"
    gd_lr = 0.01
    gd_eig_freq = 50


    dataset = "cifar10-5k"
    arch = "fc-tanh"
    loss = "mse"
    gd_lr = 0.01
    gd_eig_freq = 50

    gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"


    

    gd_train_loss = torch.load(f"{gd_directory}/train_loss_final")
    gd_train_acc = torch.load(f"{gd_directory}/train_acc_final")
    gd_sharpness = torch.load(f"{gd_directory}/eigs_final")[:,0]

    plt.figure(figsize=(5, 5), dpi=100)

    plt.subplot(3, 1, 1)
    plt.plot(gd_train_loss)
    plt.title("train loss")

    plt.subplot(3, 1, 2)
    plt.plot(gd_train_acc)
    plt.title("train accuracy")

    plt.subplot(3, 1, 3)
    plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5)
    plt.axhline(2. / gd_lr, linestyle='dotted')
    plt.title("sharpness")
    plt.xlabel("iteration")


def plot_2():
    dataset = "cifar10-5k"
    arch = "fc-tanh"
    loss = "mse"
    gd_lr = 0.01
    gd_eig_freq = 50


    dataset = "cifar10-5k"
    arch = "fc-tanh"
    loss = "mse"
    gd_lr = 0.01
    gd_eig_freq = 50

    #gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_0/{loss}/gd/lr_{gd_lr}"
    gd_directory = f"{environ['RESULTS']}/cifar10-5k/fc-tanh/seed_0/mse/adam/lr_5e-05_beta1_0.9_beta2_0.99_eps_1e-07"


    gd_directory = f"{environ['RESULTS']}/cifar10-5k/fc-tanh/seed_0/mse/vadam/lr_0.001_beta1_0.9_beta2_0.995_eps_1e-07_vbeta3_1.0_vpower_2_vnormgrad_True_vlrcutoff_19.0"


    gd_directory = f"{environ['RESULTS']}/cifar10-5k/fc-tanh/seed_0/mse/vadam/lr_0.0002_beta1_0.9_beta2_0.995_eps_1e-07_vbeta3_1.0_vpower_2_vnormgrad_True_vlrcutoff_19.0"


    gd_directory = f"{environ['RESULTS']}/cifar10-5k/fc-tanh/seed_0/mse/adam/lr_0.0002_beta1_0.9_beta2_0.995_eps_1e-07"



    gd_train_loss = torch.load(f"{gd_directory}/train_loss_final")
    gd_train_acc = torch.load(f"{gd_directory}/train_acc_final")
    gd_sharpness = torch.load(f"{gd_directory}/eigs_final")[:,0]

    plt.figure(figsize=(5, 5), dpi=100)

    plt.subplot(3, 1, 1)
    plt.plot(gd_train_loss)
    plt.title("train loss")

    plt.subplot(3, 1, 2)
    plt.plot(gd_train_acc)
    plt.title("train accuracy")

    plt.subplot(3, 1, 3)
    plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5)
    plt.axhline(2. / gd_lr, linestyle='dotted')
    plt.title("sharpness")
    plt.xlabel("iteration")

if __name__ == "__main__":
    plot_2()
    plt.savefig("plot_adam_eos_long.png")
