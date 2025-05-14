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


def plot_2(gd_directory, gd_lr, gd_eig_freq, axline= None):


    dataset = "cifar10-5k"
    arch = "fc-tanh"
    loss = "mse"


    gd_train_loss = torch.load(f"{gd_directory}/train_loss_final")
    gd_train_acc = torch.load(f"{gd_directory}/train_acc_final")
    gd_sharpness = torch.load(f"{gd_directory}/eigs_final")[:,0]

    try:
        gd_learning_rates = torch.load(f"{gd_directory}/learning_rates_final")
    except:
        gd_learning_rates = None

    plt.figure(figsize=(5, 5), dpi=100)

    plt.subplot(4, 1, 1)
    plt.plot(gd_train_loss)
    plt.title("train loss")
    #plt.yscale("log")

    plt.subplot(4, 1, 2)
    plt.plot(gd_train_acc)
    plt.title("train accuracy")
    plt.yscale("log")

    plt.subplot(4, 1, 3)
    plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5)
    if axline is not None:
        plt.axhline(axline, linestyle='dotted')
    plt.title("sharpness")
    plt.xlabel("iteration")
    plt.yscale("log")



    plt.subplot(4, 1, 4)
    plt.plot(gd_learning_rates)
    plt.title("learning rate")
    #plt.yscale("log")

if __name__ == "__main__":


    gd_lr = 0.002
    correction_factor= 2/38
    gd_eig_freq = 5

    gd_directory = f"{environ['RESULTS']}/cifar10-5k/resnet32/seed_0/mse/vadam/lr_0.002_beta1_0.9_beta2_0.999_eps_1e-07_vbeta3_1.0_vpower_2_vnormgrad_True_vlrcutoff_19.0"
    plot_2(gd_directory, gd_lr, gd_eig_freq, axline= 38/gd_lr)
    plt.savefig("plot_Vadam_eos_large_v8.png")

    gd_lr = 0.001
    gd_directory = f"{environ['RESULTS']}/cifar10-5k/resnet32/seed_0/mse/adam/lr_0.001_beta1_0.9_beta2_0.999_eps_1e-07"
    plot_2(gd_directory, gd_lr, gd_eig_freq, axline= 38/gd_lr)
    plt.savefig("plot_adam_eos_large_v8.png")

    print("done")
