'''

https://github.com/locuslab/edge-of-stability.git

https://github.com/locuslab/edge-of-stability.git

python src/gd.py cifar10-5k fc-tanh  mse  0.01 30 --acc_goal 0.99 --neigs 2  --eig_freq 20



export RESULTS="/home/ubuntu/EOS/googleeos/results"

export DATASETS="/home/ubuntu/EOS/googleeos/datasets"


export DATASETS="/home/ubuntu/EOS/googleeos/datasets"

python src/adam.py cifar10-5k fc-tanh mse 5e-5 20 --loss_goal 0.05 --neigs 4  --eig_freq 5 --beta1 0.9 --beta2 0.99



python src/adam.py cifar10-5k fc-tanh mse 2e-4 20000 --optimizer_type vadam --vadam_normgrad False --eig_freq 5 --neigs 1 --seed 0 --beta1 0.9 --beta2 0.995 --epsilon 1e-7 --vadam_beta3 1.0 --vadam_power 2 --vadam_lr_cutoff 19.0 --physical_batch_size 1000 --nproj 0 --iterate_freq -1 --abridged_size 5000 --save_freq -1



python src/adam.py cifar10-5k fc-tanh mse 2e-4 20000 --optimizer_type adam --vadam_normgrad False --eig_freq 5 --neigs 1 --seed 0 --beta1 0.9 --beta2 0.995 --epsilon 1e-7 --vadam_beta3 1.0 --vadam_power 2 --vadam_lr_cutoff 19.0 --physical_batch_size 1000 --nproj 0 --iterate_freq -1 --abridged_size 5000 --save_freq -1


'''
