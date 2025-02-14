# python fed_perfedmask.py --pd_nuser 10 --pr_nuser 5 --partition_mode dir --pu_nclass 10 --con_test_cls True --data Fmnist --partition_alpha 0.05&
# python fed_perfedmask.py --pd_nuser 100 --pr_nuser 5 --partition_mode dir --pu_nclass 10 --con_test_cls True --data Fmnist --partition_alpha 0.1&
# python fed_perfedmask.py --pd_nuser 100 --pr_nuser 5 --partition_mode dir --pu_nclass 10 --con_test_cls True --data Fmnist --partition_alpha 0.05&
# python fed_perfedmask.py --pd_nuser 100 --pr_nuser 5 --partition_mode dir --pu_nclass 10 --con_test_cls True --data Cifar10 --partition_alpha 0.1&
# python fed_perfedmask.py --pd_nuser 100 --pr_nuser 5 --partition_mode dir --pu_nclass 10 --con_test_cls True --data Cifar10 --partition_alpha 0.05&
# python fed_perfedmask.py --pd_nuser 100 --pr_nuser 5 --partition_mode dir --pu_nclass 100 --con_test_cls True --data Cifar100 --partition_alpha 0.1&
# python fed_perfedmask.py --pd_nuser 100 --pr_nuser 5 --partition_mode dir --pu_nclass 100 --con_test_cls True --data Cifar100 --partition_alpha 0.05&

# python fed_perfedmask.py --pd_nuser 10 --pr_nuser 5 --partition_mode dir --pu_nclass 200 --con_test_cls True --data Tiny-ImageNet --partition_alpha 0.1&
# python fed_perfedmask.py --pd_nuser 10 --pr_nuser 5 --partition_mode dir --pu_nclass 200 --con_test_cls True --data Tiny-ImageNet --partition_alpha 0.05&
# python fed_perfedmask.py --pd_nuser 100 --pr_nuser 5 --partition_mode dir --pu_nclass 200 --con_test_cls True --data Tiny-ImageNet --partition_alpha 0.1&
# python fed_perfedmask.py --pd_nuser 100 --pr_nuser 5 --partition_mode dir --pu_nclass 200 --con_test_cls True --data Tiny-ImageNet --partition_alpha 0.05&

# python fed_perfedmask.py --pd_nuser 10 --pr_nuser 5 --partition_mode dir --pu_nclass 10 --con_test_cls True --data Cifar10 --model mobilenet --partition_alpha 0.1&
python fed_perfedmask.py --pd_nuser 10 --pr_nuser 5 --partition_mode dir --pu_nclass 10 --con_test_cls True --data Cifar10 --model simple-cnn --partition_alpha 0.1&