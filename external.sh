# Training (Leave a client as an outside client)
python main.py --data prostate --source BIDMC HK ISBI ISBI_1.5 UCL --target I2CVB
python main.py --data prostate --source BIDMC HK I2CVB ISBI_1.5 UCL --target ISBI
python main.py --data prostate --source BIDMC ISBI I2CVB ISBI_1.5 UCL --target HK
python main.py --data prostate --source HK ISBI I2CVB ISBI_1.5 UCL --target BIDMC
python main.py --data prostate --source HK ISBI I2CVB ISBI_1.5 BIDMC --target UCL
python main.py --data prostate --source HK ISBI I2CVB UCL BIDMC --target ISBI_1.5
# Testing on the outside client
python main.py --data prostate --source BIDMC HK ISBI ISBI_1.5 UCL --target I2CVB --ood_test
python main.py --data prostate --source BIDMC HK I2CVB ISBI_1.5 UCL --target ISBI --ood_test
python main.py --data prostate --source BIDMC ISBI I2CVB ISBI_1.5 UCL --target HK --ood_test
python main.py --data prostate --source HK ISBI I2CVB ISBI_1.5 UCL --target BIDMC --ood_test
python main.py --data prostate --source HK ISBI I2CVB ISBI_1.5 BIDMC --target UCL --ood_test
python main.py --data prostate --source HK ISBI I2CVB UCL BIDMC --target ISBI_1.5 --ood_test


