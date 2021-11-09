import os


def main():

    os.system("nohup python3 -u model_trainer.py 0.01 1 1 > output_1 &")
    os.system("nohup python3 -u model_trainer.py 0.005 2 10 > output_2 &")
    os.system("nohup python3 -u model_trainer.py 0.0025 3 30 > output_3 &")
    os.system("nohup python3 -u model_trainer.py 0.001 4 60 > output_4 &")

    os.system("nohup python3 -u model_trainer.py 0.01 5 1 > output_5 &")
    os.system("nohup python3 -u model_trainer.py 0.005 6 10 > output_6 &")
    os.system("nohup python3 -u model_trainer.py 0.0025 7 30 > output_7 &")
    os.system("nohup python3 -u model_trainer.py 0.001 8 60 > output_8 &")

    os.system("nohup python3 -u model_trainer.py 0.01 9 1 > output_9 &")
    os.system("nohup python3 -u model_trainer.py 0.005 10 10 > output_10 &")
    os.system("nohup python3 -u model_trainer.py 0.0025 11 30 > output_11 &")
    os.system("nohup python3 -u model_trainer.py 0.001 12 60 > output_12 &")

    os.system("nohup python3 -u model_trainer.py 0.01 13 1 > output_13 &")
    os.system("nohup python3 -u model_trainer.py 0.005 14 10 > output_14 &")
    os.system("nohup python3 -u model_trainer.py 0.0025 15 30 > output_15 &")
    os.system("nohup python3 -u model_trainer.py 0.001 16 60 > output_16 &")
    
    return

if __name__ == '__main__':
    main()