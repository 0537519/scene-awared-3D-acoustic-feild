import pickle


path_1 = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/minmax/frl_apartment_4_minmax"
path_2 = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/minmax_2/frl_apartment_4_minmax"

with open(path_1, 'rb') as file:
    data = pickle.load(file)
    print(data)
    print("x1:")
    print(data[1][0]-data[0][0])
    print("y1:")
    print(data[1][2] - data[0][2])
    print("\n")

with open(path_2, 'rb') as file:
    data = pickle.load(file)
    print(data)
    print("x2:")
    print(data[1][0]-data[0][0])
    print("y2:")
    print(data[1][2] - data[0][2])
    print("\n")
