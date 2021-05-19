from sklearn.neighbors import KernelDensity

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

def create_train_test_data_format_scikit(train_data:list, test_data:list):
    x_train = []
    y_train = []
    for line in train_data:
        y_train.append(int(line[0]))
        coordinates = []
        for val in line[1:]:
            coordinates.append(float(val))
        x_train.append(coordinates)
    x_test = []
    y_test = []
    for line in test_data:
        y_test.append(int(line[0]))
        coordinates = []
        for val in line[1:]:
            coordinates.append(float(val))
        x_test.append(coordinates)
    return x_train, y_train, x_test, y_test

def create_sub_train_data(data:list, i):
    clean_data = []
    for line  in data:
        if int(line[0]) == i:
            clean_data.append(line)
    return clean_data

def find_best_score_kernel(train_data: list, test_data:list):
    sub_density_values = []
    for line in test_data:
        sub_density_values.append((0,0))
    for i_class in range(5):
        sub_train = create_sub_train_data(train_data, i_class+1)
        x_train, y_train, x_test, y_test = create_train_test_data_format_scikit(sub_train, test_data)
        kd1 = KernelDensity(kernel='gaussian', bandwidth=0.5)
        kd1.fit(x_train)
        values = kd1.score_samples(x_test)
        for i_val in range(len(sub_density_values)):
            if sub_density_values[i_val][0] == 0:
                sub_density_values[i_val] = (i_class+1, values[i_val])
            else:
                if sub_density_values[i_val][1] < values[i_val]:
                    sub_density_values[i_val] = (i_class+1, values[i_val])
    count=0
    for i in range(len(test_data)):
        if int(test_data[i][0]) == sub_density_values[i][0]:
            count+=1
    return count/500