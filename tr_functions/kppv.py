# local functions import
from tr_functions.general import *

class KppvModel(GeneralModel):
    
    def __init__(self):
        self._k = 1
        self._cross_val = 0
        

    def get_kppv_list(app_data, dec_point, k):
        dist_list = []
        for app_line in app_data:
            dist_list.append([app_line[0], compute_one_euclidian_dist(app_line[1:3], dec_point)])
        app_df = pd.DataFrame(dist_list)
        sorted_df = app_df.sort_values(by=1)
        kppv_list = sorted_df.head(k)
        return kppv_list

    def compute_1ppv(app_data, dec_data):
        count_top_1 = 0
        classes_num =get_unique_class_num(app_data)
        conf_matrix = np.zeros((len(classes_num), len(classes_num)))
        for line in dec_data:
            kppv_list = get_kppv_list(app_data, line[1:], 1)
            if kppv_list.iloc[0][0] == line[0]:
                count_top_1 = count_top_1 + 1
            # Conf matrix
            row_num = int(line[0]) - 1
            col_num = int(kppv_list.iloc[0][0]) -1
            conf_matrix[row_num, col_num] = conf_matrix[row_num, col_num] + 1
        conf_matrix = transform_matrix_to_df(conf_matrix, classes_num)
        print_decision_model_result(len(app_data), len(dec_data), count_top_1/len(dec_data), top2_rate=None, conf_matrix=conf_matrix)

    def get_majority_result(kppv_list):
        count_dict = Counter(kppv_list[0])
        temp = max(count_dict.values()) 
        res = [key for key in count_dict if count_dict[key] == temp] 
        return res

    def get_k_cross_validation(app_data, k_max, cv):
        best_k = [0, 0]
        df = pd.DataFrame(app_data)
        shuffled = df.sample(frac=1)
        cut_dfs = np.array_split(shuffled, cv)  
        for i in range(k_max):
            k = i + 1
            sum_error = 0
            for i in range(cv):
                df_cv = cut_dfs[i]
                df_train = df.drop(df_cv.index)
                df_cv = df_cv.values.tolist()
                df_train = df_train.values.tolist()
                count_top_1 = 0
                for line in df_cv:
                    kppv_list = get_kppv_list(df_train, line[1:], k)
                    res = get_majority_result(kppv_list)
                    if res[0] == line[0]:
                        count_top_1 = count_top_1 + 1
                error_rate = count_top_1/len(df_cv)
                sum_error = sum_error + error_rate
            if best_k[1] < sum_error/cv:
                best_k[0] = k
                best_k[1] = sum_error/cv
        return best_k

    def compute_kppv(app_data, dec_data, k):
        count_top_1 = 0
        count_top_2 = 0
        classes_num =get_unique_class_num(app_data)
        conf_matrix = np.zeros((len(classes_num), len(classes_num)))
        for line in dec_data:
            kppv_list = get_kppv_list(app_data, line[1:], k)
            k_list = list(Counter(kppv_list[0]).items())
            k_list.sort(key=lambda a: a[1], reverse=True)
            if k_list[0][0] == line[0]:
                count_top_1 = count_top_1 + 1
                count_top_2 = count_top_2 + 1
            else:
                if len(k_list) >= 2:
                    if k_list[1][0] == line[0]:
                        count_top_2 = count_top_2 + 1
            
        #     # Conf matrix
            row_num = int(line[0]) - 1
            col_num = int(k_list[0][0]) -1
            conf_matrix[row_num, col_num] = conf_matrix[row_num, col_num] + 1
        conf_matrix = transform_matrix_to_df(conf_matrix, classes_num)
        print_decision_model_result(len(app_data), len(dec_data), count_top_1/len(dec_data), count_top_2/len(dec_data), conf_matrix=conf_matrix)