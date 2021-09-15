
from utils import *
import numpy as np

def transfor_matrix_binary():
    class_dim = 50
    attribute = get_predicate_binary_mat()
    all_classes = get_all_classes()
    test_classes = get_test_classes()
    transfor_binary_matrix = np.zeros((50,85), dtype=np.int)
    transfor_all_classes = list(range(50))

    j = 0
    k = 40

    for i in range(class_dim):
        # 得到每个类名
        awa2_class = all_classes[i]
        # 训练类从0开始排序
        if  awa2_class not in test_classes:
            transfor_binary_matrix[j] = attribute[i]
            transfor_all_classes[j] = awa2_class
            j = j+1
            if j > 40:
                print("error")
                break
        # 测试类从40开始排序
        else:
            transfor_binary_matrix[k] = attribute[i]
            transfor_all_classes[k] = awa2_class
            k = k+1
            if k > 50:
                print("test error")
                break 
        
    return transfor_binary_matrix, transfor_all_classes

def train_index_40():
    class_to_index = mapping_class_to_index()
    i = 0
    transform_train_index = {}
    with open(AWA2_PATH + TRAIN_CLASS_PATH) as f:
        for line in f:
            class_name = line.strip()
            class_index = class_to_index[class_name]
            transform_train_index[class_index] = i
            i += 1
    
    return transform_train_index





if __name__ == "__main__":
    train_index_40()







