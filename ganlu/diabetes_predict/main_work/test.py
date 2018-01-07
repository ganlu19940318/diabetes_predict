
import utils as tl



data_set,test_set = tl.split_data("data/d_train_20180102.csv","data/d_test_A_20180102.csv")

tl.plot_label2(test_set)