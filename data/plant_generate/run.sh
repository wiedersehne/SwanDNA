python3 _cython_setup.py build_ext --inplace

python data_pre.py -length 1000 #-num_train 500
#python data_pre.py -length 10000
#python data_pre.py -length 50000
#python data_pre.py -length 100000

python data_label.py -plant ar #-num_train 500
#python data_label.py -plant bd
#python data_label.py -plant mh
#python data_label.py -plant sb
#python data_label.py -plant si
#python data_label.py -plant zm
#python data_label.py -plant zs
