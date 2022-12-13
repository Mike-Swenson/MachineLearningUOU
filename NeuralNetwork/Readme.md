There are two different driver files, main_nn.py and main_tensor.py

main_nn is the required driver for the assignment and is a nn implemented using only python/numpy/pandas
It has one flag "select" which switches how the weights are initialized, 0 for random weights, 1 for 0 weight initialization

eg<br>
python main_nn.py 0<br>    
python main_nn.py 1

main_tensor.py is a nn implemented with tensorflow and runs through a combination of layers and widths
It has one flag "select" which switches between Xavier(0) and HE(1) initialization

eg<br>
python main_tensor.py 0<br>
python main_tensor.py 1
