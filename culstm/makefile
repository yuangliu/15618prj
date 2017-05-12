all: 
	@python test.py
	@nvcc -o LSTM -arch=sm_52 -DPERFOPTS=31 -lcublas -lcurand LSTM.cu
	@./LSTM
clean: 
	@rm *.pyc
	@rm *.cu
	@rm LSTM
