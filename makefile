T=20
L=4
O=32
B=64
I=32

all: init seqlstm tflstm cu

init:
	@echo "T $T L $L O $O B $B I $I" > res.txt

seqlstm:
	@python seq/test.py $T $L $O  $B $I >> res.txt

tflstm:
	@python tf/tf_lstm.py $T $L $O $B $I >> res.txt

cu:
	@./rnn/LSTM2 $T $L $O $B $I >> res.txt

clean:
	@cd seq
	@rm -f *.pyc
	@cd ..
	@cd tf
	@rm -f *.pyc
	@cd ..
	@cd rnn
	@make clean
	@cd ..