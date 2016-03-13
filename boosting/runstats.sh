# for i in `seq 1 20`;
#     do
#     	python simplexgb.py train.csv test.csv $i
#     done    

for var1 in `seq 1 20`;
    do
    	for var2 in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1;
    		do
    		for var3 in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1;
    			do
    				python simplexgb.py train.csv test.csv $var1 $var2 $var3
    			done
    		done
    done    