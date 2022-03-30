
#tinystm-untouched

make clean  > /dev/null
make all    > /dev/null
make test   > /dev/null
cc -I./include -I./src ./test/custom-bank/custom-bank.c \
  -o ./test/custom-bank/custom-bank -L./lib -lstm -lpthread

for a in 200 2000 20000 200000 2000000
do
  ./test/custom-bank/custom-bank --num-threads 1 --duration 10000 --accounts $a
done
#2816
