backend=$1 # e.g: nvhtm

# cp ../backend/$backend/tm.h ./include/
# cp ../backend/$backend/thread.c ./src/
# cp ../backend/$backend/thread.h ./include/
# cp ../backend/$backend/Makefile .
# cp ../backends/$backend/Makefile.common .
# cp ../backend/$backend/Makefile.flags .
# cp ../backend/$backend/Defines.common.mk .

rm $(find . -name *.o)

cd code;
rm tpcc

make_command="make -j8 -f Makefile $MAKEFILE_ARGS"
echo " ==========> $make_command"
$make_command
