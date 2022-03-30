#!/bin/bash

# Files used in script

set -v

FNAME=$1
AGRAPH=$FNAME.in
OUTRAW=$FNAME.raw
REPS=$3
CORES=$4
ITER=$5
REPORT_OPS=$(echo "$5 * $REP_OPS" | bc )
EXEC=project

RESULTS=$FNAME.$2.results    # Delete before finish
SELECTED=$FNAME.$2.selected  # Delete before finish
SORTED=$FNAME.$2

if [ "${10}" == "all" ]
then
    REPORT_OPS=0
fi

if [ "$2" == "hydata" ]
then
    # EXEC=hybrid/project
    EXEC=hybrid/project_NO_INIT
fi

if [ "$2" == "data" ]
then
    EXEC=SA/project_ROLLBACK
fi

if [ "$2" == "data1" ]
then
    EXEC=SA/project
fi

if [ "$2" == "data2" ]
then
    # EXEC=SA/project
    EXEC=SA/project_NO_INIT
fi

if [ "$2" == "data3" ]
then
    EXEC=SA/project
fi

if [ "$2" == "greedy" ]
then
    EXEC=SA/project_GREEDY
fi

if [ "$2" == "noinit" ]
then
    EXEC=SA/project_NO_INIT
fi

if [ "$2" == "gnoinit" ]
then
    EXEC=SA/project_GREEDY_NO_INIT
fi

if [ "$2" == "bb" ]
then
    EXEC=BB/project
fi

rm ${RESULTS}*

for i in $(seq 1 $CORES $REPS)
do
    for j in $(seq 0 $(($CORES - 1)))
    do
        echo -n "G$(($i + $j))    "
        if [ "$2" == "bb" ]
        then
            echo "$HOME_PATH/$EXEC $GRAPH_PATH/$AGRAPH  $ITER $REPORT_OPS $j" > runthis.core$j
        else
            echo "$HOME_PATH/$EXEC $GRAPH_PATH/$AGRAPH $6 $7 $8 $9 $ITER $REPORT_OPS $j" > runthis.core$j
        fi
        if [ $j -eq 0 ]
        then
            cat runthis.core$j
        fi
        if [ "$2" == "bb" ]
        then
            time $HOME_PATH/$EXEC $GRAPH_PATH/$AGRAPH $ITER $REPORT_OPS $j > $RESULTS.$(($i + $j)).tmp & pid[$j]=$!
        else
            time $HOME_PATH/$EXEC $GRAPH_PATH/$AGRAPH $6 $7 $8 $9 $ITER $REPORT_OPS $j > $RESULTS.$(($i + $j)).tmp & pid[$j]=$!
        fi
        # awk -v shrink=$shrink -v seed=$RANDOM \
        #     -- 'BEGIN{srand(seed)}
        #     { if ($1 == "#" || ($6 < 5 && 0 == int(shrink * rand()))) { print }}' >> $RESULTS.$i &
    done
    for j in $(seq 0 $(($CORES - 1)))
    do
        wait ${pid[$j]}
        if [ $? -ne 0 ]
        then
            echo -n "try again G$(($i + $j))    "
            if [ "$2" == "bb" ]
            then
                time $HOME_PATH/$EXEC $GRAPH_PATH/$AGRAPH $ITER $REPORT_OPS $j > $RESULTS.$(($i + $j)).tmp & pid[$j]=$!
            else
                time $HOME_PATH/$EXEC $GRAPH_PATH/$AGRAPH $6 $7 $8 $9 $ITER $REPORT_OPS $j > $RESULTS.$(($i + $j)).tmp & pid[$j]=$!
            fi
        fi
        wait ${pid[$j]}
        if [ $? -eq 0 ]
        then
            cat $RESULTS.$(($i + $j)).tmp >> $RESULTS.$(($i + $j))
            rm $RESULTS.$(($i + $j)).tmp
        else
            echo -n "Can't run! "
            cat runthis.core$j
        fi
    done
done

:> $RESULTS
wait

rm $SORTED
touch $SORTED
for i in $(seq 1 $REPS)
do
    cat $RESULTS.$i >> $SORTED
    rm $RESULTS.$i
done

set +v

exit 0;
