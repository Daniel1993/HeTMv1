
import re
import sys
import subprocess
import numpy as np
import os
from numpy import array

##########################
# related to custom bank #
##########################

#current state of custom-bank renders these
read_set_sizes = [63, 128,257, 512, 1023, 2046, 4092, 5376, 10752, 21504, 43008, 86016, 172032, 344064]
# with these account sizes:
accounts_sizes = [42, 85, 171, 341, 682, 1364, 2728, 3824, 7648, 15296, 30592, 61184, 122368, 244736]
accounts_n = len(accounts_sizes)
'''
here is the transactional code for these r_set_sizes
TM_START(3, RW);
    for(int j = 0; j < d->bank->size; j++) {
        /*write to everyone their own balance*/
        bal = TM_LOAD(&d->bank->accounts[j].balance);
    }

    for(int j = 0; j < d->bank->size; j++) {
        /*write to everyone their own balance*/
        bal = TM_LOAD(&d->bank->accounts[j].balance);
        TM_STORE(&d->bank->accounts[j].balance, bal);
        d->nb_transfer++;
    }
    /**/
TM_COMMIT;
'''
##########################################

keywords = ['validation time']
duration = 5000
custom_bank_string_base = "test/custom-bank/custom-bank --num-threads {} --duration "+str(duration)+" --write-all-rate 50 --read-all-rate 50 --accounts {}"


#plot_x = []
plot_y_touched = []


def custom_bank(args, plot_idx, dir, plot_y):

    # tipical mod_stats data from tinystm:
    # Duration      : 2005 (ms)
    # #txs          : 13040 (6503.740648 / s)
    # aborts        : 334649 (166907.231920 / s)
    # Max retries   : 7430

    global keywords

    if(args):
        if dir is None:
            dir = "./"

        buffer = []
        line_matches = []
        # args = ["./test/custom-bank/custom-bank --num-threads 2 --duration 2000 --accounts 1024"]
        i=0
        j=0

        """execute custom-bank"""
        p = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True, cwd=dir)

        (cout, err) = p.communicate()

        ## Wait for date to terminate. Get return returncode ##
        p_status = p.wait()

        #skip all thread stat data until Duration[space]
        if(p.returncode != 1):
            print("PROGRAM EXITED WITH RETURN CODE:"+ str(p.returncode))

        #print(cout)

        for line in cout.splitlines():
            line = str(line)
            buffer.append(line)
            found = re.search(keywords[0], line)
            if(found):
                j=i
            i+=1

        #only plot data starting from Duration
        buffer = buffer[j:]

        for line in buffer:
            for s in keywords: # 'Duration', '#txs', '#aborts', 'Max retries' #
                found = re.search(s+'\s', line)
                if(found):
                    #print(re.search('\d*[.,]?\d+', line).group())
                    line_matches.append(re.search('\d*[.,]?\d+', line).group())

        #for i in range(0, len(keywords)):
            #print(keywords[i]+ " " +line_matches[i])

        #plotting throughput
        #line_matches[1] is #txs (number of transactions / throughput).
        if(len(line_matches) == 0):
            print(cout)#lets see what the custom-bank outed when it broke
            print("NO MATCH FOR \"validation time\"")
        plot_y.append(float(line_matches[0]))
        #print(keywords[0]+ " " +line_matches[0])

def run(default_rset_start=None, default_rset_end=None):
    global plot_y_touched
    
    cc_string = ""
    cwd = ""
    start = 0
    end = len(read_set_sizes)
    #read_set_sizes_idx = 0 #index for readset and account sizes

    default_rw_set_size=4096

    if(default_rset_start is not None):
        start = int(default_rset_start)
        if(default_rset_end is not None):
            end = int(default_rset_end)
        else:
            end = start + 1
            
    #for read_set_sizes_idx in range(0, len(accounts_sizes)):#LOOP ALL ACCOUNT/RSET SIZES
    for read_set_sizes_idx in range(start, end):#LOOP ALL ACCOUNT/RSET SIZES
        
        ######################################
        # work with igpu-validation tinystm  #
        ######################################
        with open(os.devnull, 'w') as devnull:
            subprocess.call(["make", "clean"], stdout=devnull);
            subprocess.call(["make", "all"], stdout=devnull);
            subprocess.call(["cc -I./include -I./src ./test/custom-bank/custom-bank.c -o ./test/custom-bank/custom-bank -L./lib -lstm -lpthread"], shell=True, stdout=devnull, stderr=devnull);
        
        cwd = "./"
        
        #print("validating " + str(accounts_sizes[read_set_sizes_idx]) +" accounts, "+ str(read_set_sizes[read_set_sizes_idx]) +" read set, 30 times:")
        for j in range(0, 30):#30 for stats
            #one is for number of threads and pass account size
            cc_string = custom_bank_string_base.format(1, str(accounts_sizes[read_set_sizes_idx]))
            #print("Launching igpu tinystm custom-bank 1 thread:%s" % (cwd+cc_string))
            custom_bank([cc_string], read_set_sizes_idx, cwd, plot_y_touched)

        mean = sum(plot_y_touched) / len(plot_y_touched)
        #sum of squared differences from the mean normalized for N
        variance = sum([((x - mean) ** 2) for x in plot_y_touched]) / len(plot_y_touched)
        stddev = variance ** 0.5

        print("Read-set:"+str(read_set_sizes[read_set_sizes_idx]) +", " + str(mean) + " " + str(stddev) + " (mean, stddev)")


#main##################################################################################################################
if __name__ == "__main__":
    argc = len(sys.argv)
    argv = sys.argv
    if argc == 1:
        # run and store
        run()
    elif argc == 2:
        run(sys.argv[1])
    elif argc == 3:
        run(sys.argv[1], sys.argv[2])

        



