#!/bin/bash


# Copyright 2013    Yajie Miao    Carnegie Mellon University
# Apache 2.0

# Decode the DNN model. The [srcdir] in this script should be the same as dir in
# build_nnet_pfile.sh. Also, the DNN model has been trained and put in srcdir.
# All these steps will be done automatically if you run the recipe file run-dnn.sh

# Modified 2018 Mirco Ravanelli Univeristé de Montréal - Mila


cfg_file=$1
out_folder=$2



# Reading the options in the cfg file
source <(grep = $cfg_file | sed 's/ *= */=/g')

cd $decoding_script_folder

./path.sh
./cmd.sh


## Begin configuration section
num_threads=1
stage=0
cmd=utils/run.pl


echo "$0 $@"  # Print the command line for logging

./parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 5)"
   echo "Usage: steps/decode_dnn.sh [options] <graph-dir> <data-dir> <ali-dir> <decode-dir>"
   echo " e.g.: steps/decode_dnn.sh exp/tri4/graph data/test exp/tri4_ali exp/tri4_dnn/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --num-threads <n>                        # number of threads to use, default 4."
   echo "  --parallel-opts <opts>                   # e.g. '-pe smp 4' if you supply --num-threads 4"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi



dir=`echo $out_folder | sed 's:/$::g'` # remove any trailing slash.
featstring=$3
srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.
sdata=$data/split$nj;

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"


mkdir -p $dir/log

arr_ck=($(ls $featstring))

nj=${#arr_ck[@]}

echo $nj > $dir/num_jobs

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


JOB=1
for ck_data in "${arr_ck[@]}"
do

    finalfeats="ark,s,cs: cat $ck_data |"
    latgen-faster-mapped$thread_string --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $alidir/final.mdl $graphdir/HCLG.fst "$finalfeats" "ark:|gzip -c > $dir/lat.$JOB.gz" &> $dir/log/decode.$JOB.log &
    JOB=$((JOB+1))
done
wait



# Copy the source model in order for scoring
cp $alidir/final.mdl $srcdir
  

if ! $skip_scoring ; then
  [ ! -x $scoring_script ] && \
    echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
  $scoring_script $scoring_opts $data $graphdir $dir
fi

exit 0;
