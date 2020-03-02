#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=true
reverse=false
word_ins_penalty=0.0
min_lmwt=5
max_lmwt=20
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --decode_mbr (true/false)       # maximum bayes risk decoding (confusion network)."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  echo "    --reverse (true/false)          # score with time reversed features "
  exit 1;
fi

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl 
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl 
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`


for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

mkdir -p $dir/scoring/log

cat $data/text | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > $dir/scoring/test_filt.txt

$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/best_path.LMWT.log \
  lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
  lattice-add-penalty --word-ins-penalty=$word_ins_penalty ark:- ark:- \| \
  lattice-best-path --word-symbol-table=$symtab \
    ark:- ark,t:$dir/scoring/LMWT.tra || exit 1;

if $reverse; then
  for lmwt in `seq $min_lmwt $max_lmwt`; do
    mv $dir/scoring/$lmwt.tra $dir/scoring/$lmwt.tra.orig
    awk '{ printf("%s ",$1); for(i=NF; i>1; i--){ printf("%s ",$i); } printf("\n"); }' \
       <$dir/scoring/$lmwt.tra.orig >$dir/scoring/$lmwt.tra
  done
fi

# Note: the double level of quoting for the sed command
#$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
#   cat $dir/scoring/LMWT.tra \| \
#    utils/int2sym.pl -f 2- $symtab \| sed 's:\<UNK\>::g' \| \
#    compute-wer --text --mode=present \
#     ark:$dir/scoring/test_filt.txt  ark,p:- ">&" $dir/wer_LMWT || exit 1;


# glm file
echo ";; empty.glm" > $dir/scoring/glm
echo "  [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token" >> $dir/scoring/glm
echo "" >> $dir/scoring/glm


# Creare scoring folders
for lmwt in `seq $min_lmwt $max_lmwt`; do
 mkdir -p $dir/score_$lmwt/
done


# ctm file (for sclite)
$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
   cat $dir/scoring/LMWT.tra \| \
    utils/int2sym.pl -f 2- $symtab \| sed 's:\<UNK\>::g' "|" awk '{for (i = 2; i <= NF; i++) {printf "%s 1 0.000 0.000 %s\n",$1,$i}}' "|" \
    tr -d . ">&" $dir/score_LMWT/ctm || exit 1


# Score the set...
$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
   $hubscr -p $hubdir -V -l english -h hub5 -g $dir/scoring/glm -r $data/stm $dir/score_LMWT/ctm || exit 1;


exit 0;
