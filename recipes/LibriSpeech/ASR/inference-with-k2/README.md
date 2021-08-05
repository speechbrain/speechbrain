# Inference with k2
In this repository, I try to combine [k2](https://github.com/k2-fsa/k2) with [speechbrain](https://github.com/speechbrain/speechbrain) to decoding well and fastly. And this repository is mainly to record and discuss the project that integrating k2 into speechbrain. **It is not the final conclusion.** 

**Notice:**, I just did a preliminary explore about integrating k2 into speechbrain. 
At the basis of the [codes](https://gist.github.com/csukuangfj/c68697cd144c8f063cc7ec4fd885fd6f) from csukuangfj (thank him!), I try to combine k2 with the pretrained transformer encoder from speechbrain and get some results on LibriSpeech. I use the public pretrained transformer encoder from the speechbrain team. I test the two datasets' samples (test-clean and test-other) one by one.

Some results I get are as follows (WER and Duration, based on 1 GPU):
``` 
                             Method                             |  test-clean(WER%) |  test-clean (h:m:s)  |  test-other(WER%)
------------------------------------------------------------------------------------------------------------------------------
    speechbrain (public, lm_weight=0.6, ctc_weight=0.52, bs=66) |       2.46        |            /         |       5.77
------------------------------------------------------------------------------------------------------------------------------
 speechbrain (reproduce, lm_weight=0.6, ctc_weight=0.52, bs=66) |       2.52        |        02:33:18      |       5.93
------------------------------------------------------------------------------------------------------------------------------
  speechbrain (reproduce, lm_weight=0.0, ctc_weight=0.0, bs=1)  |       4.43        |        00:11:26      |       10.01
------------------------------------------------------------------------------------------------------------------------------
                pre-encoder-output+softmax+greedy               |       17.42       |        00:02:30      |       23.38
------------------------------------------------------------------------------------------------------------------------------
                  k2_ctc_topo+pre-encoder (bs=8)                |       5.88        |        00:14:00      |       13.82
------------------------------------------------------------------------------------------------------------------------------
    k2_HLG+pre-encoder (use-whole-lattice=True,lm-scale=0.3)    |       4.63        |        00:11:43      |       10.93
------------------------------------------------------------------------------------------------------------------------------
```
 Prepare some packages:

 1. You can install k2 based on this [html](https://k2.readthedocs.io/en/latest/index.html).
 2. You also should install snowfall as follows:
 ```
 git clone https://github.com/k2-fsa/snowfall.git
 cd snowfall
 python setup.py install
 ```
 3. Using pip to install some other necessary packages, such as os, glob, torchaudio, kaldialm and so on.
```
pip install torchaudio
```
 

How to run:
```
bash run.sh
```

Some decoding results: ([all-results](https://drive.google.com/drive/folders/1s1dWtfgBvyziakuNhf4L7QmGglXRv2Ig?usp=sharing))

```
%WER 4.63 [ 2432 / 52576, 212 ins, 530 del, 1690 sub ]
%SER 48.32 [ 1266 / 2620 ]
Scored 2620 sentences, 0 not present in hyp.
================================================================================
ALIGNMENTS

Format:
<utterance-id>, WER DETAILS
<eps> ; reference  ; on ; the ; first ;  line
  I   ;     S      ; =  ;  =  ;   S   ;   D  
 and  ; hypothesis ; on ; the ; third ; <eps>
================================================================================
6, %WER 0.00 [ 0 / 8, 0 ins, 0 del, 0 sub ]
CONCORD ; RETURNED ; TO ; ITS ; PLACE ; AMIDST ; THE ; TENTS
   =    ;    =     ; =  ;  =  ;   =   ;   =    ;  =  ;   =  
CONCORD ; RETURNED ; TO ; ITS ; PLACE ; AMIDST ; THE ; TENTS
================================================================================
6, %WER 2.33 [ 1 / 43, 0 ins, 0 del, 1 sub ]
THE ; ENGLISH ; FORWARDED ; TO ; THE ; FRENCH ; BASKETS ; OF ; FLOWERS ; OF ; WHICH ; THEY ; HAD ; MADE ; A ; PLENTIFUL ; PROVISION ; TO ; GREET ; THE ; ARRIVAL ; OF ; THE ; YOUNG ; PRINCESS ; THE ; FRENCH ; IN ; RETURN ; INVITED ; THE ; ENGLISH ; TO ; A ; SUPPER ; WHICH ; WAS ; TO ; BE ; GIVEN ; THE ; NEXT ; DAY
 =  ;    =    ;     S     ; =  ;  =  ;   =    ;    =    ; =  ;    =    ; =  ;   =   ;  =   ;  =  ;  =   ; = ;     =     ;     =     ; =  ;   =   ;  =  ;    =    ; =  ;  =  ;   =   ;    =     ;  =  ;   =    ; =  ;   =    ;    =    ;  =  ;    =    ; =  ; = ;   =    ;   =   ;  =  ; =  ; =  ;   =   ;  =  ;  =   ;  = 
THE ; ENGLISH ;   FOLDED  ; TO ; THE ; FRENCH ; BASKETS ; OF ; FLOWERS ; OF ; WHICH ; THEY ; HAD ; MADE ; A ; PLENTIFUL ; PROVISION ; TO ; GREET ; THE ; ARRIVAL ; OF ; THE ; YOUNG ; PRINCESS ; THE ; FRENCH ; IN ; RETURN ; INVITED ; THE ; ENGLISH ; TO ; A ; SUPPER ; WHICH ; WAS ; TO ; BE ; GIVEN ; THE ; NEXT ; DAY
================================================================================
6, %WER 9.09 [ 1 / 11, 0 ins, 0 del, 1 sub ]
CONGRATULATIONS ; WERE ; POURED ; IN ; UPON ; THE ; PRINCESS ; EVERYWHERE ; DURING ; HER ; JOURNEY
       =        ;  =   ;   =    ; =  ;  =   ;  =  ;    =     ;     S      ;   =    ;  =  ;    =   
CONGRATULATIONS ; WERE ; POURED ; IN ; UPON ; THE ; PRINCESS ;   WHERE    ; DURING ; HER ; JOURNEY
================================================================================
6, %WER 1.56 [ 1 / 64, 0 ins, 0 del, 1 sub ]
FROM ; THE ; RESPECT ; PAID ; HER ; ON ; ALL ; SIDES ; SHE ; SEEMED ; LIKE ; A ; QUEEN ; AND ; FROM ; THE ; ADORATION ; WITH ; WHICH ; SHE ; WAS ; TREATED ; BY ; TWO ; OR ; THREE ; SHE ; APPEARED ; AN ; OBJECT ; OF ; WORSHIP ; THE ; QUEEN ; MOTHER ; GAVE ; THE ; FRENCH ; THE ; MOST ; AFFECTIONATE ; RECEPTION ; FRANCE ; WAS ; HER ; NATIVE ; COUNTRY ; AND ; SHE ; HAD ; SUFFERED ; TOO ; MUCH ; UNHAPPINESS ; IN ; ENGLAND ; FOR ; ENGLAND ; TO ; HAVE ; MADE ; HER ; FORGET ; FRANCE
 =   ;  =  ;    =    ;  =   ;  =  ; =  ;  =  ;   =   ;  =  ;   =    ;  =   ; = ;   =   ;  =  ;  =   ;  =  ;     =     ;  =   ;   =   ;  =  ;  =  ;    =    ; =  ;  =  ; =  ;   =   ;  =  ;    =     ; =  ;   =    ; =  ;    =    ;  =  ;   =   ;   =    ;  =   ;  =  ;   =    ;  =  ;  =   ;      =       ;     =     ;   =    ;  =  ;  =  ;   =    ;    =    ;  =  ;  =  ;  =  ;    =     ;  =  ;  =   ;      S      ; =  ;    =    ;  =  ;    =    ; =  ;  =   ;  =   ;  =  ;   =    ;   =   
FROM ; THE ; RESPECT ; PAID ; HER ; ON ; ALL ; SIDES ; SHE ; SEEMED ; LIKE ; A ; QUEEN ; AND ; FROM ; THE ; ADORATION ; WITH ; WHICH ; SHE ; WAS ; TREATED ; BY ; TWO ; OR ; THREE ; SHE ; APPEARED ; AN ; OBJECT ; OF ; WORSHIP ; THE ; QUEEN ; MOTHER ; GAVE ; THE ; FRENCH ; THE ; MOST ; AFFECTIONATE ; RECEPTION ; FRANCE ; WAS ; HER ; NATIVE ; COUNTRY ; AND ; SHE ; HAD ; SUFFERED ; TOO ; MUCH ;  HAPPINESS  ; IN ; ENGLAND ; FOR ; ENGLAND ; TO ; HAVE ; MADE ; HER ; FORGET ; FRANCE
================================================================================
6, %WER 6.45 [ 2 / 31, 0 ins, 0 del, 2 sub ]
SHE ; TAUGHT ; HER ; DAUGHTER ; THEN ; BY ; HER ; OWN ; AFFECTION ; FOR ; IT ; THAT ; LOVE ; FOR ; A ; COUNTRY ; WHERE ; THEY ; HAD ; BOTH ; BEEN ; HOSPITABLY ; RECEIVED ; AND ; WHERE ; A ; BRILLIANT ; FUTURE ; OPENED ; BEFORE ; THEM
 =  ;   =    ;  =  ;    =     ;  =   ; =  ;  =  ;  =  ;     =     ;  =  ; =  ;  =   ;  =   ;  =  ; = ;    =    ;   =   ;  =   ;  =  ;  =   ;  =   ;     =      ;    =     ;  =  ;   =   ; = ;     =     ;   =    ;   S    ;   S    ;  =  
SHE ; TAUGHT ; HER ; DAUGHTER ; THEN ; BY ; HER ; OWN ; AFFECTION ; FOR ; IT ; THAT ; LOVE ; FOR ; A ; COUNTRY ; WHERE ; THEY ; HAD ; BOTH ; BEEN ; HOSPITABLY ; RECEIVED ; AND ; WHERE ; A ; BRILLIANT ; FUTURE ;  OPEN  ;  FOR   ; THEM
================================================================================
6, %WER 0.00 [ 0 / 33, 0 ins, 0 del, 0 sub ]
THE ; COUNT ; HAD ; THROWN ; HIMSELF ; BACK ; ON ; HIS ; SEAT ; LEANING ; HIS ; SHOULDERS ; AGAINST ; THE ; PARTITION ; OF ; THE ; TENT ; AND ; REMAINED ; THUS ; HIS ; FACE ; BURIED ; IN ; HIS ; HANDS ; WITH ; HEAVING ; CHEST ; AND ; RESTLESS ; LIMBS
 =  ;   =   ;  =  ;   =    ;    =    ;  =   ; =  ;  =  ;  =   ;    =    ;  =  ;     =     ;    =    ;  =  ;     =     ; =  ;  =  ;  =   ;  =  ;    =     ;  =   ;  =  ;  =   ;   =    ; =  ;  =  ;   =   ;  =   ;    =    ;   =   ;  =  ;    =     ;   =  
THE ; COUNT ; HAD ; THROWN ; HIMSELF ; BACK ; ON ; HIS ; SEAT ; LEANING ; HIS ; SHOULDERS ; AGAINST ; THE ; PARTITION ; OF ; THE ; TENT ; AND ; REMAINED ; THUS ; HIS ; FACE ; BURIED ; IN ; HIS ; HANDS ; WITH ; HEAVING ; CHEST ; AND ; RESTLESS ; LIMBS
================================================================================
6, %WER 0.00 [ 0 / 17, 0 ins, 0 del, 0 sub ]
THIS ; HAS ; INDEED ; BEEN ; A ; HARASSING ; DAY ; CONTINUED ; THE ; YOUNG ; MAN ; HIS ; EYES ; FIXED ; UPON ; HIS ; FRIEND
 =   ;  =  ;   =    ;  =   ; = ;     =     ;  =  ;     =     ;  =  ;   =   ;  =  ;  =  ;  =   ;   =   ;  =   ;  =  ;   =   
THIS ; HAS ; INDEED ; BEEN ; A ; HARASSING ; DAY ; CONTINUED ; THE ; YOUNG ; MAN ; HIS ; EYES ; FIXED ; UPON ; HIS ; FRIEND
================================================================================
6, %WER 11.11 [ 1 / 9, 0 ins, 1 del, 0 sub ]
YOU ; WILL ; BE ; FRANK ; WITH ; ME ; I ; ALWAYS ;   AM 
 =  ;  =   ; =  ;   =   ;  =   ; =  ; = ;   =    ;   D  
YOU ; WILL ; BE ; FRANK ; WITH ; ME ; I ; ALWAYS ; <eps>
================================================================================
6, %WER 9.09 [ 1 / 11, 0 ins, 0 del, 1 sub ]
CAN ; YOU ; IMAGINE ; WHY ; BUCKINGHAM ; HAS ; BEEN ; SO ; VIOLENT ; I ; SUSPECT
 =  ;  =  ;    =    ;  S  ;     =      ;  =  ;  =   ; =  ;    =    ; = ;    =   
CAN ; YOU ; IMAGINE ;  MY ; BUCKINGHAM ; HAS ; BEEN ; SO ; VIOLENT ; I ; SUSPECT
================================================================================
6, %WER 4.17 [ 1 / 24, 0 ins, 1 del, 0 sub ]
IT ; IS ; YOU ;  WHO  ; ARE ; MISTAKEN ; RAOUL ; I ; HAVE ; READ ; HIS ; DISTRESS ; IN ; HIS ; EYES ; IN ; HIS ; EVERY ; GESTURE ; AND ; ACTION ; THE ; WHOLE ; DAY
=  ; =  ;  =  ;   D   ;  =  ;    =     ;   =   ; = ;  =   ;  =   ;  =  ;    =     ; =  ;  =  ;  =   ; =  ;  =  ;   =   ;    =    ;  =  ;   =    ;  =  ;   =   ;  = 
IT ; IS ; YOU ; <eps> ; ARE ; MISTAKEN ; RAOUL ; I ; HAVE ; READ ; HIS ; DISTRESS ; IN ; HIS ; EYES ; IN ; HIS ; EVERY ; GESTURE ; AND ; ACTION ; THE ; WHOLE ; DAY
================================================================================
6, %WER 0.00 [ 0 / 6, 0 ins, 0 del, 0 sub ]
I ; CAN ; PERCEIVE ; LOVE ; CLEARLY ; ENOUGH
= ;  =  ;    =     ;  =   ;    =    ;   =   
I ; CAN ; PERCEIVE ; LOVE ; CLEARLY ; ENOUGH
```
