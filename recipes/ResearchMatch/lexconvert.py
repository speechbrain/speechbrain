#!/usr/bin/env python
# May be run with either Python 2 or Python 3

"""lexconvert v0.33 - convert phonemes between different speech synthesizers etc
(c) 2007-21 Silas S. Brown.  License: Apache 2"""

# Run without arguments for usage information

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Old versions of this code are being kept in the E-GuideDog SVN repository at
# http://svn.code.sf.net/p/e-guidedog/code/ssb22/lexconvert
# and on GitHub at https://github.com/ssb22/lexconvert
# and on GitLab at https://gitlab.com/ssb22/lexconvert
# and on Bitbucket https://bitbucket.org/ssb22/lexconvert
# and at https://gitlab.developers.cam.ac.uk/ssb22/lexconvert
# and in China at https://gitee.com/ssb22/lexconvert
# although some early ones are missing.

def Phonemes():
   """Create phonemes by calling vowel(), consonant(),
     variant() and other().
   
     For the variants, if a particular variant does not
     exist in the destination format then we will treat it
     as equivalent to the last non-variant we created.
  
     For anything else that does not exist in the
     destination format, we will first try to break the
     source's phoneme into parts (e.g. see the treatment
     of opt_ol_as_in_gold by eSpeak and bbcmicro), and if
     that still doesn't work then we drop a character
     (warning depending on the source format's setting of
     safe_to_drop_characters).  makeDic does however warn
     about any non-variant consonants, or non-variant
     vowels that weren't marked optional, missing from a
     format. """
   a_as_in_ah = vowel()
   _, var1_a_as_in_ah = variant()
   _, var3_a_as_in_ah = variant()
   _, var4_a_as_in_ah = variant()
   _, var5_a_as_in_ah = variant()
   a_as_in_apple = vowel()
   u_as_in_but = vowel() # or the first part of un as in hunt
   _, var1_u_as_in_but = variant()
   o_as_in_orange = vowel()
   _, var1_o_as_in_orange = variant()
   _, var2_o_as_in_orange = variant()
   o_as_in_now = vowel()
   _, var1_o_as_in_now = variant()
   a_as_in_ago = vowel()
   _, var1_a_as_in_ago = variant()
   e_as_in_herd = vowel()
   _, ar_as_in_year = variant()
   eye = vowel()
   _, var1_eye = variant()
   b = consonant()
   ch = consonant()
   d = consonant()
   th_as_in_them = consonant()
   e_as_in_them = vowel()
   _, var1_e_as_in_them = variant()
   a_as_in_air = vowel()
   _, var1_a_as_in_air = variant()
   _, var2_a_as_in_air = variant()
   _, var3_a_as_in_air = variant()
   _, var4_a_as_in_air = variant()
   a_as_in_ate = vowel()
   _, var1_a_as_in_ate = variant()
   f = consonant()
   g = consonant()
   h = consonant()
   i_as_in_it = vowel()
   _, var1_i_as_in_it = variant()
   _, var2_i_as_in_it = variant()
   ear = vowel()
   _, var1_ear = variant()
   _, var2_ear = variant()
   e_as_in_eat = vowel()
   _, var1_e_as_in_eat = variant()
   j_as_in_jump = consonant()
   k = consonant()
   _, opt_scottish_loch = variant()
   l = consonant()
   _, var1_l = variant()
   m = consonant()
   n = consonant()
   ng = consonant()
   o_as_in_go = vowel()
   _, var1_o_as_in_go = variant()
   _, var2_o_as_in_go = variant()
   opt_ol_as_in_gold = opt_vowel() # see eSpeak / bbcmicro
   oy_as_in_toy = vowel()
   _, var1_oy_as_in_toy = variant()
   p = consonant()
   r = consonant()
   _, var1_r = variant()
   s = consonant()
   sh = consonant()
   t = consonant()
   _, var1_t = variant()
   th_as_in_think = consonant()
   oor_as_in_poor = vowel()
   _, var1_oor_as_in_poor = variant()
   _, opt_u_as_in_pull = variant()
   opt_ul_as_in_pull = opt_vowel() # see eSpeak / bbcmicro
   oo_as_in_food = vowel()
   _, var1_oo_as_in_food = variant()
   _, var2_oo_as_in_food = variant()
   close_to_or = vowel()
   _, var1_close_to_or = variant()
   _, var2_close_to_or = variant()
   _, var3_close_to_or = variant()
   v = consonant()
   w = consonant()
   _, var1_w = variant()
   y = consonant()
   z = consonant()
   ge_of_blige_etc = consonant()
   glottal_stop = other()
   syllable_separator = other()
   _, primary_stress = variant()
   _, secondary_stress = variant()
   text_sharp = other()
   text_underline = other()
   text_question = other()
   text_exclamation = other()
   text_comma = other()
   ipa_colon = other() # for catching missed cases
   del _ ; return locals()

def LexFormats():
  """Makes the phoneme conversion tables of each format.
     Each table has string to phoneme entries and phoneme
     to string entries.  The string to phoneme entries are
     used when converting OUT of that format, and the
     phoneme to string entries are used when converting IN
     (so you can recognise phonemes you don't support and
     convert them to something else).  By default, a tuple
     of the form (string,phoneme) will create entries in
     BOTH directions; one-directional entries are created
     via (string,phoneme,False) or (phoneme,string,False).
     The makeDic function checks the keys are unique.
     
     First parameter is always a description of the
     format, then come the phoneme entries as described
     above, then any additional settings:

       stress_comes_before_vowel (default False means any
       stress mark goes AFTER the affected vowel; set to
       True if the format requires stress placed before)

       word_separator (default same as phoneme_separator)
       phoneme_separator (default " ")
       clause_separator (default newline)

       (For a special case, clause_separator can also be
        set to a function.  If that happens, the function
        will be called whenever lexconvert needs to output
        a list of (lists of words) in this format.  See
        bbcmicro for an example function clause_separator)

       safe_to_drop_characters (default False, can be a
       string of safe characters or True = all; controls
       warnings when unrecognised characters are found)

       approximate_missing (default False) - if True,
       makeDic will attempt to compensate for missing
       phonemes by approximating them to others, instead of
       warning about them.  This is useful for American codes
       that can't cope with all the British English phonemes.
       (Approximation is done automatically anyway in the
       case of variant phonemes; approximate_missing adds in
       some additional approximations - see comments in code)

       cleanup_regexps (default none) - optional list of
       (search,replace) regular expressions to "clean up"
       after converting each word INTO this format
       cleanup_func (default none) - optional special-case
       function to pass result through after cleanup_regexps

       cvtOut_regexps (default none) - optional list of
       (search,replace) regular expressions to "clean up"
       before starting to convert OUT of this format
       cvtOut_func (default none) - optional special-case
       function to pass through before any cvtOut_regexps
  
       inline_format (default "%s") the format string for
       printing a word with --phones or --phones2phones
       (can be used to put markup around each word)
       (can also be a function taking the phonetic word
        and returning the resulting string, e.g. bbcmicro)

       output_is_binary (default False) - True if the output
       is almost certainly unsuitable for a terminal; will
       cause lexconvert to refuse to print phonemes unless
       its standard output is redirected to a file or pipe
       (affects the --phones and --phones2phones options)

       inline_header (default none) text to print first
         when outputting from --phones or --phones2phones
       inline_footer (default none) text to print last
       inline_oneoff_header (default none) text to print
         before inline_header on the first time only

       lex_filename - filename of a lexicon file.  If this
       is not specified, there is no support for writing a
       lexicon in this format: there can still be READ
       support if you define lex_read_function to open the
       lexicon by itself, but otherwise the format can be
       used only with --phones and --phones2phones.

       lex_entry_format - format string for writing each
       (word, pronunciation) entry to the lexicon file.
       This is also needed for lexicon-write support.

       lex_header, lex_footer - optional strings to write
       at the beginning and at the end of the lexicon file
       (can also be functions that take the open file as a
        parameter, e.g. for bbcmicro; lex_footer is
        allowed to close the file if it needs to do
        something with it afterwards)

       lex_word_case - optional "upper" or "lower" to
       force a particular case for lexicon words (not
       pronunciations - they're determined by the table).
       The default is to allow words to be in either case.

       lex_type (default "") - used by the --formats
       option when summarising the support for each format

       lex_read_function - Python function to READ the
       lexicon file and return a (word,phonemes) list.
       If this is not specified, there's no read support
       for lexicons in this format (but there can still be
       write support - see above - and you can still use
       --phones and --phones2phones).  If lex_filename is
       specified then this function will be given the open
       file as a parameter. """
  
  phonemes = Phonemes() ; globals().update(phonemes)
  return { "festival" : makeDic(
    "Festival's British voice",
    ('0',syllable_separator),
    ('1',primary_stress),
    ('2',secondary_stress),
    ('aa',a_as_in_ah),
    ('a',a_as_in_apple),
    ('uh',u_as_in_but),
    ('o',o_as_in_orange),
    ('au',o_as_in_now),
    ('@',a_as_in_ago),
    ('@@',e_as_in_herd),
    ('ai',eye),
    ('b',b),
    ('ch',ch),
    ('d',d),
    ('dh',th_as_in_them),
    ('e',e_as_in_them),
    (ar_as_in_year,'@@',False),
    ('e@',a_as_in_air),
    ('ei',a_as_in_ate),
    ('f',f),
    ('g',g),
    ('h',h),
    ('i',i_as_in_it),
    ('i@',ear),
    ('ii',e_as_in_eat),
    ('jh',j_as_in_jump),
    ('k',k),
    ('l',l),
    ('m',m),
    ('n',n),
    ('ng',ng),
    ('ou',o_as_in_go),
    ('oi',oy_as_in_toy),
    ('p',p),
    ('r',r),
    ('s',s),
    ('sh',sh),
    ('t',t),
    ('th',th_as_in_think),
    ('u@',oor_as_in_poor),
    ('u',opt_u_as_in_pull),
    ('uu',oo_as_in_food),
    ('oo',close_to_or),
    ('v',v),
    ('w',w),
    ('y',y),
    ('z',z),
    ('zh',ge_of_blige_etc),
    lex_filename=ifset("HOME",os.environ.get("HOME","")+os.sep)+".festivalrc",
    lex_entry_format="(lex.add.entry '( \"%s\" n %s))\n",
    lex_header=";; -*- mode: lisp -*-\n(eval (list voice_default))\n",
    lex_read_function = lambda *args:eval('['+getoutput("grep -vi parameter.set < ~/.festivalrc | grep -v '(eval' | sed -e 's/;.*//' -e 's/.lex.add.entry//' -e s/\"'\"'[(] *\"/[\"/' -e 's/\" [^ ]* /\",(\"/' -e 's/\".*$/&\"],/' -e 's/[()]/ /g' -e 's/  */ /g'")+']'),
    safe_to_drop_characters=True, # TODO: really? (could instead give a string of known-safe characters)
    cleanup_func = festival_group_stress,
  ),

  "example" : makeVariantDic(
    "A small built-in example lexicon for testing when you don't have your full custom lexicon to hand.  Use --convert to write it in one of the other formats and see if a synth can import it.",
    lex_read_function = lambda *args: [
       ("Shadrach","shei1drak"),
       ("Meshach","mii1shak"),
       ("Abednego","@be1dniigou"),
    ], cleanup_func = None,
    lex_filename=None, lex_entry_format=None, noInherit=True),

  "festival-cmu" : makeVariantDic(
    "American CMU version of Festival",
    ('ae',a_as_in_apple),
    ('ah',u_as_in_but),
    ('ax',a_as_in_ago),
    (o_as_in_orange,'aa',False),
    ('aw',o_as_in_now),
    ('er',e_as_in_herd), # TODO: check this one
    ('ay',eye),
    ('eh',e_as_in_them),
    (ar_as_in_year,'er',False),
    (a_as_in_air,'er',False),
    ('ey',a_as_in_ate),
    ('hh',h),
    ('ih',i_as_in_it),
    ('ey ah',ear),
    ('iy',e_as_in_eat),
    ('ow',o_as_in_go),
    ('oy',oy_as_in_toy),
    ('uh',oor_as_in_poor),
    ('uw',oo_as_in_food),
    ('ao',close_to_or),
  ),

  "espeak" : makeDic(
    "eSpeak's default British voice", # but eSpeak's phoneme representation isn't always that simple, hence the regexps at the end
    ('%',syllable_separator),
    ("'",primary_stress),
    (',',secondary_stress),
    # TODO: glottal_stop? (in regional pronunciations etc)
    ('A:',a_as_in_ah),
    ('A@',a_as_in_ah,False),
    ('A',var1_a_as_in_ah),
    ('a',a_as_in_apple),
    ('aa',a_as_in_apple,False),
    ('a2',a_as_in_apple,False), # TODO: this is actually an a_as_in_apple variant in espeak; festival @1 is not in mrpa PhoneSet
    ('&',a_as_in_apple,False),
    ('V',u_as_in_but),
    ('0',o_as_in_orange),
    ('aU',o_as_in_now),
    ('@',a_as_in_ago),
    ('a#',a_as_in_ago,False), # (TODO: eSpeak sometimes uses a# in 'had' when in a sentence, and this doesn't always sound good on other synths; might sometimes want to convert it to a_as_in_apple; not sure what contexts would call for this though)
    ('3:',e_as_in_herd),
    ('3',var1_a_as_in_ago),
    ('@2',a_as_in_ago,False),
    ('@-',a_as_in_ago,False), # (eSpeak @- sounds to me like a shorter version of @, TODO: double-check the relationship between @ and @2 in Festival)
    ('aI',eye),
    ('aI2',eye,False),
    ('aI;',eye,False),
    ('aI2;',eye,False),
    ('b',b),
    ('tS',ch),
    ('d',d),
    ('D',th_as_in_them),
    ('E',e_as_in_them),
    (ar_as_in_year,'3:',False),
    ('e@',a_as_in_air),
    ('eI',a_as_in_ate),
    ('f',f),
    ('g',g),
    ('h',h),
    ('I',i_as_in_it),
    ('I;',i_as_in_it,False),
    ('i',i_as_in_it,False),
    ('I2',var2_i_as_in_it,False),
    ('I2;',var2_i_as_in_it,False),
    ('i@',ear),
    ('i@3',var2_ear),
    ('i:',e_as_in_eat),
    ('i:;',e_as_in_eat,False),
    ('dZ',j_as_in_jump),
    ('k',k),
    ('x',opt_scottish_loch),
    ('l',l),
    ('L',l,False),
    ('m',m),
    ('n',n),
    ('N',ng),
    ('oU',o_as_in_go),
    ('oUl',opt_ol_as_in_gold), # (espeak says "gold" in a slightly 'posh' way though) (if dest format doesn't have opt_ol_as_in_gold, it'll get o_as_in_go + the l)
    ('OI',oy_as_in_toy),
    ('p',p),
    ('r',r),
    ('r-',r,False),
    ('s',s),
    ('S',sh),
    ('t',t),
    ('T',th_as_in_think),
    ('U@',oor_as_in_poor),
    ('U',opt_u_as_in_pull),
    ('@5',opt_u_as_in_pull,False),
    ('Ul',opt_ul_as_in_pull), # if dest format doesn't have this, it'll get opt_u_as_in_pull from the U, then the l
    ('u:',oo_as_in_food),
    ('O:',close_to_or),
    ('O@',var3_close_to_or),
    ('o@',var3_close_to_or,False),
    ('O',var3_close_to_or,False),
    ('v',v),
    ('w',w),
    ('j',y),
    ('z',z),
    ('Z',ge_of_blige_etc),
    lex_filename = "en_extra",
    lex_entry_format = "%s %s\n",
    lex_read_function = lambda lexfile: [x for x in [l.split()[:2] for l in lexfile.readlines()] if len(x)==2 and not '//' in x[0]],
    lex_footer=lambda f:(f.close(),os.system("espeak --compile=en")), # see also a bit of special-case code in mainopt_convert
    inline_format = "[[%s]]",
    word_separator=" ",phoneme_separator="",
    stress_comes_before_vowel=True,
    safe_to_drop_characters="_: !",
    cleanup_regexps=[
      ("k'a2n","k'@n"),
      ("ka2n","k@n"),
      ("gg","g"),
      ("@U","oU"), # (eSpeak uses oU to represent @U; difference is given by its accent parameters)
      ("([iU]|([AO]:))@r$","\1@"),
      ("([^e])@r",r"\1_remove_3"),("_remove_",""),
      # (r"([^iU]@)l",r"\1L") # only in older versions of espeak (not valid in more recent versions)
      ("rr$","r"),
      ("3:r$","3:"),
      ("%%+","%"),("^%",""),("%$",""),
      # TODO: 'declared' & 'declare' the 'r' after the 'E' sounds a bit 'regional' (but pretty).  but sounds incomplete w/out 'r', and there doesn't seem to be an E2 or E@
      # TODO: consider adding 'g' to words ending in 'N' (if want the 'g' pronounced in '-ng' words) (however, careful of words like 'yankee' where the 'g' would be followed by a 'k'; this may also be a problem going into the next word)
    ],
     cvtOut_regexps = [
       ("e@r$","e@"), ("e@r([bdDfghklmnNprsStTvwjzZ])",r"e@\1"), # because the 'r' is implicit in other synths (but DO have it if there's another vowel to follow)
     ],
  ),

  "sapi" : makeDic(
    "Microsoft Speech API (American English)",
    ('-',syllable_separator),
    ('1',primary_stress),
    ('2',secondary_stress),
    ('aa',a_as_in_ah),
    ('ae',a_as_in_apple),
    ('ah',u_as_in_but),
    ('ao',o_as_in_orange),
    ('aw',o_as_in_now),
    ('ax',a_as_in_ago),
    ('er',e_as_in_herd),
    ('ay',eye),
    ('b',b),
    ('ch',ch),
    ('d',d),
    ('dh',th_as_in_them),
    ('eh',e_as_in_them),
    ('ey',var1_e_as_in_them),
    (a_as_in_ate,'ey',False),
    ('f',f),
    ('g',g),
    ('h',h), # Jan suggested 'hh', but I can't get this to work on Windows XP (TODO: try newer versions of Windows)
    ('ih',i_as_in_it),
    ('iy',e_as_in_eat),
    ('jh',j_as_in_jump),
    ('k',k),
    ('l',l),
    ('m',m),
    ('n',n),
    ('ng',ng),
    ('ow',o_as_in_go),
    ('oy',oy_as_in_toy),
    ('p',p),
    ('r',r),
    ('s',s),
    ('sh',sh),
    ('t',t),
    ('th',th_as_in_think),
    ('uh',oor_as_in_poor),
    ('uw',oo_as_in_food),
    ('AO',close_to_or),
    ('v',v),
    ('w',w),
    # ('x',var1_w), # suggested by Jan, but I can't get this to work on Windows XP (TODO: try newer versions of Windows)
    ('y',y),
    ('z',z),
    ('zh',ge_of_blige_etc),
    approximate_missing=True,
    lex_filename="run-ptts.bat", # write-only for now
    lex_header = "rem  You have to run this file\nrem  with ptts.exe in the same directory\nrem  to add these words to the SAPI lexicon\n\n",
    lex_entry_format='ptts -la %s "%s"\n',
    inline_format = '<pron sym="%s"/>',
    safe_to_drop_characters=True, # TODO: really?
  ),

  "cepstral" : makeDic(
    "Cepstral's British English SSML phoneset",
    ('0',syllable_separator),
    ('1',primary_stress),
    ('a',a_as_in_ah),
    ('ae',a_as_in_apple),
    ('ah',u_as_in_but),
    ('oa',o_as_in_orange),
    ('aw',o_as_in_now),
    ('er',e_as_in_herd),
    ('ay',eye),
    ('b',b),
    ('ch',ch),
    ('d',d),
    ('dh',th_as_in_them),
    ('eh',e_as_in_them),
    ('e@',a_as_in_air),
    ('ey',a_as_in_ate),
    ('f',f),
    ('g',g),
    ('h',h),
    ('ih',i_as_in_it),
    ('i',e_as_in_eat),
    ('jh',j_as_in_jump),
    ('k',k),
    ('l',l),
    ('m',m),
    ('n',n),
    ('ng',ng),
    ('ow',o_as_in_go),
    ('oy',oy_as_in_toy),
    ('p',p),
    ('r',r),
    ('s',s),
    ('sh',sh),
    ('t',t),
    ('th',th_as_in_think),
    ('uh',oor_as_in_poor),
    ('uw',oo_as_in_food),
    ('ao',close_to_or),
    ('v',v),
    ('w',w),
    ('j',y),
    ('z',z),
    ('zh',ge_of_blige_etc),
    approximate_missing=True,
    lex_filename="lexicon.txt",
    lex_entry_format = "%s 0 %s\n",
    lex_read_function = lambda lexfile: [(word,pronunc) for word, ignore, pronunc in [l.split(None,2) for l in lexfile.readlines()]],
    lex_word_case = "lower",
    inline_format = "<phoneme ph='%s'>p</phoneme>",
    safe_to_drop_characters=True, # TODO: really?
    cleanup_regexps=[(" 1","1"),(" 0","0")],
  ),

  "mac" : makeDic(
    "approximation in American English using the [[inpt PHON]] notation of Apple's US voices",
    ('=',syllable_separator),
    ('1',primary_stress),
    ('2',secondary_stress),
    ('AA',a_as_in_ah),
    ('aa',var5_a_as_in_ah),
    ('AE',a_as_in_apple),
    ('UX',u_as_in_but),
    (o_as_in_orange,'AA',False),
    ('AW',o_as_in_now),
    ('AX',a_as_in_ago),
    (e_as_in_herd,'AX',False), # TODO: is this really the best approximation?
    ('AY',eye),
    ('b',b),
    ('C',ch),
    ('d',d),
    ('D',th_as_in_them),
    ('EH',e_as_in_them),
    ('EY',a_as_in_ate),
    ('f',f),
    ('g',g),
    ('h',h),
    ('IH',i_as_in_it),
    ('IX',var2_i_as_in_it),
    ('IY',e_as_in_eat),
    ('J',j_as_in_jump),
    ('k',k),
    ('l',l),
    ('m',m),
    ('n',n),
    ('N',ng),
    ('OW',o_as_in_go),
    ('OY',oy_as_in_toy),
    ('p',p),
    ('r',r),
    ('s',s),
    ('S',sh),
    ('t',t),
    ('T',th_as_in_think),
    ('UH',oor_as_in_poor),
    ('UW',oo_as_in_food),
    ('AO',close_to_or),
    ('v',v),
    ('w',w),
    ('y',y),
    ('z',z),
    ('Z',ge_of_blige_etc),
    approximate_missing=True,
    lex_filename="substitute.sh", # write-only for now
    lex_type = "substitution script",
    lex_header = "#!/bin/bash\n\n# I don't yet know how to add to the Apple US lexicon,\n# so here is a 'sed' command you can run on your text\n# to put the pronunciation inline:\n\nsed -E -e :S \\\n",
    lex_entry_format=r" -e 's/(^|[^A-Za-z])%s($|[^A-Za-z[12=])/\1[[inpt PHON]]%s[[inpt TEXT]]\2/g'"+" \\\n",
    # but /g is non-overlapping matches and won't catch 2 words in the lex right next to each other with only one non-alpha in between, so we put :S at start and tS at end to make the whole operation repeat until it hasn't done any more substitutions (hence also the exclusion of [, 1, 2 or = following a word so it doesn't try to substitute stuff inside the phonemes; TODO: assert the lexicon does not contain "inpt", "PHON" or "TEXT")
    lex_footer = lambda f:(f.write(" -e tS\n"),f.close(),os.chmod("substitute.sh",493)), # 493 = 0755, but no way to specify octal that works on both Python 2.5 and Python 3 (0o works on 2.6+)
    inline_format = "[[inpt PHON]]%s[[inpt TEXT]]",
    word_separator=" ",phoneme_separator="",
    safe_to_drop_characters=True, # TODO: really?
  ),

  "mac-uk" : makeDic(
    "Scansoft/Nuance British voices in Mac OS 10.7+ (system lexicon editing required, see --mac-uk option)",
    ('.',syllable_separator),
    ("'",primary_stress),
    (secondary_stress,'',False),
    ('A',a_as_in_ah),
    ('@',a_as_in_apple),
    ('$',u_as_in_but),
    (a_as_in_ago,'$',False),
    ('A+',o_as_in_orange),
    ('a&U',o_as_in_now),
    ('E0',e_as_in_herd),
    ('a&I',eye),
    ('b',b),
    ('t&S',ch),
    ('d',d),
    ('D',th_as_in_them),
    ('E',e_as_in_them),
    ('0',ar_as_in_year),
    ('E&$',a_as_in_air),
    ('e&I',a_as_in_ate),
    ('f',f),
    ('g',g),
    ('h',h),
    ('I',i_as_in_it),
    ('I&$',ear),
    ('i',e_as_in_eat),
    ('d&Z',j_as_in_jump),
    ('k',k),
    ('l',l),
    ('m',m),
    ('n',n),
    ('nK',ng),
    ('o&U',o_as_in_go),
    ('O&I',oy_as_in_toy),
    ('p',p),
    ('R+',r),
    ('s',s),
    ('S',sh),
    ('t',t),
    ('T',th_as_in_think),
    ('O',oor_as_in_poor),
    ('U',opt_u_as_in_pull),
    ('u',oo_as_in_food),
    (close_to_or,'O',False),
    ('v',v),
    ('w',w),
    ('j',y),
    ('z',z),
    ('Z',ge_of_blige_etc),
    # lex_filename not set (mac-uk code does not permanently save the lexicon; see --mac-uk option to read text)
    lex_read_function = lambda *args:[(w,p) for w,_,p in MacBritish_System_Lexicon(False,os.environ.get("MACUK_VOICE","Daniel")).usable_words()],
    inline_oneoff_header = "(mac-uk phonemes output is for information only; you'll need the --mac-uk or --trymac-uk options to use it)\n",
    word_separator=" ",phoneme_separator="",
    stress_comes_before_vowel=True,
    safe_to_drop_characters=True, # TODO: really?
    cleanup_regexps=[(r'o\&U\.Ol', r'o\&Ul')],
  ),

  "x-sampa" : makeDic(
    "General X-SAMPA notation, contributed by Jan Weiss",
    ('.',syllable_separator),
    ('"',primary_stress),
    ('%',secondary_stress),
    ('A',a_as_in_ah),
    (':',ipa_colon),
    ('A:',var3_a_as_in_ah),
    ('Ar\\',var4_a_as_in_ah),
    ('a:',var5_a_as_in_ah),
    ('{',a_as_in_apple),
    ('V',u_as_in_but),
    ('Q',o_as_in_orange),
    (var1_o_as_in_orange,'A',False),
    ('O',var2_o_as_in_orange),
    ('aU',o_as_in_now),
    ('{O',var1_o_as_in_now),
    ('@',a_as_in_ago),
    ('3:',e_as_in_herd),
    ('aI',eye),
    ('Ae',var1_eye),
    ('b',b),
    ('tS',ch),
    ('d',d),
    ('D',th_as_in_them),
    ('E',e_as_in_them),
    ('e',var1_e_as_in_them),
    (ar_as_in_year,'3:',False),
    ('E@',a_as_in_air),
    ('Er\\',var1_a_as_in_air),
    ('e:',var2_a_as_in_air),
    ('E:',var3_a_as_in_air),
    ('e@',var4_a_as_in_air),
    ('eI',a_as_in_ate),
    ('{I',var1_a_as_in_ate),
    ('f',f),
    ('g',g),
    ('h',h),
    ('I',i_as_in_it),
    ('1',var1_i_as_in_it),
    ('I@',ear),
    ('Ir\\',var1_ear),
    ('i',e_as_in_eat),
    ('i:',var1_e_as_in_eat),
    ('dZ',j_as_in_jump),
    ('k',k),
    ('x',opt_scottish_loch),
    ('l',l),
    ('m',m),
    ('n',n),
    ('N',ng),
    ('@U',o_as_in_go),
    ('oU',var2_o_as_in_go),
    ('@}',var1_u_as_in_but),
    ('OI',oy_as_in_toy),
    ('oI',var1_oy_as_in_toy),
    ('p',p),
    ('r\\',r),
    (var1_r,'r',False),
    ('s',s),
    ('S',sh),
    ('t',t),
    ('T',th_as_in_think),
    ('U@',oor_as_in_poor),
    ('Ur\\',var1_oor_as_in_poor),
    ('U',opt_u_as_in_pull),
    ('}:',oo_as_in_food),
    ('u:',var1_oo_as_in_food),
    (var2_oo_as_in_food,'u:',False),
    ('O:',close_to_or),
    (var1_close_to_or,'O',False),
    ('o:',var2_close_to_or),
    ('v',v),
    ('w',w),
    ('W',var1_w),
    ('j',y),
    ('z',z),
    ('Z',ge_of_blige_etc),
    lex_filename="acapela.txt",
    lex_entry_format = "%s\t#%s\tUNKNOWN\n", # TODO: may be able to convert part-of-speech (NOUN etc) to/from some other formats e.g. Festival
    lex_read_function=lambda lexfile:[(word,pronunc.lstrip("#")) for word, pronunc, ignore in [l.split(None,2) for l in lexfile.readlines()]],
    # TODO: inline_format ?
    word_separator=" ",phoneme_separator="",
    safe_to_drop_characters=True, # TODO: really?
  ),
  "vocaloid" : makeVariantDic(
     "X-SAMPA phonemes for Yamaha's Vocaloid singing synthesizer.  Contributed by Lorenzo Gatti, who tested in Vocaloid 4 using two American English voices.",
     ('-',syllable_separator),
     (primary_stress,'',False), # not used by Vocaloid
     (secondary_stress,'',False),
     ('Q',a_as_in_ah),
     (var3_a_as_in_ah,'Q',False),
     (var4_a_as_in_ah,'Q',False),
     (var5_a_as_in_ah,'Q',False),
     ('O@',o_as_in_orange),
     (var1_o_as_in_orange,'O@',False),
     (var2_o_as_in_orange, 'O@',False),
     ('@U',o_as_in_now),
     ('@r',e_as_in_herd),
     (var1_eye, 'aI',False),
     ('e',e_as_in_them),
     ('I@',ar_as_in_year),
     ('e@',a_as_in_air),
     (var1_a_as_in_air, 'e@',False),
     (var2_a_as_in_air, 'e@',False),
     (var3_a_as_in_air, 'e@',False),
     (var4_a_as_in_air, 'e@',False),
     (var1_a_as_in_ate, 'eI', False),
     (var1_i_as_in_it, 'I',False),
     (var1_ear, 'I@',False),
     ('i:',e_as_in_eat),
     (var1_e_as_in_eat, 'i:',False),
     (var2_o_as_in_go, '@U', False),
     ('V', var1_u_as_in_but),
     (var1_oy_as_in_toy, 'OI',False),
     ('r',r),
     ('th',t),
     (var1_oor_as_in_poor, '@U',False),
     ('u:',oo_as_in_food),
     (var1_oo_as_in_food, 'u:',False),
     (var1_close_to_or,'O:',False),
     (var2_close_to_or,'O:',False),
     (var1_w, 'w', False),
     lex_filename="vocaloid.txt",
     phoneme_separator=" ",
     noInherit=True
  ),
  "android-pico" : makeVariantDic(
    'X-SAMPA phonemes for the default \"Pico\" voice in Android (1.6+, American), wrapped in Java code', # you could put en-GB instead of en-US, but it must be installed on the phone
    ('A:',a_as_in_ah), # won't sound without the :
    (var5_a_as_in_ah,'A:',False), # a: won't sound
    ('@U:',o_as_in_go),
    ('I',var1_i_as_in_it), # '1' won't sound
    ('i:',e_as_in_eat), # 'i' won't sound
    ('u:',oo_as_in_food), # }: won't sound
    ('a_I',eye),('a_U',o_as_in_now),('e_I',a_as_in_ate),('O_I',oy_as_in_toy),(var1_oy_as_in_toy,'O_I',False),('o_U',var2_o_as_in_go),
    cleanup_regexps=[(r'\\',r'\\\\'),('"','&quot;'),('::',':')],
    lex_filename="",lex_entry_format="",
    lex_read_function=None,
    inline_oneoff_header=r'class Speak { public static void speak(android.app.Activity a,String s) { class OnInit implements android.speech.tts.TextToSpeech.OnInitListener { public OnInit(String s) { this.s = s; } public void onInit(int i) { mTts.speak(this.s, android.speech.tts.TextToSpeech.QUEUE_ADD, null); } private String s; }; if(mTts==null) mTts=new android.speech.tts.TextToSpeech(a,new OnInit(s),"com.svox.pico"); else mTts.speak(s, android.speech.tts.TextToSpeech.QUEUE_ADD, null); } private static android.speech.tts.TextToSpeech mTts = null; };'+'\n',
    inline_header=r'Speak.speak(this,"<speak xml:lang=\"en-US\">',
    inline_format=r'<phoneme alphabet=\"xsampa\" ph=\"%s\"/>',
    clause_separator=r".\n", # note r"\n" != "\n"
    inline_footer='</speak>");',
  ),

  "acapela-uk" : makeDic(
    'Acapela-optimised X-SAMPA for UK English voices (e.g. "Peter"), contributed by Jan Weiss',
    ('.',syllable_separator),('"',primary_stress),('%',secondary_stress), # copied from "x-sampa", not tested
    ('A:',a_as_in_ah),
    ('{',a_as_in_apple),
    ('V',u_as_in_but),
    ('Q',o_as_in_orange),
    ('A',var1_o_as_in_orange),
    ('O',var2_o_as_in_orange),
    ('aU',o_as_in_now),
    ('{O',var1_o_as_in_now),
    ('@',a_as_in_ago),
    ('3:',e_as_in_herd),
    ('aI',eye),
    ('A e',var1_eye),
    ('b',b),
    ('t S',ch),
    ('d',d),
    ('D',th_as_in_them),
    ('e',e_as_in_them),
    (ar_as_in_year,'3:',False),
    ('e @',a_as_in_air),
    ('e r',var1_a_as_in_air),
    ('e :',var2_a_as_in_air),
    (var3_a_as_in_air,'e :',False),
    ('eI',a_as_in_ate),
    ('{I',var1_a_as_in_ate),
    ('f',f),
    ('g',g),
    ('h',h),
    ('I',i_as_in_it),
    ('1',var1_i_as_in_it),
    ('I@',ear),
    ('I r',var1_ear),
    ('i',e_as_in_eat),
    ('i:',var1_e_as_in_eat),
    ('dZ',j_as_in_jump),
    ('k',k),
    ('x',opt_scottish_loch),
    ('l',l),
    ('m',m),
    ('n',n),
    ('N',ng),
    ('@U',o_as_in_go),
    ('o U',var2_o_as_in_go),
    ('@ }',var1_u_as_in_but),
    ('OI',oy_as_in_toy),
    ('o I',var1_oy_as_in_toy),
    ('p',p),
    ('r',r),
    ('s',s),
    ('S',sh),
    ('t',t),
    ('T',th_as_in_think),
    ('U@',oor_as_in_poor),
    ('U r',var1_oor_as_in_poor),
    ('U',opt_u_as_in_pull),
    ('u:',oo_as_in_food),
    ('O:',close_to_or),
    (var1_close_to_or,'O',False),
    ('v',v),
    ('w',w),
    ('j',y),
    ('z',z),
    ('Z',ge_of_blige_etc),
    lex_filename="acapela.txt",
    lex_entry_format = "%s\t#%s\tUNKNOWN\n", # TODO: part-of-speech (as above)
    lex_read_function=lambda lexfile:[(word,pronunc.lstrip("#")) for word, pronunc, ignore in [l.split(None,2) for l in lexfile.readlines()]],
    inline_format = "\\Prn=%s\\",
    safe_to_drop_characters=True, # TODO: really?
  ),

  "cmu" : makeDic(
    'format of the US-English Carnegie Mellon University Pronouncing Dictionary, contributed by Jan Weiss', # http://www.speech.cs.cmu.edu/cgi-bin/cmudict
    ('0',syllable_separator),
    ('1',primary_stress),
    ('2',secondary_stress),
    ('AA',a_as_in_ah),
    (var1_a_as_in_ah,'2',False),
    (ipa_colon,'1',False),
    ('AE',a_as_in_apple),
    ('AH',u_as_in_but),
    (o_as_in_orange,'AA',False),
    ('AW',o_as_in_now),
    (a_as_in_ago,'AH',False), # seems they don't use AX as festival-cmu does
    ('ER',e_as_in_herd), # TODO: check this one
    ('AY',eye),
    ('B',b),
    ('CH',ch),
    ('D',d),
    ('DH',th_as_in_them),
    ('EH',e_as_in_them),
    (ar_as_in_year,'ER',False),
    (a_as_in_air,'ER',False),
    ('EY',a_as_in_ate),
    ('F',f),
    ('G',g),
    ('HH',h),
    ('IH',i_as_in_it),
    ('EY AH',ear),
    ('IY',e_as_in_eat),
    ('JH',j_as_in_jump),
    ('K',k),
    ('L',l),
    ('M',m),
    ('N',n),
    ('NG',ng),
    ('OW',o_as_in_go),
    ('OY',oy_as_in_toy),
    ('P',p),
    ('R',r),
    ('S',s),
    ('SH',sh),
    ('T',t),
    ('TH',th_as_in_think),
    ('UH',oor_as_in_poor),
    ('UW',oo_as_in_food),
    ('AO',close_to_or),
    ('V',v),
    ('W',w),
    ('Y',y),
    ('Z',z),
    ('ZH',ge_of_blige_etc),
    # lex_filename not set (does CMU have a lex file?)
    safe_to_drop_characters=True, # TODO: really?
  ),

  # BEGIN PRE-32bit ERA SYNTHS (TODO: add an attribute to JS-hide them by default in HTML?  what about the SpeakJet which probably isn't a 32-bit chip but is post 32-bit era?  and then what about the 'approximation' formats - kana etc - would they need hiding by default also?  maybe best to just leave it)
  "apollo" : makeDic(
    'Dolphin Apollo 2 serial-port and parallel-port hardware synthesizers (in case anybody still uses those)',
    (syllable_separator,'',False), # I don't think the Apollo had anything to mark stress; TODO: control the pitch instead like bbcmicro ?
    ('_QQ',syllable_separator,False), # a slight pause
    ('_AA',a_as_in_apple),
    ('_AI',a_as_in_ate),
    ('_AR',a_as_in_ah),
    ('_AW',close_to_or),
    ('_A',a_as_in_ago),
    ('_B',b),
    ('_CH',ch),
    ('_D',d),
    ('_DH',th_as_in_them),
    ('_EE',e_as_in_eat),
    ('_EI',a_as_in_air),
    ('_ER',e_as_in_herd),
    ('_E',e_as_in_them),
    ('_F',f),
    ('_G',g),
    ('_H',h),
    ('_IA',ear),
    ('_IE',eye),
    ('_I',i_as_in_it),
    ('_J',j_as_in_jump),
    ('_K',k),
    ('_KK',k,False), # sCHool
    ('_L',l),
    ('_M',m),
    ('_NG',ng),
    ('_N',n),
    ('_OA',o_as_in_go),
    ('_OO',opt_u_as_in_pull),
    ('_OR',var3_close_to_or),
    ('_OW',o_as_in_now),
    ('_OY',oy_as_in_toy),
    ('_O',o_as_in_orange),
    ('_P',p),
    ('_PP',p,False), # sPeech (a stronger P ?)
    # _Q = k w - done by cleanup_regexps below
    ('_R',r),
    ('_SH',sh),
    ('_S',s),
    ('_TH',th_as_in_think),
    ('_T',t), ('_TT',t,False),
    ('_UU',oo_as_in_food),
    ('_U',u_as_in_but),
    ('_V',v),
    ('_W',w),
    # _X = k s - done by cleanup_regexps below
    ('_Y',y),
    ('_ZH',ge_of_blige_etc),
    ('_Z',z),
    # lex_filename not set (the hardware doesn't have one; HAL has an "exceptions dictionary" but I don't know much about it)
    approximate_missing=True,
    safe_to_drop_characters=True, # TODO: really?
    word_separator=" ",phoneme_separator="",
    cleanup_regexps=[('_K_W','_Q'),('_K_S','_X')],
    cvtOut_regexps=[('_Q','_K_W'),('_X','_K_S')],
  ),
  "dectalk" : makeDic(
    'DECtalk hardware synthesizers (American English)', # (1984-ish serial port; later ISA cards)
    (syllable_separator,'',False),
    ("'",primary_stress),
    ('aa',o_as_in_orange),
    ('ae',a_as_in_apple),
    ('ah',u_as_in_but),
    ('ao',close_to_or), # bought
    ('aw',o_as_in_now),
    ('ax',a_as_in_ago),
    ('ay',eye),
    ('b',b),
    ('ch',ch),
    ('d',d), ('dx',d,False),
    ('dh',th_as_in_them),
    ('eh',e_as_in_them),
    ('el',l,False), # -le of bottle, allophone ?
    # TODO: en: -on of button (2 phonemes?)
    ('ey',a_as_in_ate),
    ('f',f),
    ('g',g),
    ('hx',h),
    ('ih',i_as_in_it), ('ix',i_as_in_it,False),
    ('iy',e_as_in_eat), ('q',e_as_in_eat,False),
    ('jh',j_as_in_jump),
    ('k',k),
    ('l',l), ('lx',l,False),
    ('m',m),
    ('n',n),
    ('nx',ng),
    ('ow',o_as_in_go),
    ('oy',oy_as_in_toy),
    ('p',p),
    ('r',r), ('rx',r,False),
    ('rr',e_as_in_herd),
    ('s',s),
    ('sh',sh),
    ('t',t), ('tx',t,False),
    ('th',th_as_in_think),
    ('uh',opt_u_as_in_pull),
    ('uw',oo_as_in_food),
    ('v',v),
    ('w',w),
    ('yx',y),
    ('z',z),
    ('zh',ge_of_blige_etc),
    ('ihr',ear), # DECtalk makes this from ih + r
    approximate_missing=True,
    cleanup_regexps=[('yxuw','yu')], # TODO: other allophones ("x',False" stuff above)?
    cvtOut_regexps=[('yu','yxuw')],
    # lex_filename not set (depends on which model etc)
    stress_comes_before_vowel=True,
    safe_to_drop_characters=True, # TODO: really?
    word_separator=" ",phoneme_separator="",
    inline_header="[:phoneme on]\n",
    inline_format="[%s]",
  ),
  "doubletalk" : makeDic(
    'DoubleTalk PC/LT serial-port hardware synthesizers (American English; assumes DOS driver by default, otherwise set DTALK_COMMAND_CODE to your current command-code binary value, e.g. export DTALK_COMMAND_CODE=1)', # (1 is the synth's default; the DOS driver lets you put * instead)
    (syllable_separator,'',False),
    ("/",primary_stress), # TODO: check it doesn't need a balancing \ afterwards (docs do say it's a "temporary" change of pitch, but it's unclear how long a 'temporary')
    ('M',m),('N',n),('NX',ng),('O',o_as_in_go),
    ('OW',o_as_in_go,False), # allophone
    (o_as_in_orange,'O',False), # TODO: is this the best approximation we can do?
    ('OY',oy_as_in_toy),('P',p),
    ('R',r),('S',s),('SH',sh),('T',t),
    ('TH',th_as_in_think),('V',v),('W',w),('Z',z),
    ('ZH',ge_of_blige_etc),('K',k),('L',l),
    ('PX',p,False), ('TX',t,False), # aspirated allophones
    ('WH',w,False), ('KX',k,False), # ditto
    ('YY',y),('Y',y,False),
    ('UH',opt_u_as_in_pull),('UW',oo_as_in_food),
    ('AA',a_as_in_ah),('AE',a_as_in_apple),
    ('AH',u_as_in_but),('AO',close_to_or),
    ('AW',o_as_in_now),('AX',a_as_in_ago),
    ('AY',eye),('B',b),('CH',ch),('D',d),
    ('DH',th_as_in_them),
    ('DX',t,False), # an American "d"-like "t"
    ('EH',e_as_in_them),('ER',e_as_in_herd),
    ('EY',a_as_in_ate),('F',f),('G',g),('H',h),
    ('IH',i_as_in_it),('IX',i_as_in_it,False),
    ('IY',e_as_in_eat),('JH',j_as_in_jump),
    approximate_missing=True,
    stress_comes_before_vowel=True,
    inline_format=markup_doubleTalk_word,
    format_is_binary=ifset('DTALK_COMMAND_CODE',True),
    # DoubleTalk does have a loadable "exceptions dictionary" but usually relies on a DOS tool to write it; I don't have the documentation about it (and don't know how much RAM is available for it - it's taken from the input buffer)
  ),
  "keynote" : makeDic(
    'Phoneme-read and lexicon-add codes for Keynote Gold hardware synthesizers (American English)', # ISA, PCMCIA, serial, etc; non-serial models give you an INT 2Fh param to get the address of an API function to call; not sure which software can send these codes directly to it)
    (syllable_separator,'',False),
    (primary_stress,"'"),(secondary_stress,'"'),
    ('w',w),('y',y),('h',h),('m',m),('n',n),('ng',ng),
    ('l',l),('r',r),('f',f),('v',v),('s',s),('z',z),
    ('th',th_as_in_think),('dh',th_as_in_them),('k',k),
    ('ch',ch),('zh',ge_of_blige_etc),('sh',sh),('g',g),
    ('jh',j_as_in_jump),('b',b),('p',p),('d',d),('t',t),
    ('i',e_as_in_eat),('I',i_as_in_it),
    ('e',a_as_in_ate),('E',e_as_in_them),
    ('ae',a_as_in_apple),('u',oo_as_in_food),
    ('U',opt_u_as_in_pull),('o',o_as_in_go),
    ('O',close_to_or),('a',o_as_in_orange),
    ('^',u_as_in_but),('R',e_as_in_herd),
    ('ay',eye),('Oy',oy_as_in_toy),('aw',o_as_in_now),
    ('=',a_as_in_ago),
    approximate_missing=True,
    inline_format="[p]%s[t]",
    lex_filename="keynote.dat", # you have to somehow get this directly dumped to the card, see comment above
    lex_entry_format="[x]%s %s", lex_footer="[t]\n",
    stress_comes_before_vowel=False, # even though it's "'"
  ),
  "audapter" : makeVariantDic(
  "Audapter Speech System, an old hardware serial/parallel-port synthesizer (American English)", # 1989 I think.  The phonemes themselves are the same as the Keynote above, but there's an extra binary byte in the commands and the lex format is stricter.  I haven't checked but my guess is Audapter came before Keynote.
  inline_format='\x05[p] %s\x05[t]',
  format_is_binary=True,
  lex_filename="audapter.dat",
  lex_entry_format="\x05[x]%s %s\x05[t]\n", lex_footer="",
  ),
  "bbcmicro" : makeDic(
    "BBC Micro Speech program from 1985 (see comments in lexconvert.py for more details)",
    # Speech was written by David J. Hoskins and published by Superior Software.  It took 7.5k of RAM including 3.1k of samples (49 phonemes + 1 for fricatives at 64 bytes each, 4-bit ~5.5kHz), 2.2k of lexicon, and 2.2k of machine code; sounds "retro" by modern standards but quite impressive for the BBC Micro in 1985.  Samples are played by amplitude-modulating the BBC's tone generator.
    # If you use an emulator like BeebEm, you'll need diskimg/Speech.ssd.  This can be made from your original Speech disc, or you might be able to find one but beware of copyright!  Same goes with the ROM images included in BeebEm (you might want to delete ones you didn't have).  There has been considerable discussion over whether UK copyright law does or should allow "format-shifting" your own legally-purchased media, and I don't fully understand all the discussion so I don't want to give advice on it here.  The issue is "format-shifting" your legally-purchased BBC Micro ROM code and Speech disc to emulator images; IF this is all right then I suspect downloading someone else's copy is arguably allowed as long as you bought it legally "back in the day", but I'm not a solicitor so I don't know.
    # (Incidentally, yes I was the Silas Brown referred to in Beebug 11.1 p.59, 11.9 p.50/11.10 p.47 and 12.10 p.24, and, no, the question in the final issue wasn't quite how I put it, but all taken in good humour.)
    # lexconvert's --phones bbcmicro option creates *SPEAK commands which you can type into the BBC Micro or paste into an emulator, either at the BASIC prompt, or in a listing with line numbers provided by AUTO.  You have to load the Speech program first of course.
    # To script this on BeebEm, first turn off the Speech disc's boot option (by turning off File / Disc options / Write protect and entering "*OPT 4,0"; use "*OPT 4,3" if you want it back later; if you prefer to edit the disk image outside of the emulator then change byte 0x106 from 0x33 to 0x03), and then you can do (e.g. on a Mac) open /usr/local/BeebEm3/diskimg/Speech.ssd && sleep 1 && (echo '*SPEECH';python lexconvert.py --phones bbcmicro "Greetings from 19 85") | pbcopy && osascript -e 'tell application "System Events" to keystroke "v" using command down'
    # or if you know it's already loaded: echo "Here is some text" | python lexconvert.py --phones bbcmicro | pbcopy && osascript -e 'tell application "BeebEm3" to activate' && osascript -e 'tell application "System Events" to keystroke "v" using command down'
    # (unfortunately there doesn't seem to be a way of doing it without giving the emulator window focus)
    # If you want to emulate a Master, you might need a *DISK before the *SPEECH (to take it out of ADFS mode).
    # You can also put Speech into ROM, but this can cause problems: see comments on SP8000 later.
    (syllable_separator,'',False),
    ('4',primary_stress),
    ('5',secondary_stress), # (these are pitch numbers on the BBC; normal pitch is 6, and lower numbers are higher pitches, so try 5=secondary and 4=primary; 3 sounds less calm)
    ('AA',a_as_in_ah),
    ('AE',a_as_in_apple),
    ('AH',u_as_in_but),
    ('O',o_as_in_orange),
    ('AW',o_as_in_now),
    (a_as_in_ago,'AH',False),
    ('ER',e_as_in_herd),
    ('IY',eye),
    ('B',b),
    ('CH',ch),
    ('D',d),
    ('DH',th_as_in_them),
    ('EH',e_as_in_them),
    (ar_as_in_year,'ER',False),
    ('AI',a_as_in_air),
    ('AY',a_as_in_ate),
    ('F',f),
    ('G',g),
    ('/H',h),
    ('IH',i_as_in_it),
    ('IX',var2_i_as_in_it), # (IX sounds to me like a slightly shorter version of IH)
    ('IXAH',ear),
    ('EER',var2_ear), # e.g. 'hear', 'near' - near enough
    ('EE',e_as_in_eat),
    ('J',j_as_in_jump),
    ('K',k),
    ('C',k,False), # for CT as in "fact", read out as K+T
    ('L',l),
    ('M',m),
    ('N',n),
    ('NX',ng),
    ('OW',o_as_in_go),
    ('OL',opt_ol_as_in_gold), # (if dest format doesn't have this, it'll get o_as_in_orange from the O, then the l)
    ('OY',oy_as_in_toy),
    ('P',p),
    ('R',r),
    ('S',s),
    ('SH',sh),
    ('T',t),
    ('TH',th_as_in_think),
    ('AOR',oor_as_in_poor),
    ('UH',oor_as_in_poor,False), # TODO: really? (espeak 'U' goes to opt_u_as_in_pull, and eSpeak also used U for the o in good, which sounds best with Speech's default UH4, hence the line below, but where did we get UH->oor_as_in_poor from?  Low-priority though because how often do you convert OUT of bbcmicro format)
    (opt_u_as_in_pull,'UH',False),
    ('/U',opt_u_as_in_pull,False),
    ('/UL',opt_ul_as_in_pull), # if dest format doesn't have this, it'll get opt_u_as_in_pull from the /U, then l
    ('UW',oo_as_in_food),
    ('UX',oo_as_in_food,False),
    ('AO',close_to_or),
    ('V',v),
    ('W',w),
    ('Y',y),
    ('Z',z),
    ('ZH',ge_of_blige_etc),
    lex_filename=ifset("MAKE_SPEECH_ROM","SPEECH.ROM","BBCLEX"),
    lex_entry_format=as_utf8("> %s_")+chr(128)+as_utf8("%s"), # (specifying 'whole word' for now; remove the space before or the _ after if you want)
    lex_read_function = lambda lexfile: [(w[0].lstrip().rstrip('_').lower(),w[1]) for w in filter(lambda x:len(x)==2,[w.split(chr(128)) for w in getBuf(lexfile).read().split('>')])], # TODO: this reads back the entries we generate, but is unlikely to work well with the wildcards in the default lexicon that would have been added if SPEECH_DISK was set (c.f. trying to read eSpeak's en_rules instead of en_extra)
    lex_word_case = "upper",
    lex_header = bbc_prepDefaultLex,
    lex_footer = bbc_appendDefaultLex, # + ">**"
    inline_format = markup_bbcMicro_word,
    word_separator=" ",phoneme_separator="",
    clause_separator=write_bbcmicro_phones, # special case
    safe_to_drop_characters=True, # TODO: really?
    cleanup_regexps=[
      ('KT','CT'), # Speech instructions: "CT as in fact"
      ('DYUW','DUX'), # "DUX as in duke"
      ('AHR$','AH'), # usually sounds a bit better
    ],
    cvtOut_regexps=[('DUX','DYUW')], # CT handled above
  ),
  "bbcmicro-cc" : makeDic(
     "Computer Concepts Speech ROM which provided phonemes for the BBC Micro's TMS5220 \"speech chip\" add-on (less widely sold than the software-only product)", # (and harder to run on an emulator.  It wasn't the only phoneme ROM, e.g. Easytalk Speech Utility ROM by Galaxy, reviewed in Beebug Jan/Feb 1985 (3.8) p.32, expanded on Acorn's original PHROM with commands like *SAY Y.U:N.I.V.ER.S but we don't know all the phonemes; there were also some allophone-based hardware boards)
     (syllable_separator,"",False),
     ('*',primary_stress),('+',secondary_stress),
     ('E',e_as_in_eat),('i',i_as_in_it),('e',e_as_in_them),
     ('a',a_as_in_apple),('u',u_as_in_but),('AR',a_as_in_ah),
     ('o',o_as_in_orange),('OR',close_to_or),('oo',opt_u_as_in_pull),
     ('OO',oo_as_in_food),('ER',e_as_in_herd),('A',a_as_in_ate),
     ('I',eye),('O',o_as_in_go),('OY',oy_as_in_toy),
     ('AW',o_as_in_now),('EA',ear),('ea',a_as_in_air),
     ('UR',oor_as_in_poor),('UH',a_as_in_ago),
     ('P',p),('B',b),('T',t),
     ('D',d),('K',k),('G',g),
     ('CH',ch),('J',j_as_in_jump),('F',f),
     ('V',v),('TH',th_as_in_think),('DH',th_as_in_them),
     ('S',s),('Z',z),('SH',sh),
     ('ZH',ge_of_blige_etc),('H',h),('M',m),
     ('N',n),('NG',ng),('L',l),
     ('R',r),('Y',y),('W',w),
     stress_comes_before_vowel=True,
     inline_header="*UTTER <1> ",
     clause_separator="\n*UTTER <1> ", # TODO: manual does not say what the maximum length is; longest parameter in examples is 80 bytes; should we use inline_format to make each WORD a separate command?
     cleanup_regexps=[('[*] ','*'),('[+] ','+')],
     safe_to_drop_characters=' ',
  ),
     
  "amiga" : makeDic(
    'AmigaOS speech synthesizer (American English)', # shipped with the 1985 Amiga release; developed by SoftVoice Inc
    # All I had to go by for this was a screenshot on Marcos Miranda's "blog".  I once saw this synth demonstrated but never tried it.  My early background was the BBC Micro, not Amigas etc.  But I know some people are keen on Amigas so I might as well include it.
    # (By the way I think David Hoskins had it harder than SoftVoice.  Yes they were both in 1985, but the Amiga was a new 16-bit machine while the BBC was an older 8-bit one.  See the "sam" format for an even older one though, although probably not written by one person.)
    (syllable_separator,'',False),
    ('4',primary_stress),('3',secondary_stress),
    ('/H',h),
    ('EH',e_as_in_them),
    ('L',l),
    ('OW',o_as_in_go),
    ('AY',eye),
    ('AE',a_as_in_apple),
    ('M',m),
    ('DH',th_as_in_them),
    ('IY',e_as_in_eat),
    ('AH',a_as_in_ago),
    ('G',g),
    ('K',k),
    ('U',u_as_in_but),
    ('P',p),
    ('Y',y),
    ('UW',oo_as_in_food),
    ('T',t),
    ('ER',var1_a_as_in_ago),
    ('IH',i_as_in_it),
    ('S',s),
    ('Z',z),
    ('AW',o_as_in_now),
    ('AA',a_as_in_ah),
    ('R',r),
    ('D',d),('F',f),('N',n),('NX',ng),('J',j_as_in_jump),
    ('B',b),('V',v),('TH',th_as_in_think),
    ('OH',close_to_or),('EY',a_as_in_ate),
    # The following consonants were not on the screenshot
    # (or at least I couldn't find them) so I'm guessing.
    # I think this should work given the way the other
    # consonants work in this table.
    ('W',w),('CH',ch),('SH',sh),
    # The following vowels were not in the screenshot and
    # we just have to hope this guess is right - when
    # someone tries it on an Amiga and says it doesn't
    # work, maybe we can update this....
    ('O',o_as_in_orange),('OY',oy_as_in_toy),
    # and these ones we can approximate to ones we already know (given that we're having to approximate British to an American voice anyway, it can't hurt TOO much more)
     (a_as_in_air,'EH',False),
     (e_as_in_herd,'ER',False),
     (ar_as_in_year,'ER',False),
     (ear,'IYAH',False), # or try IYER, or there might be a phoneme for it
     (ge_of_blige_etc,'J',False),
     (oor_as_in_poor,'OH',False),
    # lex_filename not set (I have no idea how the Amiga lexicon worked)
    safe_to_drop_characters=True, # TODO: really?
    word_separator=" ",phoneme_separator="",
  ),
  "sam" : makeDic(
  'Software Automatic Mouth (1982 American English synth that ran on C64, Atari 400/800/etc and Apple II/etc)', # *might* be similar to Macintalk on the 1st Macintosh in 1984
  (syllable_separator,'',False),
  (primary_stress,'4'),
  (secondary_stress,'5'),
  ('IY',e_as_in_eat),
  ('IH',i_as_in_it),
  ('EH',e_as_in_them),
  ('AE',a_as_in_apple),
  ('AA',o_as_in_orange),
  ('AH',u_as_in_but),
  ('AO',close_to_or),
  ('OH',o_as_in_go),
  ('UH',opt_u_as_in_pull),
  ('UX',oo_as_in_food),
  ('ER',e_as_in_herd),
  ('AX',a_as_in_apple,False), # allophone?
  ('IX',i_as_in_it,False), # allophone?
  ('EY',a_as_in_ate),
  ('AY',eye),('OY',oy_as_in_toy),
  ('AW',o_as_in_now),('OW',o_as_in_go,False),
  ('UW',oo_as_in_food,False), # allophone?
  ('R',r),('L',l),('W',w),('WH',w,False),('Y',y),('M',m),
  ('N',n),('NX',ng),('B',b),('D',d),('G',g),('Z',z),
  ('J',j_as_in_jump),('ZH',ge_of_blige_etc),('V',v),
  ('DH',th_as_in_them),('S',s),('SH',sh),('F',f),
  ('TH',th_as_in_think),('P',p),('T',t),('K',k),
  ('CH',ch),('/H',h),('Q',glottal_stop),
  approximate_missing=True,
  word_separator=" ",phoneme_separator="",
  # TODO: inline_format etc similar to bbcmicro?
  # In Atari BASIC, you set SAM$ to the phonemes and then
  # do A=USR(8192).  I don't know about the C64 etc versions.
  # (max 255 phonemes per string; don't know max line len.)
  ),

  "cheetah" : makeDic(
     'Allophone codes for the 1983 "Cheetah Sweet Talker" SP0256-based hardware add-on for ZX Spectrum and BBC Micro home computers. The conversion from phonemes to allophones might need tweaking.',
     (syllable_separator,'',False),
     ("0",syllable_separator,False),
     ("1",syllable_separator,False),
     ("2",syllable_separator,False),
     ("3",syllable_separator,False),
     ("4",syllable_separator,False),
     ("5",oy_as_in_toy),
     ("6",eye),
     ("7",e_as_in_them),
     ("8",k,False),
     ("9",p),
     ("10",j_as_in_jump),
     ("11",n),
     ("12",i_as_in_it),
     ("13",t),
     ("14",r),
     ("15",u_as_in_but),
     ("16",m),
     ("17",t,False),
     ("18",th_as_in_them),
     ("19",e_as_in_eat),
     ("20",a_as_in_ate),
     ("21",d),
     ("22",oo_as_in_food),
     ("23",close_to_or),
     ("24",o_as_in_orange),
     ("25",y),
     ("26",a_as_in_apple),
     ("27",h),
     ("28",b),
     ("29",th_as_in_think),
     (opt_u_as_in_pull,"30",False),
     ("30",opt_ul_as_in_pull),
     ("31",oo_as_in_food,False),
     ("32",o_as_in_now),
     ("33",d,False),
     ("34",g,False),
     ("35",v),
     ("36",g),
     ("37",sh),
     ("38",ge_of_blige_etc),
     ("39",r,False),
     ("40",f),
     ("41",k),
     ("42",k,False),
     ("43",z),
     ("44",ng),
     ("45",l),
     ("46",w),
     ("47",a_as_in_air),
     ("49",y,False),
     ("50",ch),
     ("51",a_as_in_ago),
     ("52",e_as_in_herd),
     (var1_a_as_in_ago,"52",False),
     ("53",o_as_in_go),
     ("54",th_as_in_them,False),
     ("55",s),
     ("56",n,False),
     ("57",h,False),
     ("58",var3_close_to_or),
     ("59",a_as_in_ah),
     ("60",ear), # or var2_ear
     ("61",g,False),
     ("62",l,False),
     ("63",b,False),
     approximate_missing=True,
     phoneme_separator=',',safe_to_drop_characters=",",
     inline_header="DATA ",inline_footer=",0"),

  # END (?) PRE-32bit ERA SYNTHS (but see TODO above re SpeakJet, which is below)

  "speakjet" : makeDic(
    'Allophone codes for the American English "SpeakJet" speech synthesis chip (the conversion from phonemes to allophones might need tweaking).  Set the SPEAKJET_SYM environment variable to use mnemonics, otherwise numbers are used (set SPEAKJET_BINARY for binary output).',
  # TODO: might want to do something similar for the older Votrax SC-02 chip, but would need to check how exactly its phoneme interface was exposed to software by the PC cards that used it (Heathkit HV-2000 etc; not sure if any are still in use though)
    (syllable_separator,'',False), # TODO: instead of having emphasis, the Speakjet has a 'faster' code for all NON-emphasized syllables
    (speakjet('IY',128),e_as_in_eat),
    (speakjet('IH',129),i_as_in_it),
    (speakjet('EY',130),a_as_in_ate),
    (speakjet('EH',131),e_as_in_them),
    (speakjet('AY',132),a_as_in_apple),
    (speakjet('AX',133),a_as_in_ago),
    (speakjet('UX',134),u_as_in_but),
    (speakjet('OH',135),o_as_in_orange),
    (speakjet('AW',136),a_as_in_ah),
    (speakjet('OW',137),o_as_in_go),
    (speakjet('UH',138),opt_u_as_in_pull),
    (speakjet('UW',139),oo_as_in_food),
    (speakjet('MM',140),m),
    (speakjet('NE',141),n,False),
    (speakjet('NO',142),n),
    (speakjet('NGE',143),ng,False),
    (speakjet('NGO',144),ng),
    (speakjet('LE',145),l,False),
    (speakjet('LO',146),l),
    (speakjet('WW',147),w),
    (speakjet('RR',148),r),
    (speakjet('IYRR',149),ear),
    (speakjet('EYRR',150),a_as_in_air),
    (speakjet('AXRR',151),e_as_in_herd),
    (speakjet('AWRR',152),a_as_in_ah,False),
    (speakjet('OWRR',153),close_to_or),
    (speakjet('EYIY',154),a_as_in_ate,False),
    (speakjet('OHIY',155),eye),
    (speakjet('OWIY',156),oy_as_in_toy),
    (speakjet('OHIH',157),eye,False),
    (speakjet('IYEH',158),y),
    (speakjet('EHLL',159),l,False),
    (speakjet('IYUW',160),oo_as_in_food,False),
    (speakjet('AXUW',161),o_as_in_now),
    (speakjet('IHUW',162),oo_as_in_food,False),
    # TODO: 163 AYWW = o_as_in_now a_as_in_ago ? handle in cleanup_regexps + cvtOut_regexps ?
    (speakjet('OWWW',164),o_as_in_go,False),
    (speakjet('JH',165),j_as_in_jump),
    (speakjet('VV',166),v),
    (speakjet('ZZ',167),z),
    (speakjet('ZH',168),ge_of_blige_etc),
    (speakjet('DH',169),th_as_in_them),
    # TODO: get cleanup_regexps to clean up some of these according to what's coming next etc:
    (speakjet('BE',170),b,False),
    (speakjet('BO',171),b),
    (speakjet('EB',172),b,False),
    (speakjet('OB',173),b,False),
    (speakjet('DE',174),d,False),
    (speakjet('DO',175),d),
    (speakjet('ED',176),d,False),
    (speakjet('OD',177),d,False),
    (speakjet('GE',178),g,False),
    (speakjet('GO',179),g),
    (speakjet('EG',180),g,False),
    (speakjet('OG',181),g,False),
    (speakjet('CH',182),ch),
    (speakjet('HE',183),h,False),
    (speakjet('HO',184),h),
    (speakjet('WH',185),w,False),
    (speakjet('FF',186),f),
    (speakjet('SE',187),s,False),
    (speakjet('SO',188),s),
    (speakjet('SH',189),sh),
    (speakjet('TH',190),th_as_in_think),
    (speakjet('TT',191),t),
    (speakjet('TU',192),t,False),
    # TODO: 193 TS in cleanup_regexps and cvtOut_regexps
    (speakjet('KE',194),k,False),
    (speakjet('KO',195),k),
    (speakjet('EK',196),k,False),
    (speakjet('OK',197),k,False),
    (speakjet('PE',198),p,False),
    (speakjet('PO',199),p),
    # lex_filename not set (I think the front-end software might have one, but don't know if it's accessible; chip itself just takes phonemes)
    approximate_missing=True,
    word_separator=ifset('SPEAKJET_BINARY',""," "),
    phoneme_separator=ifset('SPEAKJET_BINARY',""," "),
    clause_separator=ifset('SPEAKJET_BINARY',"","\n"), # TODO: is there a pause code?
    output_is_binary=ifset('SPEAKJET_BINARY',True),
    safe_to_drop_characters=True, # TODO: really?
  ),

  "rsynth" : makeDic(
    'rsynth text-to-speech C library (American English)', # TODO: test
    (syllable_separator,'',False), # TODO: emphasis?
    ("i:",e_as_in_eat),
    ("I",i_as_in_it),
    ("eI",a_as_in_ate),
    ("E",e_as_in_them),
    ("{",a_as_in_apple),
    ("V",u_as_in_but),
    ("Q",o_as_in_orange),
    ("A:",a_as_in_ah),
    ("oU",o_as_in_go),
    ("U",opt_u_as_in_pull),
    ("u:",oo_as_in_food),
    ("m",m),
    ("n",n),
    ("N",ng),
    ("l",l),
    ("w",w),
    ("r",r),
    ("I@",ear),
    ("e@",a_as_in_air),
    ("3:",e_as_in_herd),
    ("Qr",close_to_or),
    ("OI",oy_as_in_toy),
    ("aI",eye),
    ("j",y),
    ("U@",oo_as_in_food,False),
    ("aU",o_as_in_now),
    ("@U",o_as_in_go,False),
    ("dZ",j_as_in_jump),
    ("v",v),
    ("z",z),
    ("Z",ge_of_blige_etc),
    ("D",th_as_in_them),
    ("b",b),
    ("d",d),
    ("g",g),
    ("tS",ch),
    ("h",h),
    ("f",f),
    ("s",s),
    ("S",sh),
    ("T",th_as_in_think),
    ("t",t),
    ("k",k),
    ("p",p),
    approximate_missing=True,
    # lex_filename not set (TODO: check what sort of lexicon is used by rsynth's "say" front-end)
    safe_to_drop_characters=True, # TODO: really?
    word_separator=" ",phoneme_separator="",
  ),

  "unicode-ipa" : makeDic(
    "IPA symbols in Unicode, as used by an increasing number of dictionary programs, websites etc",
    ('.',syllable_separator,False),
    (syllable_separator,'',False),
    (u'\u02c8',primary_stress),
    (u'\u02cc',secondary_stress),
    # NB the above two are "modifier", not "combining",
    # Unicode characters.  There IS a difference.  If
    # your software displays them as overprinting the
    # surrounding letters, you have a bug.
    # (E.g. WeChat v1.2.2.1 on Mac OS 10.7)
    ('#',text_sharp),
    ('_',text_underline),
    ('?',text_question),
    ('!',text_exclamation),
    (',',text_comma),
    (u'\u0251',a_as_in_ah),
    (u'\u02d0',ipa_colon),
    (u'\u0251\u02d0',var3_a_as_in_ah),
    (u'\u0251\u0279',var4_a_as_in_ah),
    (u'a\u02d0',var5_a_as_in_ah),
    (u'\xe6',a_as_in_apple),
    ('a',a_as_in_apple,False),
    (u'\u028c',u_as_in_but),
    ('\u1d27',u_as_in_but,False), # 28c sometimes mistakenly written as 1d27
    (u'\u0252',o_as_in_orange),
    (var1_o_as_in_orange,u'\u0251',False),
    (u'\u0254',var2_o_as_in_orange),
    (u'a\u028a',o_as_in_now),
    (u'\xe6\u0254',var1_o_as_in_now),
    (u'\u0259',a_as_in_ago),
    (u'\u0259\u02d0',e_as_in_herd),
    (u'\u025a',var1_a_as_in_ago),
    (u'a\u026a',eye), (u'\u028c\u026a',eye,False),
    (u'\u0251e',var1_eye),
    ('b',b),
    (u't\u0283',ch),
    (u'\u02a7',ch,False),
    ('d',d),
    (u'\xf0',th_as_in_them),
    (u'\u025b',e_as_in_them),
    ('e',var1_e_as_in_them),
    (u'\u025d',ar_as_in_year),
    (u'\u025c\u02d0',ar_as_in_year,False),
    (u'\u025b\u0259',a_as_in_air),
    (u'\u025b\u0279',var1_a_as_in_air),
    (u'e\u02d0',var2_a_as_in_air),
    (u'\u025b\u02d0',var3_a_as_in_air),
    (u'e\u0259',var4_a_as_in_air),
    (u'e\u026a',a_as_in_ate),
    (u'\xe6\u026a',var1_a_as_in_ate),
    ('f',f),
    (u'\u0261',g), ('g',g,False),
    ('h',h),
    (u'\u026a',i_as_in_it),
    (u'\u0268',var1_i_as_in_it),
    (u'\u026a\u0259',ear),
    (u'\u026a\u0279',var1_ear),
    (u'\u026a\u0279\u0259',var2_ear), # ?
    ('i',e_as_in_eat),
    (u'i\u02d0',var1_e_as_in_eat),
    (u'd\u0292',j_as_in_jump),
    (u'\u02a4',j_as_in_jump,False),
    ('k',k),
    ('x',opt_scottish_loch),
    ('l',l),
    (u'd\u026b',var1_l),
    ('m',m),
    ('n',n),
    (u'\u014b',ng),
    (u'\u0259\u028a',o_as_in_go),
    ('o',var1_o_as_in_go),
    (u'o\u028a',var2_o_as_in_go),
    (u'\u0259\u0289',var1_u_as_in_but),
    (u'\u0254\u026a',oy_as_in_toy),
    (u'o\u026a',var1_oy_as_in_toy),
    ('p',p),
    (u'\u0279',r), ('r',r,False),
    (var1_r,'r',False),
    ('s',s),
    (u'\u0283',sh),
    ('t',t),
    (u'\u027e',var1_t),
    (u'\u03b8',th_as_in_think),
    (u'\u028a\u0259',oor_as_in_poor),
    (u'\u028a\u0279',var1_oor_as_in_poor),
    (u'\u028a',opt_u_as_in_pull),
    (u'\u0289\u02d0',oo_as_in_food),
    (u'u\u02d0',var1_oo_as_in_food),
    ('u',var2_oo_as_in_food),
    (u'\u0254\u02d0',close_to_or),
    (var1_close_to_or,u'\u0254',False),
    (u'o\u02d0',var2_close_to_or),
    ('v',v),
    ('w',w),
    (u'\u028d',var1_w),
    ('j',y),
    ('z',z),
    (u'\u0292',ge_of_blige_etc),
    (u'\u0294',glottal_stop),
    lex_filename="words-ipa.html", # write-only for now
    lex_type = "HTML",
    lex_header = '<html><head><meta name="mobileoptimized" content="0"><meta name="viewport" content="width=device-width"><meta http-equiv="Content-Type" content="text/html; charset=utf-8"></head><body><table>',
    lex_entry_format="<tr><td>%s</td><td>%s</td></tr>\n",
    lex_footer = "</table></body></html>\n",
    word_separator=" ",phoneme_separator="",
    stress_comes_before_vowel=True,
    safe_to_drop_characters=True, # TODO: really? (at least '-' should be safe to drop)
    cvtOut_func=unicode_preprocess,
  ),

  "unicode-ipa-syls" : makeVariantDic(
  "Like unicode-ipa but with syllable separators preserved",
  (syllable_separator,'.'),
  cleanup_regexps=[(r"\.+",".")], # multiple . to one .
  noInherit=True),

  "yinghan" : makeVariantDic(
     "As unicode-ipa but, when converting a user lexicon, generates Python code that reads Wenlin Yinghan dictionary entries and adds IPA bands to matching words",
    lex_filename="yinghan-ipa.py", # write-only for now
    lex_type = "Python script",
    lex_header = r"""#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Works in both Python 2 and Python 3

import sys; d={""",
    lex_entry_format='u"%s":u"%s",\n',
    lex_footer = r"""}
import re
try: i,o=sys.stdin.buffer,sys.stdout.buffer # Python 3
except AttributeError: i,o=sys.stdin,sys.stdout # Python 2
for k in list(d.keys()): d[k.lower().encode('utf-8')]=d[k]
nextIsHead=False
for l in i:
 o.write(l)
 if nextIsHead and l.strip():
  w=l.split()
  if w[0]==u'ehw'.encode('utf-8'): l=u' '.encode('utf-8').join(w[1:])
  k = re.sub(u'\\([^)]*\\)$'.encode('utf-8'),u''.encode('utf-8'),l.strip()).strip().lower() # (allow parenthesised explanation after headword when matching)
  if k in d: o.write(u'ipa '.encode('utf-8')+d[k].encode('utf-8')+u'\n'.encode('utf-8'))
 if l.startswith(u'*** '.encode('utf-8')): nextIsHead=True
""",
    noInherit=True
  ),

  "unicode-rough" : makeVariantDic(
    "A non-standard notation that's reminiscent of unicode-ipa but changed so that more of the characters show in old browsers with incomplete fonts",
    ("'",primary_stress),
    (',',secondary_stress),
    ('ar-',a_as_in_ah),
    (':',ipa_colon),
    (var3_a_as_in_ah,'ar-',False),
    (var4_a_as_in_ah,'ar-',False),
    ('uh',u_as_in_but),
    (u'\u0259:',e_as_in_herd),
    ('ai',eye),
    ('ch',ch),
    ('e',e_as_in_them),
    ('3:',ar_as_in_year),
     (a_as_in_air,'e:',False),
     (var1_a_as_in_air,'e:',False),
     (var2_a_as_in_air,'e:',False),
     (var3_a_as_in_air,'e:',False),
     (var4_a_as_in_air,'e:',False),
    (u'ei',a_as_in_ate),
    (u'\xe6i',var1_a_as_in_ate),
    ('g',g),
    ('i',i_as_in_it), (var1_i_as_in_it,'i',False),
    ('eeuh-',ear), (var2_ear,'eeuh-',False),
    ('ee',e_as_in_eat), (var1_e_as_in_eat,'ee',False),
    ('j',j_as_in_jump),
    ('ng',ng),
    ('o',o_as_in_go),
    (var2_o_as_in_go,'o',False), # override unicode-ipa
    (var1_u_as_in_but,'o',False), # ditto (?? '+'?)
    ('oy',oy_as_in_toy), (var1_oy_as_in_toy,'oy',False),
    ('r',r),
    ('sh',sh),
    (var1_t,'t',False),
    ('th',th_as_in_think),
    ('or',oor_as_in_poor),
    (var1_oor_as_in_poor,'or',False),
    ('u',opt_u_as_in_pull), ('oo',oo_as_in_food),
     (var1_oo_as_in_food,'oo',False),
     (var2_oo_as_in_food,'oo',False),
     (close_to_or,'or',False),
     (var1_close_to_or,'or',False),
     (var2_close_to_or,'or',False),
     (var1_w,'w',False),
    ('y',y),
    ('3',ge_of_blige_etc),
     cleanup_regexps=[('-$','')],
    cvtOut_func=None,
  ),

  "braille-ipa" : makeDic(
    "IPA symbols in Braille (2008 BANA standard).  By default Braille ASCII is output; if you prefer to see the Braille dots via Unicode, set the BRAILLE_UNICODE environment variable.", # BANA = Braille Authority of North America.  TODO: check if the UK accepted this standard.
    # TODO: add Unicode IPA signs that aren't used in English IPA, so we can do a general IPA conversion
    ('_B',primary_stress),
    ('_2',secondary_stress),
    ('*',a_as_in_ah),
    ('3',ipa_colon),
    ('*3',var3_a_as_in_ah),
    ('*#',var4_a_as_in_ah),
    ('A3',var5_a_as_in_ah),
    ('%',a_as_in_apple),
    ('A',a_as_in_apple,False),
    ('+',u_as_in_but),
    ('4*',o_as_in_orange),
    (var1_o_as_in_orange,'*',False),
    ('<',var2_o_as_in_orange),
    ('A(',o_as_in_now),
    ('%<',var1_o_as_in_now),
    ('5',a_as_in_ago),
    ('53',e_as_in_herd),
    ('5"R.',var1_a_as_in_ago),
    ('A/',eye),
    ('*E',var1_eye),
    ('B',b),
    ('T:',ch),
    ('T":.',ch,False),
    ('D',d),
    (']',th_as_in_them),
    ('>',e_as_in_them),
    ('E',var1_e_as_in_them),
    ('4>3',ar_as_in_year), # (from \u025c\u02d0; TODO: check what happens to \u025d)
    ('>5',a_as_in_air),
    ('>#',var1_a_as_in_air),
    ('E3',var2_a_as_in_air),
    ('>3',var3_a_as_in_air),
    ('E5',var4_a_as_in_air),
    ('E/',a_as_in_ate),
    ('%/',var1_a_as_in_ate),
    ('F',f),
    ('G',g),
    ('H',h),
    ('/',i_as_in_it),
    ('0I',var1_i_as_in_it),
    ('/5',ear),
    ('/#',var1_ear),
    ('/#5',var2_ear), # ?
    ('I',e_as_in_eat),
    ('I3',var1_e_as_in_eat),
    ('D!',j_as_in_jump),
    ('K',k),
    ('X',opt_scottish_loch),
    ('L',l),
    ('D6L',var1_l),
    ('M',m),
    ('N',n),
    ('$',ng),
    ('5(',o_as_in_go),
    ('O',var1_o_as_in_go),
    ('O(',var2_o_as_in_go),
    ('50U',var1_u_as_in_but),
    ('</',oy_as_in_toy),
    ('O/',var1_oy_as_in_toy),
    ('P',p),
    ('#',r),
    (var1_r,'R',False),
    ('S',s),
    (':',sh),
    ('T',t),
    ('6R',var1_t),
    ('.?',th_as_in_think),
    ('(5',oor_as_in_poor),
    ('(#',var1_oor_as_in_poor),
    ('(',opt_u_as_in_pull),
    ('0U3',oo_as_in_food),
    ('U3',var1_oo_as_in_food),
    ('U',var2_oo_as_in_food),
    ('<3',close_to_or),
    (var1_close_to_or,'<',False),
    ('O3',var2_close_to_or),
    ('V',v),
    ('W',w),
    ('6W',var1_w),
    ('J',y),
    ('Z',z),
    ('!',ge_of_blige_etc),
    ('2',glottal_stop),
    lex_filename=ifset("BRAILLE_UNICODE","words-ipa.txt","words-ipa.brl"), # write-only for now
    lex_type = "document",
    # inline_format=",7%s7'", # -> do this in cleanup_func so it's included in BRAILLE_UNICODE if necessary
    lex_entry_format="%s = %s\n", # ditto with the markers
    word_separator=" ",phoneme_separator="",
    stress_comes_before_vowel=True,
    safe_to_drop_characters=True, # TODO: really?
    cleanup_func=lambda r:ifset("BRAILLE_UNICODE",ascii_braille_to_unicode,lambda x:x)(",7"+r+"7'"),
    cvtOut_func=unicode_to_ascii_braille,
  ),
  
  "latex-ipa" : makeDic(
    'IPA symbols for typesetting in LaTeX using the "tipa" package',
    ('.',syllable_separator,False),
    ('"',primary_stress),
    ('\\textsecstress{}',secondary_stress),
    ('\\#',text_sharp),
    ('\\_',text_underline),
    ('?',text_question),
    ('!',text_exclamation),
    (',',text_comma),
    ('A',a_as_in_ah),
    (':',ipa_colon),
    ('A:',var3_a_as_in_ah),
    ('A\\textturnr{}',var4_a_as_in_ah),
    ('a:',var5_a_as_in_ah),
    ('\\ae{}',a_as_in_apple),
    ('2',u_as_in_but),
    ('6',o_as_in_orange),
    (var1_o_as_in_orange,'A',False),
    ('O',var2_o_as_in_orange),
    ('aU',o_as_in_now),
    ('\\ae{}O',var1_o_as_in_now),
    ('@',a_as_in_ago),
    ('@:',e_as_in_herd),
    ('\\textrhookschwa{}',var1_a_as_in_ago),
    ('aI',eye),
    ('Ae',var1_eye),
    ('b',b),
    ('tS',ch),
    ('d',d),
    ('D',th_as_in_them),
    ('E',e_as_in_them),
    ('e',var1_e_as_in_them),
    ('3:',ar_as_in_year),
    ('E@',a_as_in_air),
    ('E\\textturnr{}',var1_a_as_in_air),
    ('e:',var2_a_as_in_air),
    ('E:',var3_a_as_in_air),
    ('e@',var4_a_as_in_air),
    ('eI',a_as_in_ate),
    ('\\ae{}I',var1_a_as_in_ate),
    ('f',f),
    ('g',g),
    ('h',h),
    ('I',i_as_in_it),
    ('1',var1_i_as_in_it),
    ('I@',ear),
    ('I\\textturnr{}',var1_ear),
    ('I@\\textturnr{}',var2_ear), # ?
    ('i',e_as_in_eat),
    ('i:',var1_e_as_in_eat),
    ('dZ',j_as_in_jump),
    ('k',k),
    ('x',opt_scottish_loch),
    ('l',l),
    ('d\\textltilde{}',var1_l),
    ('m',m),
    ('n',n),
    ('N',ng),
    ('@U',o_as_in_go),
    ('o',var1_o_as_in_go),
    ('oU',var2_o_as_in_go),
    ('@0',var1_u_as_in_but),
    ('OI',oy_as_in_toy),
    ('oI',var1_oy_as_in_toy),
    ('p',p),
    ('\\textturnr{}',r),
    (var1_r,'r',False),
    ('s',s),
    ('S',sh),
    ('t',t),
    ('R',var1_t),
    ('T',th_as_in_think),
    ('U@',oor_as_in_poor),
    ('U\\textturnr{}',var1_oor_as_in_poor),
    ('U',opt_u_as_in_pull),
    ('0:',oo_as_in_food),
    ('u:',var1_oo_as_in_food),
    ('u',var2_oo_as_in_food),
    ('O:',close_to_or),
    (var1_close_to_or,'O',False),
    ('o:',var2_close_to_or),
    ('v',v),
    ('w',w),
    ('\\textturnw{}',var1_w),
    ('j',y),
    ('z',z),
    ('Z',ge_of_blige_etc),
    ('P',glottal_stop),
    lex_filename="words-ipa.tex", # write-only for now
    lex_type = "document",
    lex_header = r'\documentclass[12pt,a4paper]{article} \usepackage[safe]{tipa} \usepackage{longtable} \begin{document} \begin{longtable}{ll}',
    lex_entry_format=r"%s & \textipa{%s}\\"+"\n",
    lex_footer = r"\end{longtable}\end{document}"+"\n",
    inline_format = "\\textipa{%s}",
    inline_oneoff_header = r"% In preamble, put \usepackage[safe]{tipa}"+"\n", # (the [safe] part is recommended if you're mixing with other TeX)
    word_separator=" ",phoneme_separator="",
    clause_separator=r"\\"+"\n",
    stress_comes_before_vowel=True,
    safe_to_drop_characters=True, # TODO: really?
  ),

  "pinyin-approx" : makeDic(
    "Rough approximation using roughly the spelling rules of Chinese Pinyin (for getting Chinese-only voices to speak some English words; works with some words better than others)", # write-only for now
    ('4',primary_stress),
    ('2',secondary_stress),
    ('a5',a_as_in_ah),
    ('ya5',a_as_in_apple),
    ('e5',u_as_in_but),
    ('yo5',o_as_in_orange),
    ('ao5',o_as_in_now),
    (e_as_in_herd,'e5',False),
    ('ai5',eye),
    ('bu0',b),
    ('che0',ch),
    ('de0',d),
    ('ze0',th_as_in_them),
    ('ye5',e_as_in_them),
    (a_as_in_air,'ye5',False),
    ('ei5',a_as_in_ate),
    ('fu0',f),
    ('ge0',g),
    ('he0',h),
    ('yi5',i_as_in_it),
    ('yi3re5',ear),
    (e_as_in_eat,'yi5',False),
    ('zhe0',j_as_in_jump),
    ('ke0',k),
    ('le0',l),
    ('me0',m),
    ('ne0',n),
    ('eng0',ng),
    ('ou5',o_as_in_go),
    ('ruo2yi5',oy_as_in_toy),
    ('pu0',p),
    ('re0',r),
    ('se0',s),
    ('she0',sh),
    ('te0',t),
    (th_as_in_think,'zhe0',False),
    (oor_as_in_poor,'wu5',False),
    ('yu5',oo_as_in_food),
    ('huo5',close_to_or),
    (v,'fu0',False),
    ('wu0',w),
    ('yu0',y),
    (z,'ze0',False),
    (ge_of_blige_etc,'zhe0',False),
    approximate_missing=True,
    lex_filename="words-pinyin-approx.txt", # write-only for now
    lex_type = "text",
    lex_header = "Pinyin approxmations (very approximate!)\n----------------------------------------\n",
    lex_entry_format = "%s ~= %s\n",
    word_separator=" ",phoneme_separator="",
    cleanup_regexps=[
      ("te0ye","tie"),
      ("e0e5","e5"),("([^aeiou][uo])0e(5)",r"\1\2"),
      ("yu0y","y"),
      ("wu0yo5","wo5"),
      ("([bdfghklmnpwz])[euo]0ei",r"\1ei"),
      ("([bdghklmnpstwz])[euo]0ai",r"\1ai"),
      ("([ghklmnpstyz])[euo]0ya",r"\1a"),("([ghklmnpstz])a([0-5]*)ne0",r"\1an\2"),
      ("([bdfghklmnpstwyz])[euo]0a([1-5])",r"\1a\2"),
      ("([bdjlmnpt])[euo]0yi",r"\1i"),("([bjlmnp])i([1-5]*)ne0",r"\1in\2"),
      ("([zs])he0ei",r"\1hei"),
      ("([dfghklmnprstyz])[euo]0ou",r"\1ou"),
      ("([dghklnrst])[euo]0huo",r"\1uo"),
      ("([bfpm])[euo]0huo",r"\1o"),
      ("([bdghklmnprstyz])[euo]0ao",r"\1ao"),
      ("([zcs])h[eu]0ao",r"\1hao"),
      ("re0r","r"),
      ("zhe0ne0","zhun5"),
      ("54","4"),
      ("52","2"),
      ("([bdjlmnpty])i([1-9])eng0",r"\1ing\2"),
      ("ya([1-9])eng0",r"yang\1"),
      ("ya([1-9])ne0",r"an\1"),
      ("ye([1-9])ne0",r"yan\1"),("([wr])[eu]0yan",r"\1en"),
      ("yi([1-9])ne0",r"yin\1"),
      
      ("yu0","yu5"),("eng0","eng5"), # they won't work unvoiced anyway
      ("0","5"), # comment out if the synth supports 'tone 0 for unvoiced'
      #("[euo]0","0"), # comment in if it expects consonants only when doing that
    ],
  ),

  "kana-approx" : makeDic(
  "Rough approximation using kana (for getting Japanese computer voices to speak some English words; works with some words better than others).  Set KANA_TYPE environment variable to hiragana or katakana (which can affect the sounds of some voices); default is hiragana", # for example on Mac OS 10.7+ (with Japanese voice installed in System Preferences) try PHONES_PIPE_COMMAND='say -v Kyoko' (this voice has a built-in converter from English as well, but lexconvert --phones kana-approx can work better with some complex words, although the built-in converter does seem to have access to slightly more phonemes and can therefore produce words like "to" better).  Default is hiragana because I find hiragana easier to read than katakana, although the Kyoko voice does seem to be able to say 'v' a little better when using kata.  Mac OS 10.7+'s Korean voices (Yuna and Narae) can also read kana, and you could try doing a makeVariantDic and adding in some Korean jamo letters for them (you'd be pushed to represent everything in jamo but kana+jamo seems more hopeful in theory), but again some words work better than others (not all phonetic combinations are supported and some words aren't clear at all).
    # This kana-approx format is 'write-only' for now (see comment in cleanup_regexps re possible reversal)
    (u'\u30fc',primary_stress),
    (secondary_stress,ifset('KANA_MORE_EMPH',u'\u30fc'),False), # set KANA_MORE_EMPH environment variable if you want to try doubling the secondary-stressed vowels as well (doesn't always work very well; if it did, I'd put this line in a makeVariantDic called kana-approx-moreEmph or something)
    # The following Unicode codepoints are hiragana; KANA_TYPE is handled by cleanup_func below
    (u'\u3042',a_as_in_apple),
    (u'\u3044',e_as_in_eat),
    (u'\u3046',oo_as_in_food),
    (u'\u3048',e_as_in_them),
    (u'\u304a',o_as_in_orange),
    (u'\u3042\u3044',eye), # ai
    (u'\u3042\u304a',o_as_in_now), # ao
    (u'\u3048\u3044',a_as_in_ate), # ei
    (u'\u304a\u3044',oy_as_in_toy), # oi
    (u'\u304a\u3046',o_as_in_go), # ou
    (a_as_in_ah,u'\u3042',False),
    (a_as_in_ago,u'\u3046\u304a',False), # TODO: \u3042, \u304a or \u3046 depending on the word?
    (e_as_in_herd,u'\u3042',False), # TODO: really?
    (i_as_in_it,u'\u3044',False), # TODO: really?
    (u_as_in_but,u'\u3046',False), # TODO: really?
    (ar_as_in_year,u'\u3048',False), # TODO: really?
    (ear,u'\u3044\u304a',False), # TODO: really?
    (a_as_in_air,u'\u3048',False), # TODO: really?
    (oor_as_in_poor,u'\u304a',False), # TODO: really?
    (close_to_or,u'\u304a\u30fc'), # TODO: really?
    (u'\u3076',b), # bu (with vowel replacements later)
    (u'\u3061\u3047',ch), # chu (ditto)
    (u'\u3065',d), # du (and so on)
    (u'\u3066\u3085',th_as_in_think), (th_as_in_them,u'\u3066\u3085',False),
    (u'\u3075',f),
    (u'\u3050',g),
    (u'\u306f',h), # ha (as hu == fu)
    (u'\u3058\u3085',j_as_in_jump), (ge_of_blige_etc,u'\u3058\u3085',False),
    (u'\u304f',k),
    (u'\u308b',l), (r,u'\u308b',False),
    (u'\u3080',m),
    (u'\u306c',n),
    (u'\u3093\u3050',ng),
    (u'\u3077',p),
    (u'\u3059',s),
    (u'\u3057\u3085',sh),
    (u'\u3064',t),
    (u'\u308f',w), # use 'wa' (as 'wu' == 'u')
    (v,ifset('KANA_V_AS_W',u'\u308f',u'\u3094'),False), # TODO: document KANA_V_AS_W variable.  Is vu always supported? (it doesn't seem to show up in all fonts)
    (u'\u3086',y),
    (u'\u305a',z),
    lex_filename="words-kana-approx.txt",
    lex_type = "text",
    lex_header = "Kana approxmations (very approximate!)\n--------------------------------------\n",
    lex_entry_format = "%s ~= %s\n",
    word_separator=" ",phoneme_separator="",
    clause_separator=u"\u3002\n".encode('utf-8'),
    cleanup_regexps=[(u"\u306c$",u"\u3093\u30fc"), # TODO: or u"\u3093\u3093" ?
       # now the vowel replacements (bu+a -> ba, etc) (in most cases these can be reversed into cvtOut_regexps if you want to use the kana-approx table to convert hiragana into approximate English phonemes (plus add a (u"\u3093\u30fc*",u"\u306c") and perhaps de-doubling rules to convert back to emphasis) but the result is unlikely to be any good)
       (u"\u3076\u3042",u"\u3070"),(u"\u3076\u3044",u"\u3073"),(u"\u3076\u3048",u"\u3079"),(u"\u3076\u304a",u"\u307c"),(u"\u3076\u3046",u"\u3076"),
       (u"\u3061\u3085\u3042",u"\u3061\u3083"),(u"\u3061\u3085\u3046",u"\u3061\u3085"),(u"\u3061\u3085\u3048",u"\u3061\u3047"),(u"\u3061\u3085\u304a",u"\u3061\u3087"),(u"\u3061\u3085\u3044",u"\u3061"),
       (u"\u3065\u3042",u"\u3060"),(u"\u3065\u3044",u"\u3062"),(u"\u3065\u3048",u"\u3067"),(u"\u3065\u304a",u"\u3069"),(u"\u3065\u3046",u"\u3065"),
       (u"\u3066\u3085\u3042",u"\u3066\u3083"),(u"\u3066\u3085\u3044",u"\u3066\u3043"),(u"\u3066\u3043\u3046",u"\u3066\u3085"),(u"\u3066\u3085\u3048",u"\u3066\u3047"),(u"\u3066\u3085\u304a",u"\u3066\u3087"),
       (u"\u3075\u3042",u"\u3075\u3041"),(u"\u3075\u3044",u"\u3075\u3043"),(u"\u3075\u3048",u"\u3075\u3047"),(u"\u3075\u304a",u"\u3075\u3049"),(u"\u3075\u3046",u"\u3075"),
       (u"\u306f\u3044",u"\u3072"),(u"\u306f\u3046",u"\u3075"),(u"\u306f\u3048",u"\u3078"),(u"\u306f\u304a",u"\u307b"),(u"\u306f\u3042",u"\u306f"),
       (u"\u3050\u3042",u"\u304c"),(u"\u3050\u3044",u"\u304e"),(u"\u3050\u3048",u"\u3052"),(u"\u3050\u304a",u"\u3054"),(u"\u3050\u3046",u"\u3050"),
       (u"\u3058\u3085\u3042",u"\u3058\u3083"),(u"\u3058\u3085\u3046",u"\u3058\u3085"),(u"\u3058\u3085\u3048",u"\u3058\u3047"),(u"\u3058\u3085\u304a",u"\u3058\u3087"),(u"\u3058\u3085\u304a",u"\u3058"),
       (u"\u304f\u3042",u"\u304b"),(u"\u304f\u3044",u"\u304d"),(u"\u304f\u3048",u"\u3051"),(u"\u304f\u304a",u"\u3053"),(u"\u304f\u3046",u"\u304f"),
       (u"\u308b\u3042",u"\u3089"),(u"\u308b\u3044",u"\u308a"),(u"\u308b\u3048",u"\u308c"),(u"\u308b\u304a",u"\u308d"),(u"\u308b\u3046",u"\u308b"),
       (u"\u3080\u3042",u"\u307e"),(u"\u3080\u3044",u"\u307f"),(u"\u3080\u3048",u"\u3081"),(u"\u3080\u304a",u"\u3082"),(u"\u3080\u3046",u"\u3080"),
       (u"\u306c\u3042",u"\u306a"),(u"\u306c\u3044",u"\u306b"),(u"\u306c\u3048",u"\u306d"),(u"\u306c\u304a",u"\u306e"),(u"\u306c\u3046",u"\u306c"),
       (u"\u3077\u3042",u"\u3071"),(u"\u3077\u3044",u"\u3074"),(u"\u3077\u3048",u"\u307a"),(u"\u3077\u304a",u"\u307d"),(u"\u3077\u3046",u"\u3077"),
       (u"\u3059\u3042",u"\u3055"),(u"\u3059\u3048",u"\u305b"),(u"\u3059\u304a",u"\u305d"),(u"\u3059\u3046",u"\u3059"),
       (u"\u3057\u3085\u3042",u"\u3057\u3083"),(u"\u3057\u3085\u3046",u"\u3057\u3085"),(u"\u3057\u3085\u3048",u"\u3057\u3047"),(u"\u3057\u3085\u304a",u"\u3057\u3087"),(u"\u3057\u3085\u3044",u"\u3057"),
       (u"\u3064\u3042",u"\u305f"),(u"\u3064\u3044",u"\u3061"),(u"\u3064\u3048",u"\u3066"),(u"\u3064\u304a",u"\u3068"),(u"\u3064\u3046",u"\u3064"),
       (u"\u3086\u3042",u"\u3084"),(u"\u3086\u3048",u"\u3044\u3047"),(u"\u3086\u304a",u"\u3088"),(u"\u3086\u3046",u"\u3086"),
       (u"\u305a\u3042",u"\u3056"),(u"\u305a\u3044",u"\u3058"),(u"\u305a\u3048",u"\u305c"),(u"\u305a\u304a",u"\u305e"),(u"\u305a\u3046",u"\u305a"),
       (u"\u308f\u3044",u"\u3046\u3043"),(u"\u308f\u3046",u"\u3046"),(u"\u308f\u3048",u"\u3046\u3047"),(u"\u308f\u304a",u"\u3092"),(u"\u308f\u3042",u"\u308f"),
       (u'\u3046\u3043\u3066\u3085', u'\u3046\u3043\u3065'), # sounds a bit better for words like 'with'
       (u'\u3085\u3046',u'\u3085'), # and 'the' (especially with a_as_in_ago mapping to u'\u3046\u304a'; it's hard to get a convincing 'the' though, especially in isolation)
       (u'\u3050\u3050',u'\u3050'), # gugu -> gu, sometimes comes up with 'gl-' combinations
       (u'\u30fc\u30fc+',u'\u30fc'), # in case we put 30fc in the table AND a stress mark has been applied to it
       (u'^(.)$',u'\\1\u30fc'), # lengthen any word that ends up as a single kana (otherwise can be clipped badly)
    (u'^([\u3042\u3070\u3060\u304c\u304b\u3089\u307e\u306a\u3071\u3055\u305f\u3084\u3056\u308f]\u3044)$',u'\\1\u30fc'), # ditto for -ai (TODO: -ao might need lengthening sometimes?? depends on context.  -ei, -oi, -ou seem OK)
    ],
    cleanup_func = hiragana_to_katakana
  ),

  "deva-approx" : makeDic(
  "Rough approximation using Devanagari (for getting Indian computer voices to speak some English words; works with some words better than others); can also be used to approximate Devanagari words in English phonemes",
    (u'\u02c8',primary_stress),
    (u'\u093e',a_as_in_ah),(u'\u0906',a_as_in_ah,False),
    (u'\u0905',u_as_in_but),
    (u'\u092c',b),
    (u'\u091b',ch),(u'\u091a',ch,False),
    (u'\u0926',d),(u'\u0921',d,False), # TODO: check which sounds better for reading English words
    (u'\u0920',th_as_in_them), # (very approximate)
    (u'\u0948',e_as_in_them),(u'\u0910',e_as_in_them,False),
    (u'\u0947',a_as_in_ate),(u'\u090f',a_as_in_ate,False),
    (u'\u092b\u093c',f),
    (u'\u0917',g),
    (u'\u0917\u093c',g,False), # (Hindi; differs in others)
    (u'\u0939',h),(u'\u0903',h,False),
    (u'\u093f',i_as_in_it),(u'\u0907',i_as_in_it,False),
    (u'\u0940',e_as_in_eat),(u'\u0908',e_as_in_eat,False),
    (u'\u091c',j_as_in_jump),
    (u'\u0915',k),(u'\u0916',k,False),
    (u'\u0916\u093c',opt_scottish_loch),
    (u'\u0915\u093c',opt_scottish_loch,False), # ?
    (u'\u0932',l),
    (u'\u092e',m),
    (u'\u0928',n),(u'\u0923',n,False),
    (u'\u0902',ng),
    (u'\u092a',p),
    (u'\u092b',f,False), # (Hindi; p in some others?)
    (u'\u0930',r),(u'\u0921\u093c',r,False),
    (u'\u0938',s),
    (u'\u0936',sh), (u'\u0937',sh,False),
    (u'\u091f',t),(u'\u0924',t,False),(u'\u0925',t,False),
    (u'\u0941',opt_u_as_in_pull),(u'\u0909',opt_u_as_in_pull,False),
    (u'\u0942',oo_as_in_food),(u'\u090a',oo_as_in_food,False),
    (u'\u094c',close_to_or),(u'\u0914',close_to_or,False),
    (u'\u094b',opt_ol_as_in_gold),(u'\u0913',opt_ol_as_in_gold,False),
    (u'\u0935',v),(w,u'\u0935',False),
    (u'\u092f',y),
    (u'\u091c\u093c',z),
    (u'\u091d\u093c',ge_of_blige_etc),
    (u'\u0901',ipa_colon),
    word_separator=" ",phoneme_separator="",
    stress_comes_before_vowel=True,
    safe_to_drop_characters=True, # it's an approximation
    approximate_missing=True,
    cleanup_regexps=[
       # add virama if consonant not followed by vowel, and delete default vowel after consonant:
       (u'([\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939]\u093c?)(?![\u0905\u093e-\u0942\u0947\u0948\u094b\u094c])',u'\\1\u094d'),(u'(?<=[\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939\u093c])\u0905',u''),(u'(.)\u094d\u02c8',u'\u02c8\\1'),
       # replace vowel signs with vowel letters if not preceded by consonants:
       (u'(?<![\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939\u093c])\u093e',u'\u0906'),
       (u'(?<![\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939\u093c])\u093f',u'\u0907'),
       (u'(?<![\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939\u093c])\u0940',u'\u0908'),
       (u'(?<![\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939\u093c])\u0941',u'\u0909'),
       (u'(?<![\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939\u093c])\u0942',u'\u090a'),
       (u'(?<![\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939\u093c])\u0947',u'\u090f'),
       (u'(?<![\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939\u093c])\u0948',u'\u0910'),
       (u'(?<![\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939\u093c])\u094b',u'\u0913'),
       (u'(?<![\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939\u093c])\u094c',u'\u0914')],
    cvtOut_func=unicode_preprocess,
    cvtOut_regexps=[
       # add default vowel when necessary:
       (u'([\u0902\u0903\u0915-\u0917\u091a-\u091d\u091f-\u0928\u092a-\u0930\u0932\u0935-\u0939]\u093c?)(?![\u0905\u094d\u093e-\u0942\u0947\u0948\u094b\u094c])',u'\\1\u0905'),(u'\u094d',u''),
       # 'add h' approximations:
       (u'\u092d',u'\u092c\u0939'),(u'\u0927',u'\u0922\u0939'),(u'\u0918',u'\u0917\u0939'),(u'\u091d',u'\u091c\u0939'),(u'\u0922\u093c',u'\u0921\u093c\u0939'),
    ]),

  "names" : makeDic(
    "Lexconvert internal phoneme names (sometimes useful with the --phones option while developing new formats)",
     *[(phName,phVal) for phName,phVal in phonemes.items()])}

# The mainopt_...() functions are the main options
# (if you implement a new one, main() will detect it);
# 1st line of doc string should be parameter summary
# (start the doc string with \n if no parameters); if 1st
# character of doc string is * then this function is put
# among the first in the help (otherwise alphabetically).
# If function returns a string, that's taken to be a
# message to be printed with error exit.  Same if it raises
# an exception of type Message.

def mainopt_try(i):
   """*<format> [<pronunciation>]
Convert input from <format> into eSpeak and try it out.
(Requires the 'espeak' command.)
E.g.: python lexconvert.py --try festival h @0 l ou1
 or: python lexconvert.py --try unicode-ipa '\\u02c8\\u0279\\u026adn\\u0329' (for Unicode put '\\uNNNN' or UTF-8)"""
   format = sys.argv[i+1]
   if not format in lexFormats: return "No such format "+repr(format)+" (use --formats to see a list of formats)"
   for phones in getInputText(i+2,"phonemes in "+format+" format",'maybe'):
      espeak = convert(phones,format,'espeak')
      w = os.popen("espeak -x","w")
      getBuf(w).write(markup_inline_word("espeak",espeak)+as_utf8('\n')) # separate process each item for more responsiveness from the console (sending 'maybe' to getInputText means won't lose efficiency if not read from console)

def mainopt_trymac(i):
   """*<format> [<pronunciation>]
Convert phonemes from <format> into Mac and try it using the Mac OS 'say' command"""
   format = sys.argv[i+1]
   if not format in lexFormats: return "No such format "+repr(format)+" (use --formats to see a list of formats)"
   for resp in getInputText(i+2,"phonemes in "+format+" format",'maybe'):
      mac = convert(resp,format,'mac')
      toSay = markup_inline_word("mac",mac)
      print(as_printable(toSay))
      w = os.popen(macSayCommand()+" -v Vicki","w")
      getBuf(w).write(toSay) # Need to specify a voice because the default voice might not be able to take Apple phonemes.  Vicki has been available since 10.3, as has the 'say' command (previous versions need osascript, see Gradint's code)

def mainopt_trymac_uk(i):
   """*<format> [<pronunciation>]
Convert phonemes from <format> and try it with Mac OS British voices (see --mac-uk for details)"""
   assert sys.version_info[0]==2, "--trymac-uk has not been tested with Python 3, I don't want to risk messing up your system files, please use Python 2"
   format = sys.argv[i+1]
   if not format in lexFormats: return "No such format "+repr(format)+" (use --formats to see a list of formats)"
   for resp in getInputText(i+2,"phonemes in "+format+" format",'maybe'):
     macuk = convert(resp,format,'mac-uk')
     m = MacBritish_System_Lexicon("",os.environ.get("MACUK_VOICE","Daniel"))
     try:
      try: m.speakPhones(macuk.split())
      finally: m.close()
     except KeyboardInterrupt:
      sys.stderr.write("Interrupted\n")

def mainopt_phones(i):
   """*<format> [<words>]
Use eSpeak to convert text to phonemes, and then convert the phonemes to format 'format'.
E.g.: python lexconvert.py --phones unicode-ipa This is a test sentence.
Set environment variable PHONES_PIPE_COMMAND to an additional command to which to write the phones as well as standard output.  (If standard input is a terminal then this will be done separately after each line.)
(Some commercial speech synthesizers do not work well when driven entirely from phonemes, because their internal format is different and is optimised for normal text.)
Set format to 'all' if you want to see the phonemes in ALL supported formats."""
   format = sys.argv[i+1]
   if format=="example": return "The 'example' format cannot be used with --phones; try --convert, or did you mean --phones festival" # could allow example anyway as it's basically Festival, but save confusion as eSpeak might not generate the same phonemes if our example words haven't been installed in the system's eSpeak.  (Still allow it to be used in --try etc though.)
   if not format in lexFormats and not format=="all": return "No such format "+repr(format)+" (use --formats to see a list of formats)"
   hadOneoff = False
   for response in getInputText(i+2,"text",'maybe'):
    response = pipeThroughEspeak(as_utf8(response).replace(u'\u2032'.encode('utf-8'),as_utf8('')).replace(u'\u00b4'.encode('utf-8'),as_utf8('')).replace(u'\u02b9'.encode('utf-8'),as_utf8('')).replace(u'\u00b7'.encode('utf-8'),as_utf8(''))) # (remove any 2032 and b7 pronunciation marks before passing to eSpeak)
    if not as_utf8('\n') in response.rstrip() and as_utf8('command') in response: return response.strip() # 'bad cmd' / 'cmd not found'
    if format=="all": formats = sorted(k for k in lexFormats.keys() if not k=="example")
    else: formats = [format]
    for format in formats:
       def out(doOneoff=True):
          if len(formats)>1: writeFormatHeader(format)
          if doOneoff: getBuf(sys.stdout).write(as_utf8(checkSetting(format,"inline_oneoff_header")))
          getBuf(sys.stdout).write(as_utf8(checkSetting(format,"inline_header")))
          output_clauses(format,convert(parseIntoWordsAndClauses("espeak",response),"espeak",format))
          getBuf(sys.stdout).write(as_utf8(checkSetting(format,"inline_footer")))
          print("")
          sys.stdout.flush() # in case it's being piped
       out(not hadOneoff) ; hadOneoff = True
       if os.environ.get("PHONES_PIPE_COMMAND",""):
          o,sys.stdout = sys.stdout,os.popen(os.environ["PHONES_PIPE_COMMAND"],'w')
          out()
          sys.stdout = o

def mainopt_ruby(i):
   """*<format> [<words>]
Like --phones but outputs the result as HTML RUBY markup, with each word's pronunciation symbols placed above the corresponding English word.
E.g.: python lexconvert.py --ruby unicode-ipa This is a test sentence.
This option is made more complicated by the fact that different versions of eSpeak may space the phoneme output differently, for example when handling numbers; if your eSpeak version is not recognised then all numbers are unannotated. Anyway you are advised not to rely on this option working with the new development NG versions of eSpeak. If the version you have behaves unexpectedly, words and phonemes output might lose synchronisation. However this option is believed to be stable when used with simple text and the original eSpeak.
You can optionally set the RUBY_GRADINT_CGI environment variable to the URL of an instance of Gradint Web Edition to generate audio links for each word.  If doing this in a Web Adjuster filter, see comments in the lexconvert source for setup details."""
   # htmlFilter with --htmlText of course.  Set separator to two newlines and copy the generated 'h5a' function (from a manual run or the lexconvert source) into Adjuster's headAppend option (but don't expect HTML5 audio to work from Adjuster's submitBookmarklet option; pronunciation links will take you off the page if it doesn't).
   # Use double newlines as single newlines are used in the h5a script; adding that script via bookmarklet doesn't always run it
   format = sys.argv[i+1]
   if format=="example": return "The 'example' format cannot be used with --ruby; did you mean festival?" # as above
   elif format=="all": return "The --phones all option cannot be used with --ruby" # (well you could implement it if you want but the resulting ruby would be quite unwieldy)
   if not format in lexFormats: return "No such format "+repr(format)+" (use --formats to see a list of formats)"
   text = as_utf8(getInputText(i+2,"text")).replace(u'\u2019'.encode('utf-8'),as_utf8("'")).replace(u'\u2032'.encode('utf-8'),as_utf8("'")).replace(u'\u00b4'.encode('utf-8'),as_utf8("'")).replace(u'\u02b9'.encode('utf-8'),as_utf8("'")).replace(u'\u00b7'.encode('utf-8'),as_utf8('')).replace(u'\u00a0'.encode('utf-8'),as_utf8(' '))
   # eSpeak's basic idea of an alphabetical word (most versions?) -
   wordRegexps = [r"(?:[A-Z]+['?-])*(?:(?:(?<![A-z.])(?:[A-z]\.)+[A-z](?![A-z.]))|[A-Z]+[a-z](?![A-z])|[A-Z][A-Z]+(?![a-z][A-z])|[A-Z]?(?:[a-z]['?-]?)+|[A-Z])"]
   # A dot, when not part of an elipses, followed by a letter is pronounced "dot", and two of them are pronounced "dot dot":
   wordRegexps.append(r"(?<!\.\.)\.(?=[A-z])|(?<!\.)\.(?=\.[A-z])")
   # ! followed by a letter is pronounced "exclamation", and .! is "dotexclamation"; @ symbols similarly; copyright
   atEtc = u"(?:[@!:]|\u00a9)*".encode('utf-8')
   wordRegexps.append(as_utf8(r"\.?[!@]+(?=[A-z])|(?<![A-z])@")+atEtc+as_utf8("(?![A-z])|")+unichr(0xa9).encode('utf-8')+atEtc)
   # : between numbers if NOT followed by 2 digits:
   wordRegexps.append(r"(?<![A-z]):(?![A-z]|[0-9][0-9])")
   # - between numbers
   wordRegexps.append(r"(?<=[0-9])-(?=[0-9])")
   # TODO: if you paste in (e.g.) CJK characters, eSpeak will say "symbol-symbol-symbol" etc, but this is not accounted for by the above regexp so it'll go onto following words.
   vLine = espeak_version_line()
   if "1.45." in vLine:
      # This seems to work in eSpeak 1.45:
      # (TODO: test leading 0s & leading decimal)
      # a number of 4 digits or less (with any number of digits after the decimal point) is grouped as 1 word:
      wordRegexps.append(r"(?<![0-9])[0-9]{1,4}(?:\.[0-9]+)?(?!,?[0-9])")
      # and a number of 1 to 3 digits with any number of 000 or ,000 groups, with optional decimal point followed by any number of digits, OR when placed before an integer number of 3-digit groups, is grouped as 1 word:
      wordRegexps.append(r"[0-9]{1,3}(?:,?000)*(?:\.[0-9]+)?,?(?=(?:,?[0-9]{3,3})*,?(?:[^0-9]|$))")
      text2 = text
   elif "1.48." in vLine:
      # In eSpeak 1.48 the groups are smaller.
      # Decimal point and everything after it = individual
      wordRegexps.append(r"(?<=[0-9])\.(?=[0-9])")
      for places in range(25): # TODO: really want unbounded, but (?<=...) is fixed-length
         wordRegexps.append(r"(?<=[0-9]\."+"[0-9]"*places+r")[0-9]")
      # Number with a leading dot grouped as 1 word:
      wordRegexps.append(r"(?<![0-9])\.[0-9]+")
      # TODO: leading 0s (0000048 goes to 0 000 048)
      # For normal numbers:
      # A null string w. 3 or 6 digits to go and digits b4 shld match for 'thousand', 'million' (unless 3+ digits are leading 0s, or fewer than 3 leading 0s and whole thing begins with a 0, or it's part of a decimal expansion, in which case different rules apply, but (?<=...) must be fixed-length, so we need another one of these awful loops) :
      for prevDigits in range(10):
         for beforeThat in ["^",r"[^.0-9,]"]: # beginning of string, or something OTHER than a decimal point / num
            wordRegexps.append(r"(?<="+beforeThat+"[1-9]"+"[0-9,]"*prevDigits+r")(?<!,)(?<!000)(?# empty string )(?=(?:,?(?:[0-9]{3,3}))+(?:[^0-9]|$))")
      # 1-9 (not 0) with 2, 5 or 8 etc digits to go = "N-hundred-and" :
      wordRegexps.append(r"[1-9](?=[0-9][0-9](?:,?(?:[0-9]{3,3}))*(?:[^0-9]|$))")
      # + 0 with 2 digits to go when preceded by digits = "and", as long as followed by at least one non-0:
      wordRegexps.append(r"(?<=[0-9,])0(?=(?:[0-9][1-9]|[1-9][0-9])(?:[^0-9,]|$))")
      # 1 or 2 digits with 0,3,6.. to go = "seventy-six" or whatever, as long as they're not both 0 :
      wordRegexps.append(r"(?:0[1-9]|[1-9][0-9]?)(?=(?:,?(?:[0-9]{3,3}))*(?:[^0-9]|$))")
      # 0 by itself (not preceded by digits) = "nought" :
      wordRegexps.append(r"(?<![0-9])0(?=[^0-9]|$)")
      wordRegexps.insert(0,"(?<=[^A-Za-z0-9_-])(?:of|on|in|that|with|for|was) (?:the|a)(?= )")
      wordRegexps.insert(0,"(?:Of|On|In|That|With|For|Was) (?:the|a)(?= )")
      wordRegexps.insert(0,"(?<=[^A-Za-z0-9_-])not a(?= )")
      wordRegexps.insert(0,"(?<=[^A-Za-z0-9_-])(?:some|that) one(?= )")
      wordRegexps.insert(0,"(?:Some|That) one(?= )")
      text2 = text
   else: text2 = re.sub(r"\.?[0-9]+","",text) # unknown eSpeak version: don't annotate the numbers
   response = pipeThroughEspeak(text2)
   if not as_utf8('\n') in response.rstrip() and as_utf8('command') in response: return response.strip() # 'bad cmd' / 'cmd not found'
   gradint_cgi = os.environ.get("RUBY_GRADINT_CGI","")
   if gradint_cgi:
      linkStart,linkEnd = lambda w:maybe_bytes('<a href="',w)+maybe_bytes(gradint_cgi,w)+maybe_bytes('?js=[[',w)+w.replace(maybe_bytes('%',w),maybe_bytes('%25',w)).replace(maybe_bytes('&',w),maybe_bytes('%26',w))+maybe_bytes(']]&jsl=en" onclick="return h5a(this);">',w), '</a>'
      print(r"""<script><!-- // HTML5-audio function
function h5a(link) {
 if (document.createElement) {
   var ae = document.createElement('audio');
   if (ae.canPlayType && function(s){return s!="" && s!="no"}(ae.canPlayType('audio/mpeg'))) {
     ae.setAttribute('src', link.href);
     ae.play(); return false;
   } else if (ae.canPlayType && function(s){return s!="" && s!="no"}(ae.canPlayType('audio/ogg'))) {
     ae.setAttribute('src', link.href+"&filetype=ogg");
     ae.play(); return false; }
 } return true; }
//--></script>""")
   else: linkStart,linkEnd = lambda w:maybe_bytes("",w), ""
   rubyList = []
   for clause in parseIntoWordsAndClauses("espeak",response):
      for w in clause:
         converted = convert(w,"espeak",format)
         if not converted: continue # e.g. a lone _:_:
         m = markup_inline_word(format,converted)
         rubyList.append(linkStart(w)+m.replace(maybe_bytes("&",m),maybe_bytes("&amp;",m)).replace(maybe_bytes("<",m),maybe_bytes("&lt;",m))+maybe_bytes(linkEnd,w))
   rubyList.reverse() # so can pop() left-to-right order
   # Write out re.sub ourselves, because (1) some versions of the library (e.g. on 2.7.12) try to do some things in-place, and we're using previous-context regexps that aren't compatible with previous things having been already <ruby>'ified, and (2) if we match a 0-length string, re.finditer won't ALSO return a non-0 length match starting in the same place, and we want both (so we're using wordRegexps as a list rather than an | expression)
   matches = {}
   debug = False # if True, will add ruby title=(index of the regexp that matched)
   debugCount = 0
   for r in wordRegexps:
      for match in re.finditer(maybe_bytes(r,text),text):
         matches[(match.start(),match.end())] = debugCount
      debugCount += 1
   i = 0 ; r = []
   def cmpFunc(a,b):
      (s1,e1),(s2,e2) = a,b
      if s1<s2: return -1
      if s1>s2: return 1
      if e1>e2: return -1
      if e1<e2: return 1
      return 0
   for start,end in sorted(list(matches.keys()),cmpFunc):
      if start<i: continue # overlap??
      r.append(text[i:start])
      if start==end: m = "&nbsp;"
      else: m = text[start:end].replace(maybe_bytes("&",text),maybe_bytes("&amp;",text)).replace(maybe_bytes("<",text),maybe_bytes("&lt;",text))
      try: rt = rubyList.pop()
      except: rt = "ERROR" # we've lost synchronisation
      if debug: title = as_utf8(" title=")+as_utf8(str(matches[(start,end)]))
      else: title = as_utf8("")
      r.append(as_utf8("<ruby")+title+as_utf8("><rb>")+m+as_utf8("</rb><rt>")+rt+as_utf8("</rt></ruby>"))
      i = end
   r.append(text[i:])
   while rubyList: # oops, lost synchronisation the other way (TODO: show this per-paragraph? but don't call eSpeak too many times if processing many short paragraphs)
      r.append(as_utf8("<ruby><rb>ERROR</rb><rt>")+rubyList.pop()+as_utf8("</rt></ruby>"))
   out = as_utf8("").join(r)
   if not out.endswith(as_utf8("\n")): out += as_utf8("\n")
   getBuf(sys.stdout).write(out)

def pipeThroughEspeak(inpt):
   "Writes inpt to espeak -q -x (in chunks if necessary) and returns the result"
   assert type(inpt)==bytes
   bufsize = 8192 # careful not to set this too big, as the OS might limit it (TODO can we check?)
   ret = []
   while len(inpt) > bufsize:
      splitAt = inpt.rfind('\n',0,bufsize)+1
      if not splitAt: # no newline, try to split on space
         splitAt = inpt.rfind(' ',0,bufsize)+1
         if not splitAt:
            sys.stderr.write("Note: had to split eSpeak input and couldn't find a newline or space to do it on\n")
            splitAt = bufsize
      response = pipeThroughEspeak(inpt[:splitAt])
      if not '\n' in response.rstrip() and 'command' in response: return response.strip() # 'bad cmd' / 'cmd not found'
      ret.append(response) ; inpt=inpt[splitAt:]
   try: w,r=os.popen4("espeak -q -x",bufsize=bufsize) # Python 2
   except AttributeError: # Python 3
      import subprocess
      proc=subprocess.Popen(['espeak','-q','-x'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
      w = proc.stdin
      r = None
   if r:
      getBuf(w).write(inpt) ; w.close()
      r = getBuf(r).read()
   else: # Python 3
      w.write(inpt)
      out,err=proc.communicate()
      r = as_utf8("")
      if out: r += out
      if err: r += err
   return as_utf8("\n").join(ret) + r

def espeak_version_line(): return os.popen("espeak -h 2>&1").read().strip().split("\n")[0]

def writeFormatHeader(format):
   "Writes a header for 'format' when outputting in all formats.  Assumes the output MIGHT end up being more than one line."
   global writeFormatHeader_called
   if writeFormatHeader_called: print("")
   print(format)
   print('-'*len(format))
   writeFormatHeader_called = True
writeFormatHeader_called = False

def mainopt_check_variants(i):
   # undocumented (won't appear in help text)
   groups = {}
   for k,v in lexFormats['espeak'].items():
      if type(k)==str:
         intV = int(v)
         if not intV in consonants:
            groups.setdefault(intV,[]).append((v,k))
   i = groups.items() ; i.sort()
   for k,v in i:
      if len(v)==1: continue
      v.sort()
      while True:
         print("Group "+str(k))
         es = os.popen("espeak -x","w")
         getBuf(es).write(as_utf8('\n').join([markup_inline_word("espeak",w) for _,w in v]))
         del es
         if not int(str(input("Again? 1/0: "))): break

def mainopt_check_for_similar_formats(i):
   # undocumented (won't appear in help text)
   items = lexFormats.items() ; r = []
   while items:
      k1,dic1 = items[0]
      for k2,dic2 in items[1:]:
         diff = 0
         for kk,vv in dic1.items():
            if not type(kk)==int: continue
            if kk==syllable_separator: continue
            if not dic2.get(kk,"!"+vv)==vv: diff += 1
         r.append((diff,k1,k2))
      items = items[1:]
   r.sort() ; had = set()
   for diffs,format1,format2 in r:
      if format1 in had and format2 in had: continue
      had.add(format1) ; had.add(format2)
      if "names" in had: break
      print(str(diffs)+" phoneme differences between "+format1+" and "+format2)

def festival_group_stress(pronunc):
   "Special-case cleanup_func for the Festival format"
   # TODO: do we ever need to add extra consonants to the
   # previous group instead of the next group?  (not sure
   # what difference it makes to the synthesis, but it
   # might make the entry a bit more readable)
   groups = [] ; thisGroup = [[],'0',False] # phon,stress,complete
   for phon in pronunc.split():
      if phon in ['0','1','2']:
         if groups and phon >= groups[-1][1]:
            groups[-1][1]=phon
         continue
      thisGroup[0].append(phon)
      if phon[:1] in 'aeiou@':
         thisGroup[2]=True
         groups.append(thisGroup)
         thisGroup = [[],'0',False]
   if thisGroup[0]: groups.append(thisGroup)
   if len(groups)>=2 and not groups[-1][2]:
      groups[-2][0] += groups[-1][0]
      del groups[-1]
   return "("+' '.join(("(("+' '.join(g[0])+') '+g[1]+")") for g in groups)+")"

def mainopt_convert(i):
   """*<from-format> <to-format>
Convert a user lexicon (generally from its default filename; if this cannot be found then lexconvert will tell you what it should be).
E.g.: python lexconvert.py --convert festival cepstral"""
   fromFormat = sys.argv[i+1]
   toFormat = sys.argv[i+2]
   if fromFormat==toFormat: return "Cannot convert a lexicon to its own format (that could result in it being truncated)"
   if toFormat=="mac-uk": return "Cannot permanently save a Mac-UK lexicon; please use the --mac-uk option to read text"
   if toFormat=="example": return "Cannot overwrite the built-in example lexicon"
   for f in [fromFormat,toFormat]:
      if not f in lexFormats: return "No such format "+repr(f)+" (use --formats to see a list of formats)"
   try:
      fname=getSetting(toFormat,"lex_filename")
      getSetting(toFormat,"lex_entry_format") # convert_user_lexicon will need this
   except KeyError: fname = None
   if not fname: return "Write support for lexicons of format '%s' not yet implemented (need at least lex_filename and lex_entry_format); try using --phones or --phones2phones options instead" % (toFormat,)
   if toFormat=="espeak":
      assert fname=="en_extra", "If you changed eSpeak's lex_filename in the table you also need to change the code below"
      if os.system("mv en_extra en_extra~ && (grep \" // \" en_extra~ || true) > en_extra"): sys.stderr.write("Warning: en_extra not found, making a new one\n(espeak compile will probably fail in this directory)\n") # otherwise keep the commented entries, so can incrementally update the user lexicon only
      outFile=open(fname,"a")
   else:
      l = 0
      try:
         f = open(fname)
         l = getBuf(f).read()
         del f
      except: pass
      assert not l, "File "+replHome(fname)+" already exists and is not empty; are you sure you want to overwrite it?  (Delete it first if so)" # (if you run with python -O then this is ignored, as are some other checks so be careful)
      outFile=open(fname,"w")
   print ("Writing %s lexicon entries to %s file %s" % (fromFormat,toFormat,fname))
   try: convert_user_lexicon(fromFormat,toFormat,outFile)
   except Message:
     print (" - error, deleting "+fname)
     os.remove(fname) ; raise

def mainopt_festival_dictionary_to_espeak(i):
   """<location>
Convert the Festival Oxford Advanced Learners Dictionary (OALD) pronunciation lexicon to eSpeak.
You need to specify the location of the OALD file in <location>,
e.g. for Debian festlex-oald package: python lexconvert.py --festival-dictionary-to-espeak /usr/share/festival/dicts/oald/all.scm
or if you can't install the Debian package, try downloading http://ftp.debian.org/debian/pool/non-free/f/festlex-oald/festlex-oald_1.4.0.orig.tar.gz, unpack it into /tmp, and do: python lexconvert.py --festival-dictionary-to-espeak /tmp/festival/lib/dicts/oald/oald-0.4.out
In all cases you need to cd to the eSpeak source directory before running this.  en_extra will be overwritten.  Converter will also read your ~/.festivalrc if it exists.  (You can later incrementally update from ~/.festivalrc using the --convert option; the entries from the system dictionary will not be overwritten in this case.)  Specify --without-check to bypass checking the existing eSpeak pronunciation for OALD entries (much faster, but makes a larger file and in some cases compromises the pronunciation quality)."""
   try: festival_location=sys.argv[i+1]
   except IndexError: return "Error: --festival-dictionary-to-espeak must be followed by the location of the festival OALD file (see help text)"
   try: open(festival_location)
   except: return "Error: The specified OALD location '"+festival_location+"' could not be opened"
   try: open("en_list")
   except: return "Error: en_list could not be opened (did you remember to cd to the eSpeak dictsource directory first?"
   convert_system_festival_dictionary_to_espeak(festival_location,not '--without-check' in sys.argv,not os.system("test -e ~/.festivalrc"))

def mainopt_syllables(i):
   """[<words>]
Attempt to break 'words' into syllables for music lyrics (uses espeak to determine how many syllables are needed)"""
   # As explained on mainopt_ruby's help text, espeak -x output can't be relied on to always put a space between every input word.  Rather than try to guess what espeak is going to do, here we simply put a newline after every input word instead.  This might affect eSpeak's output (so not recommended for mainopt_ruby), but it should be OK for just counting the syllables.  (Also, the assumption that the input words have been taken from song lyrics usefully rules out certain awkward punctuation cases.)
   for txt in getInputText(i+1,"word(s)",'maybe'):
      words=txt.split()
      response = pipeThroughEspeak(as_utf8('\n').join(as_utf8(w) for w in words).replace(as_utf8("!"),as_utf8("")).replace(as_utf8(":"),as_utf8("")).replace(as_utf8("."),as_utf8("")))
      if not as_utf8('\n') in response.rstrip() and as_utf8('command') in response: return response.strip() # 'bad cmd' / 'cmd not found'
      rrr = response.split(as_utf8("\n"))
      print (" ".join([hyphenate(word,sylcount(convert(line,"espeak","example"))) for word,line in zip(words,filter(lambda x:x,rrr))]))
      sys.stdout.flush() # in case piped

def wordSeparator(format):
   """Returns the effective word separator of format (remembering that it defaults to same as phoneme_separator"""
   return checkSetting(format,"word_separator",checkSetting(format,"phoneme_separator"," "))

def mainopt_phones2phones(i):
   """*<format1> <format2> [<phonemes in format1>]
Perform a one-off conversion of phonemes from format1 to format2 (format2 can be 'all' if you want)""" # If format1 is 'example' and you don't specify phonemes, we take the words from the example lexicon.  But don't say that in the help string because it might confuse the issue about phonemes being optional on the command line and prompted for if not specified and stdin is not piped in all formats other than 'example'.
   format1,format2 = sys.argv[i+1],sys.argv[i+2]
   if not format1 in lexFormats: return "No such format "+repr(format1)+" (use --formats to see a list of formats)"
   if not format2 in lexFormats and not format2=="all": return "No such format "+repr(format2)+" (use --formats to see a list of formats)"
   if format1=="example" and len(sys.argv)<=i+3:
     if stdin_is_terminal(): txt=""
     else: txt=getBuf(sys.stdin).read() # and it might still be ""
     if txt: parseIntoWordsAndClauses(format1,txt)
     else: clauses=[[x[1]] for x in getSetting('example','lex_read_function')()]
   else: clauses = parseIntoWordsAndClauses(format1,getInputText(i+3,"phonemes in "+format1+" format"))
   if format2=="all": formats = sorted(k for k in lexFormats.keys() if not k=="example")
   else: formats = [format2]
   for format2 in formats:
     if len(formats)>1: writeFormatHeader(format2)
     getBuf(sys.stdout).write(as_utf8(checkSetting(format2,"inline_header")))
     output_clauses(format2,convert(clauses,format1,format2))
     getBuf(sys.stdout).write(as_utf8(checkSetting(format2,"inline_footer"))) ; print("")

def parseIntoWordsAndClauses(format,phones):
   "Returns list of clauses, each of which is a list of words, assuming 'phones' are in format 'format'"
   wordSep = checkSetting(format,"word_separator") # don't use wordSeparator() here - we're splitting, not joining, so we don't want it to default to phoneme_separator
   clauseSep = checkSetting(format,"clause_separator","\n")
   def s(sep):
      if sep==" ": return None # " " means ANY whitespace (TODO: document this?)
      else: return maybe_bytes(sep,phones)
   if clauseSep and type(clauseSep) in [bytes,unicode]:
      clauses = phones.split(s(clauseSep))
   else: clauses = [phones]
   for i in range(len(clauses)):
      if wordSep: clauses[i]=clauses[i].split(s(wordSep))
      else: clauses[i] = [clauses[i]]
      clauses[i] = list(filter(lambda x:x, clauses[i]))
   return list(filter(lambda x:x,clauses))

def mainopt_mac_uk(i):
   """<from-format> [<text>]
Speak text in Mac OS 10.7+ British voices while using a lexicon converted in from <from-format>. As these voices do not have user-modifiable lexicons, lexconvert must binary-patch your system's master lexicon; this is at your own risk! (Superuser privileges are needed the first time. A backup of the system file is made, and all changes are restored on normal exit but if you force-quit then you might need to restore the backup manually. Text speaking needs to be under lexconvert's control because it usually has to change the input words to make them fit the available space in the binary lexicon.) By default the Daniel voice is used; Emily or Serena can be selected by setting the MACUK_VOICE environment variable."""
   # If you have xterm etc, then text will also be printed, with words from the altered lexicon underlined.
   assert sys.version_info[0]==2, "--mac-uk has not been tested with Python 3, I don't want to risk messing up your system files, please use Python 2"
   fromFormat = sys.argv[i+1]
   if not fromFormat in lexFormats: return "No such format "+repr(fromFormat)+" (use --formats to see a list of formats)"
   lex = get_macuk_lexicon(fromFormat)
   try:
      for line in getInputText(i+2,"text",True):
         m = MacBritish_System_Lexicon(line,os.environ.get("MACUK_VOICE","Daniel"))
         try: m.readWithLex(lex)
         finally: m.close()
   except KeyboardInterrupt:
      sys.stderr.write("Interrupted\n")

class Counter(object):
    "A simple class with two static members, count and subcount, for use by the consonant(), vowel() and other() functions"
    c=sc=0
def other():
    "Used by Phonemes() when creating something that is neither a vowel nor a consonant, e.g. a stress mark"
    Counter.c += 1 ; Counter.sc=0 ; return Counter.c
consonants = set() ; mainVowels = set()
def consonant():
    "Used by Phonemes() when creating a consonant"
    r = other() ; consonants.add(r) ; return r
def vowel():
    "Used by Phonemes() when creating a vowel"
    r = other() ; mainVowels.add(r) ; return r
def opt_vowel():
    "Used by Phonemes() when creating an optional vowel (one that has no warning issued if some format doesn't support it)"
    return other()
def variant():
    "Used by Phonemes() when creating a variant of the just-defined vowel/consonant/etc"
    Counter.sc += 1
    while str(Counter.sc).endswith('0'): Counter.sc += 1
    return 0, float('%d.%d' % (Counter.c,Counter.sc))
    # the 0 is so we can say _, name = variant()
    # so as to get some extra indentation

def ifset(var,a,b=""):
   "Checks the environment variable var; if it is set (non-empty), return a, otherwise return b.  Used in LexFormats to create tables with variations set by the environment."
   import os
   if os.environ.get(var,""): return a
   else: return b

def speakjet(symbol,opcode):
   "Special-case function for the Speakjet table"
   assert type(opcode)==int
   if ifset('SPEAKJET_BINARY',1):
      assert not ifset('SPEAKJET_SYM',1), "Cannot set both SPEAKJET_SYM and SPEAKJET_BINARY"
      return chr(opcode)
   else: return ifset('SPEAKJET_SYM',symbol,str(opcode))

def makeDic(doc,*args,**kwargs):
    "Make a dictionary with a doc string, default-bidirectional mappings and extra settings; see LexFormats for how this is used."
    assert type(doc)==str, "doc must be a string"
    d = {} ; duplicates = set()
    for a in args:
        assert type(a)==tuple and (len(a)==2 or len(a)==3)
        k=a[0]
        if k in d: duplicates.add(k)
        v=a[1]
        assert (type(k) in [bytes,unicode] and type(v) in [int,float]) or (type(v) in [bytes,unicode] and type(k) in [int,float]), "Wrong types "+repr(a)+" (did you forget a _, before calling variant() or something?)"
        d[k] = v
        if type(k)==unicode: d[as_utf8(k)] = v
        if len(a)==3: bidir=a[2]
        else: bidir=True
        if bidir:
            # (k,v,True) = both (k,v) and (v,k)
            if v in d: duplicates.add(v)
            d[v] = k
    assert not duplicates, " Duplicate key(s) in "+repr(doc)+": "+", ".join((repr(dup)+"".join(" (="+g+")" for g,val in globals().items() if val==dup)) for dup in sorted(list(duplicates)))+". Did you forget a ,False to suppress bidirectional mapping?" # by the way, Python does not detect duplicate keys in {...} notation - it just lets you overwrite
    missing = [l for l in (list(consonants)+list(mainVowels)) if not l in d]
    # did_approx = False
    if missing and 'approximate_missing' in kwargs:
      for miss,approxTo in [
          # TODO: put this table somewhere else?
          # (If the thing on the right is just 1 item, we could make the thing on the left a variant of it.  But that might not be a good idea unless they're really very close, since if it's a variant then the substitution is done without warning even if approximate_missing is not set.)
          (a_as_in_ago, [u_as_in_but]),
          (a_as_in_air, [e_as_in_them,r]),
          (ear, [e_as_in_eat,u_as_in_but]),
          (oor_as_in_poor, [close_to_or]), # TODO: ,r?
          (a_as_in_ah,[a_as_in_apple]), # this seems to be missing in some American voices (DecTalk, Keynote, SAM); TODO: is this the best approximation we can do?
          (a_as_in_apple,[a_as_in_ah]), # the reverse of the above, for Devanagari
          (o_as_in_orange,[oo_as_in_food]),(o_as_in_go,[oo_as_in_food]),(oy_as_in_toy,[oo_as_in_food,i_as_in_it]),(o_as_in_now,[a_as_in_ah, w]),(e_as_in_herd,[u_as_in_but,u_as_in_but]),(ar_as_in_year,[u_as_in_but,u_as_in_but]),(eye,[a_as_in_ah,y]),(th_as_in_think,[th_as_in_them]), # (Devanagari: is this really the best we can do?)
          ]:
        if miss in missing and all(x in d for x in approxTo):
          d[miss]=maybe_bytes(kwargs.get("phoneme_separator"," "),d[approxTo[0]]).join(d[x] for x in approxTo)
          # did_approx = True
          missing.remove(miss)
    # if did_approx: doc="(approx.) "+doc # and see also the code in makeVariantDic.  Commenting out because this is misleading: the formats where we didn't do a did_approx might also contain approximations of some kind.  Incidentally there are some British English voices that need approximate_missing (e.g. Apollo 2)
    d[("settings","doc")] = doc
    if missing:
       import sys ; sys.stderr.write("WARNING: Some non-optional vowels/consonants are missing from "+repr(doc)+"\nThe following are missing: "+", ".join("/".join(g for g,val in globals().items() if val==m) for m in missing)+"\n")
    for k,v in kwargs.items(): d[('settings',k)] = v
    assert type(d.get(('settings','cleanup_regexps'),[]))==list, "cleanup_regexps must be a list" # not one tuple
    assert type(d.get(('settings','cvtOut_regexps'),[]))==list, "cvtOut_regexps must be a list" # not one tuple
    wsep = d.get(('settings','word_separator'),None)
    psep = d.get(('settings','phoneme_separator'),' ')
    if not wsep==None: assert not wsep in d, "word_separator duplicates with a key in "+repr(doc)
    if not psep==None: assert not psep in d, "phoneme_separator duplicates with a key (did you forget to change the default, or to add a ,False somewhere?) in "+repr(doc)
    global lastDictionaryMade ; lastDictionaryMade = d
    return d
def makeVariantDic(doc,*args,**kwargs):
    "Like makeDic but create a new 'variant' version of the last-made dictionary, modifying some phonemes and settings (and giving it a new doc string) but keeping everything else the same.  Any list settings (e.g. cleanup_regexps) are ADDED to by the variant; other settings and phonemes are REPLACED if they are specified in the variant.  If you don't want subsequent variants to inherit the changes made by this variant, add noInherit=True to the keyword args."
    global lastDictionaryMade
    ldmOld = lastDictionaryMade
    toUpdate = lastDictionaryMade.copy()
    global mainVowels,consonants
    oldV,oldC = mainVowels,consonants
    mainVowels,consonants = [],[] # so makeDic doesn't complain if some vowels/consonants are missing
    if 'noInherit' in kwargs:
       noInherit = kwargs['noInherit']
       del kwargs['noInherit']
    else: noInherit = False
    d = makeDic(doc,*args,**kwargs)
    if noInherit: lastDictionaryMade = ldmOld
    mainVowels,consonants = oldV,oldC
    # if toUpdate[("settings","doc")].startswith("(approx.) ") and not d[("settings","doc")].startswith("(approx.) "): d[("settings","doc")]="(approx.) "+d[("settings","doc")] # TODO: always?
    for k,v in toUpdate.items():
       if type(v)==list and k in d: d[k] = v+d[k]
    toUpdate.update(d) ; return toUpdate
def getSetting(formatName,settingName):
  "Gets a setting from lexFormats, exception if not there"
  return lexFormats[formatName][('settings',settingName)]
def checkSetting(formatName,settingName,default=""):
  "Gets a setting from lexFormats, default if not there"
  return lexFormats[formatName].get(('settings',settingName),default)

import sys,re,os
try: from subprocess import getoutput
except: from commands import getoutput # Python 2
try: bytes # Python 3 and newer Python 2
except: bytes = str # older Python 2
try: unicode # Python 2
except: # Python 3
   unicode,unichr,xrange = str,chr,range
   def chr(x): return bytes([x])
   _builtin_sorted = sorted
   from functools import cmp_to_key
   def sorted(l,theCmp=None):
      if theCmp:
         return _builtin_sorted(l,key=cmp_to_key(theCmp))
      else: return _builtin_sorted(l)
   assert sys.version_info[1] > 4, "lexconvert cannot run on Python 3.4 due to lack of byte-string percent formatting (PEP 461).  Please use Python 3.5+ or stick with Python 2."
def getBuf(f):
   "Return a buffer to which bytes may be written, for Python 2 and 3 compatibility"
   try: return f.buffer # Python 3
   except AttributeError: return f # Python 2

cached_sourceName,cached_destName,cached_dict = None,None,None
def make_dictionary(sourceName,destName):
    "Uses lexFormats to make a mapping dictionary from a particular source format to a particular dest format, and also sets module variables for that particular conversion (TODO: put those module vars into an object in case someone wants to use this code in a multithreaded server)"
    global cached_sourceName,cached_destName,cached_dict
    if (sourceName,destName) == (cached_sourceName,cached_destName): return cached_dict
    source = lexFormats[sourceName]
    dest = lexFormats[destName]
    d = {}
    global dest_consonants ; dest_consonants = set()
    global dest_syllable_sep ; dest_syllable_sep = dest.get(syllable_separator,"")
    global implicit_vowel_before_NL
    implicit_vowel_before_NL = None
    for k,v in source.items():
      if type(k)==tuple: continue # settings
      if type(v) in [bytes,unicode]: continue # (num->string entries are for converting IN to source; we want the string->num entries for converting out)
      if not v in dest: v = int(v) # (try the main version of a variant)
      if not v in dest: continue # (haven't got it - will have to ignore or break into parts)
      assert type(k) in [bytes,unicode]
      d[k] = dest[v]
      if int(v) in consonants: dest_consonants.add(d[k])
      if int(v)==e_as_in_herd and (not implicit_vowel_before_NL or v==int(v)): # TODO: or u_as_in_but ?  used by festival and some other synths before words ending 'n' or 'l' (see usage of implicit_vowel_before_NL later)
        implicit_vowel_before_NL = d[k]
      d[as_utf8(k)] = d[k]
      try: d[as_unicode(k)] = d[k]
      except UnicodeDecodeError: pass
    try:
       if any(type(v)==unicode for v in d.values()): d,dest_consonants=dict((k,as_unicode(v)) for k,v in d.items()),set(as_unicode(v) for v in dest_consonants) # Python 2: if ANY dest are Unicode, make them ALL Unicode
    except UnicodeDecodeError: d,dest_consonants=dict((k,as_utf8(v)) for k,v in d.items()),set(as_utf8(v) for v in dest_consonants) # ... or make them ALL byte-strings if some were binary and not readable as UTF-8
    cached_sourceName,cached_destName,cached_dict=sourceName,destName,d
    return d

warnedAlready = set()
def convert(pronunc,source,dest):
    "Convert pronunc from source to dest.  pronunc can be a string or a list; if a list then we'll recurse on each of the list elements and return a new list (this is meant for batch-converting clauses etc)"
    assert type(pronunc) in [bytes,unicode,list], type(pronunc)
    if source==dest: return pronunc # essential for --try experimentation with codes not yet supported by lexconvert
    if type(pronunc)==list: return [convert(p,source,dest) for p in pronunc]
    func = checkSetting(source,'cvtOut_func')
    if func: pronunc=func(pronunc)
    for s,r in checkSetting(source,'cvtOut_regexps'):
        pronunc=re.sub(maybe_bytes(s,pronunc),maybe_bytes(r,pronunc),pronunc)
    ret = [] ; toAddAfter = None
    dictionary = make_dictionary(source,dest)
    maxLen=max(len(l) for l in dictionary.keys())
    debugInfo=""
    separator = checkSetting(dest,'phoneme_separator',' ')
    safe_to_drop = checkSetting(source,"safe_to_drop_characters")
    while pronunc:
        for lettersToTry in range(maxLen,-1,-1):
            if not lettersToTry:
              if safe_to_drop==True: pass
              elif (not safe_to_drop) or not pronunc[:1] in maybe_bytes(safe_to_drop,pronunc) and not (pronunc[:1],debugInfo) in warnedAlready:
                 warnedAlready.add((pronunc[:1],debugInfo))
                 sys.stderr.write("Warning: ignoring "+source+" character "+repr(pronunc[:1])+debugInfo+" (unsupported in "+dest+")\n")
              pronunc=pronunc[1:] # ignore
            elif pronunc[:lettersToTry] in dictionary:
                debugInfo=" after "+as_printable(pronunc[:lettersToTry])
                toAdd=dictionary[pronunc[:lettersToTry]]
                assert type(toAdd) in [bytes,unicode], type(toAdd)
                isStressMark=(toAdd and toAdd in [maybe_bytes(lexFormats[dest].get(primary_stress,''),toAdd),maybe_bytes(lexFormats[dest].get(secondary_stress,''),toAdd)])
                if toAdd==maybe_bytes(lexFormats[dest].get(syllable_separator,''),toAdd): pass
                elif isStressMark and not checkSetting(dest,"stress_comes_before_vowel"):
                    if checkSetting(source,"stress_comes_before_vowel"): toAdd, toAddAfter = maybe_bytes("",toAdd),toAdd # move stress marks from before vowel to after
                    else: # stress is already after, but:
                        # With Cepstral synth (and kana-approx), stress mark should be placed EXACTLY after the vowel and not any later.  Might as well do this for others also.
                        r=len(ret)-1
                        while ret[r] in dest_consonants or ret[r].endswith(maybe_bytes("*added",ret[r])): r -= 1 # (if that raises IndexError then the input had a stress mark before any vowel) ("*added" condition is there so that implicit vowels don't get the stress)
                        ret.insert(r+1,toAdd) ; toAdd=maybe_bytes("",toAdd)
                elif isStressMark and not checkSetting(source,"stress_comes_before_vowel"): # it's a stress mark that should be moved from after the vowel to before it
                    i=len(ret)
                    while i and (ret[i-1] in dest_consonants or ret[i-1].endswith(maybe_bytes("*added",ret[i-1]))): i -= 1
                    if i: i-=1
                    ret.insert(i,toAdd)
                    if dest_syllable_sep: ret.append(maybe_bytes(dest_syllable_sep,toAdd)) # (TODO: this assumes stress marks are at end of syllable rather than immediately after vowel; correct for Festival; check others; probably a harmless assumption though; mac-uk is better with syllable separators although espeak basically ignores them)
                    toAdd = maybe_bytes("",toAdd)
                # attempt to sort out the festival dictionary's (and other's) implicit_vowel_before_NL
                elif implicit_vowel_before_NL and ret and ret[-1] and toAdd in [maybe_bytes('n',toAdd),maybe_bytes('l',toAdd)] and ret[-1] in dest_consonants: ret.append(maybe_bytes(implicit_vowel_before_NL,toAdd)+maybe_bytes('*added',toAdd))
                elif len(ret)>2 and ret[-2].endswith(maybe_bytes('*added',ret[-2])) and toAdd and not toAdd in dest_consonants and not toAdd==dest_syllable_sep: del ret[-2]
                if toAdd:
                    # Add it, but if toAdd is multiple phonemes, try to put toAddAfter after the FIRST phoneme
                    if separator: toAddList=toAdd.split(separator)
                    else: toAddList = [toAdd] # TODO: won't work for formats that don't have a phoneme separator (doesn't really matter for eSpeak though)
                    ret.append(toAddList[0])
                    if toAddAfter and not toAddList[0] in dest_consonants:
                        ret.append(toAddAfter)
                        toAddAfter=None
                    ret += toAddList[1:]
                pronunc=pronunc[lettersToTry:]
                break
    if toAddAfter: ret.append(toAddAfter)
    if ret and ret[-1]==dest_syllable_sep: del ret[-1] # spurious syllable separator at end
    if not ret: ret = ""
    else: ret=maybe_bytes(separator,ret[0]).join(ret).replace(maybe_bytes('*added',ret[0]),maybe_bytes('',ret[0]))
    for s,r in checkSetting(dest,'cleanup_regexps'):
      ret=re.sub(maybe_bytes(s,ret),maybe_bytes(r,ret),ret)
    func = checkSetting(dest,'cleanup_func')
    if func: return func(ret)
    else: return ret

def unicode_preprocess(pronunc):
   "Special-case cvtOut_func for unicode-ipa etc: tries to catch \\uNNNN etc"
   if maybe_bytes("\\u",pronunc) in pronunc and not maybe_bytes('"',pronunc) in pronunc: # maybe \uNNNN copied from Gecko on X11, can just evaluate it to get the unicode
      # (NB make sure to quote the \'s if pasing in on the command line)
      try: pronunc=eval('u"'+pronunc+'"')
      except: pass
   else: # see if it makes sense as utf-8
      try: pronunc = pronunc.decode('utf-8')
      except: pass
   return pronunc

def ascii_braille_to_unicode(a):
  "Special-case cleanup_func for braille-ipa (set by braille-ipa if BRAILLE_UNICODE is set).  Converts Braille ASCII to Unicode dot patterns."
  d=dict(zip(list(" A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)="),[unichr(c) for c in range(0x2800,0x2840)]))
  return u''.join(d.get(c,c) for c in list(a))
def unicode_to_ascii_braille(u):
  d=dict(zip([unichr(c) for c in range(0x2800,0x2840)],list(" A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)=")))
  r=''.join(d.get(c,c) for c in list(as_unicode(u)))
  if r.startswith(",7") and r.endswith("7'"): r=r[2:-2]
  return r

def hiragana_to_katakana(u):
   "Special-case cleanup_func for kana-approx; converts all hiragana characters in unicode string 'u' into katakana if KANA_TYPE is set to anything beginning with a 'k'"
   assert type(u)==unicode
   if not os.environ.get("KANA_TYPE","").lower().startswith("k"): return u
   u = list(u)
   for i in xrange(len(u)):
      if 0x3041 <= ord(u[i]) <= 0x3096:
         u[i]=unichr(ord(u[i])+0x60)
   return u"".join(u)

def espeak_probably_right_already(existing_pronunc,new_pronunc):
    """Used by convert_system_festival_dictionary_to_espeak to compare a "new" pronunciation with eSpeak's existing pronunciation.  As the transcription from OALD to eSpeak is only approximate, it could be that our new pronunciation is not identical to the existing one but the existing one is actually correct; try to detect when this happens by checking if the pronunciations are the same after some simplifications."""
    if existing_pronunc==new_pronunc: return True
    def simplify(pronunc): return \
        pronunc.replace(maybe_bytes(";",pronunc),maybe_bytes("",pronunc)).replace(maybe_bytes("%",pronunc),maybe_bytes("",pronunc)) \
        .replace(maybe_bytes("a2",pronunc),maybe_bytes("@",pronunc)) \
        .replace(maybe_bytes("3",pronunc),maybe_bytes("@",pronunc)) \
        .replace(maybe_bytes("L",pronunc),maybe_bytes("l",pronunc)) \
        .replace(maybe_bytes("I2",pronunc),maybe_bytes("i:",pronunc)) \
        .replace(maybe_bytes("I",pronunc),maybe_bytes("i:",pronunc)).replace(maybe_bytes("i@",pronunc),maybe_bytes("i:@",pronunc)) \
        .replace(maybe_bytes(",",pronunc),maybe_bytes("",pronunc)) \
        .replace(maybe_bytes("s",pronunc),maybe_bytes("z",pronunc)) \
        .replace(maybe_bytes("aa",pronunc),maybe_bytes("A:",pronunc)) \
        .replace(maybe_bytes("A@",pronunc),maybe_bytes("A:",pronunc)) \
        .replace(maybe_bytes("O@",pronunc),maybe_bytes("O:",pronunc)) \
        .replace(maybe_bytes("o@",pronunc),maybe_bytes("O:",pronunc)) \
        .replace(maybe_bytes("r-",pronunc),maybe_bytes("r",pronunc))
    # TODO: rewrite @ to 3 whenever not followed by a vowel?
    if as_printable(simplify(existing_pronunc))==as_printable(simplify(new_pronunc)): return True # almost the same, and festival @/a2 etc seems to be a bit ambiguous so leave it alone

def parse_festival_dict(festival_location):
    "For OALD; yields word,part-of-speech,pronunciation"
    ret = []
    for line in open(festival_location):
        line=line.strip()
        if "((pos" in line: line=line[:line.index("((pos")]
        if line.startswith('( "'): line=line[3:]
        line=line.replace('"','').replace('(','').replace(')','')
        try:
            word, pos, pronunc = line.split(None,2)
        except ValueError: continue # malformed line
        if pos not in ['n','v','a','cc','dt','in','j','k','nil','prp','uh']: continue # two or more words
        yield (word.lower(), pos, pronunc)

class Message(Exception): pass
def convert_system_festival_dictionary_to_espeak(festival_location,check_existing_pronunciation,add_user_dictionary_also):
    "See mainopt_festival_dictionary_to_espeak"
    os.system("mv en_extra en_extra~") # start with blank 'extra' dictionary
    if check_existing_pronunciation: os.system("espeak --compile=en") # so that the pronunciation we're checking against is not influenced by a previous version of en_extra
    outFile=open("en_extra","w")
    print ("Reading dictionary lists")
    wordDic = {} ; ambiguous = {}
    el = open("en_list")
    for line in filter(lambda x:x.split() and not re.match(maybe_bytes(r'^[a-z]* *\$',x),x),getBuf(el).read().split(as_utf8('\n'))): ambiguous[line.split()[0]]=ambiguous[line.split()[0]+as_utf8('s')]=True # this stops the code below from overriding anything already in espeak's en_list.  If taking out then you need to think carefully about words like "a", "the" etc.
    for word,pos,pronunc in parse_festival_dict(festival_location):
        pronunc=pronunc.replace("i@ 0 @ 0","ii ou 2 ").replace("i@ 0 u 0","ii ou ") # (hack for OALD's "radio"/"video"/"stereo"/"embryo" etc)
        pronunc=pronunc.replace("0","") # 0's not necessary, and OALD sometimes puts them in wrong places, confusing the converter
        if word in ['mosquitoes']: continue # OALD bug (TODO: any others?)
        if word in wordDic and not wordDic[word]==(pronunc,pos):
            ambiguous[as_utf8(word)] = True
            del wordDic[word] # better not go there
        if not as_utf8(word) in ambiguous:
            wordDic[word] = (pronunc, pos)
    toDel = []
    if check_existing_pronunciation:
        print ("Checking existing pronunciation")
        proc=os.popen("espeak -q -x -v en-rp > /tmp/.pronunc 2>&1","w")
        wList = []
    progressCount=0 ; oldPercent=-1
    itemList = list(wordDic.items())
    # Make sure it's NOT sorted, to ensure eSpeak doesn't
    # cache pronunciation of previous word when add suffix
    # (which can subtly change eSpeak's pronunciation in
    # some versions of eSpeak, leading to
    # Python 2/3 differences as Python 3 sorts by default) :
    itemList.sort()
    i0,i1 = itemList[:int(len(itemList)/2)],itemList[int(len(itemList)/2):]
    itemList = []
    while i0 or i1:
       if i0: itemList.append(i0.pop())
       if i1: itemList.append(i1.pop())
    for word,(pronunc,pos) in itemList:
        if check_existing_pronunciation:
            percent = int(progressCount*100/len(wordDic))
            if not percent==oldPercent: sys.stdout.write(str(percent)+"%\r") ; sys.stdout.flush()
            oldPercent=percent
            progressCount += 1
        if not re.match("^[A-Za-z]*$",word): # (some versions of eSpeak also OK with "-", but not all)
            # contains special characters - better not go there
            toDel.append(word)
        elif word.startswith("plaque") or word in "friday saturday sunday tuesday thursday yesterday".split():
            # hack to accept eSpeak's pl'ak instead of pl'A:k - order was reversed in the March 2009 draft
            toDel.append(word)
        elif word[-1]=="s" and word[:-1] in wordDic:
            # unnecessary plural (espeak will pick up on them anyway)
            toDel.append(word)
        elif word.startswith("year") or "quarter" in word: toDel.append(word) # don't like festival's pronunciation of those (TODO: also 'memorial' why start with [m'I])
        elif check_existing_pronunciation:
            getBuf(proc).write(as_utf8(word)+as_utf8("\n"))
            proc.flush() # so the progress indicator works
            wList.append(word)
    if check_existing_pronunciation:
        proc.close() ; print("")
        oldPronDic = {}
        tp = open("/tmp/.pronunc")
        for k,v in zip(wList,getBuf(tp).read().split(as_utf8("\n"))): oldPronDic[k]=v.strip().replace(as_utf8(" "),as_utf8(""))
    for w in toDel: del wordDic[w]
    print ("Doing the conversion")
    lines_output = 0
    total_lines = 0
    not_output_because_ok = []
    items = list(wordDic.items()) ; items.sort() # necessary because of the hacks below which check for the presence of truncated versions of the word (want to have decided whether or not to output those truncated versions before reaching the hacks)
    for word,(pronunc,pos) in items:
        total_lines += 1
        new_e_pronunc = convert(pronunc,"festival","espeak")
        if new_e_pronunc.count("'")==2 and not '-' in word: new_e_pronunc=new_e_pronunc.replace("'",",",1) # if 2 primary accents then make the first one a secondary (except on hyphenated words)
        # TODO if not en-rp? - if (word.endswith("y") or word.endswith("ie")) and new_e_pronunc.endswith("i:"): new_e_pronunc=new_e_pronunc[:-2]+"I"
        unrelated_word = None
        if check_existing_pronunciation: espeakPronunc = oldPronDic.get(word,"")
        else: espeakPronunc = ""
        if word[-1]=='e' and word[:-1] in wordDic: unrelated_word, espeakPronunc = word[:-1],"" # hack: if word ends with 'e' and dropping the 'e' leaves a valid word that's also in the dictionary, we DON'T want to drop this word on the grounds that espeak already gets it right, because if we do then adding 's' to this word may cause espeak to add 's' to the OTHER word ('-es' rule).
        if espeak_probably_right_already(espeakPronunc,new_e_pronunc):
            not_output_because_ok.append(word)
            continue
        if not unrelated_word: lines_output += 1
        getBuf(outFile).write(as_utf8(word)+as_utf8(" ")+as_utf8(new_e_pronunc)+as_utf8(" // from Festival's (")+as_utf8(pronunc)+as_utf8(")"))
        if espeakPronunc: getBuf(outFile).write(as_utf8(", not [[")+as_utf8(espeakPronunc)+as_utf8("]]"))
        elif unrelated_word: getBuf(outFile).write(as_utf8(" (here to stop espeak's affix rules getting confused by Festival's \"")+as_utf8(unrelated_word)+as_utf8("\")"))
        getBuf(outFile).write(as_utf8("\n"))
    print ("Corrected(?) %d entries out of %d" % (lines_output,total_lines))
    if add_user_dictionary_also: convert_user_lexicon("festival","espeak",outFile)
    outFile.close()
    os.system("espeak --compile=en")
    if not_output_because_ok:
      print ("Checking for unwanted side-effects of those corrections") # e.g. terrible as Terr + ible, inducing as in+Duce+ing
      proc=os.popen("espeak -q -x -v en-rp > /tmp/.pronunc 2>&1","w")
      progressCount = 0
      for w in not_output_because_ok:
          getBuf(proc).write(as_utf8(w)+as_utf8("\n")) ; proc.flush()
          percent = int(progressCount*100/len(not_output_because_ok))
          if not percent==oldPercent: sys.stdout.write(str(percent)+"%\r") ; sys.stdout.flush()
          oldPercent = percent
          progressCount += 1
      proc.close()
      outFile=open("en_extra","a") # append to it
      tp = open("/tmp/.pronunc")
      for word,pronunc in zip(not_output_because_ok,getBuf(tp).read().split(as_utf8("\n"))):
        pronunc = pronunc.strip().replace(as_utf8(" "),as_utf8(""))
        if not pronunc==oldPronDic[word] and not espeak_probably_right_already(oldPronDic[word],pronunc):
          getBuf(outFile).write(as_utf8(word)+as_utf8(" ")+oldPronDic[word]+as_utf8(" // (undo affix-side-effect from previous words that gave \"")+pronunc+as_utf8("\")\n"))
      outFile.close()
      os.system("espeak --compile=en")
    return not_output_because_ok

def read_user_lexicon(fromFormat):
    "Calls the appropriate lex_read_function, opening lex_filename first if supplied"
    readFunction = checkSetting(fromFormat,"lex_read_function")
    if not readFunction: raise Message("Reading from '%s' lexicon file not yet implemented (no lex_read_function); try using --phones or --phones2phones options instead" % (fromFormat,))
    try:
       lexFilename = getSetting(fromFormat,"lex_filename")
       if lexFilename==None: lexfile = None # e.g. the example lexicon
       else:
          lexfile = open(lexFilename)
          if not os.environ.get("LEXCONVERT_OMIT_READING_FROM",""): print ("Reading from "+lexFilename) # TODO: document LEXCONVERT_OMIT_READING_FROM (might be useful for the --mac-uk option)
    except KeyError: lexfile = None # lex_read_function without lex_filename is allowed, if the read function can take null param and fetch the lexicon itself
    except IOError: raise Message(fromFormat+"'s lexicon is expected to be in a file called "+replHome(lexFilename)+" which could not be read - please fix and try again")
    return readFunction(lexfile)

def replHome(fname):
   "Format fname for printing, substituting ~ for HOME if appropriate"
   h = os.environ.get('HOME','')
   if h and fname.startswith(h+os.sep):
      return "~"+fname[len(h):]
   else: return fname
    
def get_macuk_lexicon(fromFormat):
    "Converts lexicon from fromFormat and returns a list suitable for MacBritish_System_Lexicon's readWithLex"
    return [(word,convert(pronunc,fromFormat,"mac-uk")) for word, pronunc in read_user_lexicon(fromFormat)]

def as_utf8(s):
   if type(s)==unicode: return s.encode('utf-8')
   else: return s
def as_unicode(s):
   if type(s)==unicode: return s
   else: return s.decode('utf-8')
def maybe_bytes(s,i):
   "Python 2/3 compatibility: convert s to bytes if i is bytes"
   if type(i)==unicode: return s
   else: return as_utf8(s)
def as_printable(s):
   if sys.version_info[0] < 3: return as_utf8(s)
   else: return as_utf8(s).decode('utf-8')

def convert_user_lexicon(fromFormat,toFormat,outFile):
    "See mainopt_convert"
    lex = read_user_lexicon(fromFormat)
    lex_header = checkSetting(toFormat,"lex_header")
    if type(lex_header) in [bytes,unicode]: getBuf(outFile).write(as_utf8(lex_header))
    else: lex_header(outFile)
    entryFormat=getSetting(toFormat,"lex_entry_format")
    wordCase=checkSetting(toFormat,"lex_word_case")
    for word, pronunc in lex:
        pronunc = as_utf8(convert(pronunc,fromFormat,toFormat))
        if wordCase=="upper": word=word.upper()
        elif wordCase=="lower": word=word.lower()
        getBuf(outFile).write(as_utf8(entryFormat) % (as_utf8(word),as_utf8(pronunc))) # will work in Python 3.6, but not in Python 3.4 (e.g. on jessie) which cannot do % on byte-strings
    footer = checkSetting(toFormat,"lex_footer")
    if type(footer) in [bytes,unicode]: getBuf(outFile).write(as_utf8(footer))
    else: footer(outFile)

def bbcMicro_partPhonemeCount(pronunc):
   """Returns the number of 'part phonemes' (at least that's what I'm calling them) for the BBC Micro phonemes in pronunc.  The *SPEAK command cannot take more than 117 part-phonemes at a time before saying "Line too long", and in some cases it takes less than that (I'm not sure why); 115 is a safer limit."""
   partCount = 0 ; pronunc0 = pronunc
   while pronunc:
      found = 0
      for p in ' ,AA,AE,AH,AI,AO,AW,AY,B,CH,CT,DH,DUX,D,EE,EH,ER,F,G,/H,IH,IX,IY,J,K,L,M,NX,N,OW,OL,OY,O,P,R,SH,S,TH,T,UH,/UL,/U,UW,UX,V,W,Y,ZH,Z'.split(','): # phonemes and space count, but pitch numbers do not count
         if pronunc.startswith(as_utf8(p)):
            partCount += {
               # *SPEAK can take 117 of most single-letter phonemes, or 116 (limited by the 232+6-character input limit) of most 2-letter phonemes
               'AW':2,'IY':2,'OW':2,'OL':2,'UW':2,'/UL':2, # *SPEAK can take 58 of these
               'DUX':3,'AY':3,'CH':3,'J':3,'OY':3, # *SPEAK can take 39 of these
               'CT':4, # *SPEAK can take 29 of these
            }.get(p,1)
            pronunc=pronunc[len(p):] ; found=1 ; break
      if not found:
         assert as_printable(pronunc[:1]) in '12345678',"Unrecognised BBC Micro phoneme at "+str(pronunc)+" in "+str(pronunc0)
         pronunc=pronunc[1:]
   return partCount

def markup_inline_word(format,pronunc):
    "Returns pronunc with any necessary markup for putting it in a text (using the inline_format setting)"
    pronunc = as_utf8(pronunc) # UTF-8 output - ok for pasting into Firefox etc *IF* the terminal/X11 understands utf-8 (otherwise redirect to a file, point the browser at it, and set encoding to utf-8, or try --convert'ing which will o/p HTML)
    format = checkSetting(format,"inline_format","%s")
    if type(format) in [bytes,unicode]:
       if type(format)==unicode: format=format.encode('utf-8') # see above
       return format % pronunc
    else: return format(pronunc)
def markup_doubleTalk_word(pronunc):
   "Special-case function set as inline_format in doubletalk (checks environment variables for command code)"
   cmd = os.environ.get('DTALK_COMMAND_CODE','')
   if cmd: cmd=chr(int(cmd))
   else: cmd = as_utf8('*')
   return as_utf8("%sD%s%sT") % (cmd,pronunc,cmd)
def markup_bbcMicro_word(pronunc):
   "Special-case function set as inline_format in bbcmicro.  Begins a new *SPEAK command when necessary.  See also write_bbcmicro_phones."
   global bbc_partsSoFar,bbc_charsSoFar
   thisPartCount = bbcMicro_partPhonemeCount(pronunc)
   if (not bbc_partsSoFar or bbc_partsSoFar+thisPartCount > 115) or (not bbc_charsSoFar or bbc_charsSoFar+len(pronunc) > 238): # 238 is max len of BBC BASIC prompt (both the immediate prompt and the one with line number supplied by AUTO, in both BASIC II and BASIC IV); re other limit see bbcMicro_partPhonemeCount
      if bbc_charsSoFar: r="\n"
      else: r=""
      cmd="*SPEAK" # (could add a space if want to make it more readable, at the expense of an extra keystroke in the paste buffer; by the way, when not using the ROM version you must use *SPEAK not OS.("SPEAK"), at least on a Model B; seems OSCLI doesn't go through quite the same vectors as star)
      bbc_charsSoFar = len(cmd)+len(pronunc)+1 # +1 for the space that'll be after this word if we don't start a new line
      bbc_partsSoFar = thisPartCount+1 # ditto
      return as_utf8(r+cmd)+pronunc
   else:
      bbc_charsSoFar += len(pronunc)+1
      bbc_partsSoFar += thisPartCount+1
      return pronunc
bbc_partsSoFar=bbc_charsSoFar=0

def sylcount(example_format_festival):
  """Tries to count the number of syllables in a Festival string (see mainopt_syllables).  We treat @ as counting the same as the previous syllable (e.g. "fire", "power"), but this can vary in different songs, so the result will likely need a bit of proofreading."""
  count = inVowel = maybeCount = hadAt = 0
  festival = example_format_festival.split() # no brackets, emphasis by vowels, but spaces between each syllable
  for phone,i in zip(festival,range(len(festival))):
    if phone[:1] in "aeiou": inVowel=0 # unconditionally start new syllable
    if phone[:1] in "aeiou@12":
      if not inVowel: count += 1
      elif phone[:1]=="@" and not hadAt: maybeCount = 1 # (e.g. "loyal", but NOT '1', e.g. "world")
      if "@" in phone: hadAt = 1 # for words like "cheerful" ("i@ 1 @" counts as one)
      inVowel = 1
      if phone[:1]=="@" and i>=3 and festival[i-2:i]==["ai","1"] and festival[i-3] in ["s","h"]: # special rule for higher, Messiah, etc - like "fire" but usually 2 syllables
        maybeCount = 0 ; count += 1
    else:
      if not phone[:1] in "drz": count += maybeCount # not 'r/z' e.g. "ours", "fired" usually 1 syllable in songs, "desirable" usually 4 not 5
      # TODO steward?  y u@ 1 d but usally 2 syllables
      inVowel = maybeCount = hadAt = 0
  return count
def hyphenate(word,numSyls):
  "See mainopt_syllables"
  orig = word
  try: word,isu8 = word.decode('utf-8'),True
  except: isu8 = False
  pre=[] ; post=[]
  while word and not 'a'<=word[:1].lower()<='z':
    pre.append(word[:1]) ; word=word[1:]
  while word and not 'a'<=word[-1].lower()<='z':
    post.insert(0,word[-1:]) ; word=word[:-1]
  if numSyls>len(word): return orig # probably numbers or something
  l = int((len(word)+numSyls/2)/numSyls) ; syls = []
  for i in range(numSyls):
    if i==numSyls-1: syls.append(word[i*l:])
    else: syls.append(word[i*l:(i+1)*l])
    if len(syls)>1:
      if syls[-1].startswith('-') or (len(syls[-1])>2 and syls[-1][:1]==syls[-1][1:2] and not syls[-1][:1].lower() in "aeiou"):
        # repeated consonant at start - put one on previous
        # (or hyphen at start - move it to the previous)
        syls[-2] += syls[-1][:1]
        syls[-1] = syls[-1][1:]
      elif len(syls[-1])>2 and syls[-1][1]=='-':
        # better move this splitpoint after that hyphen (TODO: move more than one character?)
        syls[-2] += syls[-1][:2]
        syls[-1] = syls[-1][2:]
      elif ((len(syls[-2])>2 and syls[-2][-1]==syls[-2][-2] and not syls[-2][-1].lower() in "aeiou") \
            or (syls[-1] and syls[-1][:1].lower() in "aeiouy" and len(syls[-2])>2)) \
            and list(filter(lambda x:x.lower() in "aeiou",list(syls[-2][:-1]))):
        # repeated consonant at end - put one on next
        # or vowel on right: move a letter over (sometimes the right thing to do...)
        # (unless doing so leaves no vowels)
        syls[-1] = syls[-2][-1]+syls[-1]
        syls[-2] = syls[-2][:-1]
  word = ''.join(pre)+"- ".join(syls)+''.join(post)
  if isu8: word=word.encode('utf-8')
  return word

def macSayCommand():
  """Return the environment variable SAY_COMMAND if it is set and if it is non-empty, otherwise return "say".
  E.g. SAY_COMMAND="say -o file.aiff" (TODO: document this in the help text?)
  In Gradint you can set (e.g. if you have a ~/.festivalrc) extra_speech=[("en","python lexconvert.py --mac-uk festival")] ; extra_speech_tofile=[("en",'echo %s | SAY_COMMAND="say -o /tmp/said.aiff" python lexconvert.py --mac-uk festival && sox /tmp/said.aiff /tmp/said.wav',"/tmp/said.wav")]"""
  s = os.environ.get("SAY_COMMAND","")
  if s: return s
  else: return "say"

def stdin_is_terminal():
   "Returns True if it seems the standard input is connected to a terminal (rather than piped from a file etc)"
   return (not hasattr(sys.stdin,"isatty")) or sys.stdin.isatty()

def getInputText(i,prompt,as_iterable=False):
  """Gets text either from the command line or from standard input.  Issue prompt if there's nothing on the command line and standard input is connected to a tty instead of a pipe or file.  If as_iterable, return an iterable object over the lines instead of reading and returning all text at once.  If as_iterable=='maybe', return the iterable but if not reading from a tty then read everything into one item."""
  txt = ' '.join(sys.argv[i:])
  if txt:
    if as_iterable=='maybe': return [txt]
    elif as_iterable: return txt.split('\n')
    else: return txt
  if stdin_is_terminal(): sys.stderr.write("Enter "+prompt+" (EOF when done)\n")
  elif as_iterable=='maybe': return [getBuf(sys.stdin).read()]
  if as_iterable: return my_xreadlines()
  else:
     try: return getBuf(sys.stdin).read()
     except KeyboardInterrupt: raise SystemExit

try: raw_input # Python 2
except NameError: raw_input = input # Python 3
def my_xreadlines():
   "On some platforms this might be a bit more responsive than sys.stdin.xreadlines"
   while True:
      try: yield raw_input()
      except EOFError: return
      except KeyboardInterrupt: raise SystemExit

def output_clauses(format,clauses):
   "Writes out clauses and words in format 'format' (clauses is a list of lists of words in the phones of 'format').  By default, calls markup_inline_word and join as appropriate.  If however the format's 'clause_separator' has been set to a special case, calls that."
   if checkSetting(format,"output_is_binary") and hasattr(sys.stdout,"isatty") and sys.stdout.isatty():
      print ("This is a binary format - not writing to terminal.\nPlease direct output to a file or pipe.")
      return
   clause_sep = checkSetting(format,"clause_separator","\n")
   if type(clause_sep) in [bytes,unicode]: getBuf(sys.stdout).write(as_utf8(clause_sep).join(as_utf8(wordSeparator(format)).join(markup_inline_word(format,word) for word in clause) for clause in clauses))
   else: clause_sep(clauses)
def write_bbcmicro_phones(clauses):
  """Special-case function set as clause_separator in bbcmicro format.  Must be a special case because it needs to track any extra keystrokes to avoid "Line too long".  And while we're at it, we might as well start a new *SPEAK command with each clause, using the natural brief delay between commands; this should minimise the occurrence of additional delays in arbitrary places.  Also calls print_bbc_warnings"""
  totalKeystrokes = 0 ; lines = 0
  for clause in clauses:
    global bbc_charsSoFar ; bbc_charsSoFar=0
    l=as_utf8(" ").join([markup_inline_word("bbcmicro",word) for word in clause])
    getBuf(sys.stdout).write(l.replace(as_utf8(" \n"),as_utf8("\n")))
    totalKeystrokes += len(l)+1 ; lines += 1
  print_bbc_warnings(totalKeystrokes,lines)
def print_bbc_warnings(keyCount,lineCount):
  "Print any relevant size warnings regarding sending 'keyCount' keys in 'lineCount' lines to the BBC Micro"
  sys.stdout.flush() # try to keep in sync if someone's doing 2>&1 | less
  limits_exceeded = [] ; severe=0
  if keyCount >= 32768:
    severe=1 ; limits_exceeded.append("BeebEm 32K keystroke limit") # At least in version 3, the clipboard is defined in beebwin.h as a char of size 32768 and its bounds are not checked.  Additionally, if you script a second paste before the first has finished (or if you try to use BeebEm's Copy command) then the first paste will be interrupted.  So if you really want to make BeebEm read more then I suggest setting a printer destination file, putting a VDU 2,10,3 after each batch of commands, and waiting for that \n to appear in that printer file before sending the next batch, or perhaps write a set of programs to a disk image and have them CHAIN each other or whatever.
  shadow_himem=0x8000 # if using a 'shadow mode' on the Master/B+/Integra-B (modes 128-135, which leave all main RAM free)
  mode7_himem=0x7c00 # (40x25 characters = 1000 bytes, by default starting at 7c00 with 24 bytes spare at the top, but the scrolling system uses the full 1024 bytes and can tell the video controller to start rendering at any one of them; if you get Jeremy Ruston's book and program the VIDC yourself then you could fix it at 7c18 if you really want, or just set HIMEM=&8000 and don't touch the screen, but that doesn't give you very much more room)
  default_speech_loc=0x5500
  overhead_per_program_line = 4
  for page,model in [
        (0x1900,"Model B"), # with Acorn DFS (a reasonable assumption although alternate DFS ROMs are different)
        (0xE00,"Master")]: # (the Master has 8k of special paged-in "filing system RAM", so doesn't need 2816 bytes of main RAM for DFS)
     top = page+keyCount+lineCount*(overhead_per_program_line-1)+2 # the -1 is because keyCount includes a carriage return at the end of each line
     if model=="Master": x=" (use Speech's Sideways RAM version instead, e.g. *SRLOAD SP8000 8000 7 and reset, but sound quality might be worse)" # I don't know why but SP8000 can play higher and more distorted than SPEECH, at least on emulation (and changing the emulation speed doesn't help, because that setting, at least in BeebEm3, just controls extra usleep every frame; it doesn't actually slow down the 6502 *between* frames; anyway timing of sound changes is done by CyclesToSamples stuff in beebsound.cc's SoundTrigger).  If on the Master you go into View (*WORD) and then try SP8000, it plays _lower_ than *SPEECH (even if you do *BASIC first) and *SAY can corrupt a View document; ViewSheet (*SHEET) doesn't seem to have this effect; neither does *TERMINAL but *SAY can confuse the terminal.
     # Re bank numbers, by default banks 4 to 7 are Sideways RAM (4*16k=64k) and I suppose filling up from 7 makes sense because banks 8-F are ROMs (ANFS,DFS,ViewSheet,Edit,BASIC,ADFS,View,Terminal; OS is a separate 16k so there's scope for 144k of supplied ROM).  Banks 0-3 are ROM expansion slots.  The "128" in the name "Master 128" comes from 32k main RAM, 64k Sideways RAM, 20k shadow RAM (for screen modes 128-135), 4k OS "private RAM" (paged on top of 8000-8FFF) and 8k filing system RAM (paged on top of C000-DFFF) = 128k.  Not sure what happened on the B+.
     # By the way BeebEm's beebsound.cc also shows us why SOUND was always out of tune especially in the higher pitches.  The 16-bit freqval given to the chip is 125000/freq and must be an integer, so the likely temperament in cents for non-PCM is given by [int(math.log(125000.0/math.ceil(125000/freq)/freq,2**(1.0/1200))) for freq in [440*((2**(1.0/12))**semi) for semi in range(-12*3+2,12*2+6)]] (the actual temperament will depend on the OS's implementation of mapping SOUND pitch values to freqval's, unless you program the chip directly, but this list is indicative and varies over 10% in the top 2 octaves)
     # Some other ROMs (e.g. Alan Blundell's "Informant" 1989) seem to result in a crash after the *SPEECH and/or *SPEAK commands complete, at least in some emulator configurations; this may or may not be resolved via timing adjustments or adjustments in the ROM order; not sure exactly what the problem is
     else: x=" (Speech program will be overwritten unless relocated)" # (could use Sideways RAM for it instead if you have it fitted, see above)
     if top > default_speech_loc: limits_exceeded.append("%s TOP=&%X limit%s" % (model,default_speech_loc,x)) # The Speech program does nothing to stop your program (or its variables etc) from growing large enough to overwrite &5500, nor does it stop the stack pointer (coming down from HIMEM) from overwriting &72FF. For more safety on a Model B you could use RELOCAT to put Speech at &5E00 and be sure to set HIMEM=&5E00 before loading, but then you must avoid commands that change HIMEM, such as MODE (but selecting any non-shadow mode other than 7 will overwrite Speech anyway, although if you set the mode before loading Speech then it'll overwrite screen memory and still work as long as the affected part of the screen is undisturbed).  You can't do tricks like ditching the lexicon because RELOCAT won't let you go above 5E00 (unless you fix it, but I haven't looked in detail; if you can fix RELOCAT to go above 5E00 then you can create a lexicon-free Speech by taking the 1st 0x1560 bytes of SPEECH and append two * bytes, relocate to &6600 and set HIMEM, but don't expect *SAY to work, unless you put a really small lexicon into the spare 144 bytes that are left - RELOCAT needs an xx00 address so you can't have those bytes at the bottom).  You could even relocate to &6A00 and overwrite (non-shadow) screen memory if you don't mind the screen being filled with gibberish that you'd better not erase! (well if you program the VIDC as mentioned above and you didn't re-add a small lexicon then you could get yourself 3.6 lines of usable Mode 7 display from the spare bytes but it's probably not worth the effort)
     if top > mode7_himem:
        if model=="Master":
           if top > shadow_himem: limits_exceeded.append(model+" 32k HIMEM limit (even for shadow modes)") # TODO: maybe add instructions for using BAS128 on the B+ or Master; this sets PAGE=&10000 and HIMEM=&20000 (i.e. 64k for programs), which uses all 4 SRAM slots so you can't use SP8000 (unless it's on a real ROM); if using Speech in main memory you need to RELOCAT it to leave &3000 upwards for Bas128 code; putting it at &1900 for B+/DFS leaves you only 417 bytes for lexicon (which might not matter if you're using only *SPEECH: just create a shortened lexicon); putting it at &E00 for Master allows space for the default 2204-byte lexicon with 1029 bytes to spare; TODO check if Bas128 uses any workspace between &E00 and &3000 though.  Alternatively (if you really want to store such a long program on the BBC) then you'd better split it into several programs that CHAIN each other (as mentioned above).
           else: limits_exceeded.append(model+" Mode 7 HIMEM limit (use shadow modes 128-135)")
        else: limits_exceeded.append(model+" Mode 7 HIMEM limit") # unless you overwrite the screen (see above) - let's assume the Model B hasn't been fitted with shadow modes (although the Integra-B add-on does give them to the Model B, and leaves PAGE at &1900; B+ has shadow modes but I don't know what's supposed to happen to PAGE on it).  65C02 Tube doesn't help much (it'll try to run Speech on the coprocessor instead of the host, and this results in silence because it can't send its sound back across the Tube; don't know if there's a way to make it run on the host in these circumstances or what the host's memory map is like)
  if lineCount > 32768: limits_exceeded.append("BBC BASIC line number limit") # and you wouldn't get this far without filling the memory, even with 128k (4 bytes per line)
  elif 10*lineCount > 32767: limits_exceeded.append("AUTO line number limit (try AUTO 0,1)") # (default AUTO increments in steps of 10; you can use AUTO 0,1 to start at 0 and increment in steps of 1.  BBC BASIC stores its line info in a compact form which allows a range of 0-32767.)
  if severe: warning,after="WARNING: ",""
  else: warning,after="Note: ","It should still work if pasted into BeebEm as immediate commands. "
  after = ". "+after+"See comments in lexconvert for more details.\n"
  if len(limits_exceeded)>1: sys.stderr.write(warning+"this text may be too big for the BBC Micro. The following limits were exceeded: "+", ".join(limits_exceeded)+after)
  elif limits_exceeded: sys.stderr.write(warning+"this text may be too big for the BBC Micro because it exceeds the "+limits_exceeded[0]+after)
def bbc_prepDefaultLex(outFile):
  """Special-case function set as lex_header in bbcmicro format.  If SPEECH_DISK and MAKE_SPEECH_ROM is set, then read the ROM code from SPEECH_DISK and write to outFile (meant to go before the lexicon, to make a modified BBC Micro Speech ROM with custom lexicon)"""
  if not os.environ.get("MAKE_SPEECH_ROM",0): return
  sd = open(os.environ['SPEECH_DISK'])
  d=getBuf(sd).read() # if this fails, SPEECH_DISK was not set or was set incorrectly (it's required for MAKE_SPEECH_ROM)
  i=d.index(as_utf8('LO')+chr(0x80)+as_utf8('LP')+chr(0x80)+chr(0x82)+chr(0x11)) # start of SP8000 file (if this fails, it wasn't a Speech disk)
  j=d.index(as_utf8('>OUS_'),i) # start of lexicon (ditto)
  assert j-i==0x1683, "Is this really an original disk image?"
  getBuf(outFile).write(d[i:j])
def bbc_appendDefaultLex(outFile):
  """Special-case function set as lex_footer in bbcmicro format.  If SPEECH_DISK is set, read Speech's default lexicon from it and append this to outFile.  Otherwise just write a terminating >** to outFile.  In either case, check for exceeding 16k if we're MAKE_SPEECH_ROM, close the file and call print_bbclex_instructions."""
  if os.environ.get("SPEECH_DISK",""):
     sd = open(os.environ['SPEECH_DISK'])
     d=getBuf(sd).read()
     i=d.index(as_utf8('>OUS_')) # if this fails, it wasn't a Speech disk
     j=d.index(as_utf8(">**"),i)
     assert j-i==2201, "Lexicon on SPEECH_DISK is wrong size (%d). Is this really an original disk image?" % (j-i)
     getBuf(outFile).write(d[i:j])
     # TODO: can we compress the BBC lexicon?  i.e. detect if a rule will happen anyway due to subsequent wildcard rules, and delete it if so (don't know how many bytes that would save)
  outFile.write(">**")
  fileLen = outFile.tell()
  assert not os.environ.get("MAKE_SPEECH_ROM",0) or fileLen <= 16384, "Speech ROM file got too big (%d)" % fileLen
  outFile.close()
  print_bbclex_instructions(getSetting("bbcmicro","lex_filename"),fileLen)

def bbcshortest(n):
  """Convert integer n into the shortest possible number of BBC Micro keystrokes; prefer hex if and only if the extra '&' keystroke won't make it any longer than its decimal equivalent"""
  if len(str(n)) < len('&%X'%n): return as_utf8(str(n))
  else: return as_utf8('&%X'%n)
def bbcKeystrokes(data,start):
  "Return BBC BASIC keystrokes to put data into RAM starting at address start, without using the BASIC heap in the process (although we do use one of the page-4 integer variables to save some keystrokes).  Assumes the data is mostly ASCII so the $ operator is the least-keystrokes method of getting it in (rather than ? and ! operators, assembler EQUB/EQUW/EQUS, 6502 mnemonics, etc); we don't mind about overwriting the byte after with a CHR$(13).  Keystrokes are limited to ASCII for easier copy/paste.  See comments for more details."
  # Taken to the extreme, a 'find the least keystrokes' function would be some kind of data compressor; we're not doing that here as we assume this is going to be used to poke in a lexicon, which is basically ASCII with a few CHR$(128)s thrown in; this '$ operator' method is highly likely to yield the least keystrokes for that kind of data, apart from setting and using temporary string variables, but then (1) you're in the realms of data compression and (2) you require heap memory, which might not be a good idea depending on where we're putting our lexicon.
  # I suppose it wouldn't hurt in most cases to have an A$=CHR$(128), but not doing this for now because you might be in a situation where you can't touch the heap at all (I'm not sure where the workspace for assembling strings is though).
  # However, just to be pedantic about saving a few bytes, there is one thing we CAN do: if we have a lexicon with a lot of CHR$(128)s in it, let's set up BASIC's page-4 integer variables such that $A%=CHR$(128), saving 6 keystrokes per entry without needing the heap (an additional 1 keystroke per entry could be saved if we didn't mind putting an A$ on the heap).
  use_int_hack = ((start>=1030 or start+len(data)<=1026) and len(data.split(chr(128))) >= 4)
  i=0 ; ret=[]
  if use_int_hack: thisLine = as_utf8("A%=&408:B%=&D80:") # (@% is at &400 and each is 4 byte LSB-MSB; $x reads to next 0D)
  # (If we're guaranteed to NOT be using Bas128 and therefore all memory addresses are effectively masked by &FFFF, we can instead set A%=&D800406 (using A%'s low 2 bytes to point to A%'s high 2 bytes) for a 1-off saving of 5 keystrokes and 1 page-4 variable, but this saving is not really worth the readability compromise and the risk posed by the possibility of Bas128 - I don't know how Bas128 treats addresses above &1FFFF)
  # (An even 'nastier' trick would be to put !13=&D80 and then use $13, as those bytes are used by BASIC's random number generator, which presumably isn't called during the paste and we don't mind disrupting it; again I don't know about Bas128.  But you can't do it because BASIC gives a "$ range" error on anything below 256.)
  # (I suppose one thing you _could_ do is LOMEM=&400:A$=CHR$(13) and end with LOMEM=TOP, which would overwrite 3 page-4 variables and let you use just A$ instead of $A%, saving keystrokes over A%=&D800406 after 21 more lexicon words, at the expense of losing track of any variables you had on the heap.  But this is getting silly.)
  else: thisLine = as_utf8("")
  bbc_max_line_len = 238
  inQuote=needPlus=0 ; needCmd=1
  while i<len(data):
    if needCmd:
       thisLine += (as_utf8('$')+bbcshortest(start)+as_utf8('='))
       inQuote=needPlus=needCmd=0
    if data[i:i+1]==as_utf8('"'): c,inQ = as_utf8('""'),1 # inQ MUST be 0 or 1, not False/True, because it's also used as 'len of necessary close quote' below
    elif 32<=ord(data[i:i+1])<127: c,inQ = data[i:i+1],1
    elif use_int_hack and ord(data[i:i+1])==128: c,inQ=as_utf8("$A%"),0
    else: c,inQ=(as_utf8("CHR$("+str(ord(data[i:i+1]))+")")),0
    addToLine = [] ; newNeedPlus = needPlus
    if inQ and not inQuote:
       if needPlus: addToLine.append(as_utf8('+'))
       addToLine.append(as_utf8('"'))
       newNeedPlus=0
    elif inQuote and not inQ:
       addToLine.append(as_utf8('"+'))
       newNeedPlus=1 # after what we'll add
    elif not inQ:
       if needPlus: addToLine.append(as_utf8('+'))
       newNeedPlus=1 # after what we'll add
    addToLine.append(c)
    addToLine=as_utf8('').join(addToLine)
    if len(thisLine)+len(addToLine)+inQ > bbc_max_line_len: # oops, we've gone too far, back off and end prev line
       if inQuote: thisLine += as_utf8('"')
       ret.append(thisLine)
       thisLine=as_utf8("") ; needCmd=1 ; continue
    thisLine += addToLine ; inQuote=inQ
    needPlus=newNeedPlus ; i += 1 ; start += 1
  if inQuote: thisLine += as_utf8('"')
  if not needCmd: ret.append(thisLine)
  return as_utf8('\n').join(ret)+as_utf8('\n')
def print_bbclex_instructions(fname,size):
 """Print suitable instructions for a BBC Micro lexicon of the given filename and size (the exact nature of the instructions depends on the size).  If appropriate, create a .key file containing keystrokes for transferring to an emulator."""
 if os.environ.get("MAKE_SPEECH_ROM",0): print ("%s (%d bytes, hex %X) can now installed on an emulator (set in Roms.cfg or whatever), or loaded onto a chip.  The sound quality of this might be worse than that of the main-RAM version." % (fname,size,size)) # (at least on emulation - see comment on sound quality above)
 else:
  print ("The size of this lexicon is %d bytes (hex %X)" % (size,size)) # (the default lexicon is 2204 bytes)
  bbcStart=None
  noSRAM_lex_offset=0x155F # (on the BBC Micro, SRAM means Sideways RAM, not Static RAM as it does elsewhere; for clarity we'd better say "Sideways RAM" in all output)
  SRAM_lex_offset=0x1683
  SRAM_max=0x4000 # 16k
  noSRAM_default_addr=0x5500
  noSRAM_min_addr=0xE00 # minimum supported by RELOCAT
  page=0x1900 # or 0xE00 for Master (but OK to just leave this at 0x1900 regardless of model; it harmlessly increases the range where special_relocate_instructions 'kick in')
  noSRAM_himem=0x7c00 # unless you're in a shadow mode or something (see comments on himem above), however leaving this at 0x7c00 is usually harmless (just causes the 'need to relocate' to 'kick in' earlier, although if memory is really full it might say 'too big' 1k too early)
  def special_relocate_instructions(reloc_addr):
    pagemove_min,pagemove_max = max(0xE00,page-0x1E00), page+0xE00 # if relocating to within this range, must move PAGE before loading RELOCAT. RELOCAT's supported range is 0xE00 to 0x5E00, omitting (PAGE-&1E00) to (PAGE+&E00)
    if reloc_addr < 0x1900: extra=" On a Model B with Acorn DFS you won't be able to use the disk after relocating below &1900, and you can't run star commands from tape so you have to initialise via CALL. (On a Master, DFS is not affected as it doesn't use &E00-&1900.)"
    else: extra = ""
    if not pagemove_min<=reloc_addr<pagemove_max:
      return extra # no other special instructions needed
    newpage = reloc_addr+0x1E00
    page_max = min(0x5E00,noSRAM_default_addr-0xE00)
    if newpage > page_max: return False # "Unfortunately RELOCAT can't put it at &%X even with PAGE changes." % reloc_addr
    return " Please run RELOCAT with PAGE in the range of &%X to &%X for this relocation to work.%s" % (newpage,page_max,extra)
  if noSRAM_default_addr+noSRAM_lex_offset+size > noSRAM_himem:
    reloc_addr = noSRAM_himem-noSRAM_lex_offset-size
    reloc_addr -= (reloc_addr%256)
    if reloc_addr >= noSRAM_min_addr:
      instr = special_relocate_instructions(reloc_addr)
      if instr==False: print ("This lexicon is too big for Speech in main RAM even with relocation, unless RELOCAT is rewritten to work from files.")
      else:
        bbcStart = reloc_addr+noSRAM_lex_offset
        reloc_call = reloc_addr + 0xB00
        print ("This lexicon is too big for Speech at its default address of &%X, but you could use RELOCAT to put a version at &%X and then initialise it with CALL %s (or do the suggested *SAVE, reset, and run *SP). Be sure to set HIMEM=&%X. Then *LOAD %s %X or change the relocated SP file from offset &%X.%s" % (noSRAM_default_addr,reloc_addr,bbcshortest(reloc_call),reloc_addr,fname,bbcStart,noSRAM_lex_offset,instr))
    else: print ("This lexicon is too big for Speech in main RAM even with relocation.")
  else: # fits at default location - no relocation needed
    bbcStart = noSRAM_default_addr+noSRAM_lex_offset
    print ("You can load this lexicon by *LOAD %s %X or change the SPEECH file from offset &%X. Suggest you also set HIMEM=&%X for safety." % (fname,bbcStart,noSRAM_lex_offset,noSRAM_default_addr))
  if bbcStart: # we managed to fit it into main RAM
     f = open(fname)
     keys = bbcKeystrokes(getBuf(f).read(),bbcStart)
     f = open(fname+".key","w")
     getBuf(f).write(keys)
     del f
     print ("For ease of transfer to emulators etc, a self-contained keystroke file for putting %s data at &%X has been written to %s.key" % (fname,bbcStart,fname))
     if len(keys) > 32767: print ("(This file looks too big for BeebEm to paste though)") # see comments elsewhere
  # Instructions for replacing lex in SRAM:
  if size > SRAM_max-SRAM_lex_offset: print ("This lexicon is too big for Speech in Sideways RAM.") # unless you can patch Speech to run in SRAM but read its lexicon from main RAM, or run in main RAM but page in multiple banks of SRAM for the lexicon (but even then there'll be a limit)
  else: print ("You can load this lexicon into Sideways RAM by *SRLOAD %s %X 7 (or whichever bank number you're using), or change the SP8000 file from offset &%X." % (fname,SRAM_lex_offset+0x8000,SRAM_lex_offset))
  if not os.environ.get("SPEECH_DISK",""): print ("If you want to append the default lexicon to this one, set SPEECH_DISK to the image of the original Speech disk before running lexconvert, e.g. export SPEECH_DISK=/usr/local/BeebEm3/diskimg/Speech.ssd")
  if size <= SRAM_max-SRAM_lex_offset: print ("You can also set MAKE_SPEECH_ROM=1 (along with SPEECH_DISK) to create a SPEECH.ROM file instead")
 print ("If you get 'Mistake in speech' when testing some words, try starting with '*SAY, ' (this seems to be a Speech bug)") # - can't track down which words it does and doesn't apply to
 print ("It might be better to load your lexicon into eSpeak and use lexconvert's --phones option to drive the BBC with phonemes.")

def mainopt_version(i):
   # TODO: doc string for the help? (or would this option clutter it needlessly) - just print lexconvert's version number and nothing else
   print (__doc__.split("\n")[0].split(" - ")[0])

def main():
    """Introspect the module to find the mainopt_ functions, and either call one of them or print the help.  Returns the error code to send back to the OS."""
    def funcToOpt(n): return "--"+n[n.index("_")+1:].replace("_","-")
    for k,v in globals().items():
        if k.startswith('mainopt_') and funcToOpt(k) in sys.argv:
           try: msg = v(sys.argv.index(funcToOpt(k)))
           except Message:
              # Python 2.6+ can have "except Message as e",
              # but Python 2.5 has to have "except Message,e"
              # which is disallowed in Python 3, so
              msg=sys.exc_info()[1].message
           if msg:
              sys.stdout.flush()
              sys.stderr.write(msg+"\n") ; return 1
           else: return 0
    html,markdown = ('--htmlhelp' in sys.argv), ('--mdhelp' in sys.argv) # (undocumented options used for my website, don't rely on them staying)
    def htmlify(h):
       h = h.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
       h = h.replace('\n','<br>')
       h = re.sub('(--[2A-Za-z-]*)',r'<kbd>\1</kbd>',h) # command-line option
       h = re.sub('(?<= )([A-Z]+_[A-Z_]*)(?= )',r'<code>\1</code>',h) # BRAILLE_UNICODE etc (as a word between spaces)
       h = re.sub('(?<= )(export [A-Z]+_[A-Z_]*=1)',r'<kbd>\1</kbd>',h) # export DTALK_COMMAND_CODE=1
       h = re.sub(" ('\\\\u[0-9a-fA-F\\\\un]*')",r' <kbd>\1</kbd>',h)
       h = h.replace("lexconvert.py","<code>lexconvert.py</code>").replace("[[inpt PHON]]","<code>[[inpt PHON]]</code>").replace("python <code>lexconvert.py</code>","<kbd>python lexconvert.py</kbd>").replace("~/.festivalrc","<code>~/.festivalrc</code>").replace(r"'\uNNNN'",r"<kbd>'\uNNNN'</kbd>")
       h = re.sub("(?<=[a-z])/","<wbr>/",h)
       h = re.sub("(?<=[A-Za-z])_(?=[A-Z0-9])","_<wbr>",h)
       h = re.sub(r"(?<=[a-z0-9])\\u",r"<wbr>\\u",h)
       h = h.replace(" ALL "," <em>all</em> ")
       return h
    if html: print (htmlify(__doc__).replace(" - ","<br>"))
    elif markdown:
       print ("Usage information\n=================\n")
       sys.stdout.flush()
       getBuf(sys.stdout).write(__doc__.replace(" - ",u"\u2014").replace("(c) ",u"\u00a9\u00a0").replace("\n",", ").encode('utf-8'))
       sys.stdout.flush()
       htmlify0 = htmlify
       htmlify = lambda x: \
          htmlify0(x) \
          .replace("<wbr>","") \
          .replace("<kbd>","`").replace("</kbd>","`") \
          .replace("<code>","`").replace("</code>","`") \
          .replace("``","").replace("<br>","  \n") \
          .replace("<em>","*").replace("</em>","*") \
          .replace("&lt;","<").replace("&gt;",">").replace("&amp;","&")
    else:
       print (__doc__)
       htmlify = lambda x:x
    if html: missALine = "<p>"
    else: missALine = ""
    print (missALine)
    if '--formats' in sys.argv: # non-HTML mode only (format descriptions are included in HTML anyway, and don't worry about the capability summary)
       print ("Available pronunciation formats (and support levels):")
       keys=list(lexFormats.keys()) ; keys.sort()
       for k in keys:
          types = []
          if not k=="example": types.append("phones")
          if k=="mac-uk": types.append("speaking")
          else:
             if checkSetting(k,"lex_read_function"): types.append("lex-read")
             if checkSetting(k,"lex_filename") and checkSetting(k,"lex_entry_format"):
                ltype = checkSetting(k,"lex_type")
                if ltype: ltype=" as "+ltype
                types.append("lex-write"+ltype)
          print ("\n"+k+" ("+", ".join(types)+")")
          print (getSetting(k,"doc"))
       return 0
    elif html or markdown:
       if markdown:
          print ("")
          colon = "\n-------------------------------\n"
       else: colon = ":"
       print ("Available pronunciation formats"+colon)
       if html: print ('<table id="formats">')
       keys=list(lexFormats.keys()) ; keys.sort()
       for k in keys:
          if html: print ('<tr><td valign="top"><nobr>'+k+'</nobr></td><td valign="top">'+htmlify(getSetting(k,"doc"))+"</td></tr>")
          else: print (k+'\n: '+htmlify(getSetting(k,"doc"))+"\n")
       if html: print ("</table><script><!-- try to be more readable on some smartphones\nif(((screen && screen.width<600) || navigator.userAgent.slice(-6)==\"Gecko/\" /* UC Browser? */) && document.getElementById && document.getElementById('formats').outerHTML) document.getElementById('formats').outerHTML = document.getElementById('formats').outerHTML.replace(/<table/g,'<dl').replace(/<.table/g,'<'+'/dl').replace(/<tr><td/g,'<dt').replace(/<.td><td/g,'<'+'/dt><dd').replace(/<.td><.tr/g,'<'+'/dd');\n//--></script>")
    else: print ("Available pronunciation formats: "+", ".join(sorted(list(lexFormats.keys())))+"\n(Use --formats to see their descriptions)")
    if markdown: colon = "\n---------------"
    else:
       colon = ":"
       print (missALine)
    print ("Program options"+colon)
    print (missALine)
    if html: print ("<dl>")
    for _,opt,desc in sorted([(not not v.__doc__ and not v.__doc__.startswith('*'),k,v.__doc__) for k,v in globals().items()]):
       if not opt.startswith("mainopt_"): continue
       opt = funcToOpt(opt)
       if not desc: continue # undocumented option
       params,rest = desc.split("\n",1)
       if params.startswith('*'): params=params[1:]
       if params: opt += (' '+params)
       if html: print ("<dt>"+htmlify(opt)+"</dt><dd>"+htmlify(rest)+"</dd>")
       elif markdown: print (opt.replace("<","`<").replace(">",">`")+"\n: "+htmlify(rest)+"\n")
       else: print (opt+"\n"+rest+"\n")
    if html: print ("</dl>")
    return 0

catchingSigs = inSigHandler = False
def catchSignals():
  "We had better try to catch all signals if using MacBritish_System_Lexicon so we can safely clean it up. We raise KeyboardInterrupt instead (need to catch this). Might not work with multithreaded code."
  global catchingSigs
  if catchingSigs: return
  catchingSigs = True
  import signal
  def f(sigNo,*args):
    global inSigHandler
    if inSigHandler: return
    inSigHandler = True
    os.killpg(os.getpgrp(),sigNo)
    sys.stderr.write("\nCaught signal %d\n" % sigNo)
    raise KeyboardInterrupt
  for n in xrange(1,signal.NSIG):
    if not n in [
          signal.SIGCHLD, # sent on subprocess completion
          signal.SIGTSTP,signal.SIGCONT, # Ctrl-Z / fg
          signal.SIGWINCH, # window-size change
    ] and not signal.getsignal(n)==signal.SIG_IGN:
      try: signal.signal(n,f)
      except: pass
class MacBritish_System_Lexicon(object):
    """Overwrites some of the pronunciations in the system
    lexicon (after backing up the original).  Cannot
    change the actual words in the system lexicon, so just
    alters pronunciations of words you don't intend to use
    so you can substitute these into your texts.
    Restores the lexicon on close()."""
    instances = {}
    def __init__(self,text="",voice="Daniel"):
        """text is the text you want to speak (so that any
        words used in it that are not mentioned in your
        lexicon are unchanged in the system lexicon);
        text="" means you just want to speak phonemes.
        Special value of text=False means lexicon read only.
        voice can be Daniel, Emily or Serena."""
        self.voice = False
        if not text==False:
            assert not voice in MacBritish_System_Lexicon.instances, "There is already another instance of MacBritish_System_Lexicon for the "+voice+" voice"
            assert not os.system("lockfile -1 -r 10 /tmp/"+voice+".PCMWave.lock") # in case some other process has it (note: if you run with python -O, this check won't happen!)
            self.voice = voice # (don't set this if text==False, since we won't need cleanup on __del__)
        self.filename = "/System/Library/Speech/Voices/"+voice+".SpeechVoice/Contents/Resources/PCMWave"
        assert not (not os.path.exists(self.filename) and os.path.exists("/System/Library/Speech/Voices/"+voice+"Compact.SpeechVoice/Contents/Resources/PCMWave")), "The only installation of "+voice+" found on this system was the Compact one, which lexconvert does not yet support" # TODO: could try self.wordIndexStart = findW("Abiquiu"),self.phIndexStart = findW("'@b.Ik.ju"),self.wordIndexEnd = findW("www.youtube.com",1),self.phIndexEnd = findW("'d^b.l.ju.'d^b.l.ju.'d^b.l.ju.dA+t.'ju.'tjub.dA+t.kA+m",1), but "t" in phones should be ignored, "activesync" and "afterlife" have no phones, "aqua" has TWO sets of phonemes (aquarium ok) and there are other synchronization issues.
        # TODO: some sync issues persist even on the NON-Compact version in newer versions of macOS (e.g. 10.12).  This currently leads to exceptions in findW on such systems (which do say it could be due to wrong version of the voice); fixing would need looking at more sync issues as above
        assert os.path.exists(self.filename),"Cannot find an installation of '"+voice+"' on this system"
        if os.path.exists(self.filename+"0"):
            if text==False: self.filename += "0" # (use the backup file for read-only, if we created one before; this means we don't have to worry about locks)
        elif not text==False: # create a backup
            sys.stderr.write("Backing up "+self.filename+" to "+self.filename+"0...\n") # (you'll need a password if you're not running as root)
            err = os.system("sudo mv \""+self.filename+"\" \""+self.filename+"0\"; sudo cp \""+self.filename+"0\" \""+self.filename+"\"; sudo chown "+str(os.getuid())+" \""+self.filename+"\"")
            assert not err, "Error creating backup"
        lexFile = self.filename+".lexdir"
        if not os.path.exists(lexFile) and not text==False:
            sys.stderr.write("Creating lexdir file...\n")
            err = os.system("sudo touch \""+lexFile+"\" ; sudo chown "+str(os.getuid())+" \""+lexFile+"\"")
            assert not err, "Error creating lexdir"
        compat_err = "\nThis probably means your Mac has a new version of the voice that is no longer compatible with this system-lexicon patch."
        import cPickle
        if os.path.exists(lexFile) and os.stat(lexFile).st_size: self.wordIndexStart,self.wordIndexEnd,self.phIndexStart,self.phIndexEnd = cPickle.Unpickler(open(lexFile)).load()
        else:
            f = open(self.filename)
            dat = getBuf(f).read()
            def findW(word,rtnPastEnd=0):
                i = re.finditer(re.escape(word+chr(0)),dat)
                try: n = i.next()
                except StopIteration: raise Exception(word+" not found in voice file"+compat_err)
                try:
                    n2 = i.next()
                    raise Exception("%s does not uniquely identify a byte position (has at least %d and %d)%s" % (word,n.start(),n2.start(),compat_err))
                except StopIteration: pass
                if rtnPastEnd: return n.end()
                else: return n.start()
            self.wordIndexStart = findW("808s")
            self.phIndexStart = findW("'e&It.o&U.e&Its")
            self.wordIndexEnd = findW("zombie",1)
            self.phIndexEnd = findW("'zA+m.bI",1)
            if not text==False: cPickle.Pickler(open(lexFile,"w")).dump((self.wordIndexStart,self.wordIndexEnd,self.phIndexStart,self.phIndexEnd))
        if text==False: self.dFile = open(self.filename)
        else: self.dFile = open(self.filename,'r+')
        assert len(self.allWords()) == len(self.allPh()), str(len(self.allWords()))+" words but "+str(len(self.allPh()))+" phonemes"+compat_err
        self.textToAvoid = u""
        if text==False: return
        MacBritish_System_Lexicon.instances[voice] = self
        self.textToAvoid = text.decode('utf-8').replace(unichr(160),' ') ; self.restoreDic = {}
        catchSignals()
    def allWords(self):
        "Returns a list of words that are defined in the system lexicon (which won't be changed, but see allPh)"
        self.dFile.seek(self.wordIndexStart)
        return [x for x in getBuf(self.dFile).read(self.wordIndexEnd-self.wordIndexStart).split(chr(0)) if x]
    def allPh(self):
        "Returns a list of (file position, phoneme string) for each of the primary phoneme entries from the system lexicon.  These entries can be changed in-place by writing to the said file position, and then spoken by giving the voice the corresponding word from allWords (but see also usable_words)."
        self.dFile.seek(self.phIndexStart)
        def f(l):
            last = None ; r = [] ; pos = self.phIndexStart
            for i in l:
                if re.search(r'[ -~]',i) and not i in ["'a&I.'fo&Un","'lI.@n","'so&Un.j$"] and not (i==last and i in ["'tR+e&I.si"]): r.append((pos,i)) # (the listed pronunciations are secondary ones that for some reason are in the list)
                if re.search(r'[ -~]',i): last = i
                pos += (len(i)+1) # +1 for the \x00
            assert pos==self.phIndexEnd+1 # +1 because the last \00 will result in a "" item after; the above +1 will be incorrect for that item
            return r
        return f([x for x in getBuf(self.dFile).read(self.phIndexEnd-self.phIndexStart).split(chr(0))])
    def usable_words(self,words_ok_to_redefine=[]):
        "Returns a list of (word,phoneme_file_position,original_phonemes) by combining allWords with allPh, but omitting any words that don't seem 'usable' (for example words that contain spaces, since these lexicon entries don't seem to be actually used by the voice).  Words that occur in self.textToAvoid are also considered non-usable, unless they also occur in words_ok_to_redefine (user lexicon)."
        for word,(pos,phonemes) in zip(self.allWords(),self.allPh()):
            if not re.match("^[a-z0-9]*$",word): continue # it seems words not matching this regexp are NOT used by the engine
            if not (phonemes and 32<ord(phonemes[:1])<127): continue # better not touch those, just in case
            if word in self.textToAvoid and not word in words_ok_to_redefine: continue
            yield word,pos,phonemes
    def check_redef(self,wordsAndPhonemes):
        "Diagnostic function to list on standard error the 'redefinitions' we want to make.  wordsAndPhonemes is a list of (original system-lexicon word, proposed new phonemes).  The old phonemes are also listed, fetched from allPh."
        aw = self.allWords() ; ap = 0
        for w,p in wordsAndPhonemes:
          w = w.lower()
          if not re.match("^[a-z0-9]*$",w): continue
          if not w in aw: continue
          if not ap:
            ap = self.allPh()
            sys.stderr.write("Warning: some words were already in system lexicon\nword\told\tnew\n")
          sys.stderr.write(w+"\t"+ap[aw.index(w)][1]+"\t"+p+"\n")
    def speakPhones(self,phonesList):
        "Speaks every phonetic word in phonesList"
        words = [str(x)+"s" for x in range(len(phonesList))]
        d = self.setMultiple(words,phonesList)
        msc = os.popen(macSayCommand()+" -v \""+self.voice+"\"",'w')
        getBuf(msc).write(as_utf8(" ").join(d.get(w,as_utf8("")) for w in words))
    def readWithLex(self,lex):
        "Reads the text given in the constructor after setting up the lexicon with the given (word,phoneme) list"
        # self.check_redef(lex) # uncomment if you want to know about these
        textToPrint = u' '+self.textToAvoid+u' '
        tta = ' '+self.textToAvoid.replace(u'\u2019',"'").replace(u'\u2032','').replace(u'\u00b4','').replace(u'\u02b9','').replace(u'\u00b7','').replace(u'\u2014',' ')+' ' # (ignore pronunciation marks 2032 and b7 that might be in the text, but still print them in textToPrint; also normalise apostrophes but not in textToPrint, and be careful with dashes as lex'ing the word after a hyphen or em-dash won't work BUT we still want to support hyphenated words IN the lexicon, so em-dashes are replaced here and hyphens are included in nonWordBefore below)
        words2,phonemes2 = [],[] # keep only the ones actually used in the text (no point setting whole lexicon)
        nonWordBefore=r"(?i)(?<=[^A-Za-z"+chr(0)+"-])" # see below for why chr(0) is included, and see comment above for why hyphen is at the end; (?i) = ignore case
        nonWordAfter=r"(?=([^A-Za-z'"+unichr(0x2019)+"-]|['"+unichr(0x2019)+r"-][^A-Za-z]))" # followed by non-letter non-apostrophe, or followed by apostrophe non-letter (so not if followed by "'s", because the voice won't use our custom lex entry if "'s" is added to the lex'd word, TODO: automatically add "'s" versions to the lexicon via +s or +iz?) (also not if followed by hyphen-letters; hyphen before start is handled above, although TODO preceded by non-letter + hyphen might be OK)
        ttal = tta.lower()
        for ww,pp in lex:
          ww = ww.decode('utf-8') # so you can add words with accents etc (in utf-8) to the lexicon
          if ww.lower() in ttal and re.search(nonWordBefore+re.escape(ww)+nonWordAfter,tta):
            words2.append(ww) ; phonemes2.append(pp)
        for k,v in self.setMultiple(words2,phonemes2).iteritems():
           tta = re.sub(nonWordBefore+re.escape(k)+nonWordAfter,chr(0)+v,tta)
           textToPrint = re.sub(nonWordBefore+'('+u'[\u2032\u00b4\u02b9\u00b7]*'.join(re.escape(c) for c in k)+')'+nonWordAfter,chr(0)+r'\1'+chr(1),textToPrint)
        tta = tta.replace(chr(0),'')
        term = os.environ.get("TERM","")
        if ("xterm" in term or term=="screen") and sys.stdout.isatty(): # we can probably underline words (inverse is more widely supported than underline, e.g. should work even on an old Linux console in case someone's using that to control an OS X server, but there might be a *lot* of words, which wouldn't be very good in inverse if user needs dark background and inverse is bright.  Unlike Annogen, we're dealing primarily with Latin letters.)
           import textwrap
           textwrap.len = lambda x: len(x.replace(chr(0),"").replace(chr(1),"")) # a 'hack' to make (at least the 2.x implementations of) textwrap ignore our chr(0) and chr(1) markers in their calculations.  Relies on textwrap calling len().
           print (textwrap.fill(textToPrint,stdout_width_unix(),break_on_hyphens=False).encode('utf-8').replace(chr(0),"\x1b[4m").replace(chr(1),"\x1b[0m").strip()) # break_on_hyphens=False because we don't really want hyphenated NAMES to be split across lines, and anyway textwrap in (at least) Python 2.7 has a bug that sometimes causes a line breaks to be inserted before a syllable marker symbol like 'prime'
        # else don't print anything (saves confusion)
        msc = os.popen(macSayCommand()+" -v \""+self.voice+"\"",'w')
        getBuf(msc).write(tta.encode('utf-8'))
    def setMultiple(self,words,phonemes):
        "Sets phonemes for words, returning dict of word to substitute word.  Flushes file buffer before return."
        avail = [] ; needed = []
        for word,pos,phon in self.usable_words(words):
            avail.append((len(phon),word,pos,phon))
        for word,phon in zip(words,phonemes):
            needed.append((len(phon),word,phon))
        avail.sort() ; needed.sort() # shortest phon first
        i = 0 ; wDic = {} ; iDone=set() ; mustBeAlpha=True
        # mustBeAlpha: prefer alphabetical words, since
        # these can be capitalised at start of sentence
        # (the prosody doesn't always work if it isn't)
        for l,word,phon in needed:
            while avail[i][0] < l or (mustBeAlpha and not re.match(as_utf8("[A-Za-z]"),avail[i][1])) or i in iDone:
                i += 1
                if i==len(avail):
                    if mustBeAlpha: # desperate situation: we HAVE to use the non-alphabetical slots now (ideally we should pick words that never occur at start of sentence for them, but this branch is hopefully a rare situation in practice)
                       mustBeAlpha=False ; i=0; continue
                    sys.stderr.write("Could not find enough lexicon slots!\n") # TODO: we passed 'words' to usable_words's words_ok_to_redefine - this might not be the case if we didn't find enough slots
                    self.dFile.flush() ; return wDic
            iDone.add(i)
            _,wSubst,pos,oldPhon = avail[i] ; i += 1
            if avail[i][2] in self.restoreDic: oldPhon=None # shouldn't happen if setMultiple is called only once, but might be useful for small experiments in the Python interpreter etc
            self.set(pos,phon,oldPhon)
            wDic[word] = wSubst[:1].upper()+wSubst[1:] # always capitalise it so it can be used at start of sentence too (TODO: copy original capitalisation of each instance instead, in case it happens to come directly after a dotted abbreviation? although if it's something that's always capitalised anyway, e.g. most names, then this won't make any difference)
        self.dFile.flush() ; return wDic
    def set(self,phPos,val,old=None):
        """Sets phonemes at position phPos to new value.
        Caller should flush the file buffer when done."""
        # print "Debugger: setting %x to %s" % (phPos,val)
        if old:
            assert not phPos in self.restoreDic, "Cannot call set() twice on same phoneme while re-specifying 'old'"
            assert len(val) <= len(old), "New phoneme is too long!"
            self.restoreDic[phPos] = old
        else: assert phPos in self.restoreDic, "Must specify old values (for restore) when setting for first time"
        self.dFile.seek(phPos)
        getBuf(self.dFile).write(val+as_utf8(chr(0)))
    def __del__(self):
        "WARNING - this might not be called before exit - best to call close() manually"
        if not self.voice: return
        self.close()
    def close(self):
        for phPos,val in self.restoreDic.items():
            self.set(phPos,val)
        self.dFile.close()
        del MacBritish_System_Lexicon.instances[self.voice]
        assert not os.system("rm -f /tmp/"+self.voice+".PCMWave.lock")
        self.voice=None
def stdout_width_unix(): # assumes isatty
   import struct,fcntl,termios
   return struct.unpack('hh', fcntl.ioctl(1,termios.TIOCGWINSZ,'1234'))[1]

lexFormats = LexFormats() # at end, in case it refers to anything that was defined later

if __name__ == "__main__": sys.exit(main())