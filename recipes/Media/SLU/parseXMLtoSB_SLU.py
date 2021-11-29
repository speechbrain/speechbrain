# -*- coding: utf-8 -*-

"""
Data preparation.
Download: See README.md

Author
------
Gaelle Laperriere
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom as DOM
from tqdm import tqdm
import subprocess
import num2words
import argparse
import csv
import os
import glob
import re

def parse_turns(root, path, filename, channels, IDs, ID, wav_folder, skip_wav, specifiers, method):
    '''
    Prepares the data for the csv files of the Media dataset.
    Download: http

    Arguments
    ---------
    root : Document
        Object representing the content of the Media xml document being processed.
    path : str
        Path of the original Media file without the extension ".wav" nor ".trs".
    filename : str
        Name of the Media recording.
    channels : list of str
        Channels (Right / Left) of the stereo recording to keep.
    IDs : list of str
        Linked IDs of the recordings, for the channels to keep.
    ID : str
        Current ID of the recording being processed.

    Returns
    -------
    list of str
    '''

    data = []
    speaker_id, speaker_name = get_speaker(root)
    channel = get_channel(ID, channels, IDs)
    if not(skip_wav):
        split_audio_channels(path, filename, channel, wav_folder)
    for turn in root.getElementsByTagName('Turn'):
        speakers = turn.getAttribute('speaker').split(' ')
        if speaker_id in speakers:
            time_beg = turn.getAttribute('startTime')
            time_end = turn.getAttribute('endTime')
            wav = wav_folder + '/' + channel + filename + '.wav'
            sentences = parse_sentences(turn.childNodes, speakers, time_beg, time_end, filename, specifiers, method)
            IDs = get_IDs(speaker_name, sentences[speaker_id], channel, filename)
            # Append only if speaker (not compère)
            for speaker in sentences:
                if speaker == speaker_id:
                    for n in range(len(sentences[speaker])):
                        out = subprocess.Popen(
                            ['soxi', '-D', wav_folder + '/' + channel + filename + '.wav'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT
                        )
                        stdout, stderr = out.communicate()
                        wav_duration = str("%.2f" % float(stdout))
                        duration = str("%.2f" % (float(sentences[speaker][n][3]) - float(sentences[speaker][n][2])))
                        if float(wav_duration) >= float(sentences[speaker][n][3]) and float(duration) != 0.0:
                            data.append([
                                IDs[n],
                                duration,
                                sentences[speaker][n][2], sentences[speaker][n][3],
                                wav, 'wav',
                                speaker_name, 'string',
                                sentences[speaker][n][0], 'string',
                                sentences[speaker][n][1], 'string'])
    return data

def parse_sentences(nodes, speakers, time_beg, time_end, filename, specifiers, method):
    '''
    Get the sentences spoken by the speaker (not the "Compère" aka Woz).

    Arguments:
    -------
    nodes : list of Document
        All the xml following nodes present in the turn.
    speakers : dictionnary
        ID and corresponding name of the speakers in the recording.

    Returns
    -------
    dictionnary of str
    '''

    currently_cut_concept = {}
    currently_open_concept = {}
    has_speech = {}
    current_speaker = speakers[0]
    sentences = {}
    concept = {}
    concept_added = {}
    n = {} # Number of segments per speaker in the turn
    for speaker in speakers:
        sentences[speaker] = [['', '', time_beg, time_end]]
        n[speaker] = 0
        concept[speaker] = 'null'
        currently_open_concept[speaker] = False
        currently_cut_concept[speaker] = False
        has_speech[speaker] = False
        concept_added[current_speaker] = False

    # For each node in the Turn
    for node in nodes:

        # Check concept
        if node.nodeName == 'SemDebut':
            concept[current_speaker] = node.getAttribute('concept')
            identifiant = node.getAttribute('identifiant')
            if (filename + '#' + identifiant in specifiers
                    and method == 'relax'
                    and concept[current_speaker][-len(specifiers[filename + '#' + identifiant]):] == specifiers[filename + '#' + identifiant]):
                concept[current_speaker] = concept[current_speaker][:-len(specifiers[filename + '#' + identifiant])]
            #if (filename + '#' + identifiant in specifiers
                    #and method == 'full'
                    #and concept[current_speaker][-len(specifiers[filename + '#' + identifiant]):] != specifiers[filename + '#' + identifiant]):
                #concept[current_speaker] += specifiers[filename + '#' + identifiant]
            if concept[current_speaker] != 'null':
                concept_added[current_speaker] = False

        # Check transcription
        if node.nodeType == node.TEXT_NODE and node.data.replace('\n', '') != '':
            # Add a new concept, when speech following
            # (useful for 'SemDebut + Sync + Speech' and 'SemDebut + Speech + Sync + Speech' sequences)
            if concept[current_speaker] != 'null' and not(concept_added[current_speaker]):
                sentences[current_speaker][n[current_speaker]][0] += '<' + concept[current_speaker] + '> '
                sentences[current_speaker][n[current_speaker]][1] += '<' + concept[current_speaker] + '> _ '
                concept_added[current_speaker] = True
            if currently_open_concept[current_speaker]:
                currently_cut_concept[current_speaker] = True
            sentence = node.data.replace("'", "' ") # Join the apostrophe to the previous word only
            sentence = sentence.replace("c' est", "c'est") # Re-join this word
            sentence = ' '.join(sentence.split()) # Remove double spaces made from the first replacement
            sentences[current_speaker][n[current_speaker]][0] += sentence + ' '
            sentences[current_speaker][n[current_speaker]][1] += ' '.join(list(sentence.replace(' ','_'))) + ' _ ' # Convert spaces to underscores
            sentences[current_speaker][n[current_speaker]][3] = time_end
            has_speech[current_speaker] = True

        # Check speaker if there are many in the Turn
        if node.nodeName == 'Who':
            current_speaker = speakers[int(node.getAttribute('nb'))-1]

        # Save audio segment for a speaker
        if node.nodeName == 'SemFin':
            # Prevent adding a closing concept if Sync followed by SemFin generate a new segment without speech yet
            if concept[current_speaker] != 'null' and has_speech[current_speaker]:
                sentences[current_speaker][n[current_speaker]][0] += '> '
                sentences[current_speaker][n[current_speaker]][1] += '> _ '
            currently_open_concept[current_speaker] = False
            concept[current_speaker] = 'null' # Indicate there is no currently open concept

        if node.nodeName == 'Sync':
            for speaker in speakers:
                # If the segment has no speech yet
                if not(has_speech[speaker]):
                    sentences[speaker][n[speaker]][2] = node.getAttribute('time') # Change time_beg for the last segment of each speaker
                # If the segment has speech
                else:
                    sentences[speaker][n[speaker]][3] = node.getAttribute('time') # Change time_end for the last segment of each speaker
                    sentences[speaker].append(['', '', sentences[speaker][n[speaker]][3], time_end]) # Create new segment for each speaker
                    has_speech[speaker] = False

                    # Currently open concept
                    if concept[current_speaker] != 'null' and concept_added[speaker]:
                        # Close the concept for the latest segment processed (Sync before SemFin)
                        sentences[speaker][n[speaker]][0] += '> '
                        sentences[speaker][n[speaker]][1] += '> _ '
                        currently_open_concept[speaker] = True
                        concept_added[speaker] = False
                    if currently_cut_concept[speaker]:
                        sentences[speaker][n[speaker]-1][0], sentences[speaker][n[speaker]-1][1] = make_exception(sentences[speaker][n[speaker]-1][0], sentences[speaker][n[speaker]-1][1], filename)
                        currently_cut_concept[speaker] = False
                    n[speaker] += 1

    # If the Turn end without a Sync, check concepts
    for speaker in speakers:
        if currently_cut_concept[speaker] == True:
            sentences[speaker][n[speaker]-1][0], sentences[speaker][n[speaker]-1][1] = make_exception(sentences[speaker][n[speaker]-1][0], sentences[speaker][n[speaker]-1][1], filename)

    # Normalize sentences
    for speaker in sentences:
        for n in range(len(sentences[speaker])):
            if sentences[speaker][n][0] != '':
                sentence = sentences[speaker][n][0][:-1] # Remove last ' '
                sentences[speaker][n][0] = normalize_sentence(sentence, filename)
                sentence = sentences[speaker][n][1][:-3] # Remove last ' _ '
                sentences[speaker][n][1] = normalize_sentence(sentence, filename)
            else:
                del sentences[speaker][n] # Might be usefull for the last appended segment
    return sentences

def normalize_sentence(sentence, filename):
    sentence = re.sub('\(.*?\)', '', sentence) # Remove round brackets and all in between
    sentence = re.sub(r"[^\w\s'-><_]",'', sentence) # Remove punctuation except '-><_
    sentence = sentence.replace('*', '')
    sentence = sentence.lower() # Lowercase letters
    words = sentence.split() # Get each 'word' in a list
    for index in range(len(words)):
        if words[index].isdigit():
            words[index] = num2words.num2words(words[index]) # Change numbers to words
    sentence = ' '.join(words) # Re-join each 'words' without double spaces
    return sentence

def write_first_row(csv_folder):
    for corpus in ['train', 'dev', 'test']:
        SB_file = open(csv_folder + '/' + corpus + '.csv', 'w')
        writer = csv.writer(SB_file, delimiter=',')
        writer.writerow([
            'ID',
            'duration',
            'start_seg', 'end_seg',
            'wav', 'wav_format',
            'spk_id', 'spk_id_format',
            'wrd', 'wrd_format',
            'char', 'char_format'])
        SB_file.close()
    return None

def append_data(path, data):
    '''
    Make the csv corpora using data retrieved previously for one Media file.

    Arguments:
    -------
    path : str
        Path of the folder to store csv.
    data : list of str
        Data retrieved from the original xml file.

    Returns
    -------
    None
    '''
    SB_file = open(path, 'a')
    writer = csv.writer(SB_file, delimiter=',')
    writer.writerows(data)
    SB_file.close()
    return None

def split_audio_channels(path, filename, channel, wav_folder):
    '''
    Split the stereo wav Media files from the dowloaded dataset.
    Keep only the speaker channel.

    Arguments:
    -------
    path : str
        Path of the original Media file without the extension ".wav" nor ".trs".
    filename : str
        Name of the Media recording.
    channel : str
        "R" or "L" following the channel of the speaker in the stereo wav file.

    Returns
    -------
    None
    '''

    channel_int = '1'
    if channel == 'R':
        channel_int = '2'
    path = path.replace('1 ',"'1 ")
    path = path.replace('2 ',"'2 ")
    path = path.replace(' 2'," 2'")
    os.system('sox ' + path + filename + '.wav ' + wav_folder + '/' + channel + filename + '_8khz.wav remix ' + channel_int)
    os.system('sox -G ' + wav_folder + '/' + channel + filename + '_8khz.wav -r 16000 ' + wav_folder + '/' + channel + filename + '.wav 2>/dev/null')
    os.system('rm ' + wav_folder + '/' + channel + filename + '_8khz.wav')
    return None

def get_root(path):
    with open(path , "rb") as  fin :
        text = fin.read()
        text2 = text.decode('ISO-8859-1')
        tree = DOM.parseString(text2)
        root = tree.childNodes[1]
    return root

def get_channels(path):
    channels = []
    IDs = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            channels.append(row[0])
            IDs.append(row[1])
    return channels, IDs

def get_specifiers(path):
    specifiers = {}
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[1][0] != '-':
                dial = row[0]
            else:
                specifiers[dial + "#" + row[0]] = row[1]
    return specifiers

def get_corpora():
    corpora = {}
    for corpus in ['train', 'dev', 'test']:
        FN_file = open('../' + corpus + '_filenames.txt', 'r')
        corpora[corpus] = FN_file.readlines()
        corpora[corpus] = [i[:-1] for i in corpora[corpus]]
        FN_file.close()
    return corpora

def get_speaker(root):
    for speaker in root.getElementsByTagName("Speaker"):
        if speaker.getAttribute('name')[0] == 's':
            return speaker.getAttribute('id'), '_'.join(speaker.getAttribute('name').split('#')).replace('-','_')

def get_channel(ID, channels, IDs):
    channel = channels[IDs.index(ID)]
    return channel

def get_corpus(corpora, filename):
    for corpus in corpora:
        if filename in corpora[corpus]:
            return corpus
    return None

def get_IDs(speaker_name, sentences, channel, filename):
    IDs = []
    for sentence in sentences:
        IDs.append(channel + filename + '#' + speaker_name + '#' + sentence[2] + '_' + sentence[3])
    return IDs

def make_exception(a, b, filename):
    if filename == '104':
        a = "hum "
        b = "h u m _ "
    elif filename == '262':
        a = "<reponse> oui > je voudrais savoir si je peux <command-dial> annuler > la "
        b = "<reponse> _ o u i _ > _ j e _ v o u d r a i s _ s a v o i r _ s i _ j e _ p e u x _ <command-dial> _ a n n u l e r _ > _ l a _ "
    elif filename == '476':
        a = "euh du "
        b = "e u h _ d u _ "
    elif filename == '606':
        a = "bon ben tant pis je euh "
        b = "b o n _ b e n _ t a n t _ p i s _ j e _ e u h _ "
    elif filename == '661':
        a = "<reponse> d' accord > euh c "
        b = "<reponse> _ d ' _ a c c o r d _ > _ e u h _ c _ "
    elif filename == '877':
        a = "<reponse> non > <paiement-montant-entier-reservation-chambre> cinquante > <paiement-monnaie> euros > c' est <temps-unite-reservation> pour le week-end > <connectProp> parce que > il y a <sejour-nbNuit-reservation> deux nuits > normalement <nombre-temps-reservation> un > <temps-unite-reservation> week-end > ah <reponse> non > il y en a qu' <sejour-nbNuit-reservation> une > <reponse> non non > ça sera très bien je r "
        b = "<reponse> _ n o n _ > _ <paiement-montant-entier-reservation-chambre> _ c i n q u a n t e _ > _ <paiement-monnaie> _ e u r o s _ > _ c ' _ e s t _ <temps-unite-reservation> _ p o u r _ l e _ w e e k - e n d _ > _ <connectProp> _ p a r c e _ q u e _ > _ i l _ y _ a _ <sejour-nbNuit-reservation> _ d e u x _ n u i t s _ > _ n o r m a l e m e n t _ <nombre-temps-reservation> _ u n _ > _ <temps-unite-reservation> _ w e e k - e n d _ > _ a h _ <reponse> _ n o n _ > _ i l _ y _ e n _ a _ q u ' _ <sejour-nbNuit-reservation> _ u n e _ > _ <reponse> _ n o n _ n o n _ > _ ç a _ s e r a _ t r è s _ b i e n _ j e _ r _ "
    elif filename == '921':
        a = "bon ben très bien <command-tache> je réserve > euh comme je dois faire <evenement> le concours de pêche > à tout prix <rang-temps-reservation> pour la deuxième > <temps-unite-reservation> semaine > <temps-mois-reservation> de de juillet > <localisation-ville-hotel> à Dole > ben donc je "
        b = "b o n _ b e n _ t r è s _ b i e n _ <command-tache> _ j e _ r é s e r v e _ > _ e u h _ c o m m e _ j e _ d o i s _ f a i r e _ <evenement> _ l e _ c o n c o u r s _ d e _ p ê c h e _ > _ à _ t o u t _ p r i x _ <rang-temps-reservation> _ p o u r _ l a _ d e u x i è m e _ > _ <temps-unite-reservation> _ s e m a i n e _ > _ <temps-mois-reservation> _ d e _ d e _ j u i l l e t _ > _ <localisation-ville-hotel> _ à _ D o l e _ > _ b e n _ d o n c _ j e _ "
    elif filename == '936':
        a = "ah je je préfère je préfère être <localisation-distanceRelative-hotel> près de > <localisation-lieuRelatif-general-hotel> la rivière > *avoir trop de retard etc <connectProp> donc > <command-tache> je vais prendre > <lienRef-coRef> la > <objet> chambre > <command-tache> je vais réserver > <nombre-chambre-reservation> une chambre > du "
        b = "a h _ j e _ j e _ p r é f è r e _ j e _ p r é f è r e _ ê t r e _ <localisation-distanceRelative-hotel> _ p r è s _ d e _ > _ <localisation-lieuRelatif-general-hotel> _ l a _ r i v i è r e _ > _ a v o i r _ t r o p _ d e _ r e t a r d _ e t c _ <connectProp> _ d o n c _ > _ <command-tache> _ j e _ v a i s _ p r e n d r e _ > _ <lienRef-coRef> _ l a _ > _ <objet> _ c h a m b r e _ > _ <command-tache> _ j e _ v a i s _ r é s e r v e r _ > _ <nombre-chambre-reservation> _ u n e _ c h a m b r e _ > _ d u _ "
    elif filename == '1084':
        a = "<reponse> d' accord > bien ben <command-tache> je réserve > donc <lienRef-coRef> cette > <objet> chambre > pour "
        b = "<reponse> _ d ' _ a c c o r d _ > _ b i e n _ b e n _ <command-tache> _ j e _ r é s e r v e _ > _ d o n c _ <lienRef-coRef> _ c e t t e _ > _ <objet> _ c h a m b r e _ > _ p o u r _ "
    elif filename == '1095':
        a = "<command-tache> je souhaite réserver > <lienRef-coRef> les > <nombre-chambre-reservation> cinq chambres > à d à "
        b = "<command-tache> _ j e _ s o u h a i t e _ r é s e r v e r _ > _ <lienRef-coRef> _ l e s _ > _ <nombre-chambre-reservation> _ c i n q _ c h a m b r e s _ > _ à _ d _ à _ "
    elif filename == '1317' or filename == '1432':
        a = "ah "
        b = "a h _ "
    elif filename == '1459':
        a = "alors <temps-date-debut> pour le trente et un mai > <temps-date-fin> au trois juin > <localisation-ville-hotel-debut> à Lyon > <connectProp> et > <temps-date-debut> le *trois du trois > <temps-date-fin> au quatre juin > <localisation-ville-fin> à Chambéry > je souhaite <nombre-chambre> dix > <chambre-type> chambres simples > <chambre-standing> bon standing > pour <paiement-montant-entier-chambre> deux cents > <paiement-monnaie> euros > la chambre avec "
        b = "a l o r s _ <temps-date-debut> _ p o u r _ l e _ t r e n t e _ e t _ u n _ m a i _ > _ <temps-date-fin> _ a u _ t r o i s _ j u i n _ > _ <localisation-ville-hotel-debut> _ à _ L y o n _ > _ <connectProp> _ e t _ > _ <temps-date-debut> _ l e _ t r o i s _ d u _ t r o i s _ > _ <temps-date-fin> _ a u _ q u a t r e _ j u i n _ > _ <localisation-ville-fin> _ à _ C h a m b é r y _ > _ j e _ s o u h a i t e _ <nombre-chambre> _ d i x _ > _ <chambre-type> _ c h a m b r e s _ s i m p l e s _ > _ <chambre-standing> _ b o n _ s t a n d i n g _ > _ p o u r _ <paiement-montant-entier-chambre> _ d e u x _ c e n t s _ > _ <paiement-monnaie> _ e u r o s _ > _ l a _ c h a m b r e _ a v e c _ "
    return a, b

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_folder', type=str, help='Path where folders S0272 and E0024 are stored.')
    parser.add_argument('wav_folder', type=str, help='Path where the wavs will be stored.')
    parser.add_argument('csv_folder', type=str, help='Path where the csv will be stored.')
    parser.add_argument('-w', '--skip_wav', action='store_true', required=False, help='Skip the wav files storing if already done with ASR Media processor.')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-r', '--relax', action='store_true', required=False, help='Remove specifiers from concepts.')
    group.add_argument('-f', '--full', action='store_false', required=False, help='Keep specifiers in concepts. Method used by default.')
    args = parser.parse_args()

    data_folder = args.data_folder
    wav_folder = args.wav_folder
    csv_folder = args.csv_folder
    skip_wav = args.skip_wav
    if args.relax:
        method = 'relax'
    else:
        method = 'full'
    print("You are processing Media Dataset using " + method + " method for the concepts.")

    paths = glob.glob(data_folder + '/S0272/**/*.wav', recursive = True)
    channels, IDs = get_channels('../channels.csv')
    specifiers = get_specifiers('./specifiers.csv')
    corpora = get_corpora()
    write_first_row(csv_folder)

    for path in tqdm(paths):
        filename = path.split('/')[-1][:-4]
        path = path[:-len(filename)-4]
        root = get_root(data_folder + '/E0024/MEDIA1FR_00/MEDIA1FR/DATA/semantizer_files/' + filename + '_HC.xml')
        filename = filename.split('_')[0]
        data = parse_turns(root, path, filename, channels, IDs, filename, wav_folder, skip_wav, specifiers, method)
        if data != None:
            corpus = get_corpus(corpora, filename)
            # Check if file used in corpora.
            if corpus != None:
                append_data(csv_folder + '/' + corpus + '.csv', data)
