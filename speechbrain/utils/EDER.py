"""Calculates Emotion Diarization Error Rate (EDER) which is the sum of Missed Emotion (ME),
False Alarm (FA), and Confusion (CF).

Authors
 * Yingzhi Wang 2023
"""


def EDER(prediction, id, duration, emotion, window_length, stride):
    """ Calculates the EDER value

    Args:
        prediction (list): a list of frame-wise predictions of the utterance
        id (str): id of the utterance
        duration (float): duration of the utterance
        emotion (list of dicts): the ground truth emotion and its duration,
            e.g. [{'emo': 'angry', 'start': 1.016, 'end': 6.336}]
        window_length (float): the frame length used for frame-wise prediction
        stride (float): the frame length used for frame-wise prediction

    Returns:
        float: the calculted EDER for the utterance

    Example
    -------
    >>> from speechbrain.utils.EDER import EDER
    >>> prediction=['n', 'n', 'n', 'a', 'a', 'a']
    >>> id="spk1_1"
    >>> duration=1.22
    >>> emotion=[{'emo': 'angry', 'start': 0.39, 'end': 1.10}]
    >>> window_length = 0.2
    >>> stride = 0.2
    >>> EDER(prediction, id, duration, emotion, window_length, stride)
    0.2704918032786885
    """

    duration = float(duration)  # for recipe tests
    lol = []
    for i in range(len(prediction)):
        start = stride * i
        end = start + window_length
        lol.append([id, start, end, prediction[i]])

    lol = merge_ssegs_same_emotion_adjacent(lol)
    if len(lol) != 1:
        lol = distribute_overlap(lol)

    ref = reference_to_lol(id, duration, emotion)

    good_preds = 0
    for i in ref:
        candidates = [element for element in lol if element[3] == i[3]]
        ref_interval = [i[1], i[2]]

        for candidate in candidates:
            overlap = getOverlap(ref_interval, [candidate[1], candidate[2]])
            good_preds += overlap
    return 1 - good_preds / duration


def getOverlap(a, b):
    """ get the overlapped length of two intervals

    Arguments
    ---------
    a : list
    b : list

    Returns:
        float: overlapped length

    Example
    -------
    >>> from speechbrain.utils.EDER import getOverlap
    >>> interval1=[1.2, 3.4]
    >>> interval2=[2.3, 4.5]
    >>> getOverlap(interval1, interval2)
    1.1
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def is_overlapped(end1, start2):
    """Returns True if segments are overlapping.

    Arguments
    ---------
    end1 : float
        End time of the first segment.
    start2 : float
        Start time of the second segment.

    Returns
    -------
    overlapped : bool
        True of segments overlapped else False.

    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> diar.is_overlapped(5.5, 3.4)
    True
    >>> diar.is_overlapped(5.5, 6.4)
    False
    """

    if start2 > end1:
        return False
    else:
        return True


def merge_ssegs_same_emotion_adjacent(lol):
    """Merge adjacent sub-segs if they are the same emotion.
    Arguments
    ---------
    lol : list of list
        Each list contains [utt_id, sseg_start, sseg_end, emo_label].
    Returns
    -------
    new_lol : list of list
        new_lol contains adjacent segments merged from the same emotion ID.
    Example
    -------
    >>> from speechbrain.utils.EDER import merge_ssegs_same_emotion_adjacent
    >>> lol=[['u1', 0.0, 7.0, 'a'],
    ... ['u1', 7.0, 9.0, 'a'],
    ... ['u1', 9.0, 11.0, 'n'],
    ... ['u1', 11.0, 13.0, 'n'],
    ... ['u1', 13.0, 15.0, 'n'],
    ... ['u1', 15.0, 16.0, 'a']]
    >>> merge_ssegs_same_emotion_adjacent(lol)
    [['u1', 0.0, 9.0, 'a'], ['u1', 9.0, 15.0, 'n'], ['u1', 15.0, 16.0, 'a']]
    """
    new_lol = []

    # Start from the first sub-seg
    sseg = lol[0]
    flag = False
    for i in range(1, len(lol)):
        next_sseg = lol[i]
        # IF sub-segments overlap AND has same emotion THEN merge
        if is_overlapped(sseg[2], next_sseg[1]) and sseg[3] == next_sseg[3]:
            sseg[2] = next_sseg[2]  # just update the end time
            # This is important. For the last sseg, if it is the same emotion then merge
            # Make sure we don't append the last segment once more. Hence, set FLAG=True
            if i == len(lol) - 1:
                flag = True
                new_lol.append(sseg)
        else:
            new_lol.append(sseg)
            sseg = next_sseg
    # Add last segment only when it was skipped earlier.
    if flag is False:
        new_lol.append(lol[-1])

    return new_lol


def reference_to_lol(id, duration, emotion):
    """change reference to a list of list
    Arguments
    ---------
    id (str): id of the utterance
    duration (float): duration of the utterance
    emotion (list of dicts): the ground truth emotion and its duration,
        e.g. [{'emo': 'angry', 'start': 1.016, 'end': 6.336}]

    Returns
    -------
    lol : list of list
        It has each list structure as [rec_id, sseg_start, sseg_end, spkr_id].

    Example
    -------
    >>> from speechbrain.utils.EDER import reference_to_lol
    >>> id="u1"
    >>> duration=8.0
    >>> emotion=[{'emo': 'angry', 'start': 1.016, 'end': 6.336}]
    >>> reference_to_lol(id, duration, emotion)
    [['u1', 0, 1.016, 'n'], ['u1', 1.016, 6.336, 'a'], ['u1', 6.336, 8.0, 'n']]
    """
    assert (
        len(emotion) == 1
    ), "NotImplementedError: The solution is only implemented for one-emotion utterance for now."
    lol = []

    start = emotion[0]["start"]
    end = emotion[0]["end"]
    if start > 0:
        lol.append([id, 0, start, "n"])
    lol.append([id, start, end, emotion[0]["emo"][0]])

    duration = float(duration)  # for recipe tests
    if end < duration:
        lol.append([id, end, duration, "n"])
    return lol


def distribute_overlap(lol):
    """Distributes the overlapped speech equally among the adjacent segments
    with different emotions.

    Arguments
    ---------
    lol : list of list
        It has each list structure as [rec_id, sseg_start, sseg_end, spkr_id].

    Returns
    -------
    new_lol : list of list
        It contains the overlapped part equally divided among the adjacent
        segments with different emotion IDs.

    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> lol = [['r1', 5.5, 9.0, 's1'],
    ... ['r1', 8.0, 11.0, 's2'],
    ... ['r1', 11.5, 13.0, 's2'],
    ... ['r1', 12.0, 15.0, 's1']]
    >>> diar.distribute_overlap(lol)
    [['r1', 5.5, 8.5, 's1'], ['r1', 8.5, 11.0, 's2'], ['r1', 11.5, 12.5, 's2'], ['r1', 12.5, 15.0, 's1']]
    """

    new_lol = []
    sseg = lol[0]

    # Add first sub-segment here to avoid error at: "if new_lol[-1] != sseg:" when new_lol is empty
    # new_lol.append(sseg)

    for i in range(1, len(lol)):
        next_sseg = lol[i]
        # No need to check if they are different emotions.
        # Because if segments are overlapped then they always have different emotions.
        # This is because similar emotion's adjacent sub-segments are already merged by "merge_ssegs_same_emotion()"

        if is_overlapped(sseg[2], next_sseg[1]):

            # Get overlap duration.
            # Now this overlap will be divided equally between adjacent segments.
            overlap = sseg[2] - next_sseg[1]

            # Update end time of old seg
            sseg[2] = sseg[2] - (overlap / 2.0)

            # Update start time of next seg
            next_sseg[1] = next_sseg[1] + (overlap / 2.0)

            if len(new_lol) == 0:
                # For first sub-segment entry
                new_lol.append(sseg)
            else:
                # To avoid duplicate entries
                if new_lol[-1] != sseg:
                    new_lol.append(sseg)

            # Current sub-segment is next sub-segment
            sseg = next_sseg

        else:
            # For the first sseg
            if len(new_lol) == 0:
                new_lol.append(sseg)
            else:
                # To avoid duplicate entries
                if new_lol[-1] != sseg:
                    new_lol.append(sseg)

            # Update the current sub-segment
            sseg = next_sseg

    # Add the remaining last sub-segment
    new_lol.append(next_sseg)

    return new_lol
