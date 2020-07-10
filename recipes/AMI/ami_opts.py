import numpy as np  # noqa F401
import commands as cmd
import sys
from sets import Set  # noqa F401


class PreProcess:
    """ Preprocesing utilities """

    def getInputSegs(self, sampleName, inDir, outDir, mulFact):
        # sampleName=   #'temp' #"CMU_20020319-1400"
        # inDir = #'/SpeakerID-IIT/nauman/TISD/code_0_1/code_0_1/scripts/otherScripts/'
        # outDir = #'/SpeakerID-IIT/nauman/TISD/code_0_1/code_0_1/scripts/otherScripts/outTemp/'
        mulFactor = mulFact  # "100" # depends on frame rate.

        # Since we don't need header in rttm file. The tempFile will have start(msec) end(msec) and name of speaker
        mycmd = (
            "less "
            + inDir
            + sampleName
            + ".rttm"
            + "| grep SPEAKER | awk -v ss="
            + mulFactor
            + ' \'{print int($4*ss) " " int(($4+$5)*ss) " " $8 }\'  | sort -nk1  > '
            + outDir
            + sampleName
            + ".tempFile"
        )
        # print (mycmd)
        cmd.getoutput(mycmd)

        fileName = outDir + sampleName + ".tempFile"
        listOfTuples = []
        with open(fileName, "r") as f:  # Now read ^ file
            for line in f.readlines():
                line = line.strip("\n")
                tmp = line.split(" ")
                myTuple = (int(tmp[0]), int(tmp[1]), str(tmp[2]))
                # myList  = [float(tmp[0]), float(tmp[1]), str(tmp[2]) ]
                listOfTuples.append(myTuple)
        # print listOfTuples
        f.close()
        sorted(
            listOfTuples, key=lambda x: x[0]
        )  # Not needed since BASH command has already sorted on start time. (Just making sure)

        return listOfTuples

        # allSegs= np.loadtxt(fileName)
        # allSegs.astype(int)  # contains all speech segments
        # return allSegs


class CreateSegs:
    """ Utilities """

    def chopSegs(self, ptr1, ptr2, Arr):  # Not Used
        print(ptr1, ptr2)
        A = Arr[ptr1 : ptr2 + 1, :]
        A = A.flatten()
        # print (A)
        A.sort()
        # print (A)

        segList = []
        # Start chopping
        for i in range(0, A.size):
            segStart = A[i]
            if i >= A.size - 1:
                break
            segEnd = A[i + 1] - 1  # Since we don't want overlaps
            if (
                segStart >= segEnd
            ):  # i.e. 1 frame segment or those with less start time than their endtime
                pass
            else:
                seg = [segStart, segEnd]
                segList.append(seg)
        print("chopped", segList)
        return segList

    def startProcessing(self, allInv):  # Not Used
        mergedSegs = []
        choppedSegs = []
        i = 0
        breakedFlag = 0
        while i < allInv.shape[0]:
            firstSegStart = allInv[i, 0]  # 1st seg of every contigous block
            BackPtr = i  # Points at the start of contiguos segments
            currMaxEnd = allInv[i, 1]  #
            i = i + 1
            if i >= allInv.shape[0]:  # i.e. if its pointing to out of array
                breakedFlag = 1
                lastSeg = [allInv[i - 1, 0], allInv[i - 1, 1]]
                break
            startTime = allInv[i, 0]  # 2nd seg's start time
            while startTime < currMaxEnd:  # Overlap condition
                endTime = allInv[i, 1]  # end time of current segment
                currMaxEnd = max(currMaxEnd, endTime)
                i = i + 1
                startTime = allInv[i, 0]
            FrontPtr = i - 1  # No need to update i here!!!

            mergedSegs.append([firstSegStart, currMaxEnd])

            choppedTempList = self.chopSegs(
                BackPtr, FrontPtr, allInv
            )  # send only contiguos/Overlapping segs for chopping
            choppedSegs.append(choppedTempList)

        if breakedFlag == 1:
            mergedSegs.append(lastSeg)

        print("\nMerged: ", mergedSegs)
        print("Chopped: ", choppedSegs)

    def chopper(self, inList):
        # I/P: This func takes sorted timings strt/end.
        # O/P: Ouputs disjoint segments (each output segment may have multiple speaker)

        currLabSet = set()  # Make a set as a container for labels
        # print (currLabSet)
        finalSegs = []
        for i in range(0, len(inList) - 1):
            a = inList[i]  # Current interval where we are looking for
            b = inList[i + 1]
            spkr = a[1]
            if spkr in currLabSet:  # that is a is ending # MAIN LOGIC
                currLabSet.remove(spkr)
                labels = list(currLabSet)
                labels.sort(
                    key=str
                )  # Sorting on speaker name (Just to keep sequence for audacity Lab). Not realted to logic of code.
            else:
                currLabSet.add(spkr)
                labels = list(currLabSet)
                labels.sort(key=str)

            myseg = (
                a[0],
                b[0] - 1,
                labels,
                len(labels),
            )  # (strt, end, {speakers set}, numOfSpeakers)
            # print myseg
            finalSegs.append(myseg)

        # Removing smaller (1 frame) segments and adding label for silence
        # SIL : Since we sorted all the times (strt and end) togther and hence it introduces SIL part as well.
        # Thus, there will be some segments without even a single speaker. So label them as SIL
        newList = []
        for i in range(0, len(finalSegs)):
            strt = finalSegs[i][0]  # Current interval where we are looking for
            endF = finalSegs[i][1]
            numSpkr = finalSegs[i][3]

            tmpSeg = finalSegs[i]
            mySeg = ()
            if (
                strt >= endF
            ):  # Not considering smaller clusters in final newList
                pass
            elif numSpkr <= 0:
                mySeg = (
                    tmpSeg[0],
                    tmpSeg[1],
                    ["SIL"],
                    tmpSeg[3],
                )  # Adding Silence label
                newList.append(mySeg)
            else:
                newList.append(tmpSeg)
        return newList

    def printInTextFormat(self, newList):
        # Printing in textual format. Space separated. With Multi-Speaker segments (without overlaps)
        myList = []
        print(
            "Final disjoint segments in Format:- strtTime endTime {Speaker(s)} NumOfSpkr(s)"
        )
        for i in range(len(newList)):
            stri = ""
            stri = str(newList[i][0]) + " " + str(newList[i][1]) + " " + "{"
            for k in newList[i][2]:
                stri += k + " "
            stri += "} "
            stri += str(newList[i][3])
            # print (stri)
            myList.append(stri)
        # return myList	 # No need to return (Its just for printing)

    def writeToSCP_nonOverlap(self, sampleName, outSCPPath, newList):
        # write to SCP file (non-overlapping segments). Only 1 speaker in a segment.
        # print 'OLD', newList
        newList = [
            i for i in newList if i[3] == 1
        ]  # Removes SIL and multi-speaker segments. Takes with exactly 1 speaker
        # print 'NEW', newList
        SCP_List = []
        for i in range(len(newList)):
            strt = newList[i][0]
            end = newList[i][1]
            # spkr = newList[i][2][0]
            # print speaker
            # line = "SPEAKER " + sampleName +" 1 " +  str (float(strt) ) + " " + str (float(end-strt)) + " <NA> <NA> " + str(spkr) +" <NA>"
            line = sampleName + ".fea[" + str(strt) + "," + str(end) + "]"
            SCP_List.append(line)

        thefile = open(outSCPPath, "w")
        for item in SCP_List:
            thefile.write("%s\n" % item)
        thefile.close()

    def writeToRttm_nonOverlap(
        self, sampleName, outRttmPath, newList, divFactor
    ):
        # write to RTTM file (non-overlapping segments). Only 1 speaker in a segment.
        # print 'OLD', newList
        newList = [
            i for i in newList if i[3] == 1
        ]  # Removes SIL and multi-speaker segments. Takes with exactly 1 speaker
        # print 'NEW', newList
        rttmList = []
        for i in range(len(newList)):
            strt = float(float(newList[i][0]) / float(divFactor))
            end = float(float(newList[i][1]) / float(divFactor))
            spkr = newList[i][2][0]
            # print speaker
            line = (
                "SPEAKER "
                + sampleName
                + " 1 "
                + str(float(strt))
                + " "
                + str(float(end - strt))
                + " <NA> <NA> "
                + str(spkr)
                + " <NA>"
            )
            rttmList.append(line)

        thefile = open(outRttmPath, "w")
        for item in rttmList:
            thefile.write("%s\n" % item)

    def writeToAudacity_ALL(
        self,
        sampleName,
        outAudacityPath,
        newList,
        divFactor,
        disjointSpeakersFlag,
    ):
        # write to RTTM file (non-overlapping segments). Only 1 speaker in a segment.
        # print 'OLD', newList
        if (
            disjointSpeakersFlag == 1
        ):  # If we want disjoint speakers then only update newList
            newList = [
                i for i in newList if i[3] == 1
            ]  # Removes SIL and multi-speaker segments. Takes with exactly 1 speaker
        # print 'NEW', newList
        audacityList = []
        # print (newList)
        for i in range(len(newList)):
            strt = float(float(newList[i][0]) / float(divFactor))
            end = float(float(newList[i][1]) / float(divFactor))
            sList = newList[i][2]  # SPEAKER LIST
            numSpkr = newList[i][3]
            myLAB = str(numSpkr) + "-" + "-".join(str(x) for x in sList)
            line = str(strt) + " " + str(end) + " " + myLAB
            audacityList.append(line)

        thefile = open(outAudacityPath, "w")
        for item in audacityList:
            thefile.write("%s\n" % item)

    def startChops(self, listOfTuples):
        lot = listOfTuples
        myList = []
        for i in range(0, len(lot)):
            strtTuple = (
                lot[i][0],
                lot[i][2],
                "strt",
            )  # strt and end NOT required.
            endTuple = (lot[i][1], lot[i][2], "end")
            myList.append(
                strtTuple
            )  # Mixing all times (str and end) and then sorting
            myList.append(endTuple)
            # print (myList)
        # The below sorting introduces silence segments also.
        myList = sorted(
            myList, key=lambda x: x[0]
        )  # sorting time (irrespective whether its start or end)
        # print ("sorted\n", myList)
        finalSegs = self.chopper(
            myList
        )  # sending this sorted list to make disjoint segments (output segments can have multiple speakers)
        return finalSegs


def main(argv):
    sampleName = argv[1]
    inDir = argv[2]
    outDir = argv[3]
    multFactor = argv[4]
    outOption = argv[5]  # rttm, scp, audacity

    print("SampleName= ", sampleName, " : outOption= ", outOption)

    # Preprocesing (inpInv will have segments as per rttm file). It have overlapping segments and speakers as well.
    inpInv = PreProcess().getInputSegs(
        sampleName, inDir, outDir, multFactor
    )  # It just reads from rttm
    # print (inpInv)

    # Cutting the segments. This is main part where the non-overlapping segments are made.
    # The outList contains the disjoint segments (non-overlapping segments). **Each segment can have multiple speakers**.
    c = CreateSegs()
    outList = c.startChops(inpInv)  # This outList will be used in all modules
    # c.printInTextFormat(outList)

    ## All below are non-overlapping segments
    divFactor = float(multFactor)
    # o/p: Non-overlapping SPEAKERS (and obviuosly non-overlapping segments)
    if outOption in ("rttm", "all"):  # if var in ('stringone', 'stringtwo')
        # Passing outList to write in RTTM format
        outRTTMpath = (
            outDir + sampleName + ".rttm"
        )  # './' + sampleName + '.rttm_nonOverlap'
        c.writeToRttm_nonOverlap(sampleName, outRTTMpath, outList, divFactor)

    # o/p: N#on-overlapping SPEAKERS (and obviuosly non-overlapping segments)

    if outOption in ("scp", "all"):
        # Passing outList to write in SCP format
        outSCP_path = (
            outDir + sampleName + ".scp"
        )  # './' + sampleName + '.scp_nonOverlap'
        c.writeToSCP_nonOverlap(sampleName, outSCP_path, outList)

    # o/p: Non-overlapping AND/OR Overlapping SPEAKERS (and obviuosly non-overlapping segments)
    if outOption in ("audacity", "all"):
        # Passing outList to write in Audacity label format
        disjointSpeakersFlag = 0  # 1: homogeneuos speaker segments i.e.label is 1 speaker, 0: multi-speaker i.e. lab is a set of speakers
        # if disjointSpeakersFlag == 0: # ALL SEGMENTS with MULTIPLE SPEAKERS
        outAudacity_Path = (
            outDir + sampleName + ".audacity_multi_spkrs.txt"
        )  # './' + sampleName + '.audacity_all.txt'
        c.writeToAudacity_ALL(
            sampleName,
            outAudacity_Path,
            outList,
            divFactor,
            disjointSpeakersFlag,
        )

        disjointSpeakersFlag = 1  # Lets write both audacity labels
        outAudacity_Path = (
            outDir + sampleName + ".audacity_homogeneuos_spkrs.txt"
        )  # './' + sampleName + '.audacity_all.txt'
        c.writeToAudacity_ALL(
            sampleName,
            outAudacity_Path,
            outList,
            divFactor,
            disjointSpeakersFlag,
        )
    else:
        print("Incorrect output Option (rttm, scp, audacity or all)")


if __name__ == "__main__":
    main(sys.argv)
