from mpi4py import MPI

def getRankBounds(nodeList):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        numRanks = comm.Get_size()

        dataSize = len(nodeList)

        # if (verbose):
        #     print("Node Count: " + str(dataSize))

        dataSizePerRank = dataSize / numRanks
        leftOver = dataSize % numRanks
        startIndex = dataSizePerRank * rank
        lastIndex = (dataSizePerRank * (rank + 1)) - 1
        if (rank < leftOver):
            dataSizePerRank += 1
            if (rank != 0):
                startIndex += 1
                lastIndex += 2
            else:
                lastIndex += 1
        else:
            startIndex += leftOver
            lastIndex += leftOver

        return startIndex, lastIndex

def filter_rtree(nodeList,_asn_data_global):
    start_index,last_index = getRankBounds(nodeList)
    return nodeList[int(start_index):int(last_index) + 1]