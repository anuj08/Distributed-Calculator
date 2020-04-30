import os
import sys
import socket
import math
import pickle
from multiprocessing import Pool
import traceback
import time
import numpy as np
import signal
import random
from timeit import default_timer as timer
from datetime import timedelta
from functools import reduce

# first host of main process
iport = 10000
sport = 60000
endian = 'little'
commsize = 1024
rankToSock = dict()
sockToRank = dict()
myrank = -1
buffer = dict()

def signal_handler(signal, frame):
    # print('You pressed Ctrl+C!')
    sys.exit(0)



def getCommsize():
    return len(rankToSock)

def distributeP(tp, cord, corp):
    mh = min(cord, key=cord.get)
    n = len(cord)
    mp = cord[mh]
    if mp >= math.ceil(tp/n):
        for key in cord.keys():
            corp[key] += math.ceil(tp/n)
        return 
    else:
        for key in cord.keys():
            corp[key] += mp
        del cord[mh]
        distributeP(tp - n*mp, cord, corp)


def setRanks(corp):
    nextRank = 0
    for val in corp.keys():
        for i in range(corp[val]):
            s = (val, sport + i)
            rankToSock[nextRank] = s
            nextRank += 1


def sendToAll(data_d, conns):
    ds = pickle.dumps(data_d)
    for conn in conns:
        conn.sendall(ds)


def MPI_Send(data, dest, tag):
    # print("Sending data {} from {} to {} tag {}".format(data, myrank, dest, tag))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(rankToSock[myrank])
    while True:
        try:
            sock.connect(rankToSock[dest])
        except:
            # traceback.print_exc()
            # print(myrank)
            time.sleep(0.5)
            continue
        break

    try:
        dat = [data, tag]
        message = pickle.dumps(dat)
        t = len(message)
        t = int(t)
        t = t.to_bytes(commsize, endian)
        sock.sendall(t)
        sock.sendall(message)
        s = sock.recv(1024).decode('utf-8')
        if(s != 'ok'):
            sock.close()
            time.sleep(0.5)
            MPI_Send(data, dest, tag)
        else:
            sock.close()

    except Exception as e:
        # traceback.print_exc()
        # print(myrank)
        time.sleep(0.5)
        sock.close()
        MPI_Send(data, dest, tag)
        
    # if(mesg != None):
    #     print(mesg)

def MPI_Recv(source, tag):
    if (source, tag) in buffer:
        # print((source, tag))
        data = buffer[(source,tag)]
        # print("Deleting {}:{}".format((source, tag), buffer[(source,tag)]))
        del buffer[(source,tag)]
        return data

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(rankToSock[myrank])
    # print("Bind address: ", rankToSock[myrank])
    sock.listen(100)
    # print(client)
    # count = 14
    # print("Bind address: ", rankToSock[myrank])
    while (source, tag) not in buffer:
        # print("accept b")
        # print(buffer)
        conn, client = sock.accept()
        # print("accept c")
        with conn:
            chunks = []
            bytes_recd = 0
            while bytes_recd < commsize:
                chunk = conn.recv(min(commsize - bytes_recd, commsize))
                if chunk == b'':
                    # raise RuntimeError("socket connection broken")
                    break
                chunks.append(chunk)
                bytes_recd = bytes_recd + len(chunk)
            s = b''.join(chunks)
            count = int.from_bytes(s, endian)
            # print(count)
            chunks = []
            bytes_recd = 0
            while bytes_recd < count:
                chunk = conn.recv(min(count - bytes_recd, commsize))
                if chunk == b'':
                    # raise RuntimeError("socket connection broken")
                    break
                chunks.append(chunk)
                bytes_recd = bytes_recd + len(chunk)
            s = b''.join(chunks)
            # print(s)
            d = pickle.loads(s)
            # print(d)
            da = d[0]
            ta = d[1]
            buffer[(sockToRank[client], ta)] = da
            conn.sendall('ok'.encode('utf-8'))
        # print(buffer[(sockToRank[client], tag)])
    sock.close()
    data = buffer[(source,tag)]
    # print("Recived data {} from rank {} for source {}".format(t, sockToRank[cly], source))
    # print("Deleting {}:{}".format((source, tag), buffer[(source,tag)]))
    del buffer[(source,tag)]
    return data
        
def MPI_Bcast(data, root, tag):
    # print("in bcast for rank {}".format(myrank))
    if myrank == root:
        for i in rankToSock.keys():
            if i != root:
                # print("Send posted")
                MPI_Send(data, i, tag)
        return data
    else:
        # print("Recv posted")
        dat = MPI_Recv(root, tag)
        return dat
        
def MPI_Scatterv(data, offsets, root):
  #   print("Scatterv called for {}".format(myrank))
  if myrank == root:
    for i in rankToSock.keys():
      if i != root:
        # print("Send posted for rank {} by {}".format(i, myrank))
        MPI_Send(data[offsets[i][0] : offsets[i][1] + 1], i, "scatter")
    return data[offsets[root][0] : offsets[root][1] + 1]
  else:
    # print("Recv posting for rank {}".format(myrank))
    dat = MPI_Recv(root, "scatter")
    return dat

def MPI_Gatherv(data, offsets, root):
  if myrank != root:
    MPI_Send(data, root, "gather")
    return data
  else:
    #  temp gathered data
    gath_data = []
    for i in rankToSock.keys():
      if i != root:
        gath_data.append([offsets[i][0], MPI_Recv(i, "gather")])
      else:
        gath_data.append([offsets[i][0], data])
    gath_data.sort(key=lambda x: x[0])
    dat = []
    for i in gath_data:
      dat.append(i[1])

    
    return dat

def MPI_Reduce(data, root, func, tag):
    if myrank == root:
        l = [data]
        for i in range(len(rankToSock)):
            if i != myrank:
                l.append(MPI_Recv(i, 'MPI_Reduce'+tag))
        try:
            res = reduce(func, l) 
        except Exception as e:
            traceback.print_exc()
        return res
    else:
        MPI_Send(data, root, 'MPI_Reduce'+tag)

def decor(l):
    global myrank
    myrank = l[1]
    f = l[0]
    f()

def spawnProcesses(myip, f):
    global rankToSock
    npp = 0
    rl = []
    for key,val in rankToSock.items():
        if(val[0] == myip):
            rl.append(key)
            npp += 1

    with Pool(processes=npp) as pool:
        pool.map(decor, zip([f for i in range(len(rl))], rl))

def init(f):
    signal.signal(signal.SIGINT, signal_handler)
    global rankToSock
    global sockToRank
    servIp = sys.argv[1]
    tot_hosts = sys.argv[2]
    tp = sys.argv[3]
    ip = sys.argv[4]
    tp = int(tp)
    tot_hosts = int(tot_hosts)

    cores = os.cpu_count() - 1
    server_address = (servIp, iport)
    
    # print(ip)
    # print(servIp)
    if servIp == ip:
        corp = dict()
        cord = dict()
        conns = []
        cord[ip] = cores
        corp[ip] = 0

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(server_address)
        sock.listen(1)
        total_conn = 0
        while(True):
            if(total_conn == tot_hosts - 1):
                break
            conn, client = sock.accept()
            conns.append(conn)
            s = conn.recv(commsize)
            s = int.from_bytes(s, endian)
            total_conn += 1
            cord[client[0]] = s
            # print(cord[client[0]])
            corp[client[0]] = 0

            # print(total_conn)

        tc = sum(cord.values())
        # print(cord)
        if(tc < tp):
            print("LESS CORES")
            #not possible
            pass
        else:
            distributeP(tp, cord, corp)
            # print(corp)
            setRanks(corp)
            print(rankToSock)
            sendToAll(rankToSock,conns)
            sockToRank = dict((v, k) for k, v in rankToSock.items())
        sock.close()


    else:
        # print("On remote")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(server_address)
        try:
            message = cores.to_bytes(commsize, endian)
            sock.sendall(message)

            r2s = sock.recv(commsize)
            rankToSock = pickle.loads(r2s)
            sockToRank = dict((v, k) for k, v in rankToSock.items())
            # print(rankToSock)

        finally:
            sock.close()

    spawnProcesses(ip, f)

def mul(a=None, b=None):
    size = getCommsize()
    rank = myrank

    offsets = []
    aa = np.zeros(shape=(1,2))
    bb = np.zeros(shape=(2,1))
    if(rank == 0):
        # a = np.zeros(shape=(N,N))
        # b = np.zeros(shape=(N,N))
        a = pickle.load(open('mult_a', 'rb'))
        b = pickle.load(open('mult_b', 'rb'))
        N = a.shape[0]
        # a[0][0] = 1
        # a[0][1] = 2
        # a[1][0] = 3
        # a[1][1] = 4
        # b[0][0] = 2
        # b[0][1] = 3
        # b[1][0] = 4
        # b[1][1] = 1
        # print(a)
        # for i in range(N):
        #     for j in range(N):
        #         a[i,j] = i+j
        #         b[i,j] = i-j
        #print("init done")

    tagc = 0
    initOff = 0

    c = 0
    r = 0
    if(rank == 0):
        r, c = a.shape
        bb = b   
        a = a.flatten()

    r = MPI_Bcast(r, 0, tagc)
    tagc += 1
    c = MPI_Bcast(c, 0, tagc)
    tagc += 1

    base = r//size
    extra = r%size


    for i in range(size):
        if(extra > 0):
            tmp = base + 1
        else:
            tmp = base
        extra -= 1
        offsets.append([initOff, initOff+tmp*c - 1])
        initOff += tmp*c

    
    print(offsets)
    bb = MPI_Bcast(bb, 0, tagc)
    tagc += 1
    aa = MPI_Scatterv(a, offsets, 0)
    # print("Print aa for rank {} is {}".format(myrank, aa))
    cc = np.zeros((len(aa)//c)*bb.shape[1])
    sum = 0
    try:
        for k in range(len(aa)//c):
            for i in range(bb.shape[1]):
                for j in range(c):
                    sum += aa[j + k*c]*bb[j][i]
                
                cc[i + k*bb.shape[1]] = sum
                sum = 0
    except Exception as e:
        traceback.print_exc()
        print(myrank)

    C = np.zeros(r*bb.shape[1])

    base = r//size
    extra = r%size
    initOff = 0
    for i in range(size):
        if(extra > 0):
            tmp = base + 1
        else:
            tmp = base
        extra -= 1
        offsets.append([initOff, initOff+tmp*bb.shape[1] - 1])
        initOff += tmp*bb.shape[1]

    #print("cc val for rank {} is {}".format(myrank, cc))

    C = MPI_Gatherv(cc, offsets, 0)

    if(rank == 0):
        CC = []
        for ls in C:
            for i in ls:
                CC.append(i)
        C = np.reshape(CC, (r, bb.shape[1]))
        print("ans: ", C)
        pickle.dump(C, open('mult_ab', 'wb'))


def gauss():
    size = getCommsize()
    rank = myrank
    c = 0
    r = 0
    tagc = 0
    offsets = []
    initOff=0
    aa = 0
    A = 0
    if(rank == 0):
        # random.seed(0)
        # a = np.zeros(shape=(N,N))
        # b = np.zeros(shape=(N,1))
        a = pickle.load(open('gauss_a', 'rb'))
        b = pickle.load(open('gauss_b', 'rb'))
        N = a.shape[0]
        # a = np.array([[1,1], [3, -2]],dtype='float64')
        # b = np.array([3, 4], dtype='float64')


        # for i in range(N):
        #     b[i,0] = random.randint(10000, 10000000)
        #     for j in range(N):
        #         a[i,j] = random.randint(10000,1000000)

        A = np.column_stack((a,b))

        r, c = A.shape
        A = A.flatten()

    c = MPI_Bcast(c, 0, tagc) 
    tagc += 1
    r = MPI_Bcast(r, 0, tagc)
    tagc += 1


    base = r//size
    extra = r%size


    for i in range(size):
        if(extra > 0):
            tmp = base + 1
        else:
            tmp = base
        extra -= 1
        offsets.append([initOff, initOff+tmp*c - 1])
        initOff += tmp*c
    # print(offsets)
    try:
        aa = MPI_Scatterv(A, offsets, 0)
    except Exception as e:
        traceback.print_exc()

    
    # if(len(aa) > c):
    aa = np.reshape(aa, (len(aa)//c, c))

    start = timer()
    tagc = 5
    # print("aa for rank {} is {}".format(myrank, aa))
    maxpiv = []
    col = -1
    eflag = False
    sol = ''
    sf = 2
    try:
        for idx, key in enumerate(offsets):
            # print("IDX IS {} for rank {}".format(idx, myrank))
            if(eflag):
                break
            for i1, _ in enumerate(range(key[0]//c, (key[1]+1)//c)):
                col += 1
                if col == r-1:
                    if(aa[i1,col] == 0):
                        if(aa[i1, c-1] != 0):
                            sf *= 5
                        else:
                            sf *= 3
                    eflag = True
                    break
                if idx == myrank:
                    #pivot swapping requiredh
                    # print(i1, col)
                    if aa[i1,col] == 0:
                        # print("AAAAAAAAAAAAAAAAAAAAAAAAAA")
                        z = False
                        for k in range(i1+1, aa.shape[0]):
                            if aa[k][col] != 0:
                                z = True
                                #swap arrays
                                aa[i1,:], aa[k,:] = aa[k,:].copy(), aa[i1,:].copy()
                                break
                        #No non zero in own chunk
                        if not z:
                            for s in range(myrank+1, size):
                                MPI_Send('help', s, tagc)
                            for s in range(myrank+1, size):
                                maxpiv.append([MPI_Recv(s, 'poll'), s])
                            # tagc += 1
                            # print("MAXpiv is ", maxpiv)
                            maxn = 0
                            mrow = -1
                            for it in maxpiv:
                                if(it[0] > maxn):
                                    maxn = it[0]
                                    mrow = it[1]
                            if(maxn == 0):
                                if(aa[i1, c-1] == 0):
                                    sol = 'Infinte'
                                    sf *= 3
                                    for s in range(myrank+1, size):
                                        MPI_Send('skip', s, 'pollres')
                                    continue
                                else:
                                    sol = 'No solution'
                                    sf *= 5
                                    eflag = True
                                    for s in range(myrank+1, size):
                                        MPI_Send('break', s, 'pollres')
                                    break
                            # recive row from mrow
                            for s in range(myrank+1, size):
                                if s != mrow:
                                    MPI_Send('not_needed', s, 'pollres')

                            MPI_Send(aa[i1], mrow, 'pollres')
                            ra = []
                            ra = MPI_Recv(mrow, 'swap')
                            aa[i1,:] = ra
                    for s in range(myrank+1, size):
                        MPI_Send(aa[i1,:], s, tagc)

                    #operate on row
                    aa[i1,:] = aa[i1,:] / aa[i1,col]
                    for iy in range(i1+1, aa.shape[0]):
                        aa[iy,:] -= (aa[iy,col]/aa[i1,col])*aa[i1,:]

                #not my rank
                elif idx < myrank:
                    stat = ''
                    # print("Reciveing")
                    stat = MPI_Recv(idx, tagc)
                    # print(stat, idx, tagc)
                    div = []
                    #send values
                    if(stat == 'help'):
                        # print("BBBBBBBBBBBBBBBBBBBB, ",col)
                        maxn = -1
                        mrow = -1
                        for ix, item in enumerate(aa[:, col]):
                            if item > maxn:
                                maxn = item
                                mrow = ix
                        MPI_Send(maxn, idx, 'poll')
                        stat2 = ''
                        stat2 = MPI_Recv(idx, 'pollres')
                        # print("stat2 for rank {} is {}".format(myrank, stat2))
                        #not selected
                        if(stat2 == 'not_needed'):
                            # tagc += 1
                            print("IN wrong")
                            div = MPI_Recv(idx, tagc)

                            for iy in range(aa.shape[0]):
                                aa[iy,:] -= (aa[iy,col]/div[col])*div

                        elif(stat2 == 'skip'):
                            continue

                        elif(stat2 == 'break'):
                            eflag = True
                            break


                        #Selected
                        else:
                            tmp = stat2
                            #print("recv posted")
                            # tmp = MPI_Recv(idx, 'swap')
                            MPI_Send(aa[mrow], idx, 'swap')
                            aa[mrow, :] = tmp
                            #print("IN wrongselce")

                            div = MPI_Recv(idx, tagc)

                        
                    else:
                        div = stat
                        # print(div)
                        # print(aa[0])
                        for iy in range(aa.shape[0]):
                            aa[iy,:] -= (aa[iy,col]/div[col])*div
    except Exception as e:
        traceback.print_exc()

    try:
        sfx = 0
        A = MPI_Gatherv(aa, offsets, 0)
        sfx = MPI_Reduce(sf, 0, (lambda x, y: x*y), 'sfx')
    except Exception as e:
        traceback.print_exc()

    if(rank == 0):
        if(sfx % 5 == 0):
            print("No solutions")
            sol = "No solutions"
            pickle.dump(sol, open('gauss_x', 'wb'))
        elif(sfx % 3 == 0):
            sol = "Infinite Solutions"
            print("Infinite Solutions") 
            pickle.dump(sol, open('gauss_x', 'wb'))
        else:
        #back subs
            # print(A)
            ass = np.zeros(shape=(1,N+1))
            try:
                for item in A:
                    ass = np.vstack((ass,item))

                ass = np.delete(ass, 0, 0)
                A = np.reshape(ass, (r,c))
                sol = np.zeros(shape=(r,1))
                for i in reversed(range(r)):
                    sol[i] = A[i][c-1]
                    for j in range(i+1, r):
                        sol[i] -= A[i][j]*sol[j]
                    
                    sol[i] =sol[i]/A[i][i]
                # print(A)
                print(sol)
                pickle.dump(sol, open('gauss_x', 'wb'))
                # pickle.dump(sol, open('sol.dat', 'wb'))
                
                end = timer()
                # print("time taken: ", timedelta(seconds=end-start))
            except Exception as e:
                traceback.print_exc()



def det():
    size = getCommsize()
    rank = myrank
    c = 0
    r = 0
    tagc = 0
    offsets = []
    initOff=0
    aa = 0
    A = 0
    if(rank == 0):
        # a = np.zeros(shape=(2,2))
        # for i in range(3):
        #     for j in range(3):
        #         a[i,j] = i*3 + j
        # a[0][0] = 
        # b = np.zeros(shape=(3,1))
        # a = np.array([[1,1,1], [1,1,3], [1,2,4]],dtype='float64')
        # b = np.array([1,1,1], dtype='float64')
        # random.seed(0)
        # N = 22
        # a = np.zeros(shape=(N,N))
        # a = np.array([[1,1,1], [1,1,1], [1,1,1]],dtype='float64')
        # b = np.array([1,1,2], dtype='float64')

        # for i in range(N):
        #     for j in range(N):
        #         a[i,j] = random.randint(1,5)

        # print("SOl is {}".format(np.linalg.det(a)))
        a = pickle.load(open('det_a', 'rb'))
        N = a.shape[0]
        A = a
        try:
            r, c = A.shape
        except Exception as e:
            traceback.print_exc()

        A = A.flatten()

    c = MPI_Bcast(c, 0, tagc) 
    tagc += 1
    r = MPI_Bcast(r, 0, tagc)
    tagc += 1


    base = r//size
    extra = r%size


    for i in range(size):
        if(extra > 0):
            tmp = base + 1
        else:
            tmp = base
        extra -= 1
        offsets.append([initOff, initOff+tmp*c - 1])
        initOff += tmp*c
    # print(offsets)
    try:
        aa = MPI_Scatterv(A, offsets, 0)
    except Exception as e:
        traceback.print_exc()

    
    # if(len(aa) > c):
    aa = np.reshape(aa, (len(aa)//c, c))

    tagc = 5
    # print("aa for rank {} is {}".format(myrank, aa))
    maxpiv = []
    col = -1
    divs = 1
    swaps = 1
    eflag = False
    try:
        for idx, key in enumerate(offsets):
            # print("IDX IS {} for rank {}".format(idx, myrank))
            if(eflag):
                break
            for i1, _ in enumerate(range(key[0]//c, (key[1]+1)//c)):
                # print(aa)
                col += 1
                if col == r - 1:
                  if(idx == myrank):
                    divs *= aa[i1][col]
                  eflag = True
                  break
                if idx == myrank:
                    #pivot swapping requiredh
                    # print(i1, col)
                    if aa[i1,col] == 0:
                        swaps *= -1
                        # print("AAAAAAAAAAAAAAAAAAAAAAAAAA")
                        z = False
                        for k in range(i1+1, aa.shape[0]):
                            if aa[k][col] != 0:
                                z = True
                                #swap arrays
                                aa[i1,:], aa[k,:] = aa[k,:].copy(), aa[i1,:].copy()
                                break
                        #No non zero in own chunk
                        if not z:
                            for s in range(myrank+1, size):
                                MPI_Send('help', s, tagc)
                            for s in range(myrank+1, size):
                                maxpiv.append([MPI_Recv(s, 'poll'), s])
                            # tagc += 1
                            # print("MAXpiv is ", maxpiv)
                            maxn = 0
                            mrow = -1
                            for it in maxpiv:
                                if(it[0] > maxn):
                                    maxn = it[0]
                                    mrow = it[1]
                            if(maxn == 0):
                                # print("Infinite")
                                for s in range(myrank+1, size):
                                    MPI_Send('skip', s, 'pollres')
                                    divs = 0
                                    swaps *= -1
                                # continue
                                eflag = True
                                break
                            # recive row from mrow
                            for s in range(myrank+1, size):
                                if s != mrow:
                                    MPI_Send('not_needed', s, 'pollres')

                            MPI_Send(aa[i1], mrow, 'pollres')
                            ra = []
                            ra = MPI_Recv(mrow, 'swap')
                            aa[i1,:] = ra
                    for s in range(myrank+1, size):
                        MPI_Send(aa[i1,:], s, tagc)

                    #operate on row
                    divs *= aa[i1,col]
                    aa[i1,:] = aa[i1,:] / aa[i1,col]
                    # print("DIVS IS ", divs)
                    for iy in range(i1+1, aa.shape[0]):
                        aa[iy,:] -= (aa[iy,col]*aa[i1,:])/aa[i1,col]

                #not my rank
                elif idx < myrank:
                    stat = ''
                    # print("Reciveing")
                    stat = MPI_Recv(idx, tagc)
                    # print(stat, idx, tagc)
                    div = []
                    #send values
                    if(stat == 'help'):
                        # print("BBBBBBBBBBBBBBBBBBBB, ",col)
                        maxn = -1
                        mrow = -1
                        for ix, item in enumerate(aa[:, col]):
                            if item > maxn:
                                maxn = item
                                mrow = ix
                        MPI_Send(maxn, idx, 'poll')
                        stat2 = ''
                        stat2 = MPI_Recv(idx, 'pollres')
                        # print("stat2 for rank {} is {}".format(myrank, stat2))
                        #not selected
                        if(stat2 == 'not_needed'):
                            # tagc += 1
                            # print("IN wrong")
                            div = MPI_Recv(idx, tagc)
                            for iy in range(aa.shape[0]):
                                aa[iy,:] -= (aa[iy,col]/div[col])*div
                        elif(stat2 == 'skip'):
                            eflag = True
                            break

                        #Selected
                        else:
                            tmp = stat2
                            print("recv posted")
                            # tmp = MPI_Recv(idx, 'swap')
                            MPI_Send(aa[mrow], idx, 'swap')
                            aa[mrow, :] = tmp
                            # print("IN wrongselce")

                            div = MPI_Recv(idx, tagc)

                        
                    else:
                        div = stat
                        # print(div)
                        # print(aa[0])
                        for iy in range(aa.shape[0]):
                            aa[iy,:] -= (aa[iy,col]*div)/div[col]
    except Exception as e:
        traceback.print_exc()

    try:
        swaptot = 0
        divtot = 0
        swaptot = MPI_Reduce(swaps, 0, (lambda x, y: x*y), 'one')
        divtot = MPI_Reduce(divs, 0, (lambda x, y: x*y), 'two')
        
    except Exception as e:
        traceback.print_exc()


    if(rank == 0):
        #back subs
        ans = swaptot * divtot
        pickle.dump(ans, open('det_ans', 'wb'))
        # print(ans)

def inverse():
    size = getCommsize()
    rank = myrank
    c = 0
    r = 0
    tagc = 0
    offsets = []
    initOff=0
    aa = 0
    A = 0
    if(rank == 0):
        random.seed(0)
        # a = np.zeros(shape=(N,N))
        a = pickle.load(open('inverse_a', 'rb'))
        N = a.shape[0]
        # b = np.zeros(shape=(N,1))
        # a = np.array([[0,1,1], [2,3,1], [1,1,1]], dtype='float64')
        b = np.eye(N, dtype='float64')
        # b = np.array([1,1,1], dtype='float64')


        # for i in range(N):
        #     for j in range(N):
        #         a[i,j] = random.randint(10,100)

        # print(np.linalg.inv(a))
        A = np.column_stack((a,b))
        r, c = A.shape
        A = A.flatten()

    c = MPI_Bcast(c, 0, tagc) 
    tagc += 1
    r = MPI_Bcast(r, 0, tagc)
    tagc += 1


    base = r//size
    extra = r%size


    for i in range(size):
        if(extra > 0):
            tmp = base + 1
        else:
            tmp = base
        extra -= 1
        offsets.append([initOff, initOff+tmp*c - 1])
        initOff += tmp*c
    # print(offsets)
    try:
        aa = MPI_Scatterv(A, offsets, 0)
    except Exception as e:
        traceback.print_exc()

    
    # if(len(aa) > c):
    aa = np.reshape(aa, (len(aa)//c, c))

    start = timer()
    tagc = 5
    # print("aa for rank {} is {}".format(myrank, aa))
    maxpiv = []
    col = -1
    eflag = False
    poss = 1
    try:
        for idx, key in enumerate(offsets):
            # print("IDX IS {} for rank {}".format(idx, myrank))
            if(eflag):
                break
            for i1, _ in enumerate(range(key[0]//c, (key[1]+1)//c)):
                col += 1
                if col == r-1:
                    eflag = True
                    break
                if idx == myrank:
                    #pivot swapping requiredh
                    # print(i1, col)
                    if aa[i1,col] == 0:
                        # print("AAAAAAAAAAAAAAAAAAAAAAAAAA")
                        z = False
                        for k in range(i1+1, aa.shape[0]):
                            if aa[k][col] != 0:
                                z = True
                                #swap arrays
                                aa[i1,:], aa[k,:] = aa[k,:].copy(), aa[i1,:].copy()
                                break
                        #No non zero in own chunk
                        if not z:
                            for s in range(myrank+1, size):
                                MPI_Send('help', s, tagc)
                            for s in range(myrank+1, size):
                                maxpiv.append([MPI_Recv(s, 'poll'), s])
                            # tagc += 1
                            # print("MAXpiv is ", maxpiv)
                            maxn = 0
                            mrow = -1
                            for it in maxpiv:
                                if(it[0] > maxn):
                                    maxn = it[0]
                                    mrow = it[1]
                            if(maxn == 0):
                                for s in range(myrank+1, size):
                                    MPI_Send('skip', s, 'pollres')
                                    poss = 2
                                eflag = True
                                break
                            # recive row from mrow
                            for s in range(myrank+1, size):
                                if s != mrow:
                                    MPI_Send('not_needed', s, 'pollres')

                            MPI_Send(aa[i1], mrow, 'pollres')
                            ra = []
                            ra = MPI_Recv(mrow, 'swap')
                            aa[i1,:] = ra
                    for s in range(myrank+1, size):
                        MPI_Send(aa[i1,:], s, tagc)

                    #operate on row
                    aa[i1,:] = aa[i1,:] / aa[i1,col]
                    for iy in range(i1+1, aa.shape[0]):
                        aa[iy,:] -= (aa[iy,col]/aa[i1,col])*aa[i1,:]

                #not my rank
                elif idx < myrank:
                    stat = ''
                    # print("Reciveing")
                    stat = MPI_Recv(idx, tagc)
                    # print(stat, idx, tagc)
                    div = []
                    #send values
                    if(stat == 'help'):
                        # print("BBBBBBBBBBBBBBBBBBBB, ",col)
                        maxn = -1
                        mrow = -1
                        for ix, item in enumerate(aa[:, col]):
                            if item > maxn:
                                maxn = item
                                mrow = ix
                        MPI_Send(maxn, idx, 'poll')
                        stat2 = ''
                        stat2 = MPI_Recv(idx, 'pollres')
                        # print("stat2 for rank {} is {}".format(myrank, stat2))
                        #not selected
                        if(stat2 == 'not_needed'):
                            # tagc += 1
                            #print("IN wrong")
                            div = MPI_Recv(idx, tagc)
                            for iy in range(aa.shape[0]):
                                aa[iy,:] -= (aa[iy,col]/div[col])*div

                        elif(stat2 == 'skip'):
                            eflag = True
                            break

                        #Selected
                        else:
                            tmp = stat2
                            print("recv posted")
                            # tmp = MPI_Recv(idx, 'swap')
                            MPI_Send(aa[mrow], idx, 'swap')
                            aa[mrow, :] = tmp
                            #print("IN wrongselce")

                            div = MPI_Recv(idx, tagc)

                        
                    else:
                        div = stat
                        # print(div)
                        # print(aa[0])
                        for iy in range(aa.shape[0]):
                            aa[iy,:] -= (aa[iy,col]/div[col])*div
    except Exception as e:
        traceback.print_exc()
    #print("FORWARD DONE {}".format(aa)) 
    ps = 0
    ps = MPI_Reduce(poss, 0, (lambda x, y: x*y), 'one')
    ps = MPI_Bcast(ps, 0, 'ip')

    if(ps%2  == 0):
        if(myrank == 0):
            print("Inverse does not exist")
            sol = "Inverse does not exist"
            pickle.dump(sol, open('inverse_ans', 'wb'))

    else:
        maxpiv = []
        col = r

        eflag = False
        try:
            for idx, key in reversed(list(enumerate(offsets))):
                if(eflag):
                    break
                for i1, _ in reversed(list(enumerate(range(key[0]//c, (key[1]+1)//c)))):
                    col -= 1
                    # print("col IS {} for rank {} and i1 {}".format(col, myrank, i1))
                    if col == 0:
                        if(aa[i1,col] != 0):
                            aa[i1,:] = aa[i1,:] / aa[i1,col]
                            eflag = True
                            break
                    if idx == myrank:
                        # if(aa[i1,col] == 0):
                        #     print("Not invertible")
                        #     for s in range(myrank-1, -1,-1):
                        #         MPI_Send('noti', s, tagc)
                        #     eflag = True
                        #     break

                        for s in range(myrank-1, -1, -1):
                            # print("sending to {}".format(s))
                            MPI_Send(aa[i1,:], s, tagc)

                        #operate on row
                        aa[i1,:] = aa[i1,:] / aa[i1,col]
                        for iy in range(i1-1, -1,-1):
                            aa[iy,:] -= (aa[iy,col]/aa[i1,col])*aa[i1,:]
                        print(aa)

                    #not my rank
                    elif idx > myrank:
                        stat = ''
                        stat = MPI_Recv(idx, tagc)
                        # print("rank {} recieved {}".format(myrank, stat))
                        # print(stat, idx, tagc)
                        div = []
                        #send values
                        # if(stat == 'noti'):
                        #     eflag = True
                        #     print("BBBBBBBBBBBBBBBBBBBB, ",col)
                        #     break
                        div = stat
                        for iy in range(aa.shape[0]):
                            aa[iy,:] -= (aa[iy,col]/div[col])*div
        except Exception as e:
            traceback.print_exc()
        # print("After loop for rank {}".format(myrank))

        try:
            A = MPI_Gatherv(aa, offsets, 0)
        except Exception as e:
            traceback.print_exc()

        if(rank == 0):
            #back subs
            # print(A)
            ass = np.zeros(shape=(1,N+N))
            try:
                for item in A:
                    ass = np.vstack((ass,item))

                ass = np.delete(ass, 0, 0)
                A = np.reshape(ass, (r,c))
                sol = A[:, N:]
                pickle.dump(sol, open('inverse_ans', 'wb'))
                # print("inverse matrix:  ", sol)
                
                #end = timer()
                # print("time taken: ", timedelta(seconds=end-start))
            except Exception as e:
                traceback.print_exc()


def checkDiagonal(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if i == j:
                continue
            else:
                if abs(arr[i][j]) > 0.001:
                    return False
    return True

def printLambda(arr):
    count = 1
    ei_vals = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if(i == j):
                temp = arr[i][j]
                if(abs(temp) < 0.000000000001):
                    temp = 0
                print("Lamda"+str(count) +": " + str(temp))
                ei_vals.append(temp)
                count += 1

       
    with open("ei_values", 'rb') as f:
      pickle.dump(ei_vals, f)
 
def qrFactorization(arr):
    temp = arr
    i = 0
    prev_q = arr

    while(True):
        Q, R = qr(temp)
        # print('QR: ', Q, R)
        if i == 0:
          prev_q = Q
        else:
          prev_q = np.dot(prev_q, Q)

        temp = np.dot(R, Q)
        print("temp: ", temp)
        if(checkDiagonal(temp)):
            print("Number of Factorizations: " + str(i+1))
            break
        else:
            i += 1
    
    with open('ei_vectors', 'rb') as f:
      pickle.dump(prev_q, f)

    print("eigen vectors: ", prev_q)
    return temp

def ei_val_nd_vec():
    # original matrix to transpose
      # a = np.array([[0, 1], [-2, -3]])
    # a = np.array([[1, 3, 0], [3, 2, 1], [0, 1, 3]])
    with open('ei_a', 'rb') as f:
      a = pickle.load(f)

    print("read a: ", a)
    printLambda(qrFactorization(a))

# @params : eis = array of all eis till k
# @return eis till k + 1 
def single_columnn_qr_decomp(ei, ak_plus_1, start = False):
  # compute ei for the first column of A
  if start:
    ei = np.transpose([ak_plus_1/np.linalg.norm(ak_plus_1)])
    return ei
  else:
    uk_plus_1 = ak_plus_1
    for i in range(ei.shape[1]):
      uk_plus_1 = np.subtract(uk_plus_1, np.dot(ak_plus_1, ei[:, i])*ei[:, i])
    ei = np.hstack((ei, np.transpose([uk_plus_1/np.linalg.norm(uk_plus_1)])))
    return ei


def qr(a = None, dump = True):
  if myrank == 0:

    # original matrix to transpose
    # a = np.array([[1, 1, 0, 1, 2], [1, 0, 1, 1, 1], [0, 1, 1, -4, 6]])
    if a is None:
      with open('qr_decompose_a', 'rb') as f:
        a = pickle.load(f)
    else:
      dump = False
    # print("a in qr: ", a)

    num_of_rows = a.shape[0]
    num_of_cols = a.shape[1]
    # a = np.zeros(shape=(num_of_rows, num_of_cols))
    # populate_matrix(a)
    # ei will eventually become Q
    ei = np.array(single_columnn_qr_decomp(None, a[:, 0], True))

    if getCommsize() == 1:
      for i in range(1, num_of_cols):
        ei = single_columnn_qr_decomp(ei, a[:, i])
    else:
      #  create slices
      num_of_slices = int(num_of_cols / getCommsize())
      last_slice = num_of_cols % getCommsize()


      # sent number of slices to other rank
      for i in range(1, getCommsize()):
        MPI_Send(num_of_slices, i, tag="slices")

      # print("num of slices: ", num_of_slices)
      if num_of_slices <= 1:
        for i in range(1, num_of_cols):
          ei = single_columnn_qr_decomp(ei, a[:, i])
      else:
        #  if equally divisible into slices/chunks
        if last_slice == 0:
          for i in range(getCommsize()):
            if i == 0:
              # compute eis
              for j in range(1, num_of_slices):
                ei = single_columnn_qr_decomp(ei, a[:, j])
              # print("after rank 0: ", ei)
            else:
              #  send eis
              #  send ai first
              MPI_Send(a[:, (i)*num_of_slices:(i)*num_of_slices + num_of_slices], i, tag="ai")
              # MPI_Send(ei[:, (i - 1)*num_of_slices:(i - 1)*num_of_slices + num_of_slices], i, tag="ei")
              MPI_Send(ei[:, :], i, tag="ei")
              ei = MPI_Recv(i, tag="ei")
 
        else:
          for i in range(getCommsize()):
            if i == 0:
              # compute eis
              for j in range(1, num_of_slices):
                ei = single_columnn_qr_decomp(ei, a[:, j])

            elif i == getCommsize() - 1:
              MPI_Send(a[:, (i) * num_of_slices: (i) * num_of_slices + num_of_slices + last_slice], i, tag="ai")
              # MPI_Send(ei[:, (i - 1) * num_of_slices : (i - 1) * num_of_slices + num_of_slices], i, tag="ei")
              MPI_Send(ei[:, :], i, tag="ei")
              ei = MPI_Recv(i, tag = "ei")
            else:
              #  send eis
              #  send ai first
              MPI_Send(a[:, (i)*num_of_slices:(i)*num_of_slices + num_of_slices], i, tag="ai")
              # MPI_Send(ei[:, (i - 1)*num_of_slices:(i - 1)*num_of_slices + num_of_slices], i, tag="ei")
              MPI_Send(ei[:, :], i, tag="ei")
              ei = MPI_Recv(i, tag="ei")
 
    # print("Q: ", ei)
    if dump:
      with open('qr_decomposed_q', 'wb') as f:
        pickle.dump(ei, f)

    #  compute R
    R = np.zeros((ei.shape[1], ei.shape[1]))
    # print(R)
    start = 0
    for i in range(R.shape[0]):
      for j in range(start, R.shape[1]):
        R[i, j] = np.dot(a[:, j], ei[:, i])
      start += 1
    
    # print("R: ", R)
    if dump:
      with open('qr_decomposed_r', 'wb') as f:
        pickle.dump(R, f)

    #  return Q and R matrix 
    if not dump:
      return ei, R


  else:
    if MPI_Recv(0, "slices") > 1:
      # print("slice > 1")
      ai = MPI_Recv(0, "ai")
      # print('ai: ', ai)
      ei = MPI_Recv(0, "ei")
      # print("ei: ", ei)
      for i in range(ai.shape[1]):
        ei = single_columnn_qr_decomp(ei, ai[:, i])
      
      # send eis
      MPI_Send(ei, 0, "ei")
  

def transpose():
  if myrank == 0:
    # original matrix to transpose
    # a = np.zeros(shape=(num_of_rows, num_of_cols))

    # populate_matrix(a)
    # load a
    with open('transpose_a', 'rb') as f:
      a = pickle.load(f)
    
    num_of_rows = a.shape[0]
    num_of_cols = a.shape[1]


    # print("original matrix: ", a)
    t_matrix = a
    if getCommsize() == 1:
      t_matrix = a.transpose()
    else:
      #  create slices
      num_of_slices = int(num_of_rows / getCommsize())
      last_slice = num_of_rows % getCommsize()


      # sent number of slices to other rank
      for i in range(1, getCommsize()):
        MPI_Send(num_of_slices, i, tag=i)
        
      if num_of_slices == 0:
        t_matrix = t_matrix.transpose()
      else:
        #  if equally divisible into slice
        if last_slice == 0:
          for i in range(1, getCommsize()):
            MPI_Send(a[(i)*num_of_slices:(i)*num_of_slices + num_of_slices, :], i, tag=i)

          t_matrix = a[0:num_of_slices, :].transpose()
          for i in range(1, getCommsize()):
            t_matrix = np.hstack((t_matrix, MPI_Recv(i, tag=i)))
          
        else:
          for i in range(1, getCommsize()):
            if i == getCommsize() - 1:
              MPI_Send(a[(i) * num_of_slices : (i) * num_of_slices + num_of_slices + last_slice, :], i, tag=i)
            else:
              MPI_Send(a[(i)*num_of_slices:(i)*num_of_slices + num_of_slices, :], i, tag=i)


          t_matrix = a[0:num_of_slices, :].transpose()
          for i in range(1, getCommsize()):
            t_matrix = np.hstack((t_matrix, MPI_Recv(i, tag=i)))
        
    
    print("transpose: ", t_matrix)
    with open('transposed_a', 'wb') as f:
      pickle.dump(t_matrix, f)
    # print("transposed matrix: ", t_matrix)
  else:
    if MPI_Recv(0, tag = myrank) != 0:
      MPI_Send(MPI_Recv(0, tag=myrank).transpose(), 0, tag=myrank)
  

if __name__ == '__main__':
    init(eval(sys.argv[5]))




