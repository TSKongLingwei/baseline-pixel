from skimage import segmentation
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf



def np_edge_create(img,segments):
    a1, a2 = segmentation.mark_boundaries(img, segments, getab=1)
    edge_number=[]
    s1 = a1[a1!=a2]
    s2 = a2[a2!=a1]
    h1 = np.array([s1,s2])
    h1 = np.unique(h1,axis=1)
    h2 = np.copy(h1)
    tmp = np.copy(h2[0])
    h2[0] = h2[1]
    h2[1] = tmp
    edge = np.concatenate((h1,h2),axis=1)
    edge = np.unique(edge,axis=1)
    sender = np.copy(edge[0])
    receiver = np.copy(edge[1])
    n= np.shape(edge)[1]
    edge_number.append(n)
    edge_number = np.array(edge_number)
    edge  = np.ones([n,1])
    return edge_number,receiver,sender,edge


def np_Kedge_create(img,segments):
    a1, a2 = segmentation.mark_boundaries(img, segments, getab=1)
    edge_number = []
    s1 = a1[a1 != a2]
    s2 = a2[a2 != a1]
    h1 = np.array([s1, s2])
    h1 = np.unique(h1, axis=1)
    h2 = np.copy(h1)
    tmp = np.copy(h2[0])
    h2[0] = h2[1]
    h2[1] = tmp
    edge = np.concatenate((h1, h2), axis=1)
    edge = np.unique(edge, axis=1)
    sender = np.copy(edge[0])
    receiver = np.copy(edge[1])
    two_sender = []
    two_receiver = []
    for i in range(np.max(segments)+1):
        k = np.where(sender == i)
        m = receiver[k]
        s = np.where(sender == m)
        for x in m:
            s = np.where(sender == x)
            r = receiver[s]
            num = np.shape(r)[0]
            two_sender = two_sender + ([i] * num)
            two_receiver = two_receiver + r.tolist()
    f1 = np.array(two_sender)
    f1 = f1[:, np.newaxis]
    f2 = np.array(two_receiver)
    f2 = f2[:, np.newaxis]
    edge2 = np.concatenate((f1, f2), axis=1)
    edge2 = edge2.transpose()
    edge2 = np.unique(edge2, axis=1)
    sender_2 = np.copy(edge2[0])
    receiver_2 = np.copy(edge2[1])
    n = np.shape(edge2)[1]
    edge_number.append(n)
    edge_number = np.array(edge_number)
    edge2 = np.ones([n, 1])
    return edge_number, receiver_2, sender_2, edge2



def node2img(node,segments):
    a = segments
    m= np.float32(a)
    for i in range(a.max()+1):
        m[a==i] = node[i]
    return m