#import graph_nets as gn
#这里假定将一个图片分割为一个batch的graph
#每个小块内部全连接
#需要加位置权重比
import numpy as np
import scipy.misc as misc
from skimage import segmentation
import tensorflow as tf
import graph_nets as gn
import time
import sonnet as snt

def create_edge(node_xy,classnum,node_num,K):
    edge = []
    receiver = []
    sender = []
    edge_num =np.zeros(classnum,dtype=int)
    for i in range(classnum):
        for h in range(node_num[i]):
            for z in range(node_num[i]):
                x=abs(node_xy[i][z][0] - node_xy[i][h][0])
                y=abs(node_xy[i][z][1] - node_xy[i][h][1])
                if((x<=K)and(y<=K)and((x+y)!=0)):
                    if i ==0 :
                        sender.append(h)
                        receiver.append(z)
                        edge.append(1/(x+y))
                        edge_num[i]=edge_num[i]+1
                    else:
                        sender.append(h+node_num[i-1])
                        receiver.append(z+node_num[i-1])
                        edge.append(1/(x+y))
                        edge_num[i]=edge_num[i]+1
    return edge_num,edge,receiver,sender


# def create_graph_fn(feature):
#     n_nodes=tf.Variable(256,dtype=int),
#     edge = tf.variable()
#     nodes =tf.









def create_graph(segment,map):
    map2segment = []
    map_xy = []
    max=np.max(segment)
    node_number = np.zeros(max+1,dtype=int)
    for f in range(max+1):
        map2segment.append([])
        map_xy.append([])

    time2 = time.time()
    for x in range(map.shape[0]):
        for y in range(map.shape[1]):
            dic = segment[x][y]
            map2segment[dic].append(map[x][y])
            map_xy[dic].append([x, y])
            node_number[dic] = node_number[dic]+1
    time3 = time.time()
    print(time3 - time2)
    Nedges,edge_w,receivers,senders=create_edge(map_xy,(max+1),node_number,2)

    a = gn.graphs.GraphsTuple(
        nodes=map2segment,
        edges=edge_w,
        globals=tf.zeros([max, 20], tf.float32),
        n_node=node_number,
        n_edge=Nedges,
        receivers=receivers,
        senders=senders
     )
    return a








def tf_create_graphnodes(segment,map,chennal):
    max = 0
    enum = 0
    for i in segment:
        for m in i:
            if m > max:
                max = m
    node_number = [(max + 1) * 0]



    for x in range(map.shape[0]):
        for y in range(map.shape[1]):
            dic = segment[x][y]
            node_number[dic] = node_number[dic]+1

    for h in range(max):
        enum=enum+node_number[h]*(node_number[h]-1)


    a= gn.graphs.GraphsTuple(
        nodes = tf.placeholder([map.shape[0]*map.shape[1],chennal],dtype = tf.float32),
        edges = tf.zeros([enum,12],dtype=tf.float32),
        globals = tf.zeros([max,20],tf.float32),
        N_node = tf.placeholder([max],dtype=tf.int32),
        N_edge = tf.placeholder([max],dtype=tf.int32),
        receivers = tf.range(5, dtype=tf.int32) ,
        senders = tf.range(5, dtype=tf.int32)
    )
    return a







#segments = segmentation.slic(img,n_segments=1,compactness=10)



def mani_edge(K):
    edge=[],
    receive =[],
    sender = [],
    for  i in range(1024):
        if((i%32<K)or(1%32>=(32-K))or(i<32*K)or(i>=1024-32*K)):
            edge.append([1])
            receive.append(i+1)
            sender.append(i)

            edge.append([1])
            receive.append(i+2)
            sender.append(i)


def fc_edge():
    edge=[]
    receive=[]
    sender=[]
    for i in range(256):
        for m in range(256):
            if i!=m:
                edge.append([1])
                receive.append(i)
                sender.append(m)
    return edge,receive,sender




def fc_GCN_feature(x,y,z,h):
    return gn.graphs.GraphsTuple(
                nodes=x,
                edges=y,
                globals=tf.Variable(tf.zeros([1, 20], dtype=tf.float32)),
                receivers=z,
                senders=h,
                n_node=tf.constant([256], dtype=tf.int32),
                n_edge=tf.constant([256*255], dtype=tf.int32),
            )









def mani_GCN_feature(feature):
    return gn.graphs.GraphsTuple(
        nodes=tf.squeeze(tf.reshape(feature,[32*32])),
        edges=tf.Variable(tf.zeros([5, 20], dtype=tf.float32)),
        globals=tf.Variable(tf.zeros([1, 20], dtype=tf.float32)),
        receivers=tf.range(1024, dtype=tf.int32) // 3,
        senders=tf.range(1024, dtype=tf.int32) % 3,
        n_node=tf.constant([32], dtype=tf.int32),
        n_edge=tf.constant([], dtype=tf.int32),
    )









def loadNodes (map):
    node=[]
    num = 0
    for i in map:
        for m in i:
            node.append(m)
            num = num +1
    return node,num


def create_edge(map,direction):   #initial edge without using deep learning
    if(direction==0):
        edge=np.ones(map.shape[0]*map.shape[1])
    else:   #在初始化的时候构建有向图,考虑到节点之间暂时没有生长的机制,这里采用yi阶邻域的方式,也可以使用一阶邻域的方式
        edge=np.ones(12+10*(map.shape[0]+map.shape[1]-4)+(map.shape[0]-2)*(map.shape[1]-2))
    return edge








    # for h in range(max):
    #     edge_number[h]=node_number[h]*(node_number[h]-1)
    #     enum=enum+edge_number[h]
    # edge_feature=[enum*1]



img=misc.imread('/home/kong/datasets/DUTS/DUT-test/DUT-test-Image/ILSVRC2012_test_00000003.jpg')
print(np.shape(img))
img=misc.imresize(img, (16, 16))
img= np.reshape(img,(256,3))
edge,receive,sender=fc_edge()
edge= np.array(edge)
receive=np.array(receive)
sender=np.array(sender)
x=tf.placeholder(tf.float32,[256, 3] )

y= tf.placeholder(tf.float32, [256 * 255, 1])

z = tf.placeholder(tf.int32, [256 * 255])
h = tf.placeholder(tf.int32, [256 * 255])

a=fc_GCN_feature(x,y,z,h)
input_graphs = a

# Create the graph network.
graph_net_module = gn.modules.GraphNetwork(
    edge_model_fn=lambda: snt.nets.MLP([32, 32]),
    node_model_fn=lambda: snt.nets.MLP([32, 32]),
    global_model_fn=lambda: snt.nets.MLP([32, 32]))

# Pass the input graphs to the graph network, and return the output graphs.
output_graphs = graph_net_module(input_graphs)
print('hello world')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    out = sess.run(output_graphs,
             {x: img, y: edge, z: receive, h: sender})
    print("hello")