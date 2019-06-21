#coding:utf-8
import os
import csv

def create_csv_full(dirname):
    path = dirname + 'segments-ssn' +'/'
    name = os.listdir(path)
    #name.sort(key=lambda x: int(x.split('.')[0]))
    #print(name)
    with open ('DUTS'+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['data', 'label','segments'])
        for n in name:
            if n[-4:] == '.png':
                print(n)
                # with open('data_'+dirname+'.csv','rb') as f:
                writer.writerow([str(dirname) + 'im/' + str(n[:-4] + '.jpg'), str(dirname) + 'mask/' + str(n[:-4] + '.png'),str(dirname) + 'segments-ssn/' + str(n[:-4] + '.png')])
            else:
                pass




def create_csv(dirname):
    path = dirname + 'im' +'/'
    name = os.listdir(path)
    #name.sort(key=lambda x: int(x.split('.')[0]))
    #print(name)
    with open ('DUTS-na'+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['data', 'label'])
        for n in name:
            if n[-4:] == '.jpg':
                print(n)
                # with open('data_'+dirname+'.csv','rb') as f:
                writer.writerow([str(dirname) + 'im/' + str(n), str(dirname) + 'mask/' + str(n[:-4] + '.png')])
            else:
                pass

if __name__ == "__main__":
    create_csv('/home/kong/Downloads/caffe-sal/data/DUT-train/')
