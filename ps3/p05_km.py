from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np 


'''
returns:
    means => clusterSize length array with centroid of each cluster
'''
def kmeans(A, clusterSize):
    #---initialize 16 points as cluster means---
    means = []
    while(len(means) != clusterSize):
        i = np.random.randint(0, len(A)-1)
        j = np.random.randint(0, len(A[0])-1)

        uniq = True        
        # check if i,j is unique
        for point in means:
            if(np.all(point == A[i][j])):
                print('here')
                uniq = False
                break

        if(uniq == True):
            means.append(A[i][j])
    #---initialize 16 points as cluster means END---

    
    means = np.array(means)
    pointCluster = np.zeros((len(A), len(A[0])))
    prevLoss = 1000
    loss = 100
    it = 0
    while(abs(loss - prevLoss) > 100 and it < 50):

        # Set clusters for each element
        for i in range(len(A)):
            for j in range(len(A[i])):
                minNorm = 100000
                clust = 0
                for i1 in range(len(means)):
                    norm = np.linalg.norm((means[i1] - A[i][j]))
                    if(norm < minNorm):
                        clust = i1
                        minNorm = norm
                pointCluster[i][j] = clust


        # Set means for each cluster
        for c in range(len(means)):
            num = np.zeros(3)
            den = 0
            for i in range(len(A)):
                for j in range(len(A[i])):
                    if(c == pointCluster[i][j]):
                        num += A[i][j]
                        den += 1

            num = num / den
            
            means[c] = np.round(num)


        #loss
        prevLoss = loss
        loss = 0    
        for i in range(len(A)):
            for j in range(len(A[i])):
                loss += np.linalg.norm((means[int(pointCluster[i][j])] - A[i][j]))

        it += 1
        print("iteration: ", it)
        print(loss)
        # print(means[0])

    return means
            

            


if __name__ == "__main__":
    A = imread('data/peppers-small.tiff')
    A = np.array(A)
    means = kmeans(A, 16)

    A2 = imread('data/peppers-large.tiff')

    B = np.zeros(A2.shape)

    for i in range(len(A2)):
        for j in range(A2.shape[1]):
            minNorm = 100000.0
            clust = 0
            for i1 in range(len(means)):
                norm = np.linalg.norm((means[i1] - A2[i][j]))
                if(norm < minNorm):
                    clust = i1
                    minNorm = norm
            B[i][j] = means[clust]

    B = B.astype(int)
    plt.imshow(B)
    plt.show()
