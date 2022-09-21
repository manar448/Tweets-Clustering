import re
import matplotlib.pyplot as plt
import math
import random as rd
import pandas as pd
import string

def Preprocessing_dataset(url):

    df = pd.read_excel(url)
    df.info()
    tweets = list(df['tweets'])
    
    list_of_cleanTweets =[]
    for i in range(len(tweets)): 
        #remove words start with sympol @, hashtag sympol #
        tweets[i] = tweets[i].replace("#","")
        tweets[i] = tweets[i].replace("@","")
        #remove any url
        tweets[i] = re.sub(r"http\S+"," ",tweets[i])
        #every word is converted to lowercase
        tweets[i] = tweets[i].lower()
        #remove the punctuation (any marks. , -) from the text
        tweets[i] = tweets[i].translate(str.maketrans('', '', string.punctuation))
        #remove extra spaces
        #put all the string in list of characters including spaces
        #join it all to gether with string seperator between each word
        tweets[i] = " ".join(tweets[i].split())
        #remove new lines
        tweets[i] = tweets[i].strip('\n')
        #split the string and put it in list of strings(words)     
        list_of_cleanTweets.append(tweets[i].split(' '))
    
    return(list_of_cleanTweets)

# KMeans function
# tweets:transformed data
# k:number of the clusters
# iterations:number of iteration(experiments)
def kmeans(tweets, k , min_iterations = 20):
    count = 0 
    centroids = []
    while count < k:
        rand_index = rd.randint(0, len(tweets) - 1)
        count += 1
        centroids.append(tweets[rand_index])  

    # Randomly choosing Centroids
    # Centroid with the minimum Distance
    iter_count = 0
    prev_centers = []

    # Repeating the above steps for a defined number of iterations or until converged.
    #with checking the convergence of new centers and previous cetroids.
    while ((converged(prev_centers, centroids)) == False) and (iter_count < min_iterations):
        clusters = assigncluster(tweets, centroids)
        prev_centers = centroids
        # Updating Centroids
        centroids = update_centroids(clusters)
        iter_count += 1

    if (iter_count == min_iterations and converged == True) or (converged(prev_centers, centroids) == True):
        print("converged")        
    elif (iter_count == min_iterations):
        print("iterations completed but clusters not converged")         
        
    sse = calc_SSE(clusters,centroids)
    
    return clusters , sse

#check the convergence of the centers

def converged(prev_centers, centroids):

    if len(prev_centers) != len(centroids):
        return False 
 
    for tweet in range(len(centroids)): 
        if (prev_centers[tweet] != centroids[tweet]): 
            return False 
            
    return True

#compute sse

def calc_SSE(clusters,centroids):

    sse = 0

    for c in range(len(clusters)):
        for tweet in range(len(clusters[c])):
            sse = sse + (jaccard_distance(clusters[c][tweet][0] , centroids[c]) * jaccard_distance(clusters[c][tweet][0] , centroids[c])) 
    return sse 

#get distance function

def jaccard_distance(tweet1, tweet2):

    # the union between the two tweets
    intersection = set(tweet1).intersection(tweet2)
    # the intersection between the two tweets
    union = set().union(tweet1,tweet2)
    return 1 - (len(intersection) / len(union))

#assign function

def assigncluster(tweets, centroids): 
 
    clusters = {} 
    
    for i in range(len(tweets)): 
        minimumDis = 1
        cluster = -1
        for j in range(len(centroids)): 
            dis = jaccard_distance(centroids[j], tweets[i]) 
            if dis < minimumDis: 
                minimumDis = dis 
                cluster = j 
            if centroids[j] == tweets[i]: 
                cluster = j 
                minimumDis = 0 
                break
        if minimumDis == 1: 
             cluster = rd.randint(0, len(centroids) - 1) 
        clusters.setdefault(cluster, []).append([tweets[i]])
        #0 : [twwt[i] , tweet[]]
         
    return clusters        

#update centers of the clusters

def update_centroids(clusters):

    centroids = []

    # iterate each cluster and check for a tweet with closest distance sum with all other tweets in the same cluster
    # select that tweet as the centroid for the cluster
    for c in range(len(clusters)):
        #positive floating point
        min_dis_sum = math.inf
        centroid_idx = 0
        for t1 in range(len(clusters[c])):
            sumOf_distances = 0
            # get distances sum for every of tweet t1 with every tweet t2 in a same cluster
            for t2 in range(len(clusters[c])):
                if t1 != t2:
                    dis = jaccard_distance(clusters[c][t1][0], clusters[c][t2][0])   
                    sumOf_distances += dis
                else:
                    sumOf_distances += 0

            # select the tweet with the minimum distances sum as the centroid for the cluster
            if sumOf_distances < min_dis_sum:
                min_dis_sum = sumOf_distances
                centroid_idx = t1

        # append the selected tweet to the centroid list
        centroids.append(clusters[c][centroid_idx][0])

    return centroids

def inputUser():
    experiments = input("number of expriments: ")
    url = input("Enter name of file: ")
    return experiments,url

experiments, url = inputUser()
k = 1
experiment = experiments
file = url + '.xlsx'
#start range from k to (k + experiments)-1
numberOfCluster = range(k, k + int(experiment)) 
tweets = Preprocessing_dataset(file)
sse_list = []
for e in range(int(experiment)):
    clusters , sse = kmeans(tweets, k)

    for c in range(len(clusters)):
        
        print(str(c+1) + ": " , str(len(clusters[c])) + "tweets")
    print(" k = " + str(k))    
    print("SSE = " + str(sse))
    k = k + 1
    sse_list.append(sse)

    
plt.rcParams["figure.figsize"] = (10,8)
plt.plot(numberOfCluster, sse_list ,color='red')
plt.title('The Elbow Method')
plt.xlabel("Number Of Cluster")
plt.ylabel("SSE")
plt.grid()
plt.show()