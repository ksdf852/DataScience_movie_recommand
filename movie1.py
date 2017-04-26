import pandas as pd
import math
import operator

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
from pandas import DataFrame

def movieLensDataLoad(type) :

    ratings = pd.read_csv('~/movie/ml-latest-small/ratings.csv')

    movies = pd.read_csv('~/movie/ml-latest-small/movies.csv')
   
    tags = pd.read_csv('~/movie/ml-latest-small/tags.csv')

    return(ratings,movies,tags)

ratings,movies,tags = movieLensDataLoad('ml-latest-small')

#for i in ratings :
#print(ratings['userId'][0:10])
UM_matrix_ds = ratings.pivot(index='userId',columns='movieId',values = 'rating')


def nearest_neighbor_user(user, top_n,similarity_func) :

    base_user = UM_matrix_ds.loc[user].dropna()
    rated_index = base_user.index
    nearest_neighbor = {}

    for user_id, row_data in UM_matrix_ds.iterrows() :
        intersection_user1 = []
        intersection_user2 = []

        if user == user_id :
            continue

        for index in rated_index :
            if math.isnan(row_data[index]) == False :
                intersection_user1.append(base_user[index])
                intersection_user2.append(row_data[index])
        if len(intersection_user1) <3 :
            continue
        similarity = similarity_func(intersection_user1, intersection_user2)

        if math.isnan(similarity) == False :
            nearest_neighbor[user_id] = similarity
    return sorted(nearest_neighbor.items(), key=operator.itemgetter(1))[:-(top_n+1):-1]


#######   ALl USer nearest_neighbor_user TOp_10 
"""
print("All User nearest_neighbor_user Top_10")

for i in range(len(UM_matrix_ds[1])) :
	print "\n",i+1,"User nearest_neighbor_user Top_10"
	print(nearest_neighbor_user(i+1,10,distance.euclidean))
"""	



######    11,18,24 nearest_neighbor_user Top_3

print("11 nearest_neighbor_user Top_3")
for index in nearest_neighbor_user(11,3,distance.euclidean) :
	print(index)

print("\n18 nearest_neighbor_user Top_3")
for index in nearest_neighbor_user(18,3,distance.euclidean) :
        print(index)

print("\n24 nearest_neighbor_user Top_3")
for index in nearest_neighbor_user(24,3,distance.euclidean) :
        print(index)



def predict_rating(user_id,nearest_neighbor=300, similarity_func=distance.euclidean) : 

#	print(user_id)
	neighbor = nearest_neighbor_user(user_id,nearest_neighbor,similarity_func)
	neighbor_id = [id for id, sim in neighbor]
	neighbor_movie = UM_matrix_ds.loc[neighbor_id].dropna(1,how='all',thresh = 1)

	neighbor_dic = (dict(neighbor))
	ret = []
	for movieId, row in neighbor_movie.iteritems() :
#		print(movieId,row)
		jsum,wsum = 0,0
		for v in row.dropna().iteritems() :
#			print(movieId,v)
#			print "\n"			
			sim = neighbor_dic.get(v[0],0)
			jsum +=sim
			wsum +=(v[1]*sim)

#		print(truth[state][1],movieId)
		if(truth[state][1] < movieId) :
			break
		if(truth[state][1] == movieId) :	
			ret=[truth[state][0],movieId, wsum/jsum]
		#	state+=1
			return ret#[:-(nearest_neighbor+1):-1]


print("\npredict_rating")
predict=[]
truth=[]
matrix=[]
cnt=1000
state=0
for i in range(10):

	truth.append([ratings['userId'][cnt],ratings['movieId'][cnt],ratings['rating'][cnt]])
	cnt+=1000


for j in range(10) :
	predict.append(predict_rating(truth[j][0]))
	state+=1
#state=2
#predict.append(predict_rating(truth[2][0]))	
for i in predict :
	print(i)
print("\ntruth_rating")
for i in truth :
	print(i)

print("\nError rate : "+str(mean_squared_error(predict,truth)))

