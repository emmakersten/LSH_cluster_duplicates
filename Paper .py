#!/usr/bin/env python
# coding: utf-8

# # Paper
# 

# In[776]:


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict 
from random import randint, seed, shuffle
import hashlib
import time
from itertools import combinations
import re
from sklearn.cluster import AgglomerativeClustering


# In[380]:


# loading the data
# Open and read the JSON file
with open('TVs-all-merged.json', 'r') as json_file:
    tv_data = json.load(json_file)


# In[381]:


# visulizing
# Create a matrix to store the data
df = pd.DataFrame()
matrix = []
titles = []
shops = []
keys = []
f_map = [] # feauture map
m_id = [] #model_id

# Iterate through the JSON data and extract and transform information into the matrix
for key, products in tv_data.items():
    for product_info in products:
        # Extract the model ID
        model_id = product_info['modelID']
        # Extract the other information as a dictionary
        other_info = {
            'shop': product_info['shop'],
            #'url': product_info['url'],
            'featuresMap': product_info['featuresMap'],
            'title': product_info['title']
        }
        
        # Append the model ID and other information as a tuple to the matrix
        keys.append(key)
        m_id.append(product_info['modelID'])
        titles.append(product_info['title'])
        shops.append(product_info['shop'])
        f_map.append(product_info['featuresMap'])


# In[382]:


# product represenation
# Create a product representation as a string
# creat shingles of each title in the tv data

def clean_data(string):
    string = string.replace(" ", "").lower()
    string = string.replace('-', '')
    string = string.replace(',', '')
    string = string.replace('.', '')
    string = string.replace('"', '')
    string = string.replace(" ' ", '')
    string = string.replace('/', '')
    string = string.replace(')', '')
    string = string.replace('(', '')
    string = string.replace('bestbuy', '')
    string = string.replace('newegg', '')
    string = string.replace(':', "")
    return string
    
    
def create_product_representation(data):
    representation = []

    title_information = clean_data(data)
    representation.append(create_shingles_from_list(title_information, k = 7))
    return representation


# In[383]:


# Shingles 
def create_shingles_from_list(text_list, k):
    for i in range(0, len(text_list) - k + 1):
        shingle = text_list[i:i + k]  # Use tuples as shingles for lists
    return shingle


# In[384]:


#voor input: alla data bij elkaar gevoegd
def simplify_and_clean(data):
    simplified_data = []
    for category, items in data.items():
        for item in items:
            simplified_data.append((category, clean_title(item.get('title')), clean_features(item), item.get("shop")))
    return simplified_data

def clean_title(title):
    title = title.lower()
    title = title.replace("\"", "inch").replace("inches", "inch")
    title = title.replace("hertz", "hz")
    title = title.replace(" ", "").lower()
    title = title.replace('-', '')
    title = title.replace('.', '')
    title = title.replace('"', '')
    title = title.replace(" ' ", '')
    title = title.replace('/', '')
    title = title.replace(')', '')
    title = title.replace('(', '')
    title = title.replace(':', "")
    title = title.replace('bestbuy', '')
    title = title.replace('newegg', '')
    title = re.sub('[^\sa-zA-z0-9.]+', '', title)
    return title

def clean_features(key_value_pairs):
    key_value_pairs = key_value_pairs['featuresMap']
    for key, value in key_value_pairs.items():
        if key != "title":
            value = clean_title(value)
            key_value_pairs[key] = value
    return key_value_pairs


# In[385]:


def extract_model_words(title, key_pair):
    word_pattern = r'\b(?:\d+[0-9.]*[a-z]+\b|\b[a-z.]+\b)'
    matches = re.findall(word_pattern, title)
    model_words = [word for word in matches if word != '']
    return model_words


def input_for_siganture_matrix(simpliefied_data):
    shSet = set()
    for i in simpliefied_data:
        mw=extract_model_words(i[1], i[2])
        shSet.update(mw)
    
    completeShingle = list(shSet)
    matrix = np.zeros((len(simpliefied_data), len(completeShingle)))
    
    for iterItem, item in enumerate(simpliefied_data):
        shinglesOfTitle = extract_model_words(item[1], item[2])
        matrix[iterItem] = np.isin(completeShingle, shinglesOfTitle)
    return matrix


# In[386]:


### word stuff
MWtitles = []
for i in range(len(titles)):
    MW = re.findall('((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9]*)', str(titles[i]))
    MWtitles.append(MW)

MWpairs = []
MWpairs2 = []
for i in range(len(f_map)):
    MW2 = re.findall("((?:[a-zA-Z]+[\x21-\x7E]+[0-9]|[0-9]+[\x21-\x7E]+[a-zA-Z])[a-zA-Z0-9]*)", str(f_map[i]))
    MWpairs2.append(MW2)

for i in range(len(MWpairs2)):
    result = re.findall(r'\d+', str(MWpairs2[i]))
    MWpairs.append(result)

titleplusfeatures = []
for i in range(len(titles)):
    titleplusfeatures.append(MWtitles[i] + MWpairs[i])
    


# In[387]:


brands = []
all_brands = set()

for category_key in tv_data:
    for item in tv_data[category_key]:
        features_map = item['featuresMap']
        brand = features_map.get('Brand', 'a')
        brands.append(brand)
        all_brands.add(brand.lower() if brand != 'a' else 'a')

brand_from_title = {index: next((brand for brand in all_brands if brand in title), 0) for index, title in enumerate(titles)}
combi_brands = [title_brand if title_brand != 0 else brand for title_brand, brand in zip(brand_from_title.values(), brands)]
combi_brands = [brand if brand != 'a' else 0 for brand in combi_brands]


# In[388]:


# create Dataframe
df['keys'] = m_id
df['shop'] = shops
df['title'] = titleplusfeatures
df['featuresMap'] = f_map


# In[664]:


# Hash function for minhashing
def hash_func(a, b, x):
    return (a * x + b) % 2147483647  # Large prime number for modulo


# Minhashing function
def minhash(shingles, num_hashes):
    num_shingles = len(shingles)
    hash_functions = []  # Generate hash functions for minhashing
    for i in range(num_hashes):
        a = np.random.randint(1, 1000)
        b = np.random.randint(0, 1000)
        hash_functions.append((a, b))
    
    signature_matrix = np.full((num_hashes, num_shingles), np.inf)  # Initialize with infinity
    
    for i, shingle in enumerate(shingles):
        for hash_idx in range(num_hashes):
            a, b = hash_functions[hash_idx]
            hash_val = hash_func(a, b, i) % num_shingles
            if shingle in signature_matrix[hash_idx]:
                signature_matrix[hash_idx, i] = min(signature_matrix[hash_idx, i], hash_val)
            else:
                signature_matrix[hash_idx, i] = hash_val
    
    return signature_matrix


# In[778]:


def test_signature(shingleMatrix, permutations):
    num_items, num_shingles = shingle_matrix.shape
    
    shingle_matrix[shingle_matrix == 0] = np.nan
    
    signature_matrix = np.zeros((num_permutations, num_items), dtype=int)
    
    hash_objects = list(range(0, num_shingles))
    
    # Generate signatures for permutations
    for perm in range(num_permutations):
        shuffle(hash_objects)  # Shuffle hash objects
        hashed_matrix = shingle_matrix * hash_objects
        min_indices = np.nanargmin(hashed_matrix, axis=1)
        signature_matrix[perm] = np.transpose(min_indices)
    
    return signature_matrix


# In[672]:


def hash_func_lsh(a, b, x, num_buckets):
    return ((a * x + b) % 2147483647) % num_buckets

def LSH_2(signature_matrix, num_bands, rows_per_band):
    num_hashes = signature_matrix.shape[0]
    all_pairs = set()  # To store pairs that hash to the same bucket
    
    for band_idx in range(num_bands):
        band_start = band_idx * rows_per_band
        band_end = (band_idx + 1) * rows_per_band
        band_signature = signature_matrix[band_start:band_end, :]
        
        # Hash each band separately
        pairs = hash_band(band_signature)
        all_pairs.update(pairs)
    
    return all_pairs

# Function to hash each band and find pairs in the same bucket
def hash_band(band_signature):
    num_buckets = 2000  # Number of buckets for hashing
    num_shingles = band_signature.shape[1]
    band_buckets = {}
    pairs = set()
    
    for col in range(num_shingles):
        hashed_values = tuple(hash_func_lsh(band_signature[row][col], row, col, num_buckets) for row in range(band_signature.shape[0]))
        hashed_str = '_'.join(map(str, hashed_values))  # Convert tuple to a hashable string
        
        # Find pairs within the same bucket
        if hashed_str in band_buckets:
            for pair_col in band_buckets[hashed_str]:
                if pair_col != col:  # Avoid adding identical pairs
                    pair = (min(pair_col, col), max(pair_col, col))  # Ensure consistent pair ordering
                    pairs.add(pair)  # Add the pair to the set
            band_buckets[hashed_str].add(col)  # Add the current column to the bucket
        else:
            band_buckets[hashed_str] = {col}  # Initialize a new set for the bucket
    
    return pairs


# In[714]:


def lsh_3(signature_matrix, num_bands, rows_per_band):
    hash_count = signature_matrix.shape[0]
    product_count = signature_matrix.shape[1]
    
    potential_pairs = set()
    bands = np.array_split(signature_matrix, bands_count, axis=0)
    
    for band in bands:
        hash_buckets = {}
        
        # Group items into hash buckets
        for product_idx, column_values in enumerate(band.transpose()):
            hashed_column = column_values.tobytes()
            
            if hashed_column in hash_buckets:
                hash_buckets[hashed_column] = np.append(hash_buckets[hashed_column], product_idx)
            else:
                hash_buckets[hashed_column] = np.array([product_idx])
        
        # Find potential pairs within buckets
        for potential_pair in hash_buckets.values():
            if len(potential_pair) > 1:
                for i, item1 in enumerate(potential_pair):
                    for j in range(i + 1, len(potential_pair)):
                        if potential_pair[i] < potential_pair[j]:
                            potential_pairs.add(tuple(sorted((potential_pair[i], potential_pair[j]))))
    
    return potential_pairs


# In[641]:


# Locality Sensitive Hashing (LSH) function with banding
def hash_func_lsh(a, b, x, num_buckets):
    return ((a * x + b) % 2147483647) % num_buckets

def LSH(signature_matrix, num_bands, rows_per_band):
    num_hashes = signature_matrix.shape[0]
    band_buckets = []  # To store buckets for each band
    total_duplicates = 0  # To count total duplicate buckets
    
    for band_idx in range(num_bands):
        band_start = band_idx * rows_per_band
        band_end = (band_idx + 1) * rows_per_band
        band_signature = signature_matrix[band_start:band_end, :]
        
        # Hash each band separately
        band_result = hash_band(band_signature)
        band_buckets.append(band_result)
        
        # Count buckets with more than one item (duplicates)
        total_duplicates += sum(1 for bucket in band_result.values() if len(bucket) > 1)
    
    return band_buckets, total_duplicates

# Function to hash each band
def hash_band(band_signature):
    num_buckets = 1000  # Number of buckets for hashing
    num_shingles = band_signature.shape[1]
    band_buckets = defaultdict(list)
    
    for col in range(num_shingles):
        hashed_values = tuple(hash_func_lsh(band_signature[row][col], row, col, num_buckets) for row in range(band_signature.shape[0]))
        band_buckets[hashed_values].append(col)
    
    return band_buckets


# In[662]:


def hash_func_lsh(a, b, x, num_buckets):
    return ((a * x + b) % 2147483647) % num_buckets

def LSH_4(signature_matrix, num_bands, rows_per_band):
    num_hashes = signature_matrix.shape[0]
    band_buckets = []  # To store buckets for each band
    duplicate_pairs = set()  # To store pairs of duplicates
    
    for band_idx in range(num_bands):
        band_start = band_idx * rows_per_band
        band_end = (band_idx + 1) * rows_per_band
        band_signature = signature_matrix[band_start:band_end, :]
        
        # Hash each band separately
        band_result = hash_band(band_signature)
        band_buckets.append(band_result)
        
        # Find pairs within each bucket
        for bucket in band_result.values():
            if len(bucket) > 1:
                for i, item1 in enumerate(bucket):
                    for j in range(i + 1, len(bucket)):
                        item2 = bucket[j]
                        if item1 < item2:
                            duplicate_pairs.add((item1, item2))
    
    
    return band_buckets, duplicate_pairs

# Function to hash each band
def hash_band(band_signature):
    num_buckets = 1000  # Number of buckets for hashing
    num_shingles = band_signature.shape[1]
    band_buckets = defaultdict(list)
    
    for col in range(num_shingles):
        hashed_values = tuple(hash_func_lsh(band_signature[row][col], row, col, num_buckets) for row in range(band_signature.shape[0]))
        band_buckets[hashed_values].append(col)
    
    return band_buckets


# In[643]:


#jaccuard distance
def jac_distance(words1, words2):
    C = words1.intersection(words2)
    D = words1.union(words2)
    return 1.0 - float(len(C))/(len(D))


# In[644]:


def dissimmatrix(Candidate_pairs, modelW, k):
    # Calculate dissimilarity matrix using Jaccard distance
    num_samples = len(modelW)
    dissimilarity_matrix = np.full((num_samples, num_samples), 1000, dtype=float)

    for pair in Candidate_pairs:
        test_item1 = pair[0]
        test_item2 = pair[1]

        dissimilarity_matrix[test_item1, test_item2] = jac_distance(set(create_shingles_from_list(modelW[test_item1][1], k)),
                                                    set(create_shingles_from_list(modelW[test_item2][1], k)))

        if(modelW[test_item1][3]==modelW[test_item2][3]):
            dissimilarity_matrix[test_item1, test_item2]=1000
        if "Brand" in modelW[test_item1][2] and "Brand" in modelW[test_item2][2]:
            if (modelW[test_item1][2].get("Brand") == modelW[test_item2][2].get("Brand")):
                dissimilarity_matrix[test_item1, test_item2] = 1000

        dissimilarity_matrix[test_item2, test_item1] = dissimilarity_matrix[test_item1, test_item2]
    return dissimilarity_matrix



# In[645]:


def clusterMethod(dissimilarity_matrix, t):
    linkage = AgglomerativeClustering(n_clusters=None, affinity="precomputed",
                                      linkage='single', distance_threshold=threshold)
    clusters = linkage.fit_predict(dissimilarity_matrix)

    dict_clusters = {}
    for index, cluster_nr in enumerate(clusters):
        if cluster_nr in dict_clusters:
            dict_clusters[cluster_nr] = np.append(dict_clusters[cluster_nr], index)
        else:
            dict_clusters[cluster_nr] = np.array([index])

    candidate_pairs = set()
    for potential_pair in dict_clusters.values():
        if len(potential_pair) > 1:
            for i, item1 in enumerate(potential_pair):
                for j in range(i + 1, len(potential_pair)):
                    if potential_pair[i] < potential_pair[j]:
                        candidate_pairs.add((potential_pair[i], potential_pair[j]))
                    else:
                        candidate_pairs.add((potential_pair[j], potential_pair[i]))

    return candidate_pairs



# In[646]:


def evaluate_results_cluster(potential_pairs_lsh, potential_pairs_cluster, data):
    true_positives_lsh = sum(1 for pair in potential_pairs_lsh if data[pair[0]][0] == data[pair[1]][0])
    true_positives_cluster = sum(1 for pair in potential_pairs_cluster if data[pair[0]][0] == data[pair[1]][0])

    number_duplicates = sum(1 for i, word1 in enumerate(data) for j, word2 in enumerate(data) if word1[0] == word2[0] and i < j)
    
    number_candidates_lsh = len(potential_pairs_lsh) + 0.000001
    tp_and_fn = len(potential_pairs_cluster) + 0.000001

    pc_lsh = true_positives_lsh / number_duplicates
    pq_lsh = true_positives_lsh / number_candidates_lsh
    f1_star_lsh = 2 * (pq_lsh * pc_lsh) / (pq_lsh + pc_lsh + 0.000001)

    precision_cluster = true_positives_cluster / number_duplicates
    recall_cluster = true_positives_cluster / tp_and_fn
    f1 = 2 * (precision_cluster * recall_cluster) / (precision_cluster + recall_cluster + 0.000001)
    
    n = len(data)
    total_comparisons = (n * (n - 1)) / 2
    print(true_positives_lsh)
    print(number_candidates_lsh)
    print(true_positives_cluster)
    print(tp_and_fn)
    print(number_duplicates)
    fraction = number_candidates_lsh / total_comparisons
    
    return [pc_lsh, pq_lsh, f1_star_lsh, precision_cluster, recall_cluster, f1, fraction]


# In[761]:


def evaluateResults_2(candidatepairs, candidate_pairsCluster, data):
    # Evaluate LSH
    tpLSH = sum(1 for pair in candidatepairs if data[pair[0]][0] == data[pair[1]][0])
    tpCluster = sum(1 for pair in candidate_pairsCluster if data[pair[0]][0] == data[pair[1]][0])

    numberDuplicates = sum(1 for i, word1 in enumerate(data) for j, word2 in enumerate(data) if word1[0] == word2[0] and i < j)
    
    numberCandidatesLSH = len(candidatepairs) + 0.000001
    
    print(numberDuplicates)
    print(numberCandidatesLSH)
    print(tpLSH)
    tpandfn = len(candidate_pairsCluster) + 0.0000001

    PC = tpLSH / numberDuplicates
    PQ = tpLSH / numberCandidatesLSH 
    F1StarLSH = (2 * (PQ * PC)) / (PQ + PC + 0.0000001)

    precision = tpCluster / numberDuplicates
    recall = tpCluster / tpandfn
    F1 = 2 * (precision * recall) / (precision + recall + 0.0000001)

    N = len(data)
    totalComparisons = (N * (N - 1)) / 2
    fraction = min(len(candidatepairs) / totalComparisons, 1)
    
    return [PC, PQ, F1StarLSH, precision, recall, F1, fraction]


# In[762]:


def bootstrap_data(data, num_bootstraps, train_split):
    train_sets = []
    test_sets = []

    data_size = len(data)
    train_size = int(data_size * train_split)

    for _ in range(num_bootstraps):
        # Generate a bootstrap sample by randomly sampling with replacement
        bootstrap_indices = [random.randint(0, data_size - 1) for _ in range(data_size)]
        bootstrap_sample = [data[i] for i in bootstrap_indices]

        # Splitting into train and test sets
        train_set = bootstrap_sample[:train_size]
        test_set = bootstrap_sample[train_size:]

        train_sets.append(train_set)
        test_sets.append(test_set)

    return train_sets, test_sets


# In[782]:


#testing
n = 800
b = 750
length = 3
r = int(n/b)
threshold = 0.6


Input_signature = input_for_siganture_matrix(simple_data)
signatureMatrix = test_signature(Input_signature, b * r)
pairsLSH = LSH_2(signatureMatrix, b, r)
dissimilarityMatrix = dissimmatrix(pairsLSH, simple_data, length)
pairsCluster = clusterMethod(dissimilarityMatrix, threshold)
    
AAA_test = evaluateResults_2(pairsLSH, pairsCluster, simple_data)


# In[799]:


#running full specs
n = 800
length = 3

diffRowsresults_zonderboot = pd.DataFrame(columns=['PC', 'PQ', 'F1*', 'Precision', 'Recall', 'F1', 'fraction'])

PC = []
PQ = []
F1_Star_LSH = []
Precision = []
Recallv = []
F1 = []
Fraction = []

print(len(simple_data))
train_set,test_set=bootstrap_data(simple_data,5,0.63)

b_test = [1,2,4,10,100,200,400,800]
for b in b_test:
    
    r = int(n/b)
    threshold = 0.6


    Input_signature = input_for_siganture_matrix(simple_data)
    signatureMatrix = test_signature(Input_signature, b * r)
    pairsLSH = lsh_3(signatureMatrix, b, r)
    dissimilarityMatrix = dissimmatrix(pairsLSH, simple_data, length)
    pairsCluster = clusterMethod(dissimilarityMatrix, threshold)
    
    AAA = evaluateResults_2(pairsLSH, pairsCluster, simple_data)
    
    PC.append(AAA[0])
    PQ.append(AAA[1])

    F1_Star_LSH.append(AAA[2])
    Precision.append(AAA[3])
    Recallv.append(AAA[4])
    F1.append(AAA[5])
    Fraction.append(AAA[6])

diffRowsresults_zonderboot['PC'] = PC
diffRowsresults_zonderboot['PQ'] = PQ
diffRowsresults_zonderboot['F1*'] = F1_Star_LSH
diffRowsresults_zonderboot['Precision'] = Precision
diffRowsresults_zonderboot['Recall'] = Recallv
diffRowsresults_zonderboot['F1'] = F1
diffRowsresults_zonderboot['fraction'] = Fraction


# In[700]:


##plotting
diffRowsresults = pd.DataFrame(columns=['PC', 'PQ', 'F1*', 'Precision', 'Recall', 'F1', 'fraction'])

PC = []
PQ = []
F1_Star_LSH = []
Precision = []
Recallv = []
F1 = []
Fraction = []


length = 3
numberBootstraps = 5
percentage = 0.63


for b in range(250,500,50):
    n = 500
    r = int(n/b)
    threshold = 0.6

    train_set,test_set=bootstrap_data(simple_data,numberBootstraps,percentage)
    results = np.zeros((numberBootstraps, 7))
    for i in range(numberBootstraps):
        Input_signature = input_for_siganture_matrix(train_set[i])
        signatureMatrix = minhash(Input_signature, b * r)
        pairsLSH = LSH_2(signatureMatrix, b, r)
        dissimilarityMatrix = dissimmatrix(pairsLSH, train_set[i], length)
        pairsCluster = clusterMethod(dissimilarityMatrix, threshold)
    
        AAA = evaluateResults_2(pairsLSH, pairsCluster, train_set[i])
        #results = AAA[i]
    #AAA_1 = np.mean(results)
    
    #mean neaded when we bootstrap over multiple samples
    PC.append(AAA[0])
    PQ.append(AAA[1])

    F1_Star_LSH.append(AAA[2])
    Precision.append(AAA[3])
    Recallv.append(AAA[4])
    F1.append(AAA[5])
    Fraction.append(AAA[6])

diffRowsresults['PC'] = PC
diffRowsresults['PQ'] = PQ
diffRowsresults['F1*'] = F1_Star_LSH
diffRowsresults['Precision'] = Precision
diffRowsresults['Recall'] = Recallv
diffRowsresults['F1'] = F1
diffRowsresults['fraction'] = Fraction
diffRowsresults['treshold'] = t



# In[702]:


#plotting with bootstrap
plt.plot(diffRowsresults["fraction"],diffRowsresults['PC'])
plt.ylabel("Pair completeness")
plt.xlabel("Fraction of comparisons")
plt.show()
plt.plot(diffRowsresults["fraction"],diffRowsresults['PQ'])
plt.ylabel("Pair quality")
plt.xlabel("Fraction of comparisons")
plt.show()
plt.plot(diffRowsresults["fraction"],diffRowsresults['F1*'])
plt.ylabel("F1*")
plt.xlabel("Fraction of comparisons")
plt.show()
plt.plot(diffRowsresults["fraction"],diffRowsresults['F1'])
plt.ylabel("F1")
plt.xlabel("Fraction of comparisons")
plt.show()

