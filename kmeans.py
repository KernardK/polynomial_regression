import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function that reads the file and returns the data sets
def read_csv(file):
    dataSet = pd.read_csv(file)
    # print(dataSet.describe())

    countries = dataSet.iloc[:, [0]].values
    X = dataSet.iloc[:, [1, 2]].values

    return X, countries, dataSet


# Function that calculates the K Means algorithm
def k_means(K, max_iters):

    # number of training examples
    m = X.shape[0]
    # number of features (This is set as 2)
    n = X.shape[1]

    # Initialized the centroids randomly from the data points
    centroids = np.array([]).reshape(n, 0)

    # Each column becomes the centroid for a cluster
    for i in range(K):
        rand = np.random.randint(0, m - 1)
        centroids = np.c_[centroids, X[rand]]

    clusters = {}

    # Loops through the number of iterations
    for i in range(max_iters):

        # The distances of the values
        distance = np.array([]).reshape(m, 0)

        # Calculates the squared distance from each data point.
        for k in range(K):
            tempDist = np.sum((X - centroids[:, k]) ** 2, axis=1)
            distance = np.c_[distance, tempDist]

        # Collects the cluster number closest to the centroids
        closest = np.argmin(distance, axis=1) + 1

        # Regroup data points based on the cluster's index in dictionary
        # Stores the clusters and indexes for each iteration
        temp = {}
        for k in range(K):
            temp[k + 1] = np.array([]).reshape(2, 0)
        # display the clusters and countries
        for i in range(m):
            temp[closest[i]] = np.c_[temp[closest[i]], X[i]]

        # display the clusters and countries
        for k in range(K):
            temp[k + 1] = temp[k + 1].T
        # display the clusters and countries
        for k in range(K):
            centroids[:, k] = np.mean(temp[k + 1], axis=0)

        # Clusters
        clusters = temp

        # Plots the data ont the graph
        plot_clustered(K, clusters, centroids)

    # List of countries from the csv file
    country_list = data['Countries']

    # Empty Lists to store values of clusters
    countries_list = []
    birth_rate = []
    mean_life_expectancy = []

    # Loops through each of the clusters
    for cluster in clusters:
        num = 0

        # For each of the points in the clusters
        for featureset in clusters[cluster]:
            num += 1
            countries_list.append(country_list[(data['BirthRate(Per1000 - 2008)'] == featureset[0])
                                  & (data['LifeExpectancy(2008)'] == featureset[1])])

            # Adds values into the birth rate list
            birth_rate.append(featureset[0])
            # Calculates the mean of the birth rates
            mean_birth_rate_val = np.mean(birth_rate)

            # Adds the values into the life expectancy
            mean_life_expectancy.append(featureset[1])
            # Calculates the mean of the
            mean_life_expectancy_val = np.mean(mean_life_expectancy)

        # Displays the appropriate information
        print("---------------------------------")
        print(f"Total Countries in cluster {i}: {num}")
        print("---------------------------------")
        print(f"Mean Life Expectancy: {mean_life_expectancy_val:.2f}")
        print("---------------------------------")
        print(f"Mean Birth Rate: {mean_birth_rate_val}")

        # Displays all the countries in each clusters
        print(f"List of Countries and their Birth Rates in Cluster: {i}: ")
        i += 1
        for c, b in zip(countries_list, birth_rate):
            print(f"{c.to_string(index=False)}: {b:.2f} ")

    return clusters, centroids


# Function displays the un-clustered graph
def plot_unclustered():
    # Scatter plot graph colors and values
    plt.scatter(X[:, 0], X[:, 1], color='blue', label='Unclustered Data')

    # Labels the graphs
    plt.xlabel('Birth Rate')
    plt.ylabel('Life Expectancy')
    plt.legend()

    # Plots the graph
    plt.title('Initial Data points plot')
    plt.show()


# Function displays the clustered data
def plot_clustered(K, clusters, centroids):

    # Color palette
    colors = ['green', 'purple', 'red', 'cyan', 'blue']
    # Labels the Clusters
    labels = ['Cluster - 1', 'Cluster - 2', 'Cluster - 3', 'Cluster - 4', 'Cluster - 5']

    # For all the clusters in the iteration
    for k in range(K):
        # Plots each point in on the graph with the appropriate color for the cluster
        plt.scatter(clusters[k + 1][:, 0], clusters[k + 1][:, 1], color=colors[k], label=labels[k])

    # Plots the centroids on the graph
    plt.scatter(centroids[0, :], centroids[1, :], marker='+', color='black', label='Centroids')

    # Labels the graphs
    plt.xlabel('Birth Rate')
    plt.ylabel('Life Expectancy')
    plt.legend()

    # Plots the graph
    plt.title('Clustered data')
    plt.show()


# Csv Files
file_1 = "data1953.csv"
file_2 = "data2008.csv"
file_3 = "dataBoth.csv"

# Methpd Calls
X, Countries, data = read_csv(file_2)

# Gathers the k and max iterations
while True:
    try:
        K = int(input("Enter the number of clusters for the algorithm: "))
        break
    except ValueError:
        print("Sorry invalid values. Please try again.")

# Take in a user-defined number of iterations
while True:
    try:
        max_iters = int(input("Enter the maximum number of iterations: "))
        break
    except ValueError:
        print("Sorry invalid values. Please try again.")


# Plots un-clustered data
plot_unclustered()

# collects the centroids and clustered
Clusters, Centroids = k_means(K, max_iters)

# Plots clustered data
plot_clustered(K, Clusters, Centroids)


