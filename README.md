# ML-Introduction
Introduction to ML using a simple example



## Supervised Learning
  Requires both input and output values to be used in the training process / giving the algorithm the "correct" answer.

## For Example / Regression model
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    # fit final model
    model = LinearRegression()
    model.fit(X, y)
    Xnew = [[-1.07296862, -0.52817175]]
    ynew = model.predict(Xnew)
    print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(X[:, 0], y, label="Feature 1 against Y")
    ax[1].scatter(X[:, 1], y, label="Feature 2 against Y")
    ax[0].set_xlabel("Feature 1")
    ax[0].set_ylabel("Y")
    ax[1].set_xlabel("Feature 2")
    ax[1].set_ylabel("Y")
    plt.show()

![image](https://user-images.githubusercontent.com/99629762/156682223-9276fb1f-e7a5-444a-bd86-ae6f37607178.png)


## Code Explanation
    
    # import 
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression
    
    # generate regression dataset
        X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    
    # fit final model
        model = LinearRegression()
        model.fit(X, y)
    
    # define one new data instance
        Xnew = [[-1.07296862, -0.52817175]]
    
    # make a prediction
        ynew = model.predict(Xnew)
    
    # show the inputs and predicted outputs
        print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(X[:, 0], y, label="Feature 1 against Y")
        ax[1].scatter(X[:, 1], y, label="Feature 2 against Y")
        ax[0].set_xlabel("Feature 1")
        ax[0].set_ylabel("Y")
        ax[1].set_xlabel("Feature 2")
        ax[1].set_ylabel("Y")
        plt.show()
     
     
## Unsupervised learning
    System which can be used to find hidden patterns / clusters within data
    
    
## Example
    - Convenience store records data on customers
    - Each customers spends x amount of time in store
    - Unsupervised learning algorithm can be used to group similiar customers together based on amount of time
    
## Code Example

    import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

    # create dataset
X, y = make_blobs(
   n_samples=150, n_features=2,
   centers=3, cluster_std=0.5,
   shuffle=True, random_state=0
)

    # plot
plt.scatter(
   X[:, 0], X[:, 1],
   c='white', marker='o',
   edgecolor='black', s=50
)
plt.show()
from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)

    # plot the 3 clusters
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

    # plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

![image](https://user-images.githubusercontent.com/99629762/156688798-7fefc870-4c67-4a53-9326-6bcf42f8a8c4.png)


    
## Reinforcement learning

    Learns on the basis of reward + punishment for taking different actions
    
<img src="https://github.com/xprilion/random-storage/raw/master/images/dct1_3.png" 


    
    
    
    





