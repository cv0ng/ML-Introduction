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


    
## Reinforcement learning

    Learns on the basis of reward + punishment for taking different actions
    
    
    
    





