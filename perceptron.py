from numpy import *

def plot_classification_2D(input_data, output_data, weights):
    """ Allows to plot 2-dimensional data (format (1, x1, x2)) using matplotlib, as well as the hyperplane defined by the weights (b, w1, w2). 
    """
    import matplotlib.pyplot as plt
    # Analyse training data
    xmin=min(input_data[:,1])
    xmax=max(input_data[:,1])
    ymin=min(input_data[:,2])
    ymax=max(input_data[:,2])
    # Plot the data
    positive_class=input_data[output_data>0]
    negative_class=input_data[output_data<=0]
    plt.plot(positive_class[:,1], positive_class[:,2], 'ro')
    plt.plot(negative_class[:,1], negative_class[:,2], 'bo')
    # Plot the hyperplane 
    x=linspace(xmin-0.1, xmax+0.1, 2)
    y=-weights[1]/weights[2] * x - weights[0]/weights[2]
    plt.plot(x, y, 'g-')
    plt.axis([xmin-0.1, xmax+0.1, ymin-0.1, ymax+0.1])
    plt.show()
 
def load_training_set(filename):
    """ Reads an input file called filename, containing the training examples in rows, the first column being always 1.0, and the last column being the output value.
    """
    input_data=[]
    output_data=[]
    with open(filename, 'r') as ifile:
        for line in ifile:
            values=line.split()
            input_value= [float(values[i]) for i in range(len(values)-1)]
            output_value= float(values[len(values)-1])
            input_data.append([float(values[i]) for i in range(len(values)-1)])
            output_data.append(float(values[len(values)-1]))
    return array(input_data), array(output_data)
 
 
def project(input_data, weights):
    """ Computes the projection of the input vector on the hyperplane defined by weights.
    """
    return sum(input_data*weights)
        
###########################################        
## Start of the perceptron algorithm ######
###########################################
# Value of the learning rate eta.
learning_rate = 0.9
# Retrieve the data set from the file.
input_data, output_data = load_training_set('xor.data')
# Initialize the weight vector to zero
weights = zeros(len(input_data[0])) 
# The initial number of errors made on the training set is equal to the number of examples in the training set.
error_count=len(input_data)
# Number of times the training set was used (called epoch). 
nb_epochs=0 

# Until there is no more errors on the training set, apply the perceptron learning rule
while error_count>0: 
    print '-' * 60
    error_count = 0
    # Apply the perceptron algorithm on the training set
    for index, input_vector in enumerate(input_data):
        desired_output=output_data[index]
        projection = project(input_vector, weights)
        result = 1.0 if projection > 0.0 else (-1.0 if projection< 0.0 else 0.0)
        if desired_output * projection <= 0:
            error_count += 1
            weights += learning_rate * (desired_output - result) * input_vector
        print 'input', input_vector, 'desired_output', desired_output, 'projection', projection, 'result', result, 'weights', weights
    # Increment the number of epochs 
    nb_epochs+=1
    
# Indicate the number of epochs needed to successfully classify the data.
print 'Learning successful after', nb_epochs, 'epochs'
# Show the data and the hyperplane in a figure.
plot_classification_2D(input_data, output_data, weights)
