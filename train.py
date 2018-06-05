import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=np.inf)


print("#####################################################")
print("# Deep learning on parametrized Stokes equation ")
print("#####################################################")

############################### Data loading ############################

print("# Loading input data...")
inputs = np.loadtxt("trial/input.dat", ndmin=2);
ni, si = inputs.shape;
print("# no. inputs ", ni)
print("# size of each input ", si)

print("# Loading output data...")
outputs = np.loadtxt("trial/output.dat", ndmin=2);
outputsmax = np.max(outputs);
outputsmin = np.min(outputs);
#outputs = (outputs - outputsmin)/(outputsmax-outputsmin);

no, so = outputs.shape;
print("# no. outputs ", no)
print("# size of each output ", so)

if no != ni:
    print("ERROR! number of output differs from number of inputs... exiting")
    sys.exit(1);
else:
    print("# Input and Output size are matching.")


# lecture de la matrice d'energie
print("# Loading Nrj Norm...")
rawnrj = np.loadtxt("tmp/nrj",ndmin=2,skiprows=4);
n = int(rawnrj.max(axis=0)[0]);

if so != n:
    print("[ERROR] output size differs from nrj size... exiting.")
    sys.exit(1);
    
nrj = np.zeros((n,n),dtype=float);
for row in rawnrj:
    i=int(row[0]-1);
    j=int(row[1]-1);
    c=row[2];
    nrj[i][j]=c;
#plt.spy(nrj)
#plt.show()
rawnrj = [];
print("# Convert to Tensor...")
tf_A = tf.convert_to_tensor(nrj, np.float64);
nrj = [];

# lecture des dofs
dofs = np.loadtxt("tmp/dofs.dat");
ndofs,mdofs = dofs.shape;
if 2*ndofs != n:
    print("# dofs size are different than output size ! Aborting...")    
    sys.exit(1);

pX = dofs[:,0]
pY = dofs[:,1]

#writing js/mesh.js
meshfile = open("live/js/mesh.js","w");
meshfile.write("dofs="+np.array2string(dofs, separator=",") + ";\n\n");
meshfile.close();
print("# live/js/mesh.js created !")
dofs=[];

############################### Neural network ############################

print("# Setting Neural network")
x = tf.placeholder(tf.float64, shape=(None, si));
y = tf.placeholder(tf.float64, shape=(None, so));

nhl = 32 # size of each hidden layer
h_size = [nhl, nhl, nhl, nhl, nhl, nhl, nhl] # number of hidden layers and their sizes
W = [] # weights 
b = [] # biases
layer = [] # layer container

# first layer
W.append(tf.Variable(tf.random_normal([si, h_size[0]], stddev=0.1,  dtype=tf.float64)))
b.append(tf.Variable(tf.zeros([1, h_size[0]], dtype=tf.float64)))         

# hidden layers (variable number)
for i in range(1,len(h_size)):
    W.append(tf.Variable(tf.random_normal([h_size[i-1], h_size[i]], stddev=0.1, dtype=tf.float64)))
    b.append(tf.Variable(tf.zeros([1, h_size[i]], dtype=tf.float64)))

# final layer
W.append(tf.Variable(tf.random_normal([h_size[-1], so], stddev=0.1,  dtype=tf.float64)))
b.append(tf.Variable(tf.zeros([1, so],  dtype=tf.float64)))

# feedforward 
layer.append(tf.nn.tanh(tf.matmul(x, W[0]) + b[0]));
for i in range(1,len(h_size)):
    layer.append(tf.nn.tanh(tf.matmul(layer[i-1], W[i]) + b[i]))
yhat = tf.nn.tanh(tf.matmul(layer[-1], W[-1]) + b[-1])

# rescaled y
#yf = (outputsmax-outputsmin) * yhat + outputsmin;

# loss function
loss = 0.5*(1/no)*tf.trace((tf.matmul(tf.matmul(y-yhat, tf_A),tf.transpose(y-yhat))));
#loss = tf.nn.l2_loss(y-yhat)

# Trainer
learning_rate = 0.01;
#train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) # train formula
train = tf.train.AdamOptimizer(learning_rate).minimize(loss) # train formula

# start session
tol = 0.05;
err =  1e300;
#plt.ion()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    err0 = sess.run(loss, feed_dict={x:inputs, y:outputs})
    
    print("\n\nTraining NN ...")
    print("err0 =", err0)
    while err > tol:
        # train
        sess.run(train, feed_dict={x:inputs, y:outputs})
        
        # error
        U = sess.run(yhat, feed_dict={x:[[1., 1., 1., 1.]]})
        #U = sess.run(yf, feed_dict={x:[[0.3, -0.3, 0.7, -0.7]]})
        err = sess.run(loss, feed_dict={x:inputs, y:outputs})/err0
        print ("err ", err)

        #### plot
        m = int(n/2)
        UX = U[0,0:m]
        UY = U[0,m:n]
        M = np.hypot(UX, UY)
        
        #plt.cla()
        #plt.quiver(pX, pY, UX, UY, M, linewidth=0.005, width=0.001,cmap=plt.cm.rainbow)
        #plt.pause(0.0001)
        #plt.draw()
        
    print("==== TRAINING DONE ====")

    # Exporting the neural net
    tfile = open("live/js/data.js", "w");
    tfile.write("np=" + str(int(np.size(U)/2)) + ";\n\n");

    i = 0;
    for poids in W:
        i+=1;
        print("--> write w"+str(i))
        tfile.write("w"+str(i)+"="+np.array2string(poids.eval(), separator=",") + ";\n\n");
        
    i = 0;
    for biais in b:
        i +=1;
        print("--> write b"+str(i))
        tfile.write("b"+str(i)+"="+np.array2string(biais.eval(), separator=",") + ";\n\n");

    tfile.close();
    print("NeuralNet cfg imported to ./live/js/data.js ")
