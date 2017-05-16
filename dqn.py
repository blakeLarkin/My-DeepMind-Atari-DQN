import state
import math
import numpy as np
import os
import tensorflow as tf 

gamma = .99

class GradientClippingOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, use_locking=False, name="GradientClipper"):
        super(GradientClippingOptimizer, self).__init__(use_locking, name)
        self.optimizer = optimizer

    def compute_gradients(self, *args, **kwargs):
        gradsAndVars = self.optimizer.compute_gradients(*args, **kwargs)
        clippedGradsAndVars = []
        for (grad, var) in gradsAndVars:
            if grad is not None:
                clippedGradsAndVars.append((tf.clip_by_value(grad, -1, 1), var))
            else:
                clippedGradsAndVars.append((grad, var))
        return clippedGradsAndVars

    def apply_gradients(self, *args, **kwargs):
        return self.optimizer.apply_gradients(*args, **kwargs)

class DeepQNet:
	def __init__(self, numActions, baseDir, args):

		self.numActions = numActions
		self.baseDir = baseDir
		self.saveModelFreq = args.save_model_freq
		self.targetModelUpdateFreq = args.target_model_update_freq
		self.normalizeWeights = args.normalize_weights

		self.staleSess = None

		tf.set_random_seed(123456)

		self.sess = tf.Session()

		assert (len(tf.all_variables()) == 0),"Expected zero variables"
        self.x, self.y = self.buildNetwork('policy', True, numActions)
        assert (len(tf.trainable_variables()) == 10),"Expected 10 trainable_variables"
        assert (len(tf.all_variables()) == 10),"Expected 10 total variables"
        self.x_target, self.y_target = self.buildNetwork('target', False, numActions)
        assert (len(tf.trainable_variables()) == 10),"Expected 10 trainable_variables"
        assert (len(tf.all_variables()) == 20),"Expected 20 total variables"

        self.updateTarget = []
        trainableVariables = tf.trainable_variables()
        allVariables = tf.all_variables()
        for i in range(0, len(trainableVariables)):
        	self.updateTarget.append(allVariables[len(TrainableVariables) + i].assign(trainableVariables[i]))

		self.a = tf.placeholder(tf.float32, shape=[None, numActions])
        print('a %s' % (self.a.get_shape()))
        self.y_ = tf.placeholder(tf.float32, [None])
        print('y_ %s' % (self.y_.get_shape()))

        self.y_a = tf.reduce_sum(tf.mul(self.y, self.a), reduction_indices=1)
        print('y_a %s' % (self.y_a.get_shape()))

        difference = tf.abs(self.y_a - self.y_)
        quadraticPart = tf.clip_by_value(difference, 0.0, 1.0)
        linearPart = difference - quadraticPart
        errors = (0.5 * tf.square(quadraticPart)) + linearPart
        self.loss = tf.reduce_sum(errors)

        optimizer = tf.train.RMSPropOptimizer(args.learningRate, decay = .95, epsilon = .01)
        self.trainStep = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep = 25)

        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.updateTarget)

        self.summaryWriter = tf.train.SummaryWriter(self.baseDir + '/tensorboard', self.sess.graphDef)

    	if args.model is not None:
    		print('Loading from model file %s' % (args.model))
            self.saver.restore(self.sess, args.model)

    def buildNetwork(self, name, trainable, numActions):

    	print("Building network for %s trainable=%s" % (name, trainable))

        # First layer takes a screen, and shrinks by 2x
        x = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4], name="screens")
        print(x)

        xNormalized = tf.to_float(x) / 255.0
        print(xNormalized)

        # Second layer convolves 32 8x8 filters with stride 4 with relu
        with tf.variable_scope("cnn1_" + name):
            WConv1, bConv1 = self.makeLayerVariables([8, 8, 4, 32], trainable, "conv1")

            hConv1 = tf.nn.relu(tf.nn.conv2d(xNormalized, WConv1, strides=[1, 4, 4, 1], padding='VALID') + bConv1, name="h_conv1")
            print(hConv1)

        # Third layer convolves 64 4x4 filters with stride 2 with relu
        with tf.variable_scope("cnn2_" + name):
            WConv2, bConv2 = self.makeLayerVariables([4, 4, 32, 64], trainable, "conv2")

            hConv2 = tf.nn.relu(tf.nn.conv2d(hConv1, WConv2, strides=[1, 2, 2, 1], padding='VALID') + bConv2, name="h_conv2")
            print(hConv2)

        # Fourth layer convolves 64 3x3 filters with stride 1 with relu
        with tf.variable_scope("cnn3_" + name):
            WConv3, bConv3 = self.makeLayerVariables([3, 3, 64, 64], trainable, "conv3")

            hConv3 = tf.nn.relu(tf.nn.conv2d(hConv2, WConv3, strides=[1, 1, 1, 1], padding='VALID') + bConv3, name="h_conv3")
            print(hConv3)

        hConv3Flat = tf.reshape(hConv3, [-1, 7 * 7 * 64], name="h_conv3_flat")
        print(hConv3Flat)

        # Fifth layer is fully connected with 512 relu units
        with tf.variable_scope("fc1_" + name):
            WFc1, bFc1 = self.makeLayerVariables([7 * 7 * 64, 512], trainable, "fc1")

            hFc1 = tf.nn.relu(tf.matmul(hConv3Flat, WFc1) + bFc1, name="h_fc1")
            print(hFc1)

        # Sixth (Output) layer is fully connected linear layer
        with tf.variable_scope("fc2_" + name):
            WFc2, bFc2 = self.makeLayerVariables([512, numActions], trainable, "fc2")

            y = tf.matmul(hFc1, WFc2) + bFc2
            print(y)
            
        return x, y

    def makeLayerVariables(self, shape, trainable, nameSuffix):
    	if self.normalizeWeights:
            #TODO: Check against Torch's linear and spatial convolution
            stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
            weights = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + nameSuffix)
            biases  = tf.Variable(tf.random_uniform([shape[-1]], minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + nameSuffix)
        else:
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), trainable=trainable, name='W_' + nameSuffix)
            biases  = tf.Variable(tf.fill([shape[-1]], 0.1), trainable=trainable, name='W_' + nameSuffix)
        return weights, biases
        
    def inference(self, screens):
        y = self.sess.run([self.y], {self.x: screens})
        qValues = np.squeeze(y)
        return np.argmax(qValues)
        
    def train(self, batch, stepNumber):

        x2 = [b.state2.getScreens() for b in batch]
        y2 = self.yTarget.eval(feed_dict={self.xTarget: x2}, session=self.sess)

        x = [b.state1.getScreens() for b in batch]
        a = np.zeros((len(batch), self.numActions))
        y_ = np.zeros(len(batch))
        
        for i in range(0, len(batch)):
            a[i, batch[i].action] = 1
            if batch[i].terminal:
                y_[i] = batch[i].reward
            else:
                y_[i] = batch[i].reward + gamma * np.max(y2[i])

        self.train_step.run(feed_dict={
            self.x: x,
            self.a: a,
            self.y_: y_
        }, session=self.sess)

        if stepNumber % self.targetModelUpdateFreq == 0:
			self.sess.run(self.updateTarget)

        if stepNumber % self.targetModelUpdateFreq == 0 or stepNumber % self.saveModelFreq == 0:
            dir = self.baseDir + '/models'
            if not os.path.isdir(dir):
                os.makedirs(dir)
            savedPath = self.saver.save(self.sess, dir + '/model', global_step=stepNumber)
