#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import tensorflow as tf
import cv2
import os.path


class CnnHeadPoseEstimator:
    """ Head pose estimation class which uses convolutional neural network

        It finds Roll, Pitch and Yaw of the head given an head figure as input.
        It manages input (colour) picture larger than 64x64 pixels. The CNN are robust
        to variance in the input features and can handle occlusions and bad
        lighting conditions. The output values are in the ranges (degrees): 
        ROLL=[-40, +40]
        YAW=[-100, +100] 
    """

    def __init__(self, tf_session):
        """ Init the class

        @param tf_session An external tensorflow session
        """
        self._sess = tf_session

    def print_allocated_variables(self):
        """ Print all the Tensorflow allocated variables

        """
        all_vars = tf.all_variables()

        print("[DEEPGAZE] Printing all the Allocated Tensorflow Variables:")
        for k in all_vars:
            print(k.name)     

    def _allocate_yaw_variables(self):
        """ Allocate variables in memory (for internal use)
            
        The variables must be allocated in memory before loading
        the pretrained weights. In this phase empty placeholders
        are defined and later fill with the real values.
        """
        self._num_labels = 1
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_yaw_input_vector = tf.placeholder(tf.float32, shape=(64, 64, 3))
        
        # Variables.
        #Conv layer
        #[patch_size, patch_size, num_channels, depth]
        self.hy_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
        self.hy_conv1_biases = tf.Variable(tf.zeros([64]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hy_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.hy_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hy_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)) #was[3, 3, 128, 256]
        self.hy_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Dense layer
        #[ 5*5 * previous_layer_out , num_hidden] wd1
        #here 5*5 is the size of the image after pool reduction (divide by half 3 times)
        self.hy_dense1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 256], stddev=0.1)) #was [5*5*256, 1024]
        self.hy_dense1_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Dense layer
        #[ , num_hidden] wd2
        #self.hy_dense2_weights = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        #self.hy_dense2_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Output layer
        self.hy_out_weights = tf.Variable(tf.truncated_normal([256, self._num_labels], stddev=0.1))
        self.hy_out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]))

        # dropout (keep probability)
        #self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
 
        # Model.
        def model(data):

            X = tf.reshape(data, shape=[-1, 64, 64, 3])

            # Convolution Layer 1
            conv1 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(X, self.hy_conv1_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv1_biases))
            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm1 = tf.nn.dropout(norm1, _dropout)
 
            # Convolution Layer 2
            conv2 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hy_conv2_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv2_biases))
            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm2 = tf.nn.dropout(norm2, _dropout)

            # Convolution Layer 3
            conv3 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hy_conv3_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv3_biases))
            # Max Pooling (down-sampling)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # Fully connected layer 4
            dense1 = tf.reshape(norm3, [-1, self.hy_dense1_weights.get_shape().as_list()[0]]) # Reshape conv3
            dense1 = tf.tanh(tf.matmul(dense1, self.hy_dense1_weights) + self.hy_dense1_biases)

            #Fully connected layer 5
            #dense2 = tf.tanh(tf.matmul(dense1, self.hy_dense2_weights) + self.hy_dense2_biases)

            #Output layer 6
            out = tf.tanh(tf.matmul(dense1, self.hy_out_weights) + self.hy_out_biases)

            return out
 
        # Get the result from the model
        self.cnn_yaw_output = model(self.tf_yaw_input_vector)

 
    def load_yaw_variables(self, YawFilePath):
        """ Load varibles from a tensorflow file

        It must be called after the variable allocation.
        This function take the variables stored in a local file
        and assign them to pre-allocated variables.      
        @param YawFilePath Path to a valid checkpoint
        """

        #Allocate the variables in memory
        self._allocate_yaw_variables()

        #It is possible to use the checkpoint file
        #y_ckpt = tf.train.get_checkpoint_state(YawFilePath)
        #.restore(self._sess, y_ckpt.model_checkpoint_path) 

        #For future use, allocating a fraction of the GPU
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) #Allocate only half of the GPU memory

        if(os.path.isfile(YawFilePath)==False): raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(load_yaw_variables): the yaw file path is incorrect.')

        tf.train.Saver(({"conv1_yaw_w": self.hy_conv1_weights, "conv1_yaw_b": self.hy_conv1_biases,
                         "conv2_yaw_w": self.hy_conv2_weights, "conv2_yaw_b": self.hy_conv2_biases,
                         "conv3_yaw_w": self.hy_conv3_weights, "conv3_yaw_b": self.hy_conv3_biases,
                         "dense1_yaw_w": self.hy_dense1_weights, "dense1_yaw_b": self.hy_dense1_biases,
                         "out_yaw_w": self.hy_out_weights, "out_yaw_b": self.hy_out_biases
                        })).restore(self._sess, YawFilePath) 


    def return_yaw(self, image, radians=False):
         """ Return the yaw angle associated with the input image.

         @param image It is a colour image. It must be >= 64 pixel.
         @param radians When True it returns the angle in radians, otherwise in degrees.
         """
         #Uncomment if you want to see the image
         #cv2.imshow('image',image)
         #cv2.waitKey(0)
         #cv2.destroyAllWindows()
         h, w, d = image.shape
         #check if the image has the right shape
         if(h == w and h==64 and d==3):
             image_normalised = np.add(image, -127) #normalisation of the input
             feed_dict = {self.tf_yaw_input_vector : image_normalised}
             yaw_raw = self._sess.run([self.cnn_yaw_output], feed_dict=feed_dict)
             yaw_vector = np.multiply(yaw_raw, 100.0)
             #yaw = yaw_raw #* 100 #cnn out is in range [-1, +1] --> [-100, + 100]
             if(radians==True): return np.multiply(yaw_vector, np.pi/180.0) #to radians
             else: return yaw_vector
         #If the image is > 64 pixel then resize it
         if(h == w and h>64 and d==3):
             image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
             image_normalised = np.add(image_resized, -127) #normalisation of the input
             feed_dict = {self.tf_yaw_input_vector : image_normalised}
             yaw_raw = self._sess.run([self.cnn_yaw_output], feed_dict=feed_dict)       
             yaw_vector = np.multiply(yaw_raw, 100.0) #cnn-out is in range [-1, +1] --> [-100, + 100]
             if(radians==True): return np.multiply(yaw_vector, np.pi/180.0) #to radians
             else: return yaw_vector
         #wrong shape          
         if(h != w or w<64 or h<64):
             if h != w :
                raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_yaw): the image given as input has wrong shape. Height must equal Width. Height=%d,Width=%d'%(h,w))
             else:		
                raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_yaw): the image given as input has wrong shape. Height and Width must be >= 64 pixel')
         #wrong number of channels
         if(d!=3):
             raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_yaw): the image given as input does not have 3 channels, this function accepts only colour images.')

    def _allocate_pitch_variables(self):
        """ Allocate variables in memory (for internal use)
            
        The variables must be allocated in memory before loading
        the pretrained weights. In this phase empty placeholders
        are defined and later fill with the real values.
        """
        self._num_labels = 1
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_pitch_input_vector = tf.placeholder(tf.float32, shape=(64, 64, 3))
        
        # Variables.
        #Conv layer
        #[patch_size, patch_size, num_channels, depth]
        self.hp_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
        self.hp_conv1_biases = tf.Variable(tf.zeros([64]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hp_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.hp_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hp_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)) #was[3, 3, 128, 256]
        self.hp_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Dense layer
        #[ 5*5 * previous_layer_out , num_hidden] wd1
        #here 5*5 is the size of the image after pool reduction (divide by half 3 times)
        self.hp_dense1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 256], stddev=0.1)) #was [5*5*256, 1024]
        self.hp_dense1_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Dense layer
        #[ , num_hidden] wd2
        #self.hp_dense2_weights = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        #self.hp_dense2_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Output layer
        self.hp_out_weights = tf.Variable(tf.truncated_normal([256, self._num_labels], stddev=0.1))
        self.hp_out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]))

        # dropout (keep probability)
        #self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
 
        # Model.
        def model(data):

            X = tf.reshape(data, shape=[-1, 64, 64, 3])

            # Convolution Layer 1
            conv1 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(X, self.hp_conv1_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hp_conv1_biases))
            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm1 = tf.nn.dropout(norm1, _dropout)
 
            # Convolution Layer 2
            conv2 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hp_conv2_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hp_conv2_biases))
            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm2 = tf.nn.dropout(norm2, _dropout)

            # Convolution Layer 3
            conv3 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hp_conv3_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hp_conv3_biases))
            # Max Pooling (down-sampling)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # Fully connected layer 4
            dense1 = tf.reshape(norm3, [-1, self.hp_dense1_weights.get_shape().as_list()[0]]) # Reshape conv3
            dense1 = tf.tanh(tf.matmul(dense1, self.hp_dense1_weights) + self.hp_dense1_biases)

            #Fully connected layer 5
            #dense2 = tf.tanh(tf.matmul(dense1, self.hp_dense2_weights) + self.hp_dense2_biases) 
            #if(DEBUG == True): print("SHAPE dense2: " + str(dense2.get_shape()))

            #Output layer 6
            out = tf.tanh(tf.matmul(dense1, self.hp_out_weights) + self.hp_out_biases)

            return out
 
        # Get the result from the model
        self.cnn_pitch_output = model(self.tf_pitch_input_vector)

 
    def load_pitch_variables(self, pitchFilePath):
        """ Load varibles from a tensorflow file

        It must be called after the variable allocation.
        This function take the variables stored in a local file
        and assign them to pre-allocated variables.      
        @param pitchFilePath Path to a valid checkpoint
        """

        #Allocate the variables in memory
        self._allocate_pitch_variables()

        #It is possible to use the checkpoint file
        #y_ckpt = tf.train.get_checkpoint_state(pitchFilePath)
        #.restore(self._sess, y_ckpt.model_checkpoint_path) 

        #For future use, allocating a fraction of the GPU
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) #Allocate only half of the GPU memory

        if(os.path.isfile(pitchFilePath)==False): raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(load_pitch_variables): the pitch file path is incorrect.')

        tf.train.Saver(({"conv1_pitch_w": self.hp_conv1_weights, "conv1_pitch_b": self.hp_conv1_biases,
                         "conv2_pitch_w": self.hp_conv2_weights, "conv2_pitch_b": self.hp_conv2_biases,
                         "conv3_pitch_w": self.hp_conv3_weights, "conv3_pitch_b": self.hp_conv3_biases,
                         "dense1_pitch_w": self.hp_dense1_weights, "dense1_pitch_b": self.hp_dense1_biases,
                         "out_pitch_w": self.hp_out_weights, "out_pitch_b": self.hp_out_biases
                        })).restore(self._sess, pitchFilePath) 


    def return_pitch(self, image, radians=False):
         """ Return the pitch angle associated with the input image.

         @param image It is a colour image. It must be >= 64 pixel.
         @param radians When True it returns the angle in radians, otherwise in degrees.
         """
         #Uncomment if you want to see the image
         #cv2.imshow('image',image)
         #cv2.waitKey(0)
         #cv2.destroyAllWindows()
         h, w, d = image.shape
         #check if the image has the right shape
         if(h == w and h==64 and d==3):
             image_normalised = np.add(image, -127) #normalisation of the input
             feed_dict = {self.tf_pitch_input_vector : image_normalised}
             pitch_raw = self._sess.run([self.cnn_pitch_output], feed_dict=feed_dict)
             pitch_vector = np.multiply(pitch_raw, 45.0)
             #pitch = pitch_raw #* 40 #cnn out is in range [-1, +1] --> [-45, + 45]
             if(radians==True): return np.multiply(pitch_vector, np.pi/180.0) #to radians
             else: return pitch_vector
         #If the image is > 64 pixel then resize it
         if(h == w and h>64 and d==3):
             image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
             image_normalised = np.add(image_resized, -127) #normalisation of the input
             feed_dict = {self.tf_pitch_input_vector : image_normalised}
             pitch_raw = self._sess.run([self.cnn_pitch_output], feed_dict=feed_dict)       
             pitch_vector = np.multiply(pitch_raw, 45.0) #cnn-out is in range [-1, +1] --> [-45, + 45]
             if(radians==True): return np.multiply(pitch_vector, np.pi/180.0) #to radians
             else: return pitch_vector
         #wrong shape          
         if(h != w or w<64 or h<64):
             raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_pitch): the image given as input has wrong shape. Height and Width must be >= 64 pixel')
         #wrong number of channels
         if(d!=3):
             raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_pitch): the image given as input does not have 3 channels, this function accepts only colour images.')
