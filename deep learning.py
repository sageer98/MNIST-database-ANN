#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


from tensorflow.examples.tutorials.mnist import input_data


# In[5]:


mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


# In[6]:


type(mnist)


# In[7]:


mnist.train.images


# In[8]:


mnist.train.num_examples


# In[9]:


mnist.test.num_examples


# In[10]:


mnist.validation.num_examples


# In[17]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


mnist.train.images.shape


# In[19]:


mnist.train.images[1].shape


# In[20]:


plt.imshow(mnist.train.images[1234].reshape(28,28))


# In[21]:


mnist.train.images[1].max()


# In[22]:


mnist.train.images[1].min()


# In[23]:


mnist.train.images[1]


# In[24]:


x=tf.placeholder(tf.float32,shape=[None,784])


# In[29]:


W=tf.Variable(tf.zeros([784,10]))


# In[26]:


b=tf.Variable(tf.zeros([10]))


# In[30]:


y=tf.matmul(x,W)+b


# In[31]:


y_true=tf.placeholder(tf.float32,[None,10])


# In[32]:


cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))


# In[33]:


optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5)


# In[34]:


train=optimizer.minimize(cross_entropy)


# In[35]:


init=tf.global_variables_initializer()


# In[37]:


with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        batch_x,batch_y=mnist.train.next_batch(100)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
    matches=tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    acc=tf.reduce_mean(tf.cast(matches,tf.float32))
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
    


# In[ ]:




