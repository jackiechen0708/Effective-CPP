import tensorflow as tf
from utils import masked_conv,dynamic_depth_residual_block,convolutional,get_model_searchable_params_helper,PlaceHolderFactory

def _channels_candidate_generator(n,step=16):
    """
    """
    return [16*i for i in range(1,(n//16) +1)]

def build_model(features):
    """
    build the searchable model given the input features
    features: input feature map( usually image)
    return: the routes feature map and the corresponding channels number [(route1,channel1),...]
    """
    select_phs = []
    routes = []
    x = convolutional(features, filters_shape=(3, 3,  3,  32), trainable=True, name='conv0')
    channels = [32,64,96,128,160,192]
    for i in range(5):
        x,wh_select_ph,c_select_ph = masked_conv(x,"downsample"+str(i),[7,7,channels[i],channels[i+1]],
                                                 [1,3,5,7],_channels_candidate_generator(channels[i]),strides=2)
        select_phs.append(wh_select_ph)
        select_phs.append(c_select_ph)
        # TODO: residual block may need pre-condition knowledge, to lower memory usage`
        # TODO: batch normal mean var gamma beta need mask or not ????
        x,tmp_select_phs = dynamic_depth_residual_block(x,"residual"+str(i),[7,7,channels[i+1],channels[i+1]],[1,3,5,7],_channels_candidate_generator(channels[i+1]),max_num_layer=3)
        select_phs += tmp_select_phs
        routes.append((x,channels[i+1]))
    return routes,select_phs

def nas_model_fn(features, labels, mode, params):
    """
    """
    routes, select_phs = build_model(features)
    x = routes[-1]


if __name__ == "__main__":
    import os
    import time
    import random
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    image = tf.get_variable(shape=[32,1280,704,3],name="image")
    print ("start building model ")
    start = time.time()
    routes , select_phs = build_model(image)
    end = time.time()
    print ("build model complete ",end - start)
    #placeholders, ranges = get_model_searchable_params_helper(select_phs)
    ph_factory = PlaceHolderFactory()
    placeholders,ranges = ph_factory.get_placeholder_all(),ph_factory.get_ranges()
    sample_range = [random.randint(0,i-1) for i in ranges]
    feed_dict = {placeholders:sample_range}
    #print (feed_dict)
    #feed_dict = {i:random.randint(0,j-1) for i,j in select_phs}
    print ('debug')
    print (len(feed_dict))
    print (len(ranges))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print (routes)
    #print(sess.run(select_phs[0][0],feed_dict))
    #print(sess.run(placeholders,feed_dict))
    #print(sess.run(routes,feed_dict))
    #train_writer = tf.summary.FileWriter('./train',sess.graph)
