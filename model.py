import tensorflow as tf
from utils import masked_conv,dynamic_depth_residual_block,convolutional

def _channels_candidate_generator(n,step=16):
    """
    """
    return [16*i for i in range(1,(n//16) +1)]

def build_model(features):
    """
    """
    #def masked_conv(feature,name,kernel_shape,wh_mask_shape,c_mask_shape,strides,padding='SAME',
    #                bn = True,is_training=True,decay=0.9,epsilon=1e-5,
    #                activation = True,dtype = tf.float32,data_format="NHWC"):
    select_phs = []
    routes = []
    #x,wh_select_ph,c_select_ph = masked_conv(features,"original",[7,7,3,32],[1,3,5,7],[16,32],strides=1)
    #select_phs.append(wh_select_ph)
    #select_phs.append(c_select_ph)
    x = convolutional(features, filters_shape=(3, 3,  3,  32), trainable=True, name='conv0')
    #dynamic_depth_residual_block(input_data,name,kernel_shape,wh_mask_shape,c_mask_shape,
    #                bn = True, is_training = True,decay=0.9,epsilon=1e-5,
    #                activation = True,dtype = tf.float32,data_format="NHWC"
    #                max_num_layer=2):
    channels = [32,64,128,256,512,1024]
    for i in range(4):
        x,wh_select_ph,c_select_ph = masked_conv(x,"downsample"+str(i),[7,7,channels[i],channels[i+1]],
                                                 [1,3,5,7],_channels_candidate_generator(channels[i]),strides=2)
        select_phs.append(wh_select_ph)
        select_phs.append(c_select_ph)
        # TODO: residual block may need pre-condition knowledge, to lower memory usage`
        x,tmp_select_phs = dynamic_depth_residual_block(x,"residual"+str(i),[7,7,channels[i+1],channels[i+1]],[1,3,5,7],_channels_candidate_generator(channels[i+1]))
        select_phs += tmp_select_phs
    routes.append(x)
    return routes,select_phs

def nas_model_fn(features, labels, mode, params):
    """
    """
    routes, select_phs = build_model(features)
    x = routes[-1]


if __name__ == "__main__":
    import os
    import time
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    image = tf.get_variable(shape=[32,224,224,3],name="image")
    print ("start building model ")
    start = time.time()
    routes , select_phs = build_model(image)
    end = time.time()
    print ("build model complete ",end - start)
    feed_dict = {key:0 for key in select_phs}
    print (routes,select_phs)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(routes,feed_dict))
