import glob
import time
from PIL import Image
import numpy as np
import tensorflow as tf
from build_graph_for_variable_size import CNN4SR, Config

import argparse
import os
from os import listdir
import sys

import cv2
from tqdm import tqdm

import datetime

def evaluate(sess, model):
    path = '/Users/congying/cyWang/projects/github/IVideoSR/IVideoSR/RealTimeVideo/test_image/'
    eval_files = glob.glob('/Users/congying/cyWang/projects/github/IVideoSR/IVideoSR/RealTimeVideo/test_image/*')
    outputPath = '/Users/congying/cyWang/projects/github/IVideoSR/IVideoSR/RealTimeVideo/test_image/'
    for filepath in eval_files:
        filename = filepath.split('/')[-1]
        cv_real_time(path + filename, outputPath, sess, model, IS_REAL_TIME=True, UPSCALE_FACTOR=2)

def image_sr(imputImg, sess, model):
    img_for_eval = np.array(imputImg, dtype=np.float32) / 255.0
    outputImg = np.array(model.predict_on_batch(sess, [img_for_eval])) 
    outputImg = 255.0 * outputImg
    outputImg = np.reshape(outputImg, outputImg.shape[1:])
    outputImg = Image.fromarray(np.uint8(outputImg))
    return outputImg

def cv_real_time(inputVideo, outputPath, sess, model, IS_REAL_TIME=True, UPSCALE_FACTOR=2):
    DELAY_TIME=1
    videoCapture = cv2.VideoCapture(inputVideo)
    output_name = outputPath + 'test'
    if not IS_REAL_TIME:
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR)
        output_name = outputPath + 'pre_' + inputVideo.split('.')[0] + '.avi'
        videoWriter = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, size)

    success, frame = videoCapture.read()
    count = 0
    while success:
        # count = count+1
        # if count % 3 != 0:
        #     success, frame = videoCapture.read()
        #     continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        y, cb, cr = img.split()
        print (y)
        # out = model(image)
        # out = out.cpu()
        starttime = time.time()
        out = image_sr(img, sess, model)
        endtime = time.time()
        print ("test: " + str(endtime - starttime))
        # out_img_y = out.data[0]
        # out_img_y *= 255.0
        # out_img_y = out_img_y.clip(0, 255)
        # out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
        # out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        # out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        # out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        #out_img = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
        
        print (output_name+".jpg")
        starttime = time.time()
        out.save(output_name+".jpg")

        if IS_REAL_TIME:
            cv2.imshow('LR Video ', frame)
            cv2.imshow('SR Video ', cv2.imread(output_name+".jpg"))
            cv2.waitKey(1) 
            endtime = time.time()
            print ("show: " + str(endtime - starttime))
        else:
            # save video
            videoWriter.write(out_img)
        # next frame
        success, frame = videoCapture.read()

def main():
    config = Config()
    with tf.Graph().as_default() as graph:
        print ("Building model...")
        start = time.time()
        model = CNN4SR(config)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        print ("took {:.2f} seconds\n".format(time.time() - start))
    graph.finalize()

    with tf.Session(graph=graph) as session:
        session.run(init_op)
        print ("Restoring...")
        saver.restore(session, "../edsr_model/SR_Model_Epoch_18")
        evaluate(session, model)


if __name__ == '__main__':
    main()
