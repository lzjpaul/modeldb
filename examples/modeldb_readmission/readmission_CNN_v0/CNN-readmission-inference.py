# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================

# all the data are read from csv not from URL
# occlude test are added here

# modify
# 6-20: occlude: consider samples as well
import sys, os
import traceback
import time
import urllib
import numpy as np
from argparse import ArgumentParser
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

from singa import tensor, device, optimizer
from singa import utils
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType
from data_loader import *
# from explain_occlude_area import *
from explain_occlude_area_format_out import *
from healthcare_metrics import *
import model
import json

def main():
    '''Command line options'''
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Train CNN Readmission Model")
        parser.add_argument('-inputfolder', type=str, help='inputfolder')
        parser.add_argument('-outputfolder', type=str, help='outputfolder')
        parser.add_argument('-visfolder', type=str, help='visfolder')
        parser.add_argument('-sampleid', type=int, help='the sample id for output')
        parser.add_argument('-p', '--port', default=9989, help='listening port')
        parser.add_argument('-C', '--use_cpu', action="store_true")
        parser.add_argument('--max_epoch', default=10)

        # Process arguments
        args = parser.parse_args()
        port = args.port

        use_cpu = args.use_cpu
        if use_cpu:
            print "runing with cpu"
            dev = device.get_default_device()
        else:
            print "runing with gpu"
            dev = device.create_cuda_gpu()

        # start to train
        agent = Agent(port)
        train(args.inputfolder, args.outputfolder, args.visfolder, args.sampleid, dev, agent, args.max_epoch, use_cpu)
        # wait the agent finish handling http request
        agent.stop()
    except SystemExit:
        return
    except:
        # p.terminate()
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")

def get_occlude_data(occlude_feature, occlude_label, height, width, height_idx, width_idx, kernel_y, kernel_x, stride_y, stride_x):
    '''load occlude data'''
    for n in range(occlude_feature.shape[0]): #sample
        for j in range (kernel_y):
            occlude_feature[n, ((height_idx * stride_y + j) * width + width_idx * stride_x) : ((height_idx * stride_y + j) * width + width_idx * stride_x + kernel_x)] = float(0.0)
    return occlude_feature, occlude_label

def handle_cmd(agent):
    pause = False
    stop = False
    while not stop:
        key, val = agent.pull()
        if key is not None:
            msg_type = MsgType.parse(key)
            if msg_type.is_command():
                if MsgType.kCommandPause.equal(msg_type):
                    agent.push(MsgType.kStatus,"Success")
                    pause = True
                elif MsgType.kCommandResume.equal(msg_type):
                    agent.push(MsgType.kStatus, "Success")
                    pause = False
                elif MsgType.kCommandStop.equal(msg_type):
                    agent.push(MsgType.kStatus,"Success")
                    stop = True
                else:
                    agent.push(MsgType.kStatus,"Warning, unkown message type")
                    print "Unsupported command %s" % str(msg_type)
        if pause and not stop:
            time.sleep(0.1)
        else:
            break
    return stop


def get_lr(epoch):
    '''change learning rate as epoch goes up'''
    return 0.001


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1).reshape(-1, 1)

def cal_accuracy(yPredict, yTrue):
    return np.sum(((yPredict > 0.5) == yTrue).astype(int)) / float(yTrue.shape[0])

def auroc(yPredictProba, yTrue):
    return roc_auc_score(yTrue, yPredictProba)

def train(inputfolder, outputfolder, visfolder, sampleid, dev, agent, max_epoch, use_cpu, batch_size=100):
    opt = optimizer.SGD(momentum=0.8, weight_decay=0.01)
    agent.push(MsgType.kStatus, 'Downlaoding data...')
    # all_feature, all_label = get_data(os.path.join(inputfolder, 'features_inference.txt'), os.path.join(inputfolder, 'label_inference.txt'))  # PUT THE DATA on/to dbsystem
    all_feature, all_label = get_data(os.path.join(inputfolder, 'features_inference.txt'), os.path.join(inputfolder, 'label_inference.txt'))  # PUT THE DATA on/to dbsystem
    agent.push(MsgType.kStatus, 'Finish downloading data')
    n_folds = 5
    all_label = all_label[:,1]
    for i, (train_index, test_index) in enumerate(StratifiedKFold(all_label.reshape(all_label.shape[0]), n_folds=n_folds)):
        test_index = np.arange(0,351)
        test_feature, test_label = all_feature[test_index], all_label[test_index]
        if i == 0:
            print "fold: ", i
            break
    print "test label sum: ", test_label.sum()
    in_shape = np.array([1, 12, 375])
    testx = tensor.Tensor((test_feature.shape[0], in_shape[0], in_shape[1], in_shape[2]), dev)
    testy = tensor.Tensor((test_feature.shape[0], ), dev, core_pb2.kInt)
    # num_test_batch = test_x.shape[0] / (batch_size)

    # height = 12
    # width = 375
    # kernel_y = 3
    # kernel_x = 80
    # stride_y = 1
    # stride_x = 20
    hyperpara = np.array([12, 375, 3, 10, 1, 3])
    height, width, kernel_y, kernel_x, stride_y, stride_x = hyperpara[0], hyperpara[1], hyperpara[2], hyperpara[3], hyperpara[4], hyperpara[5]
    net = model.create_net(in_shape, hyperpara, use_cpu)
    print "checkpoint path: ", os.path.join(os.path.join(os.environ.get('GEMINI_HOME'),'model/readmission_CNN_code/'), 'parameter_last240')
    net.load(os.path.join(os.path.join(os.environ.get('GEMINI_HOME'),'model/readmission_CNN_code/'), 'parameter_last240'), 20)
    net.to_device(dev)
    for name in zip(net.param_names()):
        print "init names: ", name
    
    test_epoch = 10
    occlude_test_epoch = 100
    for epoch in range(max_epoch):
        if handle_cmd(agent):
            break
        np.random.seed(10)
        print 'Epoch %d' % epoch
        
        # loss, acc = 0.0, 0.0
        # for b in range(num_train_batch):
        #     x, y = train_feature[b * batch_size:(b + 1) * batch_size], train_label[b * batch_size:(b + 1) * batch_size]
        #    x = x.reshape((batch_size, in_shape[0], in_shape[1], in_shape[2]))
        #     trainx.copy_from_numpy(x)
        #    trainy.copy_from_numpy(y)
        #     grads, (l, a), probs = net.train(trainx, trainy)
        #     loss += l
        #    acc += a
        #     for (s, p, g) in zip(net.param_specs(),
        #                          net.param_values(), grads):
        #         opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s.name))
        #     info = 'training loss = %f, training accuracy = %f' % (l, a)
        #     utils.update_progress(b * 1.0 / num_train_batch, info)
        # put training status info into a shared queue
        # info = dict(phase='train', step=epoch,
        #             accuracy=acc/num_train_batch,
        #             loss=loss/num_train_batch,
        #             timestamp=time.time())
        # agent.push(MsgType.kInfoMetric, info)
        # info = 'training loss = %f, training accuracy = %f' \
        #     % (loss / num_train_batch, acc / num_train_batch)
        # print info
        
        if epoch % test_epoch == 0 or epoch == (max_epoch-1):
            loss, acc = 0.0, 0.0
            x, y = np.copy(test_feature), np.copy(test_label)
            x = x.reshape((x.shape[0], in_shape[0], in_shape[1], in_shape[2]))
            testx.copy_from_numpy(x)
            testy.copy_from_numpy(y)
            l, a, probs = net.evaluate(testx, testy)
            loss += l
            acc += a
            print 'testing loss = %f, accuracy = %f' % (loss, acc)
            # put test status info into a shared queue
            info = dict(
                phase='test',
                step = epoch,
                accuracy = acc,
                loss = loss,
                timestamp = time.time())
            agent.push(MsgType.kInfoMetric, info)
            print 'self calculate test auc = %f' % auroc(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))
            print 'self calculate test accuracy = %f' % cal_accuracy(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))
            cnn_metric_dict = {} # for output to json
            cnn_metric_dict['Number of Samples: '] = y.shape[0]
            cnn_sensitivity, cnn_specificity, cnn_harmonic = HealthcareMetrics(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1), 0.25)
            cnn_metric_dict['AUC: '] = auroc(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))
            # cnn_metric_dict['accuracy: '] = cal_accuracy(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))
            cnn_metric_dict['Sensitivity: '] = cnn_sensitivity
            cnn_metric_dict['Specificity: '] = cnn_specificity
            
            try:
                with open(os.path.join(visfolder, 'cnn_metric_info.json'), 'w') as cnn_metric_info_writer:
                    # json.dump(cnn_metric_dict, cnn_metric_info_writer)
                    cnn_metric_info_writer.write('[')
                    cnn_metric_info_writer.write('\"Number of Patients: %d\", '%(y.shape[0]))
                    cnn_metric_info_writer.write('\"AUC: %s\", '% (str(int(100 * round((auroc(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))),2)))+'%') )
                    cnn_metric_info_writer.write('\"Sensitivity: %s\", '%(str(int(100 * round(cnn_sensitivity,2)))+'%'))
                    cnn_metric_info_writer.write('\"Specificity: %s\" '%(str(int(100* round(cnn_specificity,2)))+'%'))
                    cnn_metric_info_writer.write(']')
            except Exception as e:
                os.remove(os.path.join(visfolder, 'cnn_metric_info.json'))
                print('output cnn_metric_info.json failed: ', e)
            if epoch == (max_epoch-1):
                np.savetxt(os.path.join(os.path.join(os.environ.get('GEMINI_HOME'),'model/readmission_CNN_code/'),'readmitted_prob.csv'), softmax(tensor.to_numpy(probs))[:,1], fmt = '%6f', delimiter=",")
        
        # if epoch == (max_epoch-1):
        if epoch == (max_epoch):
            print "occclude test"
            # occlude test data
            height_dim = (height - kernel_y) / stride_y + 1; 
            width_dim = (width - kernel_x) / stride_x + 1;
            meta_data = np.array([height_dim, height, kernel_y, stride_y, width_dim, width, kernel_x, stride_x])
            np.savetxt(os.path.join(outputfolder,'meta_data.csv'), meta_data, fmt = '%6f', delimiter=",") #modify here
            true_label_prob_matrix = np.zeros([(height_dim * width_dim), 1])
            for height_idx in range(height_dim):
                for width_idx in range(width_dim):
                    occlude_test_feature, occlude_test_label = get_occlude_data(np.copy(test_feature), np.copy(test_label), \
                    height, width, height_idx, width_idx, kernel_y, kernel_x, stride_y, stride_x)
                    loss, acc = 0.0, 0.0
                    x, y = occlude_test_feature, occlude_test_label # !!! where are the labels?
                    x = x.reshape((x.shape[0], in_shape[0], in_shape[1], in_shape[2]))
                    testx.copy_from_numpy(x)
                    testy.copy_from_numpy(y)
                    l, a, probs = net.evaluate(testx, testy)
                    y_scores = softmax(tensor.to_numpy(probs))[:,1]
                    sum_true_label_prob = 0.0
                    for i in range(0, x.shape[0]): # !!! y_scores ~~ the probability of 1 !!!
                        if y[i] == 1:
                            sum_true_label_prob = sum_true_label_prob + y_scores[i]
                        elif y[i] == 0:
                            sum_true_label_prob = sum_true_label_prob + (1 - y_scores[i])
                    true_label_prob_matrix[height_idx * width_dim + width_idx, 0] = sum_true_label_prob / x.shape[0]
            print "occlude x shape: ", x.shape
            np.savetxt(os.path.join(os.path.join(os.environ.get('GEMINI_HOME'),'model/readmission_CNN_code/'),'true_label_prob_matrix.csv'), true_label_prob_matrix, fmt = '%6f', delimiter=",") #modify here
    #for (s, p) in zip(net.param_specs(), net.param_values()):
    #    print "last epoch param name: ", s
    #    print "last epoch param value: ", p.l2()
    # net.save('parameter_last')
    print "begin explain"
    # explain_occlude_area(np.copy(test_feature), np.copy(test_label), os.path.join(outputfolder,'readmitted_prob.csv'), os.path.join(outputfolder,'true_label_prob_matrix.csv'), os.path.join(outputfolder,'meta_data.csv'), top_n = 20)
    print "begin explain format out"
    top_n = 30
    print "top_n: ", top_n
    explain_occlude_area_format_out(sampleid, visfolder, np.copy(test_feature), np.copy(test_label), os.path.join(os.path.join(os.environ.get('GEMINI_HOME'),'model/readmission_CNN_code/'),'readmitted_prob.csv'), os.path.join(os.path.join(os.environ.get('GEMINI_HOME'),'model/readmission_CNN_code/'),'true_label_prob_matrix.csv'), os.path.join(os.path.join(os.environ.get('GEMINI_HOME'),'model/readmission_CNN_code/'),'meta_data.csv'), top_n = top_n)

 
if __name__ == '__main__':
    main()
