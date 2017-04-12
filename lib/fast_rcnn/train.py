
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick	
# --------------------------------------------------------
# --------------------------------------------------------
# Self-Paced Weakly Supervised Fast R-CNN
# Written by Enver Sangineto and Moin Nabi, 2017.
# See LICENSE in the project root for license information.
# --------------------------------------------------------


"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os
from fast_rcnn.test import im_detect 
import cv2  
import copy  
import utils.cython_bbox  
from utils.cython_bbox import bbox_overlaps 

from caffe.proto import caffe_pb2
import google.protobuf as pb2


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
            
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)            
            
           
		# Self-Paced learning hyper-parameters ---------------------------------------------------------------------------------------
        self._curr_easy_sample_ratio= 0.5
        self._easy_sample_rel_increment= 0.1 
        self._next_SP_iter= 0
        self._n_SP_epochs= 2  
        self._class_selection= True
		# ----------------------------------------------------------------------------------------------------------------------------
		
        
        self._n_classes= int(self.solver.net.params['cls_score'][1].data.shape[0]) # Reads the total number of categories of the net
		
        check_roidb(roidb, True)
        self._General_roidb= roidb
       
        curr_roidb= self.get_curr_roidb()
        self.solver.net.layers[0].set_roidb(curr_roidb) 
        self._next_SP_iter += int((np.math.floor(len(self.solver.net.layers[0]._roidb)) / cfg.TRAIN.IMS_PER_BATCH) * self._n_SP_epochs) # sets the number of SGD iterations using the cardinality of the current roidb


    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """

        net = self.solver.net
             
        if cfg.TRAIN.BBOX_REG:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()
            
            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis]) 
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
			

    def get_curr_roidb(self):
        
        sorted_boxes, sorted_img_inds, sorted_img_labels= self.Roidb_detect_and_sort()             
        curr_roidb= self.Roidb_selection(sorted_boxes, sorted_img_inds, sorted_img_labels) 			  
        self.update_roidb(curr_roidb)
			 
        print 'Computing bounding-box regression targets...'
        self.bbox_means, self.bbox_stds = \
                rdl_roidb.add_bbox_regression_targets(curr_roidb, num_classes)
        print 'done'
        
        # replace zero elements with EPSILON
        self.bbox_stds[np.where(self.bbox_stds == 0)[0]] = cfg.EPS    		
	  
        return curr_roidb

            
    def Roidb_detect_and_sort(self):
            
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
            '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        caffemodel = os.path.join(self.output_dir, filename)
        prototxt = os.path.join(self.output_dir, 'test_voc.prototxt')
            
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(prototxt, str(caffemodel), caffe.TEST)
        
        roidb = self._General_roidb
        n_imgs = len(roidb)             

		# initializing local data structures:
        all_dets = np.array([0,0,0,0])
        all_scores = np.array([-1]) 
        all_img_inds = np.array([-1])        
        all_labels = np.array([0])
        
        for i in xrange(n_imgs):
               
            #name = '{:06}'.format(i)
            #print "Image # %s." % name                        
            im_name = roidb[i]['image']
            im = cv2.imread(im_name)

            if roidb[i]['flipped']:
                im = im[:, ::-1, :]

            bb_boxes = roidb[i]['boxes']						
            scores, pred_boxes = im_detect(net, im, bb_boxes)	
			
            curr_img_labels= roidb[i]['img_labels'] 
            
            # select the top-score box excluding the background class
            scores[:,0]= 0 # background scores deleted 			
            bb_i,cls_j = np.unravel_index(scores.argmax(), scores.shape)            

            if len(np.where(curr_img_labels == cls_j)[0]) == 1: # does cls_j belong to curr_img_labels ?
						
                latent_box = pred_boxes[bb_i, 4 * cls_j : 4 * (cls_j + 1)] 
                latent_box_score = scores[bb_i, cls_j] 
								   
                all_dets = np.vstack((all_dets, latent_box))
                all_scores = np.hstack((all_scores, latent_box_score))
                all_img_inds = np.hstack((all_img_inds, i))
                all_labels = np.hstack((all_labels, cls_j))
				
				
	  # class selection and pruning ------
        if self._class_selection:
            selected_cls= self.class_selection(all_labels)
            pruned_dets, pruned_scores, pruned_img_inds, pruned_labels= self.remove_weak_class_samples(selected_cls,all_dets,all_scores,all_img_inds,all_labels)
        else:
            pruned_dets= all_dets
            pruned_scores= all_scores
            pruned_img_inds= all_img_inds
            pruned_labels= all_labels			
        # -----------------------------------       
				
        score_sortedIndex = np.argsort(pruned_scores,axis=0)[::-1] 
        sorted_boxes= pruned_dets[score_sortedIndex].astype(int) 
        sorted_img_inds= pruned_img_inds[score_sortedIndex].astype(int)  
        sorted_img_labels= pruned_labels[score_sortedIndex].astype(int)  
        n_pruned_items= len(score_sortedIndex)
		       
        return sorted_boxes[0:n_pruned_items-1], sorted_img_inds[0:n_pruned_items-1], sorted_img_labels[0:n_pruned_items-1] # after sorting, the last element corresponds to the dummy initialization
		
		
    def class_selection(self,all_labels): 
 
        n_imgs = len(all_labels)
        cls_hist= np.zeros(self._n_classes) # this includes class 0
		
        for i in xrange(1, n_imgs): # this loop starts from 1 because the 0-element of all_labels corresponds to the dummy initialitation of all_dets ([0,0,0,0])
            winning_class = all_labels[i] 
            cls_hist[winning_class]+= 1 
            
        sorted_cls = np.argsort(cls_hist,axis=0)[::-1] 
        n_classes_to_select= round((self._n_classes - 1) * self._curr_easy_sample_ratio) # the number of classes to be selected does not include class 0
        selected_cls= sorted_cls[0:int(n_classes_to_select)]
        
        return selected_cls		
		
		
    def remove_weak_class_samples(self,selected_cls,all_dets,all_scores,all_img_inds,all_labels): 
        pruned_dets = np.array([0,0,0,0])
        pruned_scores = np.array([-1])
        pruned_img_inds = np.array([-1])        
        pruned_labels = np.array([0])
		
        n_imgs = len(all_labels)      
		
        for i in xrange(1, n_imgs): 
            if len(np.where(selected_cls == all_labels[i])[0]) == 1:  
                pruned_dets = np.vstack((pruned_dets, all_dets[i])) 
                pruned_scores = np.hstack((pruned_scores, all_scores[i])) 
                pruned_img_inds = np.hstack((pruned_img_inds, all_img_inds[i])) 
                pruned_labels = np.hstack((pruned_labels, all_labels[i])) 
        
        return pruned_dets, pruned_scores, pruned_img_inds, pruned_labels
		
		
    def Roidb_selection(self, sorted_boxes, sorted_img_inds, sorted_img_labels):
        
        new_roidb = []		
        n_imgs_to_select= int(np.floor(self._curr_easy_sample_ratio * len(sorted_img_inds)))               
			
        n= 0 # n. copied images so far
        i= 0 # n. items of sorted_img_inds processed so far
		
        while n < n_imgs_to_select: 
            k= sorted_img_inds[i]
            curr_img_roidb = copy.deepcopy(self._General_roidb[k])
            new_roidb.append(curr_img_roidb)
				
		 # add the pseudo-ground truth:
            curr_img_roidb['boxes'] = np.vstack((curr_img_roidb['boxes'],sorted_boxes[i]))
            curr_img_roidb['max_classes'] = np.hstack((curr_img_roidb['max_classes'],sorted_img_labels[i]))
            curr_img_roidb['gt_classes'] = np.hstack((curr_img_roidb['gt_classes'],sorted_img_labels[i]))
            curr_img_roidb['max_overlaps'] = np.hstack((curr_img_roidb['max_overlaps'],np.array([1])))
            
            i+= 1 
            n+= 1 

        return new_roidb
    
	
    def update_roidb(self,roidb): 
 
        n_imgs = len(roidb)
		
        for i in xrange(n_imgs):
            bb_boxes= roidb[i]['boxes'] # the pseudo-ground truth box is included (in position len(bb_boxes))
            g_ind= len(bb_boxes) - 1
            g_box = roidb[i]['boxes'][[g_ind]]  
            g_class = roidb[i]['gt_classes'][g_ind] 

            max_overlaps = bbox_overlaps(bb_boxes.astype(np.float), g_box.astype(np.float))
			
            roidb[i]['max_overlaps']= max_overlaps
            nonzero_inds = np.where(max_overlaps > 0)[0]
            zero_inds = np.where(max_overlaps == 0)[0]
            roidb[i]['max_classes'][zero_inds]= 0
            roidb[i]['max_classes'][nonzero_inds]= g_class			
			
                 			

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
		
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter  == self._next_SP_iter:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

                if self._curr_easy_sample_ratio < 1:
                    self._curr_easy_sample_ratio = round(self._curr_easy_sample_ratio + self._easy_sample_rel_increment, 2)
                    self._curr_easy_sample_ratio = min(1, self._curr_easy_sample_ratio)

                curr_roidb= self.get_curr_roidb()
                check_roidb(curr_roidb, False)
                self.solver.net.layers[0].set_roidb(curr_roidb)    

                self._next_SP_iter += int((np.math.floor(len(self.solver.net.layers[0]._roidb)) / cfg.TRAIN.IMS_PER_BATCH) * self._n_SP_epochs)



        if last_snapshot_iter != self.solver.iter: # last snapshot before exiting
            self.snapshot()
            


def check_roidb(roidb, general_roidb):
    num_images = len(roidb)
    for im_i in xrange(num_images):
        g_inds= np.where(roidb[im_i]['gt_classes'] > 0)[0] 

        max_classes= roidb[im_i]['max_classes']
        max_overlaps= roidb[im_i]['max_overlaps']
        
        if general_roidb:
            assert(len(g_inds) == 0)
            assert all(max_overlaps == 0)
            assert all(max_classes == 0)
            curr_img_labels= roidb[im_i]['img_labels']
            assert(len(curr_img_labels) > 0)

        else:
            assert(len(g_inds) == 1) 
            bb_inds= np.where(roidb[im_i]['gt_classes'] == 0)[0] 
            assert(bb_inds[len(bb_inds) -1] + 1 == g_inds[0]) # here we chack that the pseudo-ground truth is put on top of bb_inds 
            
			# other sanity checks ---
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)
			
		 


def get_training_roidb(imdb): 
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
