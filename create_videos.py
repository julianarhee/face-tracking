#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:52:35 2020

@author: julianarhee
"""
import os
import glob
import cv2
import re
import sys
import optparse

import numpy as np
import pylab as pl
import pandas as pd
import pickle as pkl


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    

#%%
def create_run_movies(datakey, experiment,
                      dst_dir='/n/coxfs01/julianarhee/face-tracking/videos',
                      src_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
    '''
    Find all raw images in src_dir for current dataset.
    
    datakey (str): 'YYYYMMDD_JC001'
    
    experiment (str): can be blobs, gratings, rfs, rfs10, retino
    
    Saves movies with format: 20190616_JC097_fov1_blobs_f1.mp4
    
    '''
    allowed_experiments = ['blobs', 'gratings', 'rfs', 'rfs10', 'retino']
    assert experiment in allowed_experiments, \
        "Invalid exp <%s> requested. Choices: %s" % (experiment, allowed_experiments)
    datakey_full = '%s_%s' % (datakey, experiment)
    
    errors=[]
    print("...creating movies for dset: %s" % datakey_full) 
    # Find subdirs for all experiments for this FOV
    curr_srcdirs = sorted(glob.glob(os.path.join(src_dir, '%s*' % (datakey_full)))) 
    for s_dir in curr_srcdirs:
        # Get metadata
        performance_info = os.path.join(s_dir, 'times', 'performance.txt') 
        try:
            metadata = pd.read_csv(performance_info, sep="\t ")
            fps = float(metadata['frame_rate'])
            print("Frame rate: %.2f Hz" % fps)
        except Exception as e:
            print(e)
            errors.append(s_dir)
            return errors
        
        # Create name for movie file
        runkey = s_dir.split(datakey)[1].split('_')[1]
        framedir = os.path.join(s_dir, 'frames')
        movfile = os.path.join(dst_dir, '%s_%s.mp4' % (datakey, runkey))
        print(movfile)
     
        if os.path.exists(movfile):
            print("--- file exists!  skipping.")
            continue
             
        cmd='ffmpeg -y -r ' + '%.3f' % fps + ' -i ' + framedir+'/%d.png -vcodec libx264 -f mp4 -pix_fmt yuv420p ' + movfile
        os.system(cmd)
        print("... ...done")
        
    return

# Load aggregate data info
def load_aggregate_data_info():
    aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
    datainfo_fpath = os.path.join(aggregate_dir, 'dataset_info_assigned.pkl')
    with open(datainfo_fpath, 'rb') as f:
        sdata = pkl.load(f, encoding='latin1')
    return sdata

def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir [default: /n/coxfs01/2pdata]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')

    # Set specific session/run for current animal:
    parser.add_option('-S', '--session', action='store', dest='session', default='', \
                      help='session dir (format: YYYMMDD_ANIMALID')
    
    # Set src/dst dirs:
    parser.add_option('-d', '--dest', action='store', dest='dst_dir', default='/n/coxfs01/julianarhee/face-tracking/videos', \
                      help='output dir [default: /n//coxfs01/julianarhee/face-tracking/videos')
    parser.add_option('-s', '--src', action='store', dest='src_dir', default='/n/coxfs01/2p-data/eyetracker_tmp', \
                      help='src dir [default: /n//coxfs01/2p-data/eyetracker_tmp')

    (options, args) = parser.parse_args(options)

    return options

#rootdir = '/n/coxfs01/2p-data'
#src_dir = os.path.join(rootdir, 'eyetracker_tmp')
#dst_dir = '/n/coxfs01/julianarhee/face-tracking/videos'

def main(options):

    opts = extract_options(options)

    if not os.path.exists(opts.dst_dir):
        os.makedirs(opts.dst_dir)
        print(opts.dst_dir)

    sdata = load_aggregate_data_info()
    expdf = sdata[(sdata['animalid']==opts.animalid)\
                    & (sdata['session']==opts.session)]
    print(expdf.head())

    for datakey, g in expdf.groupby(['datakey']):
        print("****%s: creating videos*****" % datakey)
        experiment_list = g['experiment'].unique()
        print("    Found %i experiments in curr fov" % len(experiment_list))
        for curr_exp in experiment_list:
            create_run_movies(datakey, curr_exp,
                              src_dir = opts.src_dir,
                              dst_dir = opts.dst_dir)
            print("    FINISHED")

    return


#%%        
if __name__ == '__main__':
    main(sys.argv[1:])


# %%
