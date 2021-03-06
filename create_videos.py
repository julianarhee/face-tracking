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


def create_run_movies(datakey, 
                      dst_dir='/n/coxfs01/julianarhee/face-tracking/videos',
                      src_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
    print("...creating movies for dset: %s" % datakey)
    curr_srcdirs = sorted(glob.glob(os.path.join(src_dir, '%s*' % (datakey))))
    
    for s_dir in curr_srcdirs:
        performance_info = os.path.join(s_dir, 'times', 'performance.txt')
        metadata = pd.read_csv(performance_info, sep="\t ")
        #metadata
        fps = float(metadata['frame_rate'])

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
    datainfo_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
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

    print(expdf)

    for (animalid, session, fov), g in expdf.groupby(['animalid', 'session', 'fov']):
        print("****%s-%s: creating videos for %s*****" % (animalid, session, fov))
        fovnum = int(fov.split('_')[0][3:])
        experiment_list = g['experiment'].unique()
        print("Found %i experiments in curr fov" % len(experiment_list))
        for curr_exp in experiment_list:
            datakey = '%s_%s_fov%i_%s' % (session, animalid, fovnum, curr_exp)
            create_run_movies(datakey,
                              src_dir = opts.src_dir,
                              dst_dir = opts.dst_dir)
            print("FINISHED")

        
if __name__ == '__main__':
    main(sys.argv[1:])



