#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:41:41 2021

@author: julianarhee
"""
#%%
import sys
import os
import glob
import cv2
import re
import shutil
import traceback
import optparse
import pprint
pp = pprint.PrettyPrinter(indent=2, width=1)

import numpy as np
import pylab as pl
import pandas as pd
import dill as pkl


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

#recursively merge two folders including subfolders
def mergefolders(root_src_dir, root_dst_dir):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_dir)
               
    return

#%%
def load_metadata():
    aggregate_dir = '/n/coxfs01/julianarhee/aggregate-visual-areas'
    datainfo_fpath = os.path.join(aggregate_dir, 'dataset_info_assigned.pkl')
    with open(datainfo_fpath, 'rb') as f:
        sdata = pkl.load(f) #, encoding='latin1')
    return sdata

def identify_missing_raw(transfer_dsets, dst_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
    '''
    Identify which datasets + experiments are missing RAW datafiles.
    '''
    missing = []
    for (datakey, session, exp), _ in transfer_dsets.groupby(['datakey', 'session', 'experiment']):
        if exp[0] in ['x', 'X', '_'] or exp[-1] in ['x', 'X', '_']:
            continue
            
        if int(session) < 20190511 and 'rfs' in exp: # rfs is actually called gratings
            datakey_full = '%s_gratings' % datakey
        else:
            datakey_full = '%s_%s' % (datakey, exp) 
            
        curr_facedirs = sorted(glob.glob(os.path.join(dst_dir, '%s*' % (datakey_full))))
        if len(curr_facedirs)==0:
            #print(datakey)
            missing.append(datakey_full)    

    return missing 

#%%
def check_dirnames(dsets, experiment_list=[], 
                   src_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
    '''
    Check naming of subdirs containing raw eyetracker images.
    Return a) duplicate dirs for a given run, 
           b) wrong fname format (e.g., 'gratings_f3b')
    '''
    if len(experiment_list)==0:
        check_dsets = dsets.copy()
    else:
        check_dsets = dsets[dsets.experiment.isin(experiment_list)].copy()
        
    check_fnames = dict((k, []) for k in check_dsets['experiment'].unique())
    bad_fnames = []  
    for (experiment, datakey, session), g in check_dsets.groupby(['experiment', 'datakey', 'session']):
         
        if int(session) < 20190511 and 'rfs' in experiment: # rfs is actually called gratings
            experiment_name='gratings'
        else:
            experiment_name = experiment
            
        curr_dirs = sorted(glob.glob(os.path.join(src_dir, 
                                '%s_%s_f*' % (datakey, experiment_name)))) 
        
        run_to_file_lut = dict()
        for fp in curr_dirs:
            # First check that we can extract stimulus type and run number for name
            try:
                fparse = re.findall('_%s_f\d+_' % experiment, os.path.split(fp)[-1])
                assert len(fparse)>0, "... bad filename parsing: %s" % str(fparse)
                fnum = fparse[0].split('_')[2] 
            except Exception as e:
                #traceback.print_exc()
                #print("ERROR parsing fname: %s" % fp)
                bad_fnames.append(fp)
                continue    
            # Add identified run number to LUT 
            if fnum not in run_to_file_lut.keys():
                run_to_file_lut[fnum] = []
            run_to_file_lut[fnum].append(fp.split(src_dir)[1])
             
        # Check if any runs have >1 associated dir
        runs_to_check = [v for k, v in run_to_file_lut.items() if len(v)>1]
        if len(runs_to_check)>0:
            check_fnames[experiment].append({datakey: run_to_file_lut})
                
    return check_fnames, bad_fnames


def check_raw_datafiles(dsets, experiment_list=[],
                          data_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
   
    '''
    Cycle through datasets and identify raw data dirs with either:
        a) list: missing raw (need to find on external HD)
        b) dict: duplicate data dirs for 1 run
        c) list: unparsable filenames (e.g., 'gratings_f3b')  
    ''' 
  
    #%% Identify datasets with missing RAW   
    missing_raw = identify_missing_raw(dsets, dst_dir=data_dir)
    print("Missing raw data in src for %i datasets:" % len(missing_raw))
    for m in missing_raw:
        print('   %s' % m)
        
    #%% Check naming
    duplicate_fnames, bad_fnames = check_dirnames(dsets, 
                                            experiment_list=experiment_list, 
                                            src_dir=data_dir)
    print("Datasets with >1 found dir per run:")
    pp.pprint(duplicate_fnames)

    print("Found %i filepaths that I could not parse:" % len(bad_fnames))
    for p in bad_fnames:
        print('   %s' % p)
        
    return missing_raw, duplicate_fnames, bad_fnames



#%%
def extract_options(options):
    
    parser = optparse.OptionParser()

    parser.add_option('-R', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data',\
                      help='data root dir [default: /n/coxfs01/2pdata]')
   
    # Set src/dst dirs:
    parser.add_option('-D', '--dst', action='store', dest='dst_dir', 
                default='/n/coxfs01/2p-data/eyetracker_tmp', \
                help='dst dir for raw data (default: /n/coxfs01/2p-data/eyetracker_tmp)')
    parser.add_option('-S', '--src', action='store', dest='src_dir', 
                default=None, \
                help='src dir (default: None, specify an ext. hard drive)')
    parser.add_option('-e', '--experiment', action='store', dest='experiment', 
                default=None, \
                help='experiment name to check (default: None)')
    
    parser.add_option('--naming', action='store_true', dest='check_naming',
                default=False, 
                help='Cycle through specified datasets and find missing or funky naming (checks DST).')
    
    parser.add_option('--transfer', action='store_true', dest='do_transfer',
                default=False, 
                help='Transfer files from SRC to DST.')
    
    (options, args) = parser.parse_args(options)

    return options

#%%
rootdir = '/n/coxfs01/2p-data'
src_dir = os.path.join(rootdir, 'eyetracker_tmp')
dst_dir = '/n/coxfs01/julianarhee/face-tracking/videos'
experiment=None

def main(options):

    opts = extract_options(options)
    dst_dir = opts.dst_dir
    src_dir = opts.src_dir
    experiment = opts.experiment
    check_naming = opts.check_naming
    do_transfer = opts.do_transfer
     
    print("DST:\n    %s" % dst_dir)
    print("SRC:\n    %s" % src_dir)

    #%% Load metadata
    sdata = load_metadata()   

    # Select datasets to check
    if experiment is not None:
        dsets = sdata[sdata.experiment==experiment].copy()
    else:
        dsets = sdata.copy()
 
    if check_naming:
        experiment_list = dsets['experiment'].unique() 
        no_raw, duplicate_fnames, bad_fnames = check_raw_datafiles(sdata,
                                                        experiment_list=experiment_list,
                                                        data_dir=dst_dir)
        pp.pprint(duplicate_fnames)
        
    #pp = pprint.PrettyPrinter(indent=2, width=1, depth=10, )
    if do_transfer:
        # TODO
        print(" transfer not implemented ")
        pass        

    return


       
#%%     
if __name__ == '__main__':
    main(sys.argv[1:])

   
