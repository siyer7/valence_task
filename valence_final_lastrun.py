﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Fri Apr 11 17:08:27 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from block_start_code
import random, numpy as np
# Run 'Before Experiment' code from task_code
from psychopy.hardware import keyboard
kb = keyboard.Keyboard()
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'asymmetry_final'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'subj': '',
    'difficulty': 'patients or hard',
    'sess_type': 'A or B or C or D',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'results/%s_subj-%s_difficulty-%s_order-%s_%s' % (expName, expInfo['subj'], expInfo['difficulty'], expInfo['sess_type'], expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/f0064z8/Documents/code_repos/valence_task/valence_final_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        expInfo['frameRate'] = 120
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('block_start_resp') is None:
        # initialise block_start_resp
        block_start_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='block_start_resp',
        )
    if deviceManager.getDevice('slider_resp') is None:
        # initialise slider_resp
        slider_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='slider_resp',
        )
    if deviceManager.getDevice('submit_resp') is None:
        # initialise submit_resp
        submit_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='submit_resp',
        )
    if deviceManager.getDevice('block_end_resp') is None:
        # initialise block_end_resp
        block_end_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='block_end_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "block_start" ---
    # Run 'Begin Experiment' code from block_start_code
    block_outcome, block_bonus, disp_blockN, block_id = 0, 0, 0, 0
    
    if expInfo['sess_type'] in ['A','B']:
        block_order = [0,1,2,4,3,5]
    elif expInfo['sess_type'] in ['C','D']:
        block_order = [0,2,1,3,4,5]
    block_start_text = visual.TextStim(win=win, name='block_start_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    block_start_resp = keyboard.Keyboard(deviceName='block_start_resp')
    
    # --- Initialize components for Routine "baseline" ---
    ISI1 = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI1')
    # Run 'Begin Experiment' code from base_code
    base_dur = 0
    
    # --- Initialize components for Routine "stim" ---
    target_stim = visual.ImageStim(
        win=win,
        name='target_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=[.6,.6],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "delay" ---
    ISI2 = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI2')
    # Run 'Begin Experiment' code from delay_code
    delay_dur = 0
    
    # --- Initialize components for Routine "task" ---
    # Run 'Begin Experiment' code from task_code
    leftPressed, rightPressed, marker_moved = 0, 0, 0
    positions = []
    marker_move = .004
    debug_task_txt = ''
    img1 = visual.ImageStim(
        win=win,
        name='img1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[-.6, 0], draggable=False, size=[.45,.45],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    img2 = visual.ImageStim(
        win=win,
        name='img2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[.6, 0], draggable=False, size=[.45,.45],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    divider_line = visual.Line(
        win=win, name='divider_line',
        size=[.4,0],
        ori=90.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=3.0,
        colorSpace='rgb', lineColor=[1.0000, 1.0000, 1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-3.0, interpolate=True)
    slider_line = visual.Line(
        win=win, name='slider_line',
        size=[.8,.0],
        ori=0.0, pos=[0, 0], draggable=False, anchor='center',
        lineWidth=5.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=1.0, depth=-4.0, interpolate=True)
    marker = visual.Slider(win=win, name='marker',
        startValue=None, size=[.8,.05], pos=[0,0], units=win.units,
        labels=None, ticks=(-.4,-.3,-.2,-.1,.0,.1,.2,.3,.4), granularity=0.001,
        style='slider', styleTweaks=('labels45',), opacity=1.0,
        labelColor=None, markerColor=[-1.0000, -1.0000, -1.0000], lineColor=None, colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    slider_resp = keyboard.Keyboard(deviceName='slider_resp')
    submit_resp = keyboard.Keyboard(deviceName='submit_resp')
    
    # --- Initialize components for Routine "anticipation" ---
    anticipation_text = visual.TextStim(win=win, name='anticipation_text',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "feedback" ---
    # Run 'Begin Experiment' code from fb_code
    correct, outcome, valence = .0, .0, ''
    no_resp_text = visual.TextStim(win=win, name='no_resp_text',
        text='',
        font='Open Sans',
        pos=[0,.2], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    three_coin = visual.ImageStim(
        win=win,
        name='three_coin', 
        image='input_data/coin3.png', mask=None, anchor='center',
        ori=0.0, pos=[0, 0], draggable=False, size=[.97,.3],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    one_coin = visual.ImageStim(
        win=win,
        name='one_coin', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=[.3,.3],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    three_cross = visual.ImageStim(
        win=win,
        name='three_cross', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0, 0], draggable=False, size=[.97,.3],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    one_cross = visual.ImageStim(
        win=win,
        name='one_cross', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=[.3,.3],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    
    # --- Initialize components for Routine "block_end" ---
    block_end_text = visual.TextStim(win=win, name='block_end_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    block_end_resp = keyboard.Keyboard(deviceName='block_end_resp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=6.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "block_start" ---
        # create an object to store info about Routine block_start
        block_start = data.Routine(
            name='block_start',
            components=[block_start_text, block_start_resp],
        )
        block_start.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from block_start_code
        #print('trig10', end=', ')
        
        block_outcome, block_bonus = 0, 0
        
        disp_blockN += 1
        block_start_txt = f'Press up arrow to begin block {disp_blockN}'
        
        trial_rows = [row + (block_order[block_id]) * 40 for row in range(40)]
        block_id += 1
        block_start_text.setText(block_start_txt)
        # create starting attributes for block_start_resp
        block_start_resp.keys = []
        block_start_resp.rt = []
        _block_start_resp_allKeys = []
        # store start times for block_start
        block_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_start.tStart = globalClock.getTime(format='float')
        block_start.status = STARTED
        thisExp.addData('block_start.started', block_start.tStart)
        block_start.maxDuration = None
        # keep track of which components have finished
        block_startComponents = block_start.components
        for thisComponent in block_start.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "block_start" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        block_start.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *block_start_text* updates
            
            # if block_start_text is starting this frame...
            if block_start_text.status == NOT_STARTED and frameN >= 0.0:
                # keep track of start time/frame for later
                block_start_text.frameNStart = frameN  # exact frame index
                block_start_text.tStart = t  # local t and not account for scr refresh
                block_start_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_start_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_start_text.started')
                # update status
                block_start_text.status = STARTED
                block_start_text.setAutoDraw(True)
            
            # if block_start_text is active this frame...
            if block_start_text.status == STARTED:
                # update params
                pass
            
            # *block_start_resp* updates
            waitOnFlip = False
            
            # if block_start_resp is starting this frame...
            if block_start_resp.status == NOT_STARTED and frameN >= 0.0:
                # keep track of start time/frame for later
                block_start_resp.frameNStart = frameN  # exact frame index
                block_start_resp.tStart = t  # local t and not account for scr refresh
                block_start_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_start_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_start_resp.started')
                # update status
                block_start_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(block_start_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(block_start_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if block_start_resp.status == STARTED and not waitOnFlip:
                theseKeys = block_start_resp.getKeys(keyList=['up'], ignoreKeys=["escape"], waitRelease=False)
                _block_start_resp_allKeys.extend(theseKeys)
                if len(_block_start_resp_allKeys):
                    block_start_resp.keys = _block_start_resp_allKeys[-1].name  # just the last key pressed
                    block_start_resp.rt = _block_start_resp_allKeys[-1].rt
                    block_start_resp.duration = _block_start_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_start.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_start.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_start" ---
        for thisComponent in block_start.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_start
        block_start.tStop = globalClock.getTime(format='float')
        block_start.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block_start.stopped', block_start.tStop)
        # check responses
        if block_start_resp.keys in ['', [], None]:  # No response was made
            block_start_resp.keys = None
        blocks.addData('block_start_resp.keys',block_start_resp.keys)
        if block_start_resp.keys != None:  # we had a response
            blocks.addData('block_start_resp.rt', block_start_resp.rt)
            blocks.addData('block_start_resp.duration', block_start_resp.duration)
        # the Routine "block_start" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(
            f'input_data/trials_{expInfo["difficulty"]}.csv', 
            selection=trial_rows
        )
        , 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "baseline" ---
            # create an object to store info about Routine baseline
            baseline = data.Routine(
                name='baseline',
                components=[ISI1],
            )
            baseline.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from base_code
            base_dur = (1 + np.random.choice(np.arange(-.1, 0.11, 0.05)) ) * expInfo['frameRate']
            
            #print('trig20', end=', ')
            # store start times for baseline
            baseline.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            baseline.tStart = globalClock.getTime(format='float')
            baseline.status = STARTED
            thisExp.addData('baseline.started', baseline.tStart)
            baseline.maxDuration = None
            # keep track of which components have finished
            baselineComponents = baseline.components
            for thisComponent in baseline.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "baseline" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            baseline.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # *ISI1* period
                
                # if ISI1 is starting this frame...
                if ISI1.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    ISI1.frameNStart = frameN  # exact frame index
                    ISI1.tStart = t  # local t and not account for scr refresh
                    ISI1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ISI1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('ISI1.started', t)
                    # update status
                    ISI1.status = STARTED
                    ISI1.start(base_dur*frameDur)
                elif ISI1.status == STARTED:  # one frame should pass before updating params and completing
                    # Updating other components during *ISI1*
                    target_stim.setImage(target_file)
                    # Component updates done
                    ISI1.complete()  # finish the static period
                    ISI1.tStop = ISI1.tStart + base_dur*frameDur  # record stop time
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    baseline.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in baseline.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "baseline" ---
            for thisComponent in baseline.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for baseline
            baseline.tStop = globalClock.getTime(format='float')
            baseline.tStopRefresh = tThisFlipGlobal
            thisExp.addData('baseline.stopped', baseline.tStop)
            # the Routine "baseline" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "stim" ---
            # create an object to store info about Routine stim
            stim = data.Routine(
                name='stim',
                components=[target_stim],
            )
            stim.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from stim_code
            #print('trig30', end=', ')
            # store start times for stim
            stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            stim.tStart = globalClock.getTime(format='float')
            stim.status = STARTED
            thisExp.addData('stim.started', stim.tStart)
            stim.maxDuration = None
            # keep track of which components have finished
            stimComponents = stim.components
            for thisComponent in stim.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "stim" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            stim.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *target_stim* updates
                
                # if target_stim is starting this frame...
                if target_stim.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    target_stim.frameNStart = frameN  # exact frame index
                    target_stim.tStart = t  # local t and not account for scr refresh
                    target_stim.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_stim, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_stim.started')
                    # update status
                    target_stim.status = STARTED
                    target_stim.setAutoDraw(True)
                
                # if target_stim is active this frame...
                if target_stim.status == STARTED:
                    # update params
                    pass
                
                # if target_stim is stopping this frame...
                if target_stim.status == STARTED:
                    if frameN >= (target_stim.frameNStart + 1*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        target_stim.tStop = t  # not accounting for scr refresh
                        target_stim.tStopRefresh = tThisFlipGlobal  # on global time
                        target_stim.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'target_stim.stopped')
                        # update status
                        target_stim.status = FINISHED
                        target_stim.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    stim.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in stim.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "stim" ---
            for thisComponent in stim.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for stim
            stim.tStop = globalClock.getTime(format='float')
            stim.tStopRefresh = tThisFlipGlobal
            thisExp.addData('stim.stopped', stim.tStop)
            # the Routine "stim" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "delay" ---
            # create an object to store info about Routine delay
            delay = data.Routine(
                name='delay',
                components=[ISI2],
            )
            delay.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from delay_code
            delay_dur = ( 1.4 + np.random.choice(np.arange(-.1, 0.11, 0.05)) ) * expInfo['frameRate']
            
            #print('trig40', end=', ')
            # store start times for delay
            delay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            delay.tStart = globalClock.getTime(format='float')
            delay.status = STARTED
            thisExp.addData('delay.started', delay.tStart)
            delay.maxDuration = None
            # keep track of which components have finished
            delayComponents = delay.components
            for thisComponent in delay.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "delay" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            delay.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # *ISI2* period
                
                # if ISI2 is starting this frame...
                if ISI2.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    ISI2.frameNStart = frameN  # exact frame index
                    ISI2.tStart = t  # local t and not account for scr refresh
                    ISI2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ISI2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('ISI2.started', t)
                    # update status
                    ISI2.status = STARTED
                    ISI2.start(delay_dur*frameDur)
                elif ISI2.status == STARTED:  # one frame should pass before updating params and completing
                    # Updating other components during *ISI2*
                    img1.setImage(img1_file)
                    img2.setImage(img2_file)
                    one_coin.setImage('input_data/coin1.png')
                    three_cross.setImage('input_data/cross3.png')
                    one_cross.setImage('input_data/cross1.png')
                    # Component updates done
                    ISI2.complete()  # finish the static period
                    ISI2.tStop = ISI2.tStart + delay_dur*frameDur  # record stop time
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    delay.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in delay.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "delay" ---
            for thisComponent in delay.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for delay
            delay.tStop = globalClock.getTime(format='float')
            delay.tStopRefresh = tThisFlipGlobal
            thisExp.addData('delay.stopped', delay.tStop)
            # the Routine "delay" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "task" ---
            # create an object to store info about Routine task
            task = data.Routine(
                name='task',
                components=[img1, img2, divider_line, slider_line, marker, slider_resp, submit_resp],
            )
            task.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from task_code
            leftPressed, rightPressed, marker_moved = 0, 0, 0
            positions = []
            thrown = False
            debug_task_txt = f'Trial {trial_key+1}'
            kb.clearEvents()
            
            #print('trig50', end=', ')
            divider_line.setPos([div_pos, 0])
            marker.reset()
            # create starting attributes for slider_resp
            slider_resp.keys = []
            slider_resp.rt = []
            _slider_resp_allKeys = []
            # create starting attributes for submit_resp
            submit_resp.keys = []
            submit_resp.rt = []
            _submit_resp_allKeys = []
            # store start times for task
            task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            task.tStart = globalClock.getTime(format='float')
            task.status = STARTED
            thisExp.addData('task.started', task.tStart)
            task.maxDuration = None
            # keep track of which components have finished
            taskComponents = task.components
            for thisComponent in task.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "task" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            task.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from task_code
                # for some reason psychopy doesnt remember this assignment
                if marker.markerPos == None:
                    marker.markerPos = marker_init
                
                # FYI, by default press & release are false, only temporarily true
                if kb.getKeys(['left'], waitRelease=True, clear=True):
                # released
                    leftPressed = 0
                #    print('trig53', end=', ')
                    
                if leftPressed or kb.getKeys(['left'], waitRelease=False, clear=False):
                # pressed
                    marker_moved = 1
                    
                #    if marker_moved and not thrown:
                #        thrown = True
                #        print('trig51', end=', ')
                #    if not leftPressed:
                #        print('trig52', end=', ')
                        
                    leftPressed = 1
                    if (-.4 + marker_move) <= marker.markerPos:
                        marker.markerPos -= marker_move
                
                if kb.getKeys(['right'], waitRelease=True, clear=True):
                # released
                    rightPressed = 0
                #    print('trig55', end=', ')
                    
                if rightPressed or kb.getKeys(['right'], waitRelease=False, clear=False):
                # pressed
                
                    marker_moved = 1
                
                #    if marker_moved and not thrown:
                #        thrown = True
                #        print('trig51', end=', ')
                #    if not rightPressed:
                #        print('trig54', end=', ')
                        
                    rightPressed = 1
                    if marker.markerPos <= (.4 - marker_move):
                        marker.markerPos += marker_move   
                
                positions.append(marker.markerPos)
                    
                #if kb.getKeys(['up'], waitRelease=False, clear=False):
                #    print('trig56', end=', ')
                
                # *img1* updates
                
                # if img1 is starting this frame...
                if img1.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    img1.frameNStart = frameN  # exact frame index
                    img1.tStart = t  # local t and not account for scr refresh
                    img1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(img1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'img1.started')
                    # update status
                    img1.status = STARTED
                    img1.setAutoDraw(True)
                
                # if img1 is active this frame...
                if img1.status == STARTED:
                    # update params
                    pass
                
                # if img1 is stopping this frame...
                if img1.status == STARTED:
                    if frameN >= (img1.frameNStart + 3*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        img1.tStop = t  # not accounting for scr refresh
                        img1.tStopRefresh = tThisFlipGlobal  # on global time
                        img1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'img1.stopped')
                        # update status
                        img1.status = FINISHED
                        img1.setAutoDraw(False)
                
                # *img2* updates
                
                # if img2 is starting this frame...
                if img2.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    img2.frameNStart = frameN  # exact frame index
                    img2.tStart = t  # local t and not account for scr refresh
                    img2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(img2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'img2.started')
                    # update status
                    img2.status = STARTED
                    img2.setAutoDraw(True)
                
                # if img2 is active this frame...
                if img2.status == STARTED:
                    # update params
                    pass
                
                # if img2 is stopping this frame...
                if img2.status == STARTED:
                    if frameN >= (img2.frameNStart + 3*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        img2.tStop = t  # not accounting for scr refresh
                        img2.tStopRefresh = tThisFlipGlobal  # on global time
                        img2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'img2.stopped')
                        # update status
                        img2.status = FINISHED
                        img2.setAutoDraw(False)
                
                # *divider_line* updates
                
                # if divider_line is starting this frame...
                if divider_line.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    divider_line.frameNStart = frameN  # exact frame index
                    divider_line.tStart = t  # local t and not account for scr refresh
                    divider_line.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(divider_line, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'divider_line.started')
                    # update status
                    divider_line.status = STARTED
                    divider_line.setAutoDraw(True)
                
                # if divider_line is active this frame...
                if divider_line.status == STARTED:
                    # update params
                    pass
                
                # if divider_line is stopping this frame...
                if divider_line.status == STARTED:
                    if frameN >= (divider_line.frameNStart + 3*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        divider_line.tStop = t  # not accounting for scr refresh
                        divider_line.tStopRefresh = tThisFlipGlobal  # on global time
                        divider_line.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'divider_line.stopped')
                        # update status
                        divider_line.status = FINISHED
                        divider_line.setAutoDraw(False)
                
                # *slider_line* updates
                
                # if slider_line is starting this frame...
                if slider_line.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    slider_line.frameNStart = frameN  # exact frame index
                    slider_line.tStart = t  # local t and not account for scr refresh
                    slider_line.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(slider_line, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_line.started')
                    # update status
                    slider_line.status = STARTED
                    slider_line.setAutoDraw(True)
                
                # if slider_line is active this frame...
                if slider_line.status == STARTED:
                    # update params
                    pass
                
                # if slider_line is stopping this frame...
                if slider_line.status == STARTED:
                    if frameN >= (slider_line.frameNStart + 3*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        slider_line.tStop = t  # not accounting for scr refresh
                        slider_line.tStopRefresh = tThisFlipGlobal  # on global time
                        slider_line.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'slider_line.stopped')
                        # update status
                        slider_line.status = FINISHED
                        slider_line.setAutoDraw(False)
                
                # *marker* updates
                
                # if marker is starting this frame...
                if marker.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    marker.frameNStart = frameN  # exact frame index
                    marker.tStart = t  # local t and not account for scr refresh
                    marker.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(marker, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'marker.started')
                    # update status
                    marker.status = STARTED
                    marker.setAutoDraw(True)
                
                # if marker is active this frame...
                if marker.status == STARTED:
                    # update params
                    pass
                
                # if marker is stopping this frame...
                if marker.status == STARTED:
                    if frameN >= (marker.frameNStart + 3*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        marker.tStop = t  # not accounting for scr refresh
                        marker.tStopRefresh = tThisFlipGlobal  # on global time
                        marker.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'marker.stopped')
                        # update status
                        marker.status = FINISHED
                        marker.setAutoDraw(False)
                
                # *slider_resp* updates
                waitOnFlip = False
                
                # if slider_resp is starting this frame...
                if slider_resp.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    slider_resp.frameNStart = frameN  # exact frame index
                    slider_resp.tStart = t  # local t and not account for scr refresh
                    slider_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(slider_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_resp.started')
                    # update status
                    slider_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(slider_resp.clock.reset)  # t=0 on next screen flip
                
                # if slider_resp is stopping this frame...
                if slider_resp.status == STARTED:
                    if frameN >= (slider_resp.frameNStart + 3*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        slider_resp.tStop = t  # not accounting for scr refresh
                        slider_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        slider_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'slider_resp.stopped')
                        # update status
                        slider_resp.status = FINISHED
                        slider_resp.status = FINISHED
                if slider_resp.status == STARTED and not waitOnFlip:
                    theseKeys = slider_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                    _slider_resp_allKeys.extend(theseKeys)
                    if len(_slider_resp_allKeys):
                        slider_resp.keys = [key.name for key in _slider_resp_allKeys]  # storing all keys
                        slider_resp.rt = [key.rt for key in _slider_resp_allKeys]
                        slider_resp.duration = [key.duration for key in _slider_resp_allKeys]
                
                # *submit_resp* updates
                waitOnFlip = False
                
                # if submit_resp is starting this frame...
                if submit_resp.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    submit_resp.frameNStart = frameN  # exact frame index
                    submit_resp.tStart = t  # local t and not account for scr refresh
                    submit_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(submit_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'submit_resp.started')
                    # update status
                    submit_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(submit_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(submit_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if submit_resp is stopping this frame...
                if submit_resp.status == STARTED:
                    if frameN >= (submit_resp.frameNStart + 3*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        submit_resp.tStop = t  # not accounting for scr refresh
                        submit_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        submit_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'submit_resp.stopped')
                        # update status
                        submit_resp.status = FINISHED
                        submit_resp.status = FINISHED
                if submit_resp.status == STARTED and not waitOnFlip:
                    theseKeys = submit_resp.getKeys(keyList=['up'], ignoreKeys=["escape"], waitRelease=False)
                    _submit_resp_allKeys.extend(theseKeys)
                    if len(_submit_resp_allKeys):
                        submit_resp.keys = _submit_resp_allKeys[-1].name  # just the last key pressed
                        submit_resp.rt = _submit_resp_allKeys[-1].rt
                        submit_resp.duration = _submit_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    task.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in task.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "task" ---
            for thisComponent in task.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for task
            task.tStop = globalClock.getTime(format='float')
            task.tStopRefresh = tThisFlipGlobal
            thisExp.addData('task.stopped', task.tStop)
            # Run 'End Routine' code from task_code
            thisExp.addData('positions', positions)
            kb.clearEvents()
            trials.addData('marker.response', marker.getRating())
            trials.addData('marker.rt', marker.getRT())
            # check responses
            if slider_resp.keys in ['', [], None]:  # No response was made
                slider_resp.keys = None
            trials.addData('slider_resp.keys',slider_resp.keys)
            if slider_resp.keys != None:  # we had a response
                trials.addData('slider_resp.rt', slider_resp.rt)
                trials.addData('slider_resp.duration', slider_resp.duration)
            # check responses
            if submit_resp.keys in ['', [], None]:  # No response was made
                submit_resp.keys = None
            trials.addData('submit_resp.keys',submit_resp.keys)
            if submit_resp.keys != None:  # we had a response
                trials.addData('submit_resp.rt', submit_resp.rt)
                trials.addData('submit_resp.duration', submit_resp.duration)
            # the Routine "task" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "anticipation" ---
            # create an object to store info about Routine anticipation
            anticipation = data.Routine(
                name='anticipation',
                components=[anticipation_text],
            )
            anticipation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from anticipation_code
            #print('trig60', end=', ')
            # store start times for anticipation
            anticipation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            anticipation.tStart = globalClock.getTime(format='float')
            anticipation.status = STARTED
            thisExp.addData('anticipation.started', anticipation.tStart)
            # keep track of which components have finished
            anticipationComponents = anticipation.components
            for thisComponent in anticipation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "anticipation" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            anticipation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on frames since Routine start)
                if frameN >= .25*expInfo['frameRate']:
                    continueRoutine = False
                
                # *anticipation_text* updates
                
                # if anticipation_text is starting this frame...
                if anticipation_text.status == NOT_STARTED and frameN >= 0.0:
                    # keep track of start time/frame for later
                    anticipation_text.frameNStart = frameN  # exact frame index
                    anticipation_text.tStart = t  # local t and not account for scr refresh
                    anticipation_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(anticipation_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'anticipation_text.started')
                    # update status
                    anticipation_text.status = STARTED
                    anticipation_text.setAutoDraw(True)
                
                # if anticipation_text is active this frame...
                if anticipation_text.status == STARTED:
                    # update params
                    pass
                
                # if anticipation_text is stopping this frame...
                if anticipation_text.status == STARTED:
                    if frameN >= (anticipation_text.frameNStart + .25*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        anticipation_text.tStop = t  # not accounting for scr refresh
                        anticipation_text.tStopRefresh = tThisFlipGlobal  # on global time
                        anticipation_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'anticipation_text.stopped')
                        # update status
                        anticipation_text.status = FINISHED
                        anticipation_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    anticipation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in anticipation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "anticipation" ---
            for thisComponent in anticipation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for anticipation
            anticipation.tStop = globalClock.getTime(format='float')
            anticipation.tStopRefresh = tThisFlipGlobal
            thisExp.addData('anticipation.stopped', anticipation.tStop)
            # the Routine "anticipation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "feedback" ---
            # create an object to store info about Routine feedback
            feedback = data.Routine(
                name='feedback',
                components=[no_resp_text, three_coin, one_coin, three_cross, one_cross],
            )
            feedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from fb_code
            #print('trig70', end=', ')
            
            # C0F1 = Curve penalty, Flat reward
            if expInfo['sess_type'] in ['A','C']:
                valence = subj_C0F1_val
            elif expInfo['sess_type'] in ['B','D']:
                valence = subj_C1F0_val
                
            no_resp_txt = ''
            coin3, coin1, cross3, cross1 = 0, 0, 0, 0
            correct, outcome = 0, 0
            
            if marker_moved and submit_resp.keys is not None and\
               ( (marker.markerPos >= div_pos and target_pos >= div_pos) or
                 (marker.markerPos <= div_pos and target_pos <= div_pos) ):
                
                correct = 1
                if valence == 'rew':
                    outcome, coin3 = 3, 1
                else:
                    outcome, coin1 = 1, 1
            
                if abs(target_pos-marker.markerPos) <= .05:
                    block_bonus += 1
               
            else:
                
                correct = 0
                if valence == 'rew':
                    outcome, cross1 = -1, 1
                else:
                    outcome, cross3, = -3, 1
                
                if not marker_moved:
                    no_resp_txt = 'You must move the marker!' 
                elif submit_resp.keys is None:
                    no_resp_txt = 'Respond faster!'
                    
            block_outcome += outcome
            
            no_resp_text.setText(no_resp_txt)
            # store start times for feedback
            feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            feedback.tStart = globalClock.getTime(format='float')
            feedback.status = STARTED
            thisExp.addData('feedback.started', feedback.tStart)
            # keep track of which components have finished
            feedbackComponents = feedback.components
            for thisComponent in feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "feedback" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on frames since Routine start)
                if frameN >= 1*expInfo['frameRate']:
                    continueRoutine = False
                
                # *no_resp_text* updates
                
                # if no_resp_text is starting this frame...
                if no_resp_text.status == NOT_STARTED and frameN >= 0:
                    # keep track of start time/frame for later
                    no_resp_text.frameNStart = frameN  # exact frame index
                    no_resp_text.tStart = t  # local t and not account for scr refresh
                    no_resp_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(no_resp_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'no_resp_text.started')
                    # update status
                    no_resp_text.status = STARTED
                    no_resp_text.setAutoDraw(True)
                
                # if no_resp_text is active this frame...
                if no_resp_text.status == STARTED:
                    # update params
                    pass
                
                # if no_resp_text is stopping this frame...
                if no_resp_text.status == STARTED:
                    if frameN >= (no_resp_text.frameNStart + 1*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        no_resp_text.tStop = t  # not accounting for scr refresh
                        no_resp_text.tStopRefresh = tThisFlipGlobal  # on global time
                        no_resp_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'no_resp_text.stopped')
                        # update status
                        no_resp_text.status = FINISHED
                        no_resp_text.setAutoDraw(False)
                
                # *three_coin* updates
                
                # if three_coin is starting this frame...
                if three_coin.status == NOT_STARTED and coin3:
                    # keep track of start time/frame for later
                    three_coin.frameNStart = frameN  # exact frame index
                    three_coin.tStart = t  # local t and not account for scr refresh
                    three_coin.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(three_coin, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'three_coin.started')
                    # update status
                    three_coin.status = STARTED
                    three_coin.setAutoDraw(True)
                
                # if three_coin is active this frame...
                if three_coin.status == STARTED:
                    # update params
                    pass
                
                # if three_coin is stopping this frame...
                if three_coin.status == STARTED:
                    if frameN >= (three_coin.frameNStart + 1*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        three_coin.tStop = t  # not accounting for scr refresh
                        three_coin.tStopRefresh = tThisFlipGlobal  # on global time
                        three_coin.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'three_coin.stopped')
                        # update status
                        three_coin.status = FINISHED
                        three_coin.setAutoDraw(False)
                
                # *one_coin* updates
                
                # if one_coin is starting this frame...
                if one_coin.status == NOT_STARTED and coin1:
                    # keep track of start time/frame for later
                    one_coin.frameNStart = frameN  # exact frame index
                    one_coin.tStart = t  # local t and not account for scr refresh
                    one_coin.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(one_coin, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'one_coin.started')
                    # update status
                    one_coin.status = STARTED
                    one_coin.setAutoDraw(True)
                
                # if one_coin is active this frame...
                if one_coin.status == STARTED:
                    # update params
                    pass
                
                # if one_coin is stopping this frame...
                if one_coin.status == STARTED:
                    if frameN >= (one_coin.frameNStart + 1*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        one_coin.tStop = t  # not accounting for scr refresh
                        one_coin.tStopRefresh = tThisFlipGlobal  # on global time
                        one_coin.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'one_coin.stopped')
                        # update status
                        one_coin.status = FINISHED
                        one_coin.setAutoDraw(False)
                
                # *three_cross* updates
                
                # if three_cross is starting this frame...
                if three_cross.status == NOT_STARTED and cross3:
                    # keep track of start time/frame for later
                    three_cross.frameNStart = frameN  # exact frame index
                    three_cross.tStart = t  # local t and not account for scr refresh
                    three_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(three_cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'three_cross.started')
                    # update status
                    three_cross.status = STARTED
                    three_cross.setAutoDraw(True)
                
                # if three_cross is active this frame...
                if three_cross.status == STARTED:
                    # update params
                    pass
                
                # if three_cross is stopping this frame...
                if three_cross.status == STARTED:
                    if frameN >= (three_cross.frameNStart + 1*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        three_cross.tStop = t  # not accounting for scr refresh
                        three_cross.tStopRefresh = tThisFlipGlobal  # on global time
                        three_cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'three_cross.stopped')
                        # update status
                        three_cross.status = FINISHED
                        three_cross.setAutoDraw(False)
                
                # *one_cross* updates
                
                # if one_cross is starting this frame...
                if one_cross.status == NOT_STARTED and cross1:
                    # keep track of start time/frame for later
                    one_cross.frameNStart = frameN  # exact frame index
                    one_cross.tStart = t  # local t and not account for scr refresh
                    one_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(one_cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'one_cross.started')
                    # update status
                    one_cross.status = STARTED
                    one_cross.setAutoDraw(True)
                
                # if one_cross is active this frame...
                if one_cross.status == STARTED:
                    # update params
                    pass
                
                # if one_cross is stopping this frame...
                if one_cross.status == STARTED:
                    if frameN >= (one_cross.frameNStart + 1*expInfo['frameRate']):
                        # keep track of stop time/frame for later
                        one_cross.tStop = t  # not accounting for scr refresh
                        one_cross.tStopRefresh = tThisFlipGlobal  # on global time
                        one_cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'one_cross.stopped')
                        # update status
                        one_cross.status = FINISHED
                        one_cross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in feedback.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "feedback" ---
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for feedback
            feedback.tStop = globalClock.getTime(format='float')
            feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData('feedback.stopped', feedback.tStop)
            # Run 'End Routine' code from fb_code
            thisExp.addData('valence', valence)
            thisExp.addData('correct', correct)
            thisExp.addData('outcome', outcome)
            thisExp.addData('trial_key', trial_key)
            # the Routine "feedback" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "block_end" ---
        # create an object to store info about Routine block_end
        block_end = data.Routine(
            name='block_end',
            components=[block_end_text, block_end_resp],
        )
        block_end.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from block_end_code
        txt1 = f'You netted {block_outcome} coins.\n\n'
        txt2 = f'Based on your marker precision and speed, you gained a bonus of {block_bonus}.\n\n'
        txt3 = 'Press the up arrow to begin the next block.'
        block_end_txt = txt1 + txt2 + txt3
        
        #print('trig80', end=', ')
        block_end_text.setText(block_end_txt)
        # create starting attributes for block_end_resp
        block_end_resp.keys = []
        block_end_resp.rt = []
        _block_end_resp_allKeys = []
        # store start times for block_end
        block_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_end.tStart = globalClock.getTime(format='float')
        block_end.status = STARTED
        thisExp.addData('block_end.started', block_end.tStart)
        block_end.maxDuration = None
        # keep track of which components have finished
        block_endComponents = block_end.components
        for thisComponent in block_end.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "block_end" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        block_end.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *block_end_text* updates
            
            # if block_end_text is starting this frame...
            if block_end_text.status == NOT_STARTED and frameN >= 0.0:
                # keep track of start time/frame for later
                block_end_text.frameNStart = frameN  # exact frame index
                block_end_text.tStart = t  # local t and not account for scr refresh
                block_end_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_end_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_end_text.started')
                # update status
                block_end_text.status = STARTED
                block_end_text.setAutoDraw(True)
            
            # if block_end_text is active this frame...
            if block_end_text.status == STARTED:
                # update params
                pass
            
            # *block_end_resp* updates
            waitOnFlip = False
            
            # if block_end_resp is starting this frame...
            if block_end_resp.status == NOT_STARTED and frameN >= 0.0:
                # keep track of start time/frame for later
                block_end_resp.frameNStart = frameN  # exact frame index
                block_end_resp.tStart = t  # local t and not account for scr refresh
                block_end_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_end_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_end_resp.started')
                # update status
                block_end_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(block_end_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(block_end_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if block_end_resp.status == STARTED and not waitOnFlip:
                theseKeys = block_end_resp.getKeys(keyList=['up'], ignoreKeys=["escape"], waitRelease=False)
                _block_end_resp_allKeys.extend(theseKeys)
                if len(_block_end_resp_allKeys):
                    block_end_resp.keys = _block_end_resp_allKeys[-1].name  # just the last key pressed
                    block_end_resp.rt = _block_end_resp_allKeys[-1].rt
                    block_end_resp.duration = _block_end_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_end.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_end.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_end" ---
        for thisComponent in block_end.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_end
        block_end.tStop = globalClock.getTime(format='float')
        block_end.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block_end.stopped', block_end.tStop)
        # Run 'End Routine' code from block_end_code
        thisExp.addData('block_outcome', block_outcome)
        thisExp.addData('block_bonus', block_bonus)
        # check responses
        if block_end_resp.keys in ['', [], None]:  # No response was made
            block_end_resp.keys = None
        blocks.addData('block_end_resp.keys',block_end_resp.keys)
        if block_end_resp.keys != None:  # we had a response
            blocks.addData('block_end_resp.rt', block_end_resp.rt)
            blocks.addData('block_end_resp.duration', block_end_resp.duration)
        # the Routine "block_end" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 6.0 repeats of 'blocks'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
