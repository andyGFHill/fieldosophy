#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class for interactively visualizing 3-dimensional data.

This file is part of Fieldosophy, a toolkit for random fields.

Copyright (C) 2021 Anders Gunnar Felix Hildeman <fieldosophySPDEC@gmail.com>

This Source Code is subject to the terms of the BSD 3-Clause License.
If a copy of the license was not distributed with this file, you can obtain one at https://opensource.org/licenses/BSD-3-Clause.

"""

import numpy as np
import datetime
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class SpatiotemporalViewer(tk.Tk):
    # General class for presenting spatiotemporal data 
    
    
    def __init__(self, *args, **kwargs):
        # Initiate
        
        # Fillup with user data
        self.userdata = kwargs.get( "userdata", None )
        # Fillup with user controlers
        self.userControllers = kwargs.get( "userControllers", None )
        
        self.time = kwargs.get( "time", np.datetime64('') )        
        self.plotInstant = kwargs["plotInstant"]
        if "fig" in kwargs.keys():
            self.fig = kwargs["fig"]
        else:
            self.fig = plt.figure()
        
        tk.Tk.__init__(self)
        self.title( kwargs.get( "title", "Viewer" ) )
        
        # Setup controllers
        self.controls = self.setupWidgets( self.userControllers )
        self.controls.pack(side=tk.TOP)
        
        # Setup matplotlib canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master = self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Set exit on window exit
        self.protocol('WM_DELETE_WINDOW', self.doExit )
        
        # Set original controllers
        self.controls.slider.set(0.0)
        
        
        
        
        
    def setupWidgets( self, userControllers = None ):
        # Define buttons and other widgets
        
        controlFrame = tk.Frame(self)
        
        controlFrame.buttonBar = tk.Frame( controlFrame )
        controlFrame.buttonBar.pack( side = tk.TOP )
        
        # Create left button
        controlFrame.buttonBar.left_button = tk.Button(master = controlFrame.buttonBar, 
                             height = 2, 
                             width = 10, 
                            text = "<",
                            command = self.decreaseTime ) 
        controlFrame.buttonBar.left_button.grid(row=0,column=0)
        # Create play button
        controlFrame.buttonBar.play_button = tk.Button(master = controlFrame.buttonBar, 
                             height = 2, 
                             width = 10, 
                            text = "Play",
                            command = self.startStopAnim ) 
        controlFrame.buttonBar.play_button.grid(row=0,column=1)
        # Create right button
        controlFrame.buttonBar.right_button = tk.Button(master = controlFrame.buttonBar, 
                             height = 2, 
                             width = 10, 
                            text = ">",
                            command = self.increaseTime ) 
        controlFrame.buttonBar.right_button.grid(row=0,column=2)
        # Create step jumps
        controlFrame.buttonBar.stepLabel = tk.Label( master = controlFrame.buttonBar, text = "Step length:" )
        controlFrame.buttonBar.stepLabel.grid(row = 0, column=3)
        controlFrame.buttonBar.stepText = tk.StringVar(value="1")
        controlFrame.buttonBar.stepEntry = tk.Entry( master = controlFrame.buttonBar, textvariable = controlFrame.buttonBar.stepText, width=4 )
        controlFrame.buttonBar.stepEntry.grid(row = 0, column=4)
        
        # Create slider
        controlFrame.slider = ttk.Scale( master = controlFrame, from_ = 0.0, to_ = 1.0, command = lambda x : self.updateTime( self.getSliderInd( float(x) ) ) )
        controlFrame.slider.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Handle dates
        controlFrame.dateBar = tk.Frame( controlFrame )
        controlFrame.dateBar.pack( side = tk.TOP )
        
        controlFrame.dateBar.dateLabel = tk.Label( master = controlFrame.dateBar, text = "Date:" )
        controlFrame.dateBar.dateLabel.grid(row = 0, column=0)
        controlFrame.dateBar.dateText = tk.StringVar()
        controlFrame.dateBar.dateEntry = tk.Entry( master = controlFrame.dateBar, textvariable=controlFrame.dateBar.dateText, state='readonly' )
        controlFrame.dateBar.dateEntry.grid(row = 1, column=1)
        
        controlFrame.dateBar.timeLabel = tk.Label( master = controlFrame.dateBar, text = "Time:" )
        controlFrame.dateBar.timeLabel.grid(row = 0, column=2)
        controlFrame.dateBar.timeText = tk.StringVar()        
        controlFrame.dateBar.timeEntry = tk.Entry( master = controlFrame.dateBar, textvariable=controlFrame.dateBar.timeText, state='readonly' )
        controlFrame.dateBar.timeEntry.grid(row = 1, column=3)
        
        # If user controllers are defined
        if userControllers is not None:
            # Handle user controllers
            controlFrame.userBar = tk.Frame( controlFrame )
            controlFrame.userBar.pack( side = tk.TOP )
            
            iterItems = 0
            
            # Loop through all user-controllers
            for iterController in userControllers:
                
                iterController["controllers"] = {}
                
                if 'label' in iterController:
                    iterController["controllers"]["label"] = tk.Label( master = controlFrame.userBar, text = iterController["label"] )
                    iterController["controllers"]["label"].grid(row = 0, column=iterItems)
                    iterItems = iterItems + 1
                if iterController["type"] == "button":
                    iterController["controllers"]["button"] = tk.Button(master = controlFrame.userBar, 
                            height = 2, 
                            width = 10, 
                            text = iterController["text"],
                            command = lambda x: iterController["callback"](self, x) ) 
                    iterController["controllers"]["button"].grid(row = 0, column=iterItems)
                    iterItems = iterItems + 1
                
                if iterController["type"] == "entry":
                    iterController["controllers"]["string"] = tk.StringVar( value = iterController["text"] )
                    iterController["controllers"]["entry"] = tk.Entry( master = controlFrame.userBar, textvariable=iterController["controllers"]["string"], state= iterController["state"] )
                    iterController["controllers"]["entry"].bind('<Key-Return>', lambda x: iterController["callback"](self, x) )
                    iterController["controllers"]["entry"].grid(row = 0, column=iterItems)
                    iterItems = iterItems + 1
                
        
        
        return controlFrame
    
    
    def getSliderInd( self, x ):
        # This function retrieves the corresponding time instance of the current slider
        
        # Get index of time
        ind = int(np.round( x * (self.time.size-1) ))
        
        return ind
    
    def increaseTime( self ):
        # This function steps forward one step length
        
        # Get current slider value
        ind = self.getSliderInd( self.controls.slider.get() )
        ind = np.min( (ind+int(self.controls.buttonBar.stepText.get()), self.time.size-1) )        
        self.controls.slider.set( ind / (self.time.size-1) )
        
        return

    def decreaseTime( self ):
        # This function steps back one step length
        
        # Get current slider value
        ind = self.getSliderInd( self.controls.slider.get() )
        ind = np.max( (ind-int(self.controls.buttonBar.stepText.get()), 0) )
        self.controls.slider.set( ind / (self.time.size-1) )
        
        return

    def updateTime( self, ind ):
        # This will update the plot and time

        ts = (self.time[ind] - np.datetime64(0, 'h')) / np.timedelta64(1, 's')
        dt = datetime.datetime.utcfromtimestamp(ts)
        
        self.controls.dateBar.dateText.set( dt.strftime('%Y-%m-%d') )
        self.controls.dateBar.timeText.set( dt.strftime('%H:%M:%S') )
        
        self.plotInstantWrapper(ind)
        
        return
    
    def startStopAnim( self ):
        # This function is called when pressing the play/Stop button
        
        # If should stop animation
        if self.controls.buttonBar.play_button["text"] == "Stop":
            self.controls.buttonBar.play_button["text"] = "Play"
        else: # If should start animation
            self.controls.buttonBar.play_button["text"] = "Stop"
            self.controls.buttonBar.play_button.after(200, self.anim)
        return
    
    def anim( self ):
        # This function will update each 5:th of a second while animating
        
        # If animation is on
        if self.controls.buttonBar.play_button["text"] == "Stop":
            
            # If reached end of animation
            if self.controls.slider.get() >= 1:
                # Stop animation
#                self.startStopAnim()
                # Set slider to 0
                self.controls.slider.set( 0 )
                self.controls.buttonBar.play_button.after(200, self.anim)
            else:
                self.increaseTime()
                self.controls.buttonBar.play_button.after(200, self.anim)
        return
        
    def plotInstantWrapper(self, ind):
        # Wrapper function for calling user defined plotInstant function
        
        self.plotInstant(self, ind)
        self.canvas.draw_idle()
        
        return
    
    
    def doExit(self):
        # This function closes the window
        
        plt.close(self.fig)
        self.destroy()