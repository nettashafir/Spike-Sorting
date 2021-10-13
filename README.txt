Implemented in Aug-Sep 2021 by Netta Shafir as part of summer project in Prof. Yifat Prut in Edmond and Lily Safra 
center of Brian Science in the Hebrew University of Jerusalem. This program is a partially implementation of the 
algorithms proposed in
this paper:

Mokri Y, Salazar RF, Goodell B, Baker J, Gray CM and Yen S-C (2017) Sorting Overlapping Spike Waveforms from 
Electrode and Tetrode Recordings. Front. Neuroinform. 11:53. doi: 10.3389/fninf.2017.00053

Please send questions and notes to: netta.shafir@mail.huji.ac.il.


======================================
=====    Program description    ======
======================================
This program is a supervised method of spike sorting. The program gets true data of an extracellular record 
and the waveforms of the spikes of all of the neurons in the area that had been recorded (see details in Usage) 
and returns an estimation of the spike times of of all of the neurons in this area. If an information about the 
spikes' times is also provided, the program gives the FPR (False Positive Rate) and the FNR of the output 
relatively to the input. 


======================================
=====     Files description     ======
======================================
1. main2.py - The main module of the project. contains a function named "run" which is the basic algorithm of the 
	program. Also contain 3 functions that gives multiple analyzes about the data, usig the main function.
2. Constants.py - Is imported by all the other files in the project. Contains Magic nubers and constants that are 
	shared by all the files.Tuning parameters of the number of milliseconds that had been recorded, or the
	sampeling frequency of the electrode can be determined in this file.
3. HelperFunctions.py - Is imported by almost all the files in the project.
4. Spike.py - Contain a class named Spike, which represents an object of neuronal spike, an action potential that
	can be observed in an extracellular record.
5. SpikeFactory.py - A module in the project that responible of generating Spike objects for the program different
	needs. Contains multiple function that returns arrays of Spike objects.
6. Clustering - A module in the code that responsible for classify the different Spike objects to different
	clusters.
7. Simulator.py - A module that generate a simulated data of single-unit record, and multi-unit activity record.
8. MatlabData.py - A module that loads true data of extracellular record that saved in MatLab files. 
	Simulator.py and MatlabData.py have a similar API.
9. Visuzlization.py - A few simple functions of visualization, that prevents code duplication in the project.


==========================
=====     Usage     ======
==========================
Before using the program you should have the followings:
(1) An array of the extracellular record (can be created by the proper functions in the files Simulator.py or 
	MatlabData.py, or in any other way).
(2) Arrays of all the average spike's waveforms of the neurons in the area that was recorded, all of the same size
	and aligned by the positive peak of the spike (can be created by the proper functions in the files 
	Simulator.py or MatlabData.py, or in any other way).
(3) (Optional) Arrays of spike times of the neurons in the area that was recorded, correspondes to the arrays of 
	section 2 (can be created by the proper functions in the files Simulator.py or 
	MatlabData.py, or in any other way).
(4) Information about the millisecond that had been recorded, the frequency in kHz of the record, and the data
	before and after the peak in the spike waveforms of section (2).
In the file "Constants.py", update the relevant constants by the values in section (4) above. In addition, change 
the enum named "SingleUnits" to have number of neurons as the input, and make sure that the constants
"POINTS_BEFORE_PEAK" and "POINTS_AFTER_PEAK" is aligned with the waveforms of the input.
In the file "main2.py", use the function "run" while you send to it as an input the record of section (1) above and
a 2D array of the waveforms arrays of section (2) above. If no spike time (of section (3) above) were provided,
the function will return a 2D array of he estimated spike times of all neurons, corresponding to the waveforms
array. Otherwise, the function will print the losses (FPR and FNR) for every neuron in the area that was recorded,
and then will return a 2D array of the losses.